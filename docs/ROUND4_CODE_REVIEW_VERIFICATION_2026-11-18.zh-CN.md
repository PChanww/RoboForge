# 第四轮审查复审结论（对当前 HEAD 实跑，2026-11-18）

对象：仓库 `vla-lab` 当前工作区（复审时业务提交为 `acb366a`；其后 main 上另有并发的 SO-ARM100/SmolVLA 提交，
工作区的 `vlab/engine.py`、`vlab/tasks.py`、`assets/*cup-web*` 为并发线未提交改动，非本次审查所改）。
方法：逐条精读源码 + 直接运行审阅自带的复现脚本
`docs/review-round4-20261112/reproduce_review.py`（原位置 `outputs/review-round4-20261112/`，路径正确）。

## 结论

**`docs/ROUND4_CODE_REVIEW_2026-11-12.zh-CN.md` 的 10 项缺陷在当前代码上已全部不可复现（10/10）。**
该审阅针对的是更早的代码快照（其行号如 `object_provider.py:173` 现在指向 mock 下载处）。当前实现不仅修复了缺陷，
多实例场景节点（F04）是真落地的（进入 MuJoCo 体/碰撞并做短步检查），不是只做映射。

复现结果（脚本输出）：`preview_discarded / regeneration_reuses_old_asset / concurrent_duplicate_remote_submits /
width_depth_ignored / world_session_unreadable / clarification_answer_lost / asset_nodes_not_applied /
unknown_submit_cannot_attach_remote / download_retry_reuses_partial / confirmed_friction_overwritten`
—— **全部 `reproduced=false`**（其中重生成项现被"规格已变更需重新编译"的 `CompileError` 拦截）。

> 说明：仓库内 `docs/review-round4-20261112/` 的副本因放在 `docs/` 下，相对路径解析会找不到 `assets/`、`configs/`；
> 请以原位置脚本或从仓库根加 `PYTHONPATH=.` 运行（本次即如此）。

## 逐条核对（缺陷 → 当前修复位置与做法）

- **F01 幂等非原子 / 前端不保留确认 ID — 已修**
  `vlab/generation/object_provider.py:274-296`：`find_by_request`+占位创建+落盘在同一把 `RLock` 内（intent-before-submit），
  并发同 `client_request_id` 只建 1 个本地任务；绑定载荷不同则拒绝（`:277-278`）。
  前端 `frontend/src/components/Generation.tsx:383-402`：以 `bindingKey`（session+冻结规格+request_hash）把
  `client_request_id` 存 `localStorage`，重试复用、成功后清除；并传 `expected_compiled_revision`/`expected_request_hash`（`:396-397`）。
- **F02 重生成复用旧目录/旧资产 — 已修**
  `orchestrator.py:186-191`：slug=会话后缀+`r{spec_revision}-n{seq}-{uuid6}`，每次显式生成唯一目录；
  `object_provider.py:515-536`：导入遇已存在 manifest 时比对 `generation_id/spec_hash/sha256_glb/job_id`，不一致则另分配 slug 或拒绝复用。
- **F06 确认丢弃审核提示词 — 已修**
  `orchestrator.py:37-63` 用 `session.compiled`，校验 `base_spec_hash`/`request_hash`/provider，且 `validate_prompt_against_spec` 拦截与硬约束冲突；
  `compiler.py:41-52` 的 `compiled_request_hash` 覆盖 `prompt_en`(含 override)+`payload`+`base_spec_hash`。
- **F03 World 会话创建后读取 500 — 已修**
  `generation/api.py:358-391`：建会话先校验 kind∈允许集、给合法默认 role、`AssetSpec`/`PhysicsSpec` 双重 `model_validate`，
  未知 kind 返回 400；不再写入非法的 `environment_splat/dynamic`。
- **F07 澄清答案不更新规格/不进 LLM — 已修**
  `generation/api.py:239-294` 把问题映射为字段并把答案解析为长度/质量/布尔/摩擦/外观，
  `:443-499` 经 `apply_user_patch` 写回规格（user_explicit）；`planner.py:97-102` 把 questions/answers 作为数据送入 LLM。
- **F04 asset_nodes 不进项目/仿真、缺失资产仍通过 — 已修（真落地）**
  `scene_recipe.py:272-334` 解析节点、拒绝未标定 static/dynamic、静态节点自动生成代理；
  `scene_builder.py:39-112,238-265` 的 `resolve_asset_node`/`environment_xml` 把节点建成 MuJoCo 体（dynamic 为 freejoint+碰撞 geom），缺失/越界资产抛错；
  `validate_recipe` 真实构建 MuJoCo 并 `_short_step_stability` 检查 dynamic 节点是否在支撑区（`scene_recipe.py:659-668`）；
  tray 经 `goal_tray_node_id` 参数化到 `task.goal_*`（`:360-383`）。
- **F05 宽深尺寸被忽略却标已标定 — 已修**
  `asset_pipeline/import_object.py:97-112` 宽/深/高各向异性缩放 + 轴对齐（`:87-90`）+ 浅平面自动找平（`:81-83`）；
  `:133-150` 计算 `unsatisfied_dimensions`，未满足则不 `calibrated`；`:172` 未标定强制降级 `visual_only`；
  `:152-163` `proxy_basis` 仅在全部显式且满足时为 `user_explicit`，否则 `proxy_estimate`。
- **F08 submit_unknown 不能绑定找回 remote_id — 已修**
  `object_provider.py:321-377` `attach_remote()`（状态/活跃/provider/重复绑定/审计校验，不重提）；`orchestrator.recover`/`import_remote`（`:225-240,262-323`）；
  API `recover`/`import-remote`（`generation/api.py:580-614`）。
- **F09 下载残留被当完整缓存 — 已修**
  `object_provider.py:417-444` 仅当 `nodes.download==done` 且 sha256 匹配且能通过 `validate_mesh_file` 才复用；
  下载前清残留 `result.*`/`.part`；`force_redownload` 清哈希/路径/下游节点后重下（`:467-489`）。
- **F10 LLM 覆盖用户已确认摩擦 — 已修**
  `planner.py:209-219` friction 有硬约束保护；mass/graspable 同（`:193-208,220-230`）；物理更新统一经 `_try_set_physics` 对整份 `PhysicsSpec` 重新校验（`:181-190`）。

## 我额外审到的残留（低危，非验收阻断）

- **API 边界可跳过"审核预览"**：前端强制传 `expected_compiled_revision/expected_request_hash`；但 `orchestrator._resolve_compiled`
  在无 `compiled` 且未传 expected 时会即时编译并提交。若要求 API 层也强制"先预览再确认"，需把 expected_* 设为必填。
- **同会话长耗时阻塞**：`post_message`/`answer` 在持有 `sessions.session_lock(sid)` 期间同步调用规划器（超时 60s），会阻塞该会话其它操作。
- **`apply_reply` 直接改 `physics_spec.role`**（`planner.py:179`）未重新整体校验（其余物理字段已统一校验）。
- **`ObjectJobStore.__init__`** 只把 `submitting` 归为 `submit_unknown`；`downloading/importing/validating` 残留靠 `poll/retry` 恢复（可接受，建议文档化）。
- 命名：`CompiledGenerationRequest.spec_hash` 实际存的是 `request_hash`（在 `base_spec_hash` 存在时无影响），易混淆。

## 未验证（本轮复审未覆盖，需在线/真实栈）

- 真实 LLM 对话与在线新建付费生成（无 `AUX_MODEL_*` / `AHOLO_API_KEY`）——未验证。
- 真实 Lux3D 下载路径（复现用 mock 生成 glb）；hybrid + SceneRecipe 组合；新 tray/goal 参数的前端可视化。
- 上述残留项未改写；如需可专项处理。

## 与旧审阅文档的关系

`docs/ROUND4_CODE_REVIEW_2026-11-12.zh-CN.md` 作为历史记录保留（反映旧快照的问题与复现方法）。
本文件是其**对当前 HEAD 的复核结论**：缺陷已修、复现脚本 10/10 不再复现。两者并存，勿混淆。