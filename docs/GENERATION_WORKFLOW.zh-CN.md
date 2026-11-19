# 对话式资产规划与场景生成闭环（第四轮，2026-11-12）

本文描述「对话 → 规格 → 生成 → 导入 → 组装场景 → 仿真验证」的真实连通闭环。
原则：**语言模型负责理解与建议，程序负责校验、编排与落地**。模型估计的质量/摩擦/尺度/关节轴/可抓取性
永远标记为 `model_inferred/unverified`，不会变成"已测量事实"；远程生成成功 ≠ 已导入 ≠ 可训练，
一切以真实 MuJoCo 构建与短步稳定为准。

## 1. 闭环总览

```
用户原话(数据) ──> GenerationSession(持久化, revision)
      │  对话澄清(可选, 需配置辅助模型) 或 直接表单/JSON(跳过对话)
      ▼
  AssetSpec + PhysicsSpec(字段级 provenance: source/confirmation/verification)
      ▼
  compile(确定性, 版本化) ──> CompiledGenerationRequest(prompt + provider 载荷 + hash)
      ▼  显式"确认生成"(产品交互; 模型回复不自动触发付费)
  Orchestrator ──> Lux3D 单体 / World 背景(复用既有 provider 与 GenerationJobStore)
      ▼  节点级失败恢复 / 幂等 client_request_id / 已知 remote_id 恢复
  导入校验 ──> assets/imported/<slug>/manifest.json(哈希/单位/标定/来源)
      ▼
  SceneRecipe(composed_scene | background_scene) ──> validate(真实 MuJoCo + 布局/依赖/背景检查)
      ▼  apply(expected_project_version, 原子保存)
  项目新版本 ──> 仿真会话 / 专家抓放 / 采集(带 provenance 元数据)
```

## 2. 数据模型与 provenance

- `GenerationSession`：`schema_version`、`revision`（乐观锁）、`state`
  （draft/clarifying/ready/frozen/generating/review/ready_to_apply/applied）、`messages`、`questions`（每轮≤3）、
  `branches`（重新生成=显式新分支，记录父修订与原因）、`compiled`、`recipe_id`。
- `FieldProvenance`：`source` ∈ user_explicit/project_config/measured/model_inferred/default/provider_predicted；
  `confirmation` ∈ unconfirmed/accepted；`verification` ∈ unverified/validated/measured。
  区分"用户没说"（default/unconfirmed）与"用户接受默认"（default/accepted）。
- `AssetSpec`：kind ∈ task_object/static_prop/robot_visual/environment_splat/environment_mesh；
  role ∈ visual_only/static/dynamic；尺寸以米计并带未知标记。未知尺寸 → 未标定视觉资产，标定前不绑定物理。
- `PhysicsSpec` 与 AssetSpec 分离；质量/摩擦独立 provenance。
- `SceneRecipe`：task_asset_ref、background_ref、colliders（显式代理）、asset_nodes、layout_constraints、
  scene_mode、sensor_renderer、use_default_table、versions{generation_schema, session_id, spec_revision}、manifest_hash。
- `CompiledGenerationRequest`：compiler 版本 + provider + prompt + 载荷 + hash；
  **绝不**把 mass/friction/joint_limits 传给 3D API；不伪造 SDK 字段。

## 3. 规划器与编译器边界

- 规划器仅输出 JSON（无执行代码/工具调用）；用户/参考文本视为数据（注入防御）；保留已确认硬约束；
  不承诺生成器满足尺寸/拓扑/关节。注入真实能力表 `capability_registry`。
- 未配置辅助模型服务时显示 `waiting_for_configuration`，
  表单/JSON/本地导入仍可用；**不用模板回复冒充模型**。一次受控 JSON 修复后报错并转人工编辑。
- 后端负责提供方凭据的读取与使用，模型建议由结构化字段接收和校验。
- 编译器确定性、版本化；zh↔en 数值/否定一致；载荷排除 mass_kg/friction/joint_limits/collision_proxies/seed/layout/urdf/negative_prompt。

## 4. 任务与恢复

- 状态：queued/submitting/remote_processing/downloading/importing/validating/succeeded/failed/submit_unknown。
- confirm 绑定不可变 spec_revision + compiled payload/hash + provider + project_id/config_version + client_request_id；
  重复点击同一 client_request_id 返回同一本地任务。**本地幂等不保证远程 exactly-once 计费**。
- 未知提交结果 → submit_unknown，不盲目重提；已知 remote_id → 有界退避重试，复用任务/文件哈希。
- manifest 记录源任务 id、哈希、spec/compiler 版本、单位/标定、引用、角色、校验；签名 URL 不作长期身份、不进公共日志。

## 5. 两种场景模式

- `composed_scene`：参数化工作台/地面 + 静态道具显式代理 + 可操作对象分开组装；**无需 World/标定**即可校验与应用。
- `background_scene`：World 外观 + 显式代理；高斯背景必须 `sensor_renderer=hybrid` 且
  受限 `static_background_overlay`（仅世界固定 front，wrist/运动相机拒绝）；网格背景需 mujoco + 已标定。
- 校验项：依赖图（环/缺失）、布局（非有限位姿、穿透、悬空、重复 ID）、背景/渲染器匹配、任务对象、
  真实 MuJoCo 构建 + 250 步短稳（qpos/qvel 有限、杯高/ upright）。物理不合格资产可 visual-only，dynamic 应用被阻断并解释。

## 6. API（统一 /api/v1）

| 方法 路径 | 用途 |
| --- | --- |
| GET /generation/capabilities | 能力表（provider 支持/不支持参数、aux 状态） |
| POST /generation/sessions | 建会话 |
| GET /generation/sessions[/{id}] | 列表/读取（持久化，跨刷新/重启） |
| POST /generation/sessions/{id}/messages | 对话（expected_revision，409 冲突） |
| PATCH /generation/sessions/{id}/spec | 表单/JSON 直改（跳过对话） |
| POST /generation/sessions/{id}/answer | 回答澄清问题 |
| POST /generation/sessions/{id}/compile | 编译预览（可 prompt_override） |
| POST /generation/sessions/{id}/branch | 显式新分支 |
| POST /generation/sessions/{id}/confirm | 确认生成（幂等/恢复） |
| GET /generation/sessions/{id}/tasks | 任务列表 |
| GET /generation/sessions/{id}/tasks/{jid}/binding-guide | 入库后下载资产绑定指南（Markdown） |
| POST .../tasks/{jid}/poll · /retry | 轮询 / 节点级重试 |
| POST /generation/sessions/{id}/recipe | 由会话组装 SceneRecipe |
| GET /scene-recipes[/{id}] | 配方列表/读取 |
| POST /scene-recipes/{id}/validate | 真实 MuJoCo + 布局/依赖/背景校验 |
| POST /scene-recipes/{id}/apply | 原子应用到项目（expected_project_version，409 冲突） |

复用既有 `/lux3d/generate` 与 `/generation-jobs` 做内部编排（无 HTTP 自环）。

## 7. 页面

- 资产库 / 生成工作台（对话 + 可编辑规格 + 编译预览；无 LLM 时显示规格预览，不放假 3D 画布）/
  生成任务详情 / 场景编辑器（3D 主视图；全景标"全景预览"，不做假高斯 3D 编辑）。
- 工作台底部 RecipePanel：模式/任务 ref/渲染器/默认桌/碰撞代理 JSON/目标项目 → 创建并校验 → 应用（原子）→ 读回 + 进入仿真。
- 已入库的任务可下载资产绑定指南。应用配方后可下载对应项目版本的指南，查看关节、相机和动作顺序。详见[模型接入与资产绑定指南](MODEL_INTEGRATION.zh-CN.md)。

## 8. 本轮真实验证（摘要）

- 会话 gs_b4d4bf8981e7：表单编辑（高0.09/半径0.03/质量0.12/无把手，user_explicit/accepted/unverified）→ 编译
  （compiler 1.0/lux3d/single_object，hash 6d72294ddbc1；载荷不含质量/摩擦/关节）。
- 配方 recipe_38c00ff485f7（composed_scene，workbench+toolbox 代理，任务对象 imported/lux3d-blue-live-5887561）：
  validate 通过（真实 MuJoCo，250 步，cup_z=0.045、upright=1）→ apply 到 round4_cup **v2**（读回 colliders/桌/渲染器一致）。
- 专家在 v2 上真实抓放 **success=true**（grasped/lifted/tray_contact，stable_ticks=6，tick 93）；前视帧见 evidence。
- 短采集 round4_composed_v2：**3/3 成功、279 帧、seeds 0/1/2**（新 ID，不复用旧 3/6）；manifest 含
  task_asset（mesh/manifest 哈希+来源+物理 basis）、generation_provenance（session/spec_revision/recipe/hash）、
  renderer_version、cameras、config_sha256。
- 回归：pytest 64 通过 1 跳过（torch/WSL）；生成/配方/提交组 26 通过；test_scene_commit 脚本通过；tsc + vite build 通过。

详见 [ROUND4_GENERATION_REPORT.zh-CN.md](ROUND4_GENERATION_REPORT.zh-CN.md)。

## 9. 资产与运行范围

- Lux3D 支持环境网格导入路径，满足尺寸条件时可配置组合盒代理。3DGS 背景使用单独的物理代理，hybrid 合成限世界固定的 `front`。
- 机器人外观可以进入拆件和连杆装配。运行还需要匹配的关节、执行器、相机和控制器配置。当前桌面任务与示范专家的适用范围见[平台支持范围](PLATFORM_SCOPE.zh-CN.md)。
- 对话入口需要辅助模型配置，表单、JSON 和本地导入可直接使用。
