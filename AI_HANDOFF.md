# 交接 Brief（直接给接手 AI）

> 用法：把本文件**整段**交给接手 AI，或让它读仓库根 `AI_HANDOFF.md`。本文自包含，读完即可开工；
> 深度细节见文末「参考文档」。

---

## 0. 你现在接手的是什么（先读这一段）

你接手一个叫 **RoboForge / VLA Lab** 的机器人仿真 + 学习平台（配置驱动、MuJoCo 为物理真相、
React+R3F 编辑器、FastAPI 后端、LeRobot/PyTorch 学习侧）。最近一轮（第四轮）在做
**"对话 → 规格 → 生成 → 导入 → 组装场景 → 仿真验证"** 的闭环。

**真实状态（务必以此为准，不要相信"已完成"的乐观表述）：**
- ✅ 能跑且已验证：配置驱动仿真、会话/控制权、相机/渲染、场景导入+碰撞代理+原子保存、三维编辑器、
  生成杯进场景并在 `round4_cup` v2 上专家抓放成功、短采集 3/3。
- ❌ **生成闭环尚未验收**：独立代码审查复现了 **10 项缺陷（7×P1、3×P2）**（见 §5）。
  专家/采集成功只证明"该版本杯子任务 + 手工 box 代理"，**不能**代表生成链路合格。
- 你的第一优先级：**按 §6 顺序修缺陷**，每项转成隔离回归测试；不要先加新功能。

诚实口径是硬要求：**不把 mock 当在线成功；不把估计值当测量事实；未验证能力必须标注；不复用旧成功数字。**

---

## 1. 仓库 / Git

- 本地：`C:\Users\31295\Documents\Codex\2026-11-09\https-www-aholo3d-cn-hackathon-utm\outputs\vla-lab`
- 远程：`https://github.com/PChanww/RoboForge.git`（规范地址大写），分支 `main`
- 最近 HEAD：`a33c39c`（docs）/ 业务 `acb366a`（第四轮闭环）；提交总数 11
- 提交身份必须 `PChanww <3129555017@qq.com>`（机器默认 `R_232@noreply.gitcode.com` 会被拒）；
  用 `git -c user.name=PChanww -c user.email=3129555017@qq.com commit ...`，**不要改全局 config**。
- 推送（Windows 直连被墙）：用 WSL + Clash 代理
  `wsl -d Ubuntu-22.04 -- bash -lc 'cd "/mnt/c/.../vla-lab" && git -c http.proxy=http://127.0.0.1:7897 push origin main'`
- **仅在用户明确要求时** commit/push；不要 `reset/clean/全量 checkout`；HEAD 不是唯一基线。
- 不入库项：`outputs/`（scratch 日志）、`frontend/tsconfig.tsbuildinfo`（构建产物）、`node_modules/`。

---

## 2. 运行环境（按设计拆分）

- **仿真/后端/采集/渲染**：Windows venv `..\..\work\venv\Scripts\python.exe`（Py 3.13；MuJoCo 4.0.0、FastAPI、trimesh、PIL；**无 torch**）。
- **学习/数据（仅训练）**：WSL `Ubuntu-22.04` conda `voxcpm`（Py 3.11、torch 2.6.0+cu124、lerobot 0.4.4）。
- **相机渲染需可交互桌面**：后台/detached 启动 uvicorn 会在相机预览时 GL 崩溃；用桌面窗口/前台启动。
  无桌面时 `camera_status=render_unavailable`（物理仍跑，但不能拿相机）。
- 端口：后端 8767、Vite 5173、遗留 `server.py` 8765。WSL→后端调用需 `no_proxy=127.0.0.1,localhost`。

启动：

```powershell
# 仓库根
..\..\work\venv\Scripts\python.exe -m uvicorn vlab.api:app --host 127.0.0.1 --port 8767
cd frontend
npm ci      # 首次
npm run dev # http://localhost:5173
```

环境变量（`.env.example`，**仅后端读取；真实 key 不入库/不进浏览器/日志**）：

```
AHOLO_API_KEY=            # Lux3D 单体 + World 背景（在线）
AUX_MODEL_BASE_URL=       # 对话规划 LLM（OpenAI 兼容）
AUX_MODEL_API_KEY=
AUX_MODEL_MODEL=gpt-4o-mini   # 可配置
```

未配 `AUX_MODEL_*` → 对话显示 `waiting_for_configuration`（表单/JSON/本地导入仍可用，禁止模板冒充）。
未配 `AHOLO_API_KEY` → 在线生成不可用（可复用已有 remote_id / 本地资产）。

---

## 3. 代码结构（关键项）

```
vlab/                 引擎/API/会话/调度/任务
  api.py              create_app；装配 generation_ctx
  engine.py           PlatformEnv(MuJoCo) 观测/步进/渲染/指标
  sessions.py         会话隔离 + 控制权 token/epoch/tick
  store.py            ConfigStore 版本化 + validate_config + 原子 _commit_scene(409)
  schema.py / tasks.py / kinematics.py / scene_builder.py / scene_assets.py / scene_import.py
  collect.py          专家采集（含 provenance 元数据）
  jobs.py             后台作业(collect/train/eval/export/lux3d)
  rendering/, gaussian.py, splat_render.py   hybrid 高斯渲染
vlab/generation/      第四轮闭环（缺陷集中区）
  schema.py  planner.py  compiler.py  sessions.py
  object_provider.py   ← F01/F02/F09
  orchestrator.py      ← F06/F08
  scene_recipe.py      ← F04
  api.py  capabilities.py
asset_pipeline/import_object.py   ← F05
frontend/src/components/Generation.tsx   ← F01（前端幂等键）
tests/                回归（生成组：test_generation_sessions/jobs、test_scene_recipe）
docs/                 文档 + 审阅报告
```

关键机制：`ProjectConfig` 版本化、不可变 revision；`_commit_scene(pid, mutate, expected_version)` 一次事务、409 冲突；
场景校验会真实 `mj_step` 250 步短稳；`cup_pick_place` 托盘 `TRAY_HALF≈[0.074,0.069]` **硬编码未参数化**；
高斯背景仅 `hybrid/static_background_overlay` 固定前视（无 wrist）；前端专家由 `requestAnimationFrame` 驱动，标签在后台会停摆。

---

## 4. 前端路由（定位功能）

`/`、`/projects`、`/projects/:id/sim`、`/generation`、`/generation/:sessionId`、
`/generation/:sessionId/tasks/:jobId`、`/library`、`/scenes/:id/edit`、`/robots/:id/edit`、
`/datasets`、`/training`、`/evaluations`、`/deployments`、`/settings`。

生成闭环 API 速览（`/api/v1`）：`/generation/capabilities`、`POST /generation/sessions`、
`.../{id}/{messages,answer,compile,branch,confirm,recipe}`、`PATCH .../{id}/spec`、`GET .../{id}/tasks`、
`POST .../tasks/{jid}/{poll,retry}`、`GET /scene-recipes[/{id}]`、`POST /scene-recipes/{id}/{validate,apply}`。

---

## 5. 必须修复的缺陷（独立审查已复现，10/10）

复现脚本与机器结果：`docs/review-round4-20261112/reproduce_review.py` + `results.json`。
详细说明：`docs/ROUND4_CODE_REVIEW_2026-11-12.zh-CN.md`。**修复顺序：F01/F02/F06 → F03/F07 → F04/F05 → F08/F09/F10。**

- **F01（P1）幂等非原子 + 前端不保留确认请求 ID** — `vlab/generation/object_provider.py:173`、`frontend/src/components/Generation.tsx:250`。
  并发同 `client_request_id` → 2 个本地作业、provider 提交 2 次；前端每次点击换新 UUID。
  修：幂等查找+占位在同一临界区/唯一约束；请求 ID 绑定 session+规格+编译版本并持久化到确认明确；同 ID 不同载荷拒绝；补并发/响应丢失测试。
- **F02（P1）重生成复用旧目录，任务成功却返回旧资产** — `orchestrator.py:90`、`object_provider.py:284`。
  slug 忽略作业/分支/版本，旧 manifest 存在即跳过注册，不比对 remote_id/spec_hash/hash。
  修：每次显式新生成用不可变资产 revision/唯一输出目录；仅同作业重试且来源/hash 匹配才复用；不覆盖旧目录。
- **F06（P1）确认丢弃用户审核过的提示词** — `orchestrator.py:40`。confirm 重新 `compile_session` 而非用 `session.compiled`。
  修：确认要求 compiled revision/request hash，校验规格未变后**冻结并提交审核过的 payload**；spec_hash 需覆盖 override/payload。
- **F03（P1）World 背景会话创建后读取 500** — `api.py:185`。创建时直接赋字符串 kind 不走校验，存了非法
  `environment_splat/dynamic`，GET 反序列化 500。修：创建参数一次校验后写入；背景/机器人外观设合法 role；未知类型 4xx。
- **F07（P1）澄清答案不更新规格、也不进下一次 LLM** — `api.py:248`、`planner.py:94`。answer 仅写 questions。
  修：答案持久化为用户输入交给规划器或有类型字段更新 + 校验 + provenance；支持自由文本（含尺寸）。
- **F04（P1）SceneRecipe 的 asset_nodes 未进项目/仿真，缺失资产也能校验通过** — `scene_recipe.py:151,383`。
  只映射 colliders + 单个 background_ref + task_asset_ref；节点不解析/不装载/不落地，`imported/DOES_NOT_EXIST` 仍 valid。
  修：真正的多实例场景配置 + 资源解析 + 可视网格装载 + 位姿/缩放/代理落地；未支持节点拒绝应用；
  补"生成静态道具→配方→保存读回→MuJoCo 网格/位姿存在"测试。**未修前不要把"组合生成"描述为完整交付。**
- **F05（P1）通用导入忽略宽深尺寸却标已标定/来自用户** — `asset_pipeline/import_object.py:55,70`。仅按 height 等比缩放；
  宽深只写 manifest；box 代理标 `basis=user_explicit`；实测 0.20×0.15×0.02 被缩放成 ~0.014×0.014×0.02。
  修：明确缩放策略（单轴等比 / 允许的 bbox 非等比 / 报告不满足）；未满足硬约束不得标已标定；代理来源反映真实计算；补 bbox/碰撞尺寸数值断言。
- **F08（P2）submit_unknown 无法在原作业绑定找回的 remote_id** — `orchestrator.py:48`。
  修：受校验的 recover/attach_remote，仅合适状态绑定已知 ID，保留原作业/规格/request ID/审计，不重提；类型/来源不匹配拒绝。
- **F09（P2）下载失败残留被当完整缓存** — `object_provider.py:269`。直接写 result.glb，异常留截断文件，重试不重下。
  修：`.part` + 校验后原子改名；仅复用已完成且 hash 匹配的缓存；显式重试可废弃损坏产物。
- **F10（P2）LLM 可覆盖用户已确认摩擦** — `planner.py:174`。尺寸/质量有硬约束保护，friction 没有。
  修：所有物理字段统一硬约束保护 + 候选整体校验；建议只进待接受变更。

**下一轮最小可验收演示**：对话/表单定一个托盘或静态道具规格 → 审核实际请求 → 生成或恢复任务 →
独立资产 revision 入库 → 资产节点真正进入项目配置与 MuJoCo → 核对尺寸/碰撞 → 改外观再生成，确认新旧资产并存且来源/hash 对应。

---

## 6. 立即任务清单（按顺序）

1. `git log --oneline -5`、`git status`；通读 `docs/ROUND4_CODE_REVIEW_2026-11-12.zh-CN.md`。
2. 跑基线测试（§7）确认当前绿/红。
3. 修 F01/F02/F06 → 补并发/重生成/预览一致性回归测试。
4. 修 F03/F07 → 补会话创建/答案闭环测试。
5. 修 F04/F05 → 补多实例节点落地 + 尺寸/碰撞数值测试。**这是"组合生成"真正可交付的门槛。**
6. 修 F08/F09/F10 → 补恢复/下载完整性/物理保护测试。
7. 全部绿后，再考虑在线分层验证（需密钥）与标定/托盘参数化。

---

## 7. 验证命令（仓库根，Windows sim venv）

```powershell
..\..\work\venv\Scripts\python.exe -m pytest tests/ -q
# 最近基线：64 passed, 1 skipped（test_package_isolation 需 torch/WSL）

..\..\work\venv\Scripts\python.exe -m pytest tests/test_generation_sessions.py tests/test_generation_jobs.py tests/test_scene_recipe.py -q
# 26 passed（5+10+11）

..\..\work\venv\Scripts\python.exe tests/test_scene_commit.py     # 脚本模式（无 test_ 函数）
cd frontend; npx tsc -b; npm run build                             # 构建
```

- `tests/conftest.py` 用 socketpair 垫片（端口 51200–51400）规避 Windows TestClient `WinError 10013`。
- 隐藏浏览器视口：截图失败（`NATIVE_BROWSER_VIEWPORT_UNAVAILABLE`），用无障碍快照 + API 读回取证；
  前端专家 rAF 停摆，改用脚本直接驱动 `env.servo_expert_action()`。

---

## 8. 硬约束（违反即视为不合格）

- 凭据只在后端 env 读取；不进浏览器/消息/导出/日志/截图；模型输出不得成为路径或 shell 字符串。
- 语言模型只理解/建议；**估计的质量/摩擦/尺度不是测量事实**；未知尺寸 → 未标定视觉资产，处理前不绑物理。
- MuJoCo 是物理真相；hybrid 仅固定前视、无 wrist；高斯全景预览 ≠ 3D 编辑。
- 未配 LLM/密钥时明确 `waiting_for_configuration`，禁止模板冒充模型回复。
- 不 reset/clean/覆盖他人未提交成果；不把 mock 当在线成功；不复用旧 3/6 冒充提升；不虚报未验证能力。
- 采集必须用新目录、报新 ID/seeds；不复用旧成功数字。

---

## 9. 参考文档（仓库 `docs/`）

- `ROUND4_CODE_REVIEW_2026-11-12.zh-CN.md` — 本轮 10 缺陷全文（必读）
- `review-round4-20261112/` — 复现脚本 + `results.json`
- `GENERATION_WORKFLOW.zh-CN.md` — 生成闭环工作流/数据模型/API/两种场景模式/限制
- `ROUND4_GENERATION_REPORT.zh-CN.md` — 第四轮测试/能力报告（命令、证据、跳过原因）
- `FEATURE_STATUS.zh-CN.md` — 功能状态（已实现/部分/未实现）
- `SCENE_PIPELINE|SCENE_VALIDATION|SCENE_EDITOR|RENDERING_PIPELINE*.zh-CN.md` — 场景/渲染前序
- `README.md` / `README.zh-CN.md` — 对外说明
- 更细的交接说明：`MIGRATION_HANDOFF.zh-CN.md`

第四轮真实证据（配方校验/专家帧/采集 3/3）在 Qoder 工作区 `evidence/`，**不在仓库内**，需要时请用户提供。

---

> 接手第一步就做两件事：跑 §7 全量测试确认基线；按 §5 顺序开修 F01。其它先别动。
