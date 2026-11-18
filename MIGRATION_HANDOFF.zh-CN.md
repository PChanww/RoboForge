# 迁移交接文档 · RoboForge / VLA Lab

> 面向接手的下一位 agent。本文件独立成篇：读完即可接手，无需回溯聊天记录。
> 最近更新：2026-11-12（第四轮交付 + 独立代码审查）。语言：中文。

---

## 0. 一句话现状（务必先读）

- **可用且真实验证**：配置驱动仿真（MuJoCo 为准）、会话/控制权、相机/渲染、场景导入与碰撞代理、三维编辑器、
  生成杯 `imported/lux3d-blue-live-5887561` 组装进场景并在 `round4_cup` v2 上**专家抓放成功**、短采集 3/3。
- **生成闭环主体已实现（缺陷已修）**：2026-11-12 独立代码审查报告了 10 项缺陷（7×P1、3×P2）；**对当前 HEAD(acb366a)
  复审时用其自带复现脚本实跑，10/10 不再复现**，并逐条定位到修复实现（见 `docs/ROUND4_CODE_REVIEW_VERIFICATION_2026-11-18.zh-CN.md`）。
  其中 F04（多实例场景节点）为真落地（进入 MuJoCo 体/碰撞 + 短步检查）。
- **接手第一优先级**：跑基线测试 + 复核复现脚本（确认 10/10 不复现）+ 在线分层验证（需密钥）+ 处理少量低危残留；**不要重做已修项**。
- **仍未在线验证**：真实 LLM 对话、在线新建付费生成（无 `AUX_MODEL_*`/`AHOLO_API_KEY`）。

诚实口径：不把 mock 当在线成功；不把估计值当测量事实；不宣称已实现未验证的能力。

---

## 1. 仓库与 Git 信息

| 项 | 值 |
| --- | --- |
| 本地路径 | `C:\Users\31295\Documents\Codex\2026-11-09\https-www-aholo3d-cn-hackathon-utm\outputs\vla-lab` |
| 远程 | `https://github.com/Pchanww/RoboForge.git`（GitHub 规范地址 `https://github.com/PChanww/RoboForge.git`，旧 URL 自动重定向） |
| 分支 | `main` |
| 当前 HEAD | `acb366a`（2026-11-12，作者/提交者 `PChanww <3129555017@qq.com>`） |
| 提交总数 | 11 |
| 未提交（交接时） | `docs/ROUND4_CODE_REVIEW_2026-11-12.zh-CN.md`、`docs/review-round4-20261112/`（若尚未入库）、`docs/MIGRATION_HANDOFF.zh-CN.md`（本文件）；`outputs/`、`frontend/tsconfig.tsbuildinfo` 为故意不入库的 scratch/构建产物 |

提交历史（最新在上）：

```
acb366a Round 4: conversational generation loop + scene recipes; land prior uncommitted work
4affcac Scene pipeline: local import, static colliders, scene-in-RGB, parameterized loop
a8a506b P2: reproducible visual BC policy, multi-robot selection, deploy compat checks
1d8ae54 P1: 3D-main workspace layout, multi-page router, web background jobs
3fef1dd P0 correctness: bounded camera latency queue, config-driven IK, control epoch
ff3b51b Create legacy lab lazily to avoid background GLFW segfault
fd33578 Add closed-loop position-servo expert; imitation now succeeds closed-loop
85ea52d docs: add bilingual READMEs and separate audit reports
de58ca1 Rewrite README intro as a project overview for RoboForge
e6a2f27 Add data collection, LeRobot training pipeline, deployment package, second robot
e08f161 Add configurable robot simulation, data collection and training platform
```

**推送方式（非显而易见，已验证）**：Windows 直连 `github.com:443` 被墙，且 Windows git 无凭据；
用 WSL（`Ubuntu-22.04`，镜像网络可访问 `127.0.0.1:7897`）配合 Clash 代理推送：

```bash
wsl -d Ubuntu-22.04 -- bash -lc 'cd "/mnt/c/.../vla-lab" && git -c http.proxy=http://127.0.0.1:7897 push origin main'
```

**提交身份**：必须 `PChanww <3129555017@qq.com>`（机器默认是 `R_232@noreply.gitcode.com`，会被拒）。
用 `git -c user.name=PChanww -c user.email=3129555017@qq.com commit ...`，不要改全局 config。
**不要** `reset/clean/全量 checkout`；HEAD 不是唯一基线，曾有大块未提交成果。

---

## 2. 环境与快速启动

拆分为两套环境（按设计）：

- **仿真/后端/渲染/采集**：Windows `..\..\work\venv`（Python 3.13；MuJoCo 4.0.0、FastAPI、trimesh、PIL、numpy；**无 torch**）。
- **学习/数据（仅训练相关）**：WSL `~/miniconda3/envs/voxcpm`（Py 3.11，torch 2.6.0+cu124，lerobot 0.4.4，transformers 4.57.3），
  `PYTHONPATH=<workspace>/work/policy/deps`。视觉 BC 训练/转换在此环境。

> MuJoCo 相机渲染需要可交互桌面（GLFW 上下文）。**后台/detached 启动 uvicorn 会在相机预览时 GL 崩溃**：
> 后端要在桌面窗口里启动（普通终端或 `Start-Process -WindowStyle Minimized`）。前台测试/采集脚本渲染正常。
> 无桌面时退化为 `camera_status=render_unavailable`（物理仍运行）。

启动（在 `vla-lab`）：

```powershell
# 后端（桌面窗口）
..\..\work\venv\Scripts\python.exe -m uvicorn vlab.api:app --host 127.0.0.1 --port 8767

# 前端（另一个终端）
cd frontend
npm ci      # 首次
npm run dev # http://localhost:5173 ，Vite 代理 /api 与 /assets 到 8767
```

新环境从零搭建：

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn vlab.api:app --host 127.0.0.1 --port 8767
# 可选：World SDK 额外依赖
python -m pip install -r requirements-world.txt
```

端口：新后端 `8767`，Vite `5173`，遗留 `server.py` `8765`。
WSL→Windows 后端调用需 `no_proxy=127.0.0.1,localhost`（Clash 会拦截本地）。

**环境变量**（`.env.example` 模板，仅后端读取；真实 key 不入库/不进浏览器/日志）：

```
AHOLO_API_KEY=            # Lux3D 单体 + World 背景生成（在线）
AUX_MODEL_BASE_URL=       # 对话规划（LLM），OpenAI 兼容
AUX_MODEL_API_KEY=
AUX_MODEL_MODEL=gpt-4o-mini   # 可配置，非硬编码
VLAB_WSL_DISTRO=Ubuntu-22.04
VLAB_LINUX_LAB=/mnt/c/.../vla-lab
VLAB_LINUX_DEPS=/mnt/c/.../work/policy/deps
VLAB_LINUX_PYTHON=/root/miniconda3/envs/voxcpm/bin/python
```

未配置 `AUX_MODEL_*` 时对话显示 `waiting_for_configuration`（表单/JSON/本地导入仍可用，禁止模板冒充模型）。
未配置 `AHOLO_API_KEY` 时在线生成不可用（可复用已有 remote_id 或本地资产）。

---

## 3. 项目结构

```text
vlab/                 核心：配置驱动引擎、API、会话、调度、任务、渲染封装
  api.py              FastAPI 应用（create_app）；装配 generation_ctx
  engine.py           PlatformEnv（MuJoCo）观测/步进/渲染/指标
  sessions.py         会话隔离 + 控制权 token/epoch/tick
  store.py            ConfigStore 版本化项目；validate_config；原子 _commit_scene
  schema.py           Pydantic 项目/场景/机器人配置
  tasks.py            cup_pick_place 任务（托盘/判定硬编码于插件）
  kinematics.py       配置驱动 FK/IK
  scene_builder.py    MuJoCo 环境 XML 组装
  scene_assets.py     mesh/gaussian 资产加载
  scene_import.py     GLB/PLY 分类导入
  world_provider.py   World 场景生成 provider（复用）
  gaussian.py / splat_render.py / rendering/  高斯解析与 CPU 光栅化、hybrid 渲染
  collect.py          专家采集（含 provenance 元数据）
  jobs.py             后台作业（collect/train/eval/export/lux3d 等）
  generation/         第四轮：对话式生成闭环（见 §5、§7）
vlab/generation/
  schema.py            GenerationSession/AssetSpec/PhysicsSpec/SceneRecipe/... + provenance
  planner.py           LLM 规划器（JSON only；注入能力表；注入防御）
  compiler.py          确定性编译预览（排除物理字段）
  sessions.py          会话持久化（SQLite WAL + 乐观 revision）
  object_provider.py   Lux3D 单体 provider + 作业存储（幂等/恢复）  ← 审阅 F01/F02/F09
  orchestrator.py      confirm 编排/节点恢复                         ← 审阅 F06/F08
  capabilities.py      能力表
  scene_recipe.py      SceneRecipe 映射/校验/原子应用                ← 审阅 F04
  api.py               /api/v1/generation/* 与 /api/v1/scene-recipes/*
frontend/             React 18 + React Router + React Three Fiber（Vite）
  src/components/Generation.tsx   工作台 + 任务详情 + RecipePanel
  src/components/{Scene,SceneCanvas,pages}.tsx  编辑器/页面
asset_pipeline/       生成/导入对象与场景（import_object.py 等）      ← 审阅 F05
configs/              default.json、second_robot.json
assets/               robot/、imported/（生成杯 8.7M）、scenes/（world PLY 14M）
datasets/             collections/、models/（.pt）、reports/（.json）
scripts/              采集/转换/训练/评测/导出/渲染工具
tests/                引擎/API/渲染/场景/生成 回归
docs/                 文档与审查报告（见下）
render-service/       Node 高斯渲染服务（Spark/Three；node_modules 已被 ignore）
```

前端路由：`/`、`/projects`、`/projects/:id/sim`、`/generation`、`/generation/:sessionId`、
`/generation/:sessionId/tasks/:jobId`、`/library`、`/library/:assetId`、`/scenes/:id/edit`、
`/robots/:id/edit`、`/datasets`、`/training`、`/evaluations`、`/deployments`、`/settings`。

关键文档索引：

- `docs/GENERATION_WORKFLOW.zh-CN.md` — 生成闭环工作流（数据模型/API/两种场景模式/限制）
- `docs/ROUND4_GENERATION_REPORT.zh-CN.md` — 第四轮测试/能力报告（命令、证据、跳过原因）
- `docs/ROUND4_CODE_REVIEW_2026-11-12.zh-CN.md` — **独立代码审查（10 缺陷，接手必读）**
- `docs/review-round4-20261112/` — 审阅复现脚本 `reproduce_review.py` + `results.json`（10/10 复现）
- `docs/FEATURE_STATUS.zh-CN.md` — 功能状态（已实现/部分/未实现）
- `docs/SCENE_PIPELINE\|SCENE_VALIDATION\|SCENE_EDITOR\|RENDERING_PIPELINE*.zh-CN.md` — 场景/渲染前序
- `README.md` / `README.zh-CN.md` — 对外说明（英/中）

---

## 4. 架构要点（接手需知）

- **配置驱动**：`ProjectConfig` 版本化；保存生成不可变 revision；`_commit_scene(pid, mutate, expected_version)`
  一次事务提交整份配置，`expected_version` 不符返回 409；`validate_config` 会真实构建 PlatformEnv 并做 5R IK 可达性。
- **物理真相是 MuJoCo**：场景校验会真实 `mj_step` 短步稳定性（250 步）并检查 qpos/qvel 有限、杯高/upright。
- **任务对象解析**：`tasks.asset_params(asset, lab_root)`；`imported/<slug>` → `assets/imported/<slug>/manifest.json`，
  需 `radius_m/height_m/mass_kg`、`obj`、`color_rgb[3]`。BUILTINS=`standard/tall/wide`。
- **托盘与判定硬编码**：`cup_pick_place` 插件内 `TRAY_HALF≈[0.074,0.069]`，**未参数化**；用户自定义托盘尺寸目前不被任务采用。
- **场景渲染**：`sensor_renderer=mujoco`（网格）或 `hybrid`（高斯背景，受限 `static_background_overlay`，仅世界固定 front，
  wrist/运动相机拒绝）。高斯背景每环境构建约 10–19s（CPU），需进程级缓存，避免在请求线程反复重建。
- **前端专家循环由 requestAnimationFrame 驱动**：浏览器标签不在前台时 rAF 被节流，专家循环会停摆。取证可改用
  后台脚本直接驱动 `env.servo_expert_action()`（见 `evidence` 方式或 §9）。

---

## 5. 功能状态

### 5.1 已实现且真实验证

- 项目配置版本化、会话隔离、控制权 token/epoch/tick、动作块预校验（`test_control/test_api`）。
- 5 旋转关节臂 + 笛卡尔龙门；配置驱动 IK；不可达结构化（`test_kinematics`）。
- 相机有界缓冲/延迟/图像年龄（`test_camera_latency`）、不可变快照（`test_gaussian_snapshot`）、数值投影 ≤0.85px（`test_camera_projection`）。
- World 高斯 PLY 解析 + CPU 光栅化 + 协方差投影（`test_covariance_camera`），解析盒遮挡/半透明（`test_occlusion_translucency`）。
- hybrid 进 RGB（受限模式）、前景分割 objid/objtype（`test_segmentation`）、跨会话隔离（`test_hybrid_sessions`）。
- 场景本地导入、实例+碰撞代理、原子保存 + 409（`test_scene_commit/test_scene_editor`）。
- 三维编辑器（gizmo/世界变换/撤销重做/测量/保存）、机器人独立编辑页（浏览器实跑）。
- 视觉-only 对照（同物理仅换渲染）、视觉模型包导出（自包含+SHA+契约+runner）、作业配置快照、障碍阻挡。
- **第四轮**：生成会话持久化 + 表单/JSON 直改 + 字段级 provenance + 编译预览（`test_generation_sessions` 5）；
  SceneRecipe `composed_scene` 真实 MuJoCo 校验 + 原子 apply + 读回（`test_scene_recipe` 11）；
  生成杯进场景、`round4_cup` v2 专家抓放 `success=true`、短采集 `round4_composed_v2` 3/3（seeds 0/1/2，带 provenance）。

### 5.2 部分实现（含缺陷，见 §7）

- 生成闭环：幂等/重生成/预览一致性/World 会话创建/澄清答案/多实例场景/尺寸约束/恢复/物理保护 存在 §7 缺陷。
- 场景标定：地面平移对齐，水平/尺度未精化；hybrid 背景与工作区未完全对齐。
- 部署：视觉包可加载运行；通用 PolicyAdapter 仅封装 VisualBC。
- 异步渲染：当前帧同步渲染；异步延迟完成/丢弃未实现。

### 5.3 未实现 / 未验证

- 真实 LLM 对话（未配置 `AUX_MODEL_*`）→ **未验证**。
- 真实在线新建付费生成（无 `AHOLO_API_KEY`）→ **未验证**；本轮复用既有 remote_id/本地资产。
- 多实例场景资产节点真正进入项目/仿真（审阅 F04 的核心缺口）。
- `environment_mesh` 在线生成、SDK `part_split/auto_rig`、六轴臂物理机器人（仅 `robot_visual` 路线）、
  hybrid wrist/运动相机、自动凸分解/碰撞提取、sim2real。

---

## 6. 本轮（第四轮）改动说明

新增（新文件）：

- `vlab/generation/`：`schema` `planner` `compiler` `sessions` `object_provider` `orchestrator` `capabilities`
  `scene_recipe` `api` `__init__`。
- `tests/test_generation_sessions.py`(5)、`tests/test_generation_jobs.py`(10)、`tests/test_scene_recipe.py`(11)。
- `frontend/src/components/Generation.tsx`（工作台 + GenerationTaskDetail + RecipePanel）。
- `.env.example`、`docs/GENERATION_WORKFLOW.zh-CN.md`、`docs/ROUND4_GENERATION_REPORT.zh-CN.md`、本文件。
- `docs/ROUND4_CODE_REVIEW_2026-11-12.zh-CN.md` + `docs/review-round4-20261112/`（独立审阅产物）。

修改：

- `vlab/api.py`（装配 generation/scene-recipes 路由）、`vlab/collect.py`（采集 manifest 增补 task_asset 哈希 +
  generation_provenance）、`vlab/jobs.py`（collect 作业传 provenance）、`scripts/collect_expert.py`（`--provenance`）。
- `frontend/src/{App.tsx,api.ts}`（任务详情路由、`genPoll`、`recipeValidate/recipeApply`）。
- `README.md`/`README.zh-CN.md`、`docs/FEATURE_STATUS.zh-CN.md`。

提交 `acb366a` 同时落入了此前**未提交**的场景/渲染/视觉 BC 成果与真实资产，以保证提交树自洽可运行。

---

## 7. 历史审查的缺陷（当前 HEAD 已全部修复，保留作回归/背景）

来源：`docs/ROUND4_CODE_REVIEW_2026-11-12.zh-CN.md`（旧快照）。**复审（2026-11-18，HEAD acb366a）**：用其自带复现脚本实跑，
**10/10 不再复现**，逐条修复实现见 `docs/ROUND4_CODE_REVIEW_VERIFICATION_2026-11-18.zh-CN.md`。
下面保留为"回归要点/历史背景"，**不要**按"未修"处理。仅存低危残留（API 边界可跳过预览、同会话 LLM 调用阻塞、
`apply_reply` role 未整体复校等）见复审文档。

- **F01（P1）幂等非原子 + 前端不保留确认请求 ID** — `object_provider.py:173`、`Generation.tsx:250`。
  并发同 `client_request_id` 创建 2 个本地作业、provider 提交 2 次；前端每次点击换新 UUID。
  修：幂等查找+占位同一临界区/唯一约束；请求 ID 绑定 session+规格+编译版本并持久化到确认明确；同 ID 不同载荷拒绝；补并发/响应丢失测试。
- **F02（P1）重生成复用旧目录，第二个任务成功却返回旧资产** — `orchestrator.py:90`、`object_provider.py:284`。
  slug 忽略作业/分支/版本，旧 manifest 存在即跳过注册，不比对 remote_id/spec_hash/hash。修：每次显式新生成用不可变 revision/唯一输出目录；
  仅同作业重试且来源/hash 匹配才复用。
- **F06（P1）确认丢弃用户审核过的提示词** — `orchestrator.py:40`。confirm 重新 `compile_session` 而非用 `session.compiled`。
  修：确认要求 compiled revision/request hash，校验规格未变后冻结并提交审核过的 payload。
- **F03（P1）World 背景会话创建后读取 500** — `api.py:185`。创建时直接赋字符串 kind 不走校验，存了非法的
  `environment_splat/dynamic`，GET 反序列化 500。修：创建参数一次校验后写入，背景/机器人外观设合法 role。
- **F07（P1）澄清问题答案不更新规格也不进下一次 LLM** — `api.py:248`、`planner.py:94`。answer 仅写 questions。
  修：答案持久化为用户输入交给规划器或有类型字段更新；支持自由文本（含尺寸）。
- **F04（P1）SceneRecipe 的 asset_nodes 未进项目/仿真，缺失资产也校验通过** — `scene_recipe.py:151,383`。
  只映射 colliders + 单个 background_ref + task_asset_ref；节点不解析/不装载/不落地。修：真正的多实例场景配置 + 资源解析 +
  可视网格装载 + 位姿/缩放落地；未支持节点拒绝应用；补"生成静态道具→配方→读回→MuJoCo 网格/位姿存在"测试。
- **F05（P1）通用导入忽略宽深尺寸却标已标定/来自用户** — `asset_pipeline/import_object.py:55,70`。仅按 height 等比缩放；
  宽深只写 manifest；box 代理标 `basis=user_explicit`。修：明确缩放策略（单轴 / 允许的 bbox 非等比 / 报告不满足）；未满足硬约束不得标已标定；
  代理来源反映真实计算；补 bbox 与碰撞尺寸数值断言。
- **F08（P2）submit_unknown 无法在原作业绑定找回的 remote_id** — `orchestrator.py:48`。修：受校验的 recover/attach_remote，
  仅合适状态绑定已知 ID，保留原作业/规格/request ID/审计，不重提。
- **F09（P2）下载失败残留被当完整缓存** — `object_provider.py:269`。修：`.part` + 校验后原子改名；仅复用已完成且 hash 匹配；
  显式重试可废弃损坏产物。
- **F10（P2）LLM 可覆盖用户已确认摩擦** — `planner.py:174`。修：所有物理字段统一硬约束保护 + 候选整体校验；建议只进待接受变更。

审查建议的**下一轮最小可验收演示**：对话或表单定一个托盘/静态道具规格 → 审核实际请求 → 生成或恢复任务 →
独立资产 revision 入库 → 资产节点真正进入项目配置与 MuJoCo → 核对尺寸/碰撞 → 改外观再生成，确认新旧资产并存且来源/hash 对应。

---

## 8. 使用教程（端到端）

### 8.1 网页闭环（推荐手动体验）

1. 启动后端(桌面窗口) + 前端，打开 `http://localhost:5173`。
2. **生成**页：新建会话 → （无 LLM 时用右侧表单/JSON 直改）填规格 → 看编译预览（提示词/载荷/警告）。
3. **确认生成**：显式点击；可填"恢复已有远程任务 ID"复用（避免新建付费）。*注意 §7 F01/F02/F06 风险。*
4. 会话底部 **RecipePanel**：选场景模式、任务对象 ref、渲染器、默认桌、碰撞代理 JSON、目标项目 →
   **创建配方并校验**（真实 MuJoCo）→ **应用到项目**（原子，409 冲突）→ 读回 + **进入仿真**。
5. 仿真页 **运行专家** 验证抓放；或到 **数据集** 页发起采集作业。
6. 资产/场景/机器人编辑页可改配置并保存新版本；训练/评测/部署沿用既有流程。

### 8.2 API 速览（`/api/v1`，统一；`/docs` 有 schema）

```
GET  /capabilities                         引擎/机器人/控制器能力
GET|PUT /config?project=  /config/versions 配置读取/保存/版本
POST /sessions ; /sessions/{sid}/{reset,control/acquire,action,action_block,result,frame.jpg}
GET  /generation/capabilities
POST /generation/sessions ; GET /generation/sessions[/{id}]
POST /generation/sessions/{id}/messages|answer|compile|branch|confirm
PATCH /generation/sessions/{id}/spec
GET  /generation/sessions/{id}/tasks ; POST .../tasks/{jid}/poll|retry
POST /generation/sessions/{id}/recipe
GET  /scene-recipes[/{id}] ; POST /scene-recipes/{id}/validate|apply?expected_project_version=
POST /lux3d/generate ; POST|GET /generation-jobs          （内部编排复用，无 HTTP 自环）
POST /scene-assets/import ; /projects/{id}/bind-scene
POST /jobs (collect/train/eval/export ...)
```

### 8.3 采集（带 provenance）

```powershell
# 先保存目标项目 config 快照（如 round4_cup v2），准备 plan 与 provenance JSON
..\..\work\venv\Scripts\python.exe scripts/collect_expert.py \
  --config <config_v2.json> --plan <plan.json> --provenance <prov.json> \
  --out datasets/collections/<new_dataset>
```

manifest 会含 `task_asset`（mesh/manifest 哈希+来源+物理 basis）、`generation_provenance`（session/spec_revision/recipe/hash）、
`renderer_version`、`cameras`、`config_sha256`。**必须用新目录**（已存在数据会拒绝），报告新 ID/seeds，不复用旧 3/6 冒充提升。

---

## 9. 测试与验证

```powershell
# 全量后端（Windows sim venv）
..\..\work\venv\Scripts\python.exe -m pytest tests/ -q
# 最近一次：64 passed, 1 skipped（test_package_isolation 需 torch/WSL）

# 生成/配方重点
..\..\work\venv\Scripts\python.exe -m pytest tests/test_generation_sessions.py tests/test_generation_jobs.py tests/test_scene_recipe.py -q
# 26 passed（5+10+11）

# 原子提交（脚本模式，无 test_ 函数）
..\..\work\venv\Scripts\python.exe tests/test_scene_commit.py

# 前端
cd frontend && npx tsc -b && npm run build
```

- 测试用 `tests/conftest.py` 的 socketpair 垫片（端口 51200–51400）规避 Windows TestClient `WinError 10013`。
- 部分测试同时支持 `def test_`（pytest）与 `main()`（脚本）两种；脚本模式**不会**自动建 `tmp_path` 子目录。
- **隐藏浏览器视口**：`take_screenshot` 失败（`NATIVE_BROWSER_VIEWPORT_UNAVAILABLE`），用无障碍快照 `take_snapshot` + API 读回取证；
  前端专家 rAF 会停摆，改用后台脚本驱动 `env.servo_expert_action()`。

第四轮真实证据（交接时位于 Qoder workspace `evidence/`，非仓库内）：配方 validate（250 步稳定）、apply v2 读回、
专家 `success=true`（tick 93，stable_ticks=6）、采集 `round4_composed_v2` 3/3、前视帧 JSON+JPG。

---

## 10. 下一步建议路线

1. **修 §7**（F01/F02/F06 → F03/F07 → F04/F05 → F08/F09/F10），每项补隔离回归测试；F04/F05 完成前不要宣称"组合生成已交付"。
2. 分层在线验证（需配置密钥）：真实 LLM 对话 → 真实 Lux3D/World 新建或恢复 → 入库 revision → 多实例场景落地。
3. 场景标定精化（水平/尺度/hybrid 对齐）；托盘/任务参数化（若要支持用户自定义托盘尺寸）。
4. 如需提升策略：composed_scene 采集 → VisualBC 重训，报**新**数据集 ID/seeds/success。
5. 回归：先 26 项生成/配方 + 渲染/场景相关，再全量 64 项 + 前端构建。

---

## 11. 坑与边界（必看）

- **凭据只在后端 env**；不进浏览器/消息/导出/日志/截图。模型输出不得成为路径/shell 字符串。
- **MuJoCo 是物理真相**；无桌面后台 uvicorn 相机渲染会崩（前台/桌面窗口运行）。
- **hybrid 限制**：仅 `static_background_overlay` 固定 front，无 wrist/运动相机；高斯全景预览 ≠ 3D 编辑；构建约 10–19s 需缓存。
- **不要** reset/clean/全量 checkout/覆盖新文件；HEAD 非唯一基线。
- **提交/推送**：仅在明确要求时；身份 `PChanww`；WSL+代理推送。
- **诚实口径**：mock≠在线成功；估计值≠测量；未验证能力明确标注；不复用旧成功数字。

---

> 接手后建议第一步：`git -C <repo> log --oneline -5` 与 `git status`，通读 `docs/ROUND4_CODE_REVIEW_2026-11-12.zh-CN.md`，
> 运行 §9 全量测试确认基线，然后按 §7 顺序开修。
