# 第四轮测试与能力报告（2026-11-12）

范围：对话式资产规划与场景生成闭环。分层证据：**mock ≠ 在线成功**；真实本地资产 / 真实 MuJoCo 为准；
真实 LLM 仅在配置时验证，否则标"未验证"；真实生成 API 优先复用既有远程任务。本轮 **不 commit、不 push**。

> 复审补记（2026-11-18）：后续独立代码审查（`ROUND4_CODE_REVIEW_2026-11-12`）曾报告 10 项缺陷；
> 在当前 HEAD `acb366a` 用其复现脚本实跑已 **10/10 不再复现**。结论与逐条修复位置见
> `ROUND4_CODE_REVIEW_VERIFICATION_2026-11-18.zh-CN.md`。本报告的命令/证据仍有效。

## 1. 命令与结果

| 命令（vla-lab 目录，Windows sim venv） | 结果 |
| --- | --- |
| `python -m pytest tests/ -q` | **64 passed, 1 skipped**（78s）。跳过=test_package_isolation（torch，需 WSL） |
| `python -m pytest tests/test_generation_sessions.py tests/test_generation_jobs.py tests/test_scene_recipe.py -q` | **26 passed**（5+10+11） |
| `python tests/test_scene_commit.py`（脚本模式） | 通过：version_after_valid=7、invalid_cases=8、conflict=409、legacy 不持久化非法 |
| `cd frontend && npx tsc -b && npm run build` | tsc 无错误；vite build 成功（1237 modules；chunk>500kB 为既有警告） |
| `python <evidence>/verify_round4_expert.py` | 真实 MuJoCo 专家在 round4_cup v2：**success=true**，tick 93，stable_ticks=6 |
| `python scripts/collect_expert.py --config <v2> --plan <3ep> --provenance <json> --out datasets/collections/round4_composed_v2` | **3/3 成功、279 帧、seeds 0/1/2** |

浏览器（browser-use 无障碍快照 + API 读回；视口隐藏时截图不可用，改用结构化快照）：
工作台表单改规格→revision 递增+provenance；编译预览 hash 6d72294ddbc1；RecipePanel 预设→创建并校验→
应用 round4_cup v2→读回 colliders/桌/渲染器；进入仿真运行专家（grasped/lifted 真实出现）。

## 2. 证据文件（Qoder 工作区 evidence/）

- round4_recipe_validate.json：valid=true、issues/warnings=[]、physics{built,stable,steps=250,cup_z=0.045,upright=1}
- round4_scene_readback.json / round4_config_v2.json：v2、task.asset=imported/lux3d-blue-live-5887561、colliders=[workbench,toolbox]、table=false、renderer=mujoco
- round4_expert_server_result.json + round4_expert_front.jpg：专家 success=true 轨迹 + 前视帧（杯入托盘）
- round4_collect_plan.json / round4_collect_provenance.json：采集计划与 provenance 输入
- datasets/collections/round4_composed_v2/manifest.json：task_asset 哈希/来源/物理 basis + generation_provenance + renderer_version + cameras + config_sha256

## 3. 能力矩阵

### 已实现且真实验证
- 会话持久化（SQLite WAL+revision）、表单/JSON 直改跳过对话、字段级 provenance、编译预览（确定性/版本化/载荷排除物理字段）。
- SceneRecipe composed_scene 无需 World/标定：真实 MuJoCo 构建+250 步短稳；原子 apply（expected_project_version，409）+读回。
- 生成杯（imported/lux3d-blue-live-5887561）流入配方→项目 v2→专家抓放 success→采集 3/3（新 seeds）。
- 布局/依赖/背景校验：环/缺失/穿透/悬空/高斯需 hybrid/网格需标定（test_scene_recipe 11 项）。
- 幂等/恢复/节点重试/submit_unknown（test_generation_jobs 10 项，mock provider）；World 复用既有 provider。
- 采集 manifest 增补 task_asset 哈希+generation_provenance（recipe/spec revision）+renderer_version+cameras。

### mock 验证（非在线）
- Lux3D/World provider 成功/失败/超时/重启/不重复提交（test_generation_jobs、test_world_provider）。
- 规划器 JSON 修复/注入防御/≤3 问（test_generation_sessions，未配真实 LLM）。

### 未验证 / 未实现
- 真实 LLM 对话：未配置 AUX_MODEL_*，显示 waiting_for_configuration，**未验证**（不冒充）。
- 真实在线 Lux3D/World 新建付费任务：无 AHOLO_API_KEY，**未验证**；本轮复用既有 remote_id/本地资产。
- environment_mesh 在线生成、SDK part_split/auto_rig、六轴物理机器人、hybrid wrist/运动相机、自动凸分解、sim2real：未实现。

## 4. 风险与下一步

- 风险：本地幂等≠远程 exactly-once 计费；高斯背景仅固定 front 受限模式；未标定资产不绑物理；
  隐藏视口下浏览器 rAF 节流会使前端专家循环停摆（本轮改用服务端驱动取证）。
- 托盘语义：cup_pick_place 的目标托盘由任务插件固定（TRAY_HALF≈7.4×6.9 cm），**未参数化**；
  用户样例的"蓝托盘 20×15×2 cm"已作为 user_explicit 记录进规格/会话，但本轮验证的放置语义是插件真实托盘，
  不宣称实现了 20×15×2 托盘任务。静态道具（toolbox）与参数化工作台（workbench）以显式代理落地。
- 下一步：配置真实 AUX_MODEL/AHOLO_API_KEY 后做在线对话与生成验证；标定流程精化；
  composed_scene 视觉采集→VisualBC 重训（如需提升须报新 ID/seeds，不复用旧 3/6）。