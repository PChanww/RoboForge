# 功能状态（第二阶段，2026-11-12 第四轮）

不使用模糊完成百分比；按 已实现 / 已验证 / 部分 / 未实现 列出。基线：当前工作区（含未提交修复）。

## 已实现且已验证

- 项目配置版本化、会话隔离、控制权 token/epoch/tick、动作块预校验（test_control/test_api）。
- 五旋转关节机械臂 + 笛卡尔龙门；配置驱动 IK（改连杆/工具偏移后 FK 一致）、不可达结构化（test_kinematics）。
- 相机有界缓冲/延迟/图像年龄（test_camera_latency）；采集时刻不可变快照（test_gaussian_snapshot）。
- 相机数值投影（非中心主点/fx≠fy/非方形/铰链运动）≤0.85 px；修复 MuJoCo 主点导出偏移（test_camera_projection）。
- 真实 World 高斯 PLY 解析（199747，尺度/不透明度/SH DC/四元数）、CPU 高斯光栅化；协方差先变换到相机系（`RcᵀΣRc`）再投影，相机 roll 0/90°、俯仰/偏航、fx≠fy、离中心主点、实例旋转/缩放均数值验证（test_covariance_camera）；解析盒遮挡前/后/边缘与半透明混合（test_occlusion_translucency，仅代表解析盒，不代表 hybrid 完整遮挡）。
- hybrid 进入 observation.images（front；受限模式 `static_background_overlay`/`hybrid:3`，splat 不遮挡机器人；wrist/运动相机明确拒绝；构建守卫）（test_hybrid_renderer/test_hybrid_cache）；前景分割使用真实 geom objid + objtype（test_segmentation）；跨会话隔离（test_hybrid_sessions）。
- 场景本地导入（GLB/PLY 分类）、场景实例+碰撞代理 API/持久化/物理落地（test_scene_editor）；单请求原子保存、完整校验、`expected_version` 409 冲突（test_scene_commit）。
- 三维场景编辑画布：选取/平移/旋转 gizmo、世界变换与圆柱 Z 轴、撤销重做、测量缩放、原子保存（浏览器验证 + scene-debug 测具）。
- 机器人独立编辑页：关节/连杆/执行器/夹爪/相机数值编辑、校验、保存新版本（浏览器实跑，v1→v2）。
- 混合场景 CLI 采集（修正后新链路）：`hybrid_corrected_1` 21/21（1952 帧，`hybrid:3`）→ LeRobot（front-only，含 `frame_timing.jsonl`）→ 视觉 BC `visual_corrected_1`（val 0.0118 / gen 0.01653，重载 0）→ 闭环 **3/6**（in 1/2、spawn 1/2、asset 1/2）。旧 `hybrid_engine_v3`/4/6 为旧渲染器历史。
- 视觉-only 对照：同物理仅换渲染，hybrid 与 MuJoCo 均成功、物理最大差 0.0（test_visual_only_control）。
- 视觉模型包导出（自包含权重+SHA+契约+适配器+HTTP runner，重载差 0，删除源 lab 可运行，篡改被拒）；`visual_corrected_package_1` 部署中心列出、5R 兼容；龙门不兼容。
- 作业配置快照：提交后 current 变更不影响已入队作业；非法计划入队前拒绝（test_job_config_snapshot）。
- 障碍物阻挡：静态墙与指垫真实接触、任务失败、物体不位移（test_obstacle_block）。
- World 提供方 mock 测试（成功/失败/超时/重启/不重复提交）；Lux3D Web 入口（无 key 明确 409，命令构造正确）。

## 第四轮：对话式资产规划与场景生成闭环（已实现且真实验证）

- 生成会话持久化（SQLite WAL + 乐观 revision，跨刷新/重启）；表单/JSON 直改跳过对话；字段级
  provenance（source/confirmation/verification，区分"没说"与"接受默认"）；编译预览确定性/版本化，
  载荷排除 mass/friction/joint_limits 等（test_generation_sessions 5 项 + 浏览器实跑）。
- SceneRecipe `composed_scene` 无需 World/标定：真实 MuJoCo 构建 + 250 步短稳；布局/依赖/背景校验
  （环/缺失/穿透/悬空/高斯需 hybrid/网格需标定）；原子 apply（expected_project_version→409）+ 读回
  （test_scene_recipe 11 项）。
- 端到端真实闭环：生成杯 imported/lux3d-blue-live-5887561 → 配方 recipe_38c00ff485f7 → 项目 round4_cup v2 →
  专家抓放 success=true（tick 93、stable_ticks=6）→ 短采集 round4_composed_v2 **3/3、279 帧、seeds 0/1/2**（新 ID）。
- 采集 manifest 增补 task_asset（mesh/manifest 哈希+来源+物理 basis）与 generation_provenance
  （session/spec_revision/recipe/hash）+ renderer_version + cameras + config_sha256。
- 任务幂等/节点级恢复/submit_unknown/已知 remote_id 恢复（test_generation_jobs 10 项，mock provider）；
  World 背景复用既有 provider 与 GenerationJobStore。
- 回归：pytest 64 通过 1 跳过（torch/WSL）；tsc + vite build 通过。详见
  [GENERATION_WORKFLOW](GENERATION_WORKFLOW.zh-CN.md) 与 [ROUND4_GENERATION_REPORT](ROUND4_GENERATION_REPORT.zh-CN.md)。

### 第四轮：mock 验证 / 未验证

- Lux3D/World provider 与规划器 JSON 修复/注入防御为 mock 验证（非在线）。
- 真实 LLM 对话未配置 AUX_MODEL_*：显示 waiting_for_configuration，**未验证**，不冒充模型回复。
- 真实在线新建付费生成无 AHOLO_API_KEY：**未验证**；本轮复用既有 remote_id/本地资产。

## 部分实现

- 场景标定：地面平移对齐（方差 +6%），水平/尺度未精化；hybrid 背景与机器人工作区仍未完全对齐。
- 部署：视觉包可加载运行；通用 PolicyAdapter 仅封装 VisualBC；其他目标平台适配未做。
- 异步渲染：当前帧在采样时刻同步渲染；旧 epoch 拒绝已测，异步延迟完成/丢弃未实现。
- 遮挡：splat 与解析盒的深度遮挡已测；hybrid 为受限模式 `static_background_overlay`，splat 不按深度遮挡 MuJoCo 前景，不宣称完整遮挡。

## 未实现

- 独立离屏服务自动化：浏览器内 Spark 路径已验证（内置 Chromium WebGL2，需 ~15–20s 预热）；Playwright/Chromium CLI 自动化仍失败（下载受阻/Edge 退出）。
- wrist/运动相机 hybrid（明确拒绝，仅世界固定 front）。
- Web Lux3D 真实在线生成验证（无 AHOLO_API_KEY；仅有入口与 mock/命令构造）。
- 生成闭环：environment_mesh 在线生成、SDK part_split/auto_rig、六轴臂物理机器人（仅 robot_visual 路线）、
  真实 LLM 对话（未配置 AUX_MODEL_*）均未实现/未验证。
- 自动碰撞提取/凸分解、倾斜工作台与通用任务坐标。
- 实机迁移（无硬件，未验证；不宣称数字孪生/sim2real）。

## 已知兼容变化

- 相机主点：配置按角点像素；MuJoCo 导出转换为中心相对偏移。修复前采集的视觉数据/模型（expert21、hybrid_engine_v2、visual_hybrid）为历史产物，需用 `hybrid_engine_v3` 及后续模型。
- 高斯场景必须 `sensor_renderer=hybrid` 且物理有默认桌面或碰撞代理，否则明确拒绝。
- 导入环境后关闭默认桌面且无碰撞代理的旧配置被拒绝（需补代理或保留桌面）。
- 非 1 unit_scale 且无 normalized 标记的旧资产要求重导入；未标定网格/高斯不能直接物理绑定。
- 模型契约缺失的旧权重需显式允许历史假设（CLI）。
