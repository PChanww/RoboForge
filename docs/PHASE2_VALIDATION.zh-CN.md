# 第二阶段验收记录

日期：2026-11-12（第三轮更新）。区分：通过 / 部分 / 未完成。证据均为本地实际运行产物。第三轮的修复项、受限模式声明与新证据链见 [第三轮修复与验证说明](第三轮修复与验证说明.zh-CN.md)。

> 标记说明：T05 为**受限通过**（仅解析盒遮挡测试成立，hybrid 背景不遮挡 MuJoCo 前景）；T07 为**同步通过**（异步渲染未实现）；T13 已补**真实视觉-only 对照**（同物理、仅换渲染）；T16 已用**自包含隔离包**重验。

| 编号 | 项目 | 结果 | 证据 |
| --- | --- | --- | --- |
| T01 | 旧项目回归：默认场景/控制权/epoch/动作块 | **通过** | test_engine / test_control / test_api 全过（32 passed，含本轮 T03/T04/T05/T07/T08/T12/T14 新测） |
| T02 | 真实 World PLY 高斯显示（非点云） | **通过（CPU + 浏览器）** | CPU：`splat-auto.png`（199581/199747）、`splat-with-box.png`；浏览器：内置 Chromium WebGL2 + Spark `browser-front-clean.png`（256²，21545 色，NVIDIA RTX 4070/D3D11，`browser_render_report.json`）；Spark 需 ~15–20s 预热后才出画面 |
| T03 | 坐标/单位/统一缩放/150° 旋转/高斯方向 | **通过** | `test_gaussian_snapshot`（150° 方向、单位、scale 解码）；`tests/test_gaussian_covariance.py`：长轴足迹 90°、四元数/实例旋转后主方向各偏转 90°，伸长率 6.05，scale×2 像素比 1.99（`gaussian_covariance.json`） |
| T04 | 非中心主点/fx≠fy/非方形/运动腕部投影误差 | **通过** | `tests/test_camera_projection.py`：5 用例（含 320×240、fx≠fy、cx/cy≠中心、铰链 q=0/0.6/1.2）最大误差 **0.85 px**，轴上标记 0.707 px，内参回读差 0；`outputs/phase2-render-validation/camera_projection.json`。真实默认配置 front 相机对杯子中心投影误差 1.75 px（`camera_real_model.json`，wrist 因杯子出画不单独判定） |
| T05 | 遮挡：动态盒在前/后/边缘 | **受限通过** | `tests/test_occlusion_translucency.py`：盒在前覆盖 2601 px；盒在后可见 0 px；边缘 315 px 在前、后方 0 px；50% 透明混合误差 0.0008（`occlusion/*.png`）。限制：以上只验证独立解析盒与 splat 深度，**hybrid 受限模式（`static_background_overlay`）不按 splat 深度遮挡 MuJoCo 前景**，不宣称完整遮挡 |
| T06 | 每帧快照，capture≤available≤decision | **通过** | `test_gaussian_snapshot`（S1/S3/S5） |
| T07 | reset 后旧 epoch 帧不泄漏 | **通过（同步）** | `tests/test_stale_frames.py`：旧 epoch/旧 tick 动作在 reset 后被拒（ControlConflict），物理状态不变；reset 后缓冲帧仅含新 epoch 与新 session_id。当前为采样时刻同步渲染，**异步渲染/延迟完成/丢弃未实现** |
| T08 | 会话隔离（不同背景/物体/相机不串） | **通过** | `tests/test_hybrid_sessions.py`：两个 hybrid 会话各自帧/epoch/session_id 独立，背景缓存共享但 HybridRenderer 对象分离；不同 seed 帧均值差 1.408；reset 一个会话不改另一个 |
| T09 | 故障：服务未起/超时/非法尺寸/损坏资产 | **通过** | hybrid 构建明确报错、渲染错误可见、后端二次渲染挂起已修复；mock 世界测试覆盖失败/超时 |
| T10 | 关闭浏览器 CLI 离线采集含高斯 RGB | **通过** | 第三轮新链路 `datasets/collections/hybrid_corrected_1`（21/21 成功、1952 帧、`renderer_version=hybrid:3`、`gaussian_sha256=8895bd11…`）；`outputs/phase2-render-validation/third-round/corrected_ep0_f*.jpg`。旧 `hybrid_engine_v3` 保留为旧渲染器历史 |
| T11 | 编辑持久化（变换+多代理保存后刷新一致） | **通过** | `test_scene_editor`；浏览器保存 → 后端 v14（desk/post/proxy_3） |
| T12 | 物理：落物稳定/接触对/障碍阻挡/缺碰撞拒绝 | **通过** | `tests/test_obstacle_block.py`：静态墙（0.27×0.012×0.30 m）置于夹爪与杯之间，右指垫在 tick 3 起发生真实接触，任务 success=false，杯子位移 0（`obstacle_block.json`） |
| T13 | 数据来源：仅换场景视觉、比较原始 RGB | **通过（真实对照）** | `tests/test_scene_pipeline.scene_changes_rgb`；`tests/test_visual_only_control.py`：同一 seed/桌面/控制器，hybrid 与纯 MuJoCo 均 93 tick 成功、物理状态最大差 **0.0**、图像平均差 **131.09**（`outputs/phase2-render-validation/visual_only_control/`）。`data_checks_hybrid_corrected_1.json`：1952 帧、0 issues、图像年龄均值 8 ms |
| T14 | 作业选择非 default 项目/数据集/快照 | **通过** | `tests/test_job_config_snapshot.py`：提交后 current 配置升到 v2，作业快照仍为 v1；4 种非法 plan 均在入队前拒绝 |
| T15 | 训练契约不匹配提前拒绝 | **通过** | `compat` 对龙门判不兼容；eval_visual 运行前检查 |
| T16 | 模型包导出/重载/闭环 | **通过（自包含）** | `deploy/visual_corrected_package_1`：自带 `model.pt` + SHA256（`32a2ca2d…`），包内权重 vs 源模型重载差 **0.0**；删除源 lab 仍可推理；篡改 `model.pt` 被拒；契约含 `camera_specs/controller_spec/state_joint_names`；部署中心列出且 5R 兼容；runner seed 1 成功 94 tick、seed 0 失败，均清理会话。`tests/test_package_isolation.py` 直接执行通过（WSL） |
| T17 | 生成任务恢复（成功/失败/超时/重启/不重复提交） | **通过（mock）** | `test_world_provider`；真实在线未验证（无凭据） |
| T18 | 页面真实操作（导入/编辑/绑定/采集/训练/评估/部署） | **通过** | 机器人独立编辑页 `/robots/:id/edit`：项目切换、关节/连杆/执行器/夹爪/相机表格、校验与保存；真实页面保存 `robot_ui_check` v1→v2（waist damping 0.85），刷新后显示 v2 与内参；此前资产导入/场景绑定/部署兼容均浏览器实测 |

## 相机主点修复（2026-11-10）

数值投影测试暴露出 MuJoCo 4.0.0 的 `principalpixel` 语义：文档称“相对图像中心的偏移”，而渲染位移与之相反（实测 `content_center = resolution/2 - principalpixel`）。平台配置按角点像素存储主点，直接写入会让整幅画面偏移 `(W/2 - cx, H/2 - cy)`——历史 front 帧（如 `datasets/collections/hybrid_engine_v2/decoded-front.png`）确实整体偏到左上角。修复：`vlab/mjcf.py` 导出时写 `resolution/2 - cx/cy`，快照/高斯光栅化继续用角点内参；T04 数值验证 5 用例 ≤0.85 px。

影响：修复前的视觉数据集/模型/评测（expert21、hybrid_engine_v2、visual_hybrid 2/6 等）保留为历史产物，但对应旧相机投影；相机修复后曾重采 `hybrid_engine_v3`，第三轮又因分割/协方差修复重采 `hybrid_corrected_1` 并重跑转换/训练/评测，**以第三轮数字为准**。

## 性能（实测，256²，CPU）

- 高斯加载 67–86ms；整帧光栅化 14–19s；hybrid 背景 18.5s（每进程一次）；MuJoCo 前景合成毫秒级。未承诺实时。

## 训练/评估真实数字

- 历史（修复前相机）：`visual_hybrid` 440 步，val 0.0132 / gen 0.0147，重载差 0。
- 旧渲染器（协方差/分割修复前）：`visual_hybrid_v3`（数据 `hybrid_engine_v3`，21/21、1952 帧）：440 步，val 0.01104 / gen 0.01344，重载差 0；闭环 **4/6**（in 1/2、spawn 1/2、asset 2/2）；包 v4。**保留为历史，不与第三轮混用**。
- **第三轮（当前）**：数据 `hybrid_corrected_1`（21/21、1952 帧、`hybrid:3`）→ `visual_corrected_1`：1394 训练帧、440 步，val **0.0118**、gen **0.01653**、重载差 **0.0**，checkpoint 记录 `render_meta`（`datasets/reports/train_visual_corrected_1.json`）。闭环 **3/6**（in_distribution 1/2、heldout_spawn 1/2、heldout_asset 1/2；`datasets/reports/eval_visual_corrected_1.json`），每回合记录 `render_mode=static_background_overlay`。部署包 `deploy/visual_corrected_package_1`：自包含、SHA256、重载差 0、对 `lux3d_5r` 兼容；包内 runner seed 1 成功 94 tick。
- 视觉-only 对照：hybrid 与纯 MuJoCo 同 seed/桌面/控制器，均 93 tick 成功，物理最大差 **0.0**，图像平均差 **131.09**。

## 未完成/未验证

真实在线生成（T17，无 AHOLO_API_KEY）、异步渲染、wrist/运动相机 hybrid（明确拒绝）、标定精化与实机。hybrid 为受限模式 `static_background_overlay`：splat 不遮挡 MuJoCo 前景，不宣称完整遮挡或数字孪生。画布 gizmo 真实鼠标拖动未自动化（同一提交路径已验证）。Playwright/Chromium CLI 自动化仍失败，但内置浏览器 Spark 路径已验证。

## 第三轮回归

- Windows 全量 pytest：**38 passed, 1 skipped**（skip 为需 torch 的包隔离测试）。
- WSL：`tests/test_package_isolation.py` 直接执行通过（`package ok / isolation ok / tamper rejected / sessions 0`）；`tests/test_visual_only_control.py` 通过。
- 新增回归：`test_segmentation`、`test_covariance_camera`、`test_hybrid_cache`、`test_scene_commit`、`test_package_isolation`、`test_visual_only_control`。
