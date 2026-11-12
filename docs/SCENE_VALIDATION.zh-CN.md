# 场景链路验证记录

> 后续复核发现：scene_demo 关闭默认桌面但缺少碰撞代理，历史采集/模型结果不能直接归因为可达性。修复方案、重新采集与实际验证见 [修复草案](SCENE_REPAIR_PROPOSAL.zh-CN.md)。以下保留为原版本的历史记录。

日期：2026-11-10 · 代码基线：见提交（场景链路）。区分：已实现 / 已验证 / 未验证 / 后续增强。

## 已验证（真实运行）

`tests/test_scene_pipeline.py`：
- 四元数 xyzw→wxyz（w 在前）、绕 Z 90° 旋转矩阵、分层变换含统一缩放与平移（compose）。
- demo 场景（`assets/scenes/demo_workspace/1,2`）：环境+碰撞编译通过；杯子落在场景桌面上（高度≈height/2）；接触对含 `cup_contact` 与 `scene_table`。
- 场景进入传感器 RGB：有场景 vs 默认空白桌的 front 图像不相同。
- seed 可复现（两次 reset 同 seed 物体位姿一致）；旧默认任务回归（伺服专家抓放成功）。

`tests/test_world_provider.py`（可控 mock，非在线）：
- submit→remote_processing，poll→succeeded；重启后新 store 从持久化恢复 remote_id；poll 不重复提交。
- 失败可见（state=failed, error）；提交超时→interrupted 且 remote_id 为空；无凭据→unconfigured。

回归：`test_engine`、`test_control`、`test_camera_latency`、`test_kinematics` 全过；前端 tsc + vite build 通过。

纵向闭环（demo 场景，真实运行）：
- 采集：`scripts/collect_expert.py --config projects/scene_demo/current.json --out datasets/collect_scene_demo` → **21 episode，成功 8/21**（场景墙体外置后与首轮同值，说明失败非墙体遮挡，属该场景下专家可达性，未放宽判定）。
- 转换：`lerobot_writer --staging datasets/collect_scene_demo --out datasets/lerobot/scene_demo` → 3343 帧回读通过。
- 训练：`train_image_bc --dataset scene_demo --model-out visual_scene` → val MSE 0.0038 / gen 0.0304，重载最大绝对差 0.0。
- 评估：`eval_image_policy --model visual_scene --base-project scene_demo` → 闭环 **1/6**（训练内 0/2、留出摆位 0/2、留出资产 1/2）。真实结果，含失败。

## 已实现（代码存在，未做在线验证）

- 场景 schema（SceneAssetManifest/SceneInstance/SceneCollider，Pydantic+TS）、本地 GLB/PLY 导入与 PLY 头分类、坐标变换模块、scene_builder 环境/任务拆分、静态碰撞代理编译、场景 API（import/list/detail/validate/bind-scene）、generation-jobs POST/GET/poll（WorldProvider + 持久化不重复提交）、前后端场景列表/导入/编辑/绑定。

## 未验证

- Aholo World 真实在线生成/重建（需 `AHOLO_API_KEY` + 可选 `manycore-aholo-sdk-world`）：仅 mock 可控测试，真实调用未验证。
- 自动碰撞提取（splat-transform）：未接入。
- hybrid 的限制：splat 与解析盒的前/后/边缘遮挡和半透明混合已数值验证（`tests/test_occlusion_translucency.py`），但 hybrid 合成仍不按 splat 深度遮挡 MuJoCo 前景；wrist 相机仍不可用。
- 相机数值投影：`tests/test_camera_projection.py` 覆盖非中心主点/fx≠fy/非方形/铰链运动（≤0.85 px），并修复 MuJoCo 主点导出偏移；仅 MuJoCo 渲染器，CPU 高斯路径使用同一套角点内参（见 PHASE2_VALIDATION T04）。

## 后续增强

- 3DGS 渲染服务（Node+Playwright/Chromium，Spark/Aholo Viewer 版本核对）、深度/遮挡合成、场景三维编辑与碰撞绘制、资产版本迁移工具、更多场景格式。

## 证据文件

- `datasets/reports/train_visual_scene.json`、`datasets/reports/eval_visual_scene.json`
- `assets/scenes/demo_workspace/2/manifest.json`、`assets/scenes/demo_splat/2/manifest.json`
- 测试：`tests/test_scene_pipeline.py`、`tests/test_world_provider.py`
