# 场景管线（本地场景 → 碰撞 → 传感器图像 → 数据 → 训练 → 评估）

本文说明 RoboForge 场景链路的结构、坐标/单位、渲染时序、资源版本、操作步骤、支持格式与实际限制。审计与验证记录见 [SCENE_VALIDATION.zh-CN.md](SCENE_VALIDATION.zh-CN.md)。

## 1. 架构分层

- **场景资产（SceneAssetManifest）**：`assets/scenes/<asset_id>/<revision>/manifest.json`，含来源、格式、文件 SHA256、up_axis、unit_scale、calibrated、visual_asset、bbox、status。源文件保存在 `source/`，处理后网格在 `processed/visual.obj`。
- **项目场景绑定（ProjectConfig.scene）**：`environment`（asset_id/revision + 实例 position/quaternion_xyzw/uniform_scale）、`colliders`（静态碰撞代理）、`sensor_renderer`、`use_default_table`、`work_surface_z`。
- **环境/任务/机器人分层**：`vlab/scene_builder.environment_xml()` 生成灯光/地面/默认桌面或场景视觉网格 + 静态碰撞代理；`vlab/tasks.task_xml()` 只生成任务托盘与动态杯子；`vlab/mjcf.robot_xml()` 生成机器人。`engine._build()` 组合三者。
- **渲染**：MuJoCo 模式把标定的场景网格作为静态视觉几何加入传感器 RGB。Hybrid 模式支持把 3DGS 背景与 MuJoCo 前景合成到世界固定的 `front` 相机。有关时序和遮挡范围，见[渲染管线](RENDERING_PIPELINE.zh-CN.md)。
- **数据/训练**：`vlab/collect.py` 采集（图像/状态/动作/时间戳/成功标签，含配置快照哈希）；`scripts/lerobot_writer.py` 转 LeRobot；`scripts/train_image_bc.py` 视觉 BC；`scripts/eval_image_policy.py` 经 HTTP 闭环评估。

## 2. 坐标与单位

- 规范世界：米、右手、Z-up（MuJoCo）。
- 对外四元数统一 **xyzw**；MuJoCo 需要 **wxyz**，在 `vlab/transforms.quat_xyzw_to_wxyz` 显式转换并测试。
- 分层变换唯一方向：`T_world_from_asset = T_world_from_scene × T_scene_from_asset`；原始 up_axis/缩放与项目实例变换分层保存，避免重复转换。
- 仅支持**统一缩放**（非均匀缩放会破坏接触几何，被 schema 拒绝）。
- 前端 Three.js Y-up：内容根节点绕 X 轴 -π/2 旋转显示后端 Z-up，不做额外重复转换。
- 碰撞体 `full_size_m` 是**全尺寸**，导出 MuJoCo 时转半尺寸/半高/半径。

## 3. 渲染时序（复用 Scheduler/CameraBuffer）

- 物理按固定步长推进；相机在调度器规定的仿真时刻生成**不可变帧**（capture_time），经 `available_time = capture_time + latency` 进入 `CameraBuffer` 有界队列。
- 观测返回 `available_time ≤ decision_time` 的最新历史帧；`capture≤available≤decision`；reset/epoch 切换丢弃旧帧；队列有界（96 帧）。
- 真实渲染墙钟耗时不计入配置的传感器延迟；离线采集以数据正确为先（可慢于墙钟）。
- `frame.jpg` 表示“当前预览”，与缓冲观测帧不同（接口区分）。

## 4. 支持格式与导入

- 本地导入：`.glb`（自包含）与 `.ply`。PLY 按**头部/属性**分类：三角形网格（element face）、点云（仅 vertex）、3D 高斯（`f_dc_*`/`scale_*`/`rot_*`）。
- 网格导入后按指定单位生成 `processed/visual.obj`，使用其环境尺度。点云用于预览。高斯可作为 hybrid 的视觉背景，物理接触使用单独的桌面或代理。
- 未标定（unit 未知/未勾选）→ `status=uncalibrated`，仅预览，不作为已验证物理场景。
- 上限 256 MiB；路径/格式/自包含校验。

## 5. 操作步骤

1. 资产中心 → 「导入场景(.glb/.ply)」上传 → 生成 rev1 清单与预览。
2. 打开 `/scenes/:id/edit`，查看来源、尺寸、包围盒和标定状态。确认显式支撑代理可用后，再关闭默认桌面。校验并绑定到项目时固定资产 revision。
3. 项目仿真 `/projects/:projectId/sim` 加载该场景；碰撞代理通过配置 `scene.colliders` 提供（box/plane/cylinder，名称/位置/旋转/全尺寸/摩擦/掩码）。
4. 检查项目版本、机器人、控制器和相机，再采集并转换为 LeRobot。内置专家采集使用支持固定姿态 5R 逆运动学的控制器。其它机构的示例与边界见[平台支持范围](PLATFORM_SCOPE.zh-CN.md)。

## 6. 已知限制

- Hybrid 的范围是固定前视视觉背景合成。双相机采集使用 MuJoCo 渲染。
- 场景编辑支持手工静态代理，部分生成网格导入路径支持组合盒。高斯背景的接触几何由独立代理设置。
- 场景仅支持统一缩放的刚体位姿；不支持倾斜工作台/自动语义拆分。
- 生成外观不保证可靠关节/惯量/接触或可分离对象；机器人仍由配置定义关节与物理。
- 在线 World 生成需 `AHOLO_API_KEY` 与可选 `manycore-aholo-sdk-world`；未配置时 `unconfigured`，仅在可控 mock 下测试。
- 演示场景墙体外置于工作区，避免遮挡；遮挡验证为受限版本。
