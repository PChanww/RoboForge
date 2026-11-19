# 渲染管线（第二阶段）

## 架构

- **物理唯一真值**：MuJoCo 推进动力学；任何渲染只消费快照，不自行运行机器人动力学。
- **MujocoSensorRenderer（默认）**：现有 `engine._render_camera` 路径，场景网格进 MJCF 视觉 geom，进入 `observation.images`。
- **HybridSensorRenderer（受限模式）**：3DGS 环境作背景 + MuJoCo 机器人/杯/托盘分割前景合成。front 相机可用；wrist 明确 `render_unavailable`。实现：`vlab/rendering/hybrid.py` + `vlab/splat_render.py`（CPU 高斯光栅化，numpy）。
- **浏览器 Spark 试验**：`render-service/` 用真实高斯 PLY 测试浏览器显示，预热条件和验证截图见下文。平台传感器使用 MuJoCo 或固定前视 hybrid 路径。

## 坐标与相机

- 世界米、右手 Z-up。前端 Three 仅显示转换（根组绕 X -π/2），不重复变换。
- 对外四元数 xyzw，MuJoCo wxyz 显式转换（`vlab/transforms.py`）。
- MuJoCo 相机约定实测：视轴 -Z、+Y 上、+X 右，与 Three 相机一致，可直接用 `cam_xpos/cam_xmat` 驱动。
- 高斯实例变换：均值 `s·R·μ+t`，协方差 `s²·R·Σ·Rᵀ`（`splat_render.render(inst=...)`）。

## 快照与时序

- 采集时刻从当前 MuJoCo data 构造**只读快照**（qpos、连杆/任务物 xpos/xmat、相机世界位姿、内参、epoch、tick、capture_time），同刻多相机共享；`capture ≤ available ≤ decision`；reset 后旧 epoch 帧不进入缓冲。
- `CameraBuffer` 有界（96 帧）并保留 capture/available；延迟生效；图像年龄按返回帧计算。

## 遮挡

- 3DGS 背景 + MuJoCo 前景为**受限模式**：splat 不遮挡机器人（机器人在工作区前方）；`foreground_ids` 用分割 id（geom_id+1）区分机器人/杯/托盘。
- 解析盒遮挡已系统数值验证（`tests/test_occlusion_translucency.py`）：盒在前覆盖 2601 px、盒在后 0 px、跨两深度层边缘 315 px 在前，50% 透明混合误差 0.0008。注意深度图为最近高斯中心深度（诊断用），非逐像素表面深度。
- 未实现：按 splat 逐像素深度遮挡 MuJoCo 前景（需要 MuJoCo 深度图与 splat 表面深度合成）。

## 相机内参约定（实测）

- 配置/快照/CPU 光栅化使用**角点**像素主点 `(cx, cy)`；MuJoCo 4.0.0 把 `principalpixel` 当中心相对偏移且渲染位移取负，实测 `content_center = resolution/2 - principalpixel`。
- `vlab/mjcf.py` 导出时写 `resolution/2 - cx/cy`；`tests/test_camera_projection.py` 对非中心主点、fx≠fy、非方形、铰链运动 5 用例验证 ≤0.85 px。修复前采集的视觉数据使用了偏移画面。

## 浏览器 Spark 路径

- `render-service/` 静态页在真实 Chromium（WebGL2，ANGLE/D3D11）中加载 `SplatMesh`；`__loadCam/__autoCam/__save` 可保存 256² PNG。实测需 **15–20 s 预热**，早期渲染为全黑（此前“无可见输出”的真正原因）。
- 证据：`browser-front-clean.png`（splat-only，21545 色）、`browser-cam-box.png`（含诊断盒/平面）、`browser_render_report.json`（GPU/版本/统计）。

## 失败与性能

- hybrid 构建失败明确报错（缺高斯资产/缺默认桌面或碰撞/非 front 相机），不静默回退。
- 进程级背景缓存：同 ply+相机+实例+分辨率只渲一次（此前每次建环境 18s 导致后端请求挂起；已修复）。
- 实测（256²，CPU）：加载 67–86ms；整帧光栅化 14–19s；hybrid 背景 18.5s（首次）；MuJoCo 前景合成毫秒级。未承诺实时能力。
- 已知限制：资产未标定（地面平移仅使 front 方差 +6%），工作区对齐差；未实现 3DGS 依赖库的 offscreen 服务与精确深度合成。
