# 第二阶段状态（3DGS 渲染与训练闭环）

日期：2026-11-12（第三轮）。基线：当前工作区（含未提交修改）。本轮未 commit/push，保留供审查。

> **第三轮更新（2026-11-12）**：按 [代码审查](PHASE2_CODE_REVIEW_2026-11-11.zh-CN.md) 修复了分割 id、splat 协方差/深度、hybrid 构建与缓存、场景原子保存、画布世界变换、模型包隔离六项 P0，并以新 ID 重建证据链：`hybrid_corrected_1`（21/21、1952 帧）→ `visual_corrected_1`（val 0.0118 / gen 0.01653）→ 闭环 3/6 → 自包含包 `visual_corrected_package_1`（重载差 0）。渲染明确为受限模式 `static_background_overlay`（hybrid:3），不做完整遮挡/数字孪生声明。详见 [第三轮修复与验证说明](第三轮修复与验证说明.zh-CN.md)。
>
> 历史说明：v2 数据集/2/6 闭环为**修复前相机**记录；`hybrid_engine_v3` / `visual_hybrid_v3` / 闭环 4/6 / 包 v4 为**旧渲染器（协方差/分割修复前）**记录，均保留但不与第三轮数字混用。

## 纠偏后更新（M1 完成、M2 部分）

- **PLY 契约收窄并修正**：`vlab/gaussian.py` 现仅支持 `binary_little_endian`、单一 `vertex`、float32、必需属性齐全；对 ascii/大端、额外 element、`f_rest_*`（高阶 SH）、缺字段、截断、非有限值、零四元数**明确抛 ValueError**，不再“猜测”；sigmoid 稳定、exp 裁剪、四元数归一化（真实文件原始四元数非单位，范数≈2.1，已归一化）。bbox 明确仅为高斯中心、非表面支撑。
- **epoch 唯一所有者**：`env.epoch` 在 `PlatformEnv.reset()` 起始自增（初始帧采样即带新 epoch），`Session.epoch` 镜像 `env.epoch`；`capture_snapshot` 使用 `env.epoch`，与 `Session.reset` 返回值一致。
- **快照真正进入采集调用链**：`_push_frame` 在采样时刻调用 `capture_snapshot`，同一时刻多相机共享一份快照；帧携带 `request_id/snapshot_epoch/session_id/camera_world_pos/mat`，渲染与帧元数据同源。快照为只读数组 + `MappingProxyType` 深拷贝嵌套；新增 `to_wire/from_wire` JSON 传输契约（大数组不走 JSON）。
- **测试**：`tests/test_gaussian_snapshot.py`（pytest，7 项）：合成 G1 尺度 2/3/4、G2 opacity 0→0.5 与 DC→0.5、G3 150° 方向、拒绝 ascii/高阶 SH/缺字段/截断/NaN/零四元数、带 comment 头；真实资产 199747 高斯（缺失则 skip 并声明未验证）；S1 epoch 一致、S3 步进/reset 不改旧快照、S4 编解码往返、S5 帧用快照 request_id。**7 passed**。

### M2/M3 实拍结果（本轮完成，真实运行）

- 尝试浏览器路线：Spark 2.4.0 + Three 0.180.0 已装入 `render-service/`（页面 + 静态服务 + `save` 端点），内置浏览器中 Spark 成功加载真实 `scene.ply`（`window.__ready`），但**未见可见高斯**且 Playwright 自带 Chromium 下载被网络中断（系统 Edge 无头在 Playwright 下立即退出）。浏览器路线未作为交付依据。
- 改用**自有 CPU 高斯光栅化**（`vlab/splat_render.py`）真实渲染真实资产：
  - `splat-auto.png`：对准高斯中心，**199581/199747** 高斯可见，14.3s（256²，CPU）。
  - `splat-front.png`：MuJoCo front 相机（174666 可见），17.9s；画面偏灰，因**资产未标定**、房间与机器人/桌面未对齐（真实发现，未隐藏）。
  - `splat-with-box.png`：动态立方体（5066 像素）与高斯同屏，按射线/最近高斯深度做遮挡（18ms）。
  - 性能：加载 67–86ms；每帧 14–19s（CPU，256²，无 GPU 库）。报告：`outputs/phase2-render-validation/render_report.json`。
- **M3 最小传感器接入**（`vlab/rendering/hybrid.py` + `scripts/collect_hybrid_short.py`）：世界前视相机高斯背景渲染一次（18.5s），逐帧以 MuJoCo 分割（28 个前景 geom，机器人/杯/托盘）叠前景，写 `datasets/collections/hybrid_check/episode.npz`（6 帧，含状态/动作/时间戳），并从 npz 回读解码 `decoded-hybrid.png`——**真实高斯背景 RGB 已进入原始数据可回读**。限制：资产未标定，机器人/房间对齐差；受限模式（高斯背景 + 工作区前景），splat 不遮挡机器人。
- MuJoCo front 相机与 Three 相机轴约定一致（视图 -Z、+Y 上、+X 右），已用真实 cam_xmat 验证。

---

## 续：传感器接入、场景编辑 API/UI、标定（本轮）

- **hybrid 进入 observation.images（front）**：`engine._render_camera` 在 `sensor_renderer=hybrid` 时用高斯背景（世界前视相机，跨 reset 缓存）+ MuJoCo 分割前景（`foreground_ids`：机器人/杯/托盘）合成；wrist 明确 `render_unavailable`（不静默回退）；网格场景或缺少默认桌面/碰撞时明确报错。测试 `tests/test_hybrid_renderer.py` **3 passed**。
- **真实采集数据路径**：新建 front-only hybrid 项目 `hybrid_engine`（环境 `world_586c79650960` rev1、默认桌面），`collect_expert --config projects/hybrid_engine/current.json` → **21 episode / 21 成功 / 1952 帧**；`datasets/collections/hybrid_engine_v2` 解码 `decoded-front.png`（std 19.4，机器人+杯子前景 + 高斯房间背景），manifest 记录 `ply_kind=gaussian/preview_only` 与 front 相机。**真实高斯背景 RGB 已进入 observation 与原始数据并可回读**。
- **场景标定 + 碰撞编辑（API+UI）**：`PUT /projects/{pid}/scene/instance`、`GET /projects/{pid}/scene`、`PUT /projects/{pid}/scene/colliders`；`/scenes/:id/edit` 增加数值实例标定（position/uniform_scale）+ 碰撞代理增删改保存。测试 `tests/test_scene_editor.py`（持久化、圆柱 x≠y 拒绝、编辑后物理落地）。圆柱需 x==y；非均匀缩放不支持。
- **标定实测**：按 z 1% 分位把 World 地面平移到 z=0（floor_z≈-1.401 → 平移 +1.401），front 图像方差 0.0150→0.0159（仅 +6%）——**仅地面平移不足以对齐**，水平位置/尺度仍需人工标定（真实发现）。
- 回归：engine/control/api/scene_pipeline/scene_editor/gaussian 全过；前端 tsc+vite build 通过。

## 续：hybrid 数据转换/训练（本轮）

- **LeRobot 写入器泛化**：按 manifest 相机键（front-only 通过）、动作名/状态名来自配置；新增 `meta/frame_timing.jsonl` 伴随文件（逐帧 sim_time + 每相机 capture_time/age），时间信息随导出保留。转换 `datasets/collections/hybrid_engine_v2` → `datasets/lerobot/hybrid_engine_v2`（21 episode / 1952 帧 / image_keys 仅 front）并官方回读通过。
- **数据检查参数化**：`data_checks.py --dataset --staging`，关节名/限位取自数据清单；hybrid 数据集 **0 issue**（图像年龄均值 8ms）。
- **视觉 BC（hybrid 数据）**：`visual_hybrid`，440 优化步，val MSE 0.0132 / gen 0.0147，重载最大绝对差 0（`datasets/reports/train_visual_hybrid.json`）。
- **hybrid 背景进程级缓存**：`_BG_CACHE`（ply+相机+实例+分辨率键）使同一后进程内背景只渲一次（此前每次建环境都 18s，导致后端请求挂起；已修复）。修正缓存键对 2D 矩阵需 ravel，及 `compose/background` 被误移出类的问题。
- **hybrid 闭环评估（真实）**：`visual_hybrid` 在 hybrid 场景 `--plan full` → **in_distribution 0/2、heldout_spawn 1/2、heldout_asset 1/2，共 2/6**（`datasets/reports/eval_visual_hybrid.json`）。小样本视觉 BC，失败回合保留；资产未标定、工作区对齐差是主要限制。
- **视觉模型可加载导出包**：`scripts/export_visual_package.py` → `deploy/visual_hybrid_package_v2/`（model.pt + manifest 契约 + policy_code 适配器 + run_http.py）。固定输入两次加载最大差 **0.0**；包内 `run_http.py` 经 HTTP 实际跑 30 tick 闭环（无异常）。
- **PolicyAdapter**：`scripts/visual_policy_adapter.py` 封装图像/状态/前动作预处理、归一化与动作后处理，直连 Python 与 HTTP 共用同一契约。
- **部署中心接入视觉包**：`GET /api/v1/deployments`（含 kind/robot）、`/{name}`（契约明细）、`/{name}/compat`（按包契约 vs 项目机器人）；`export_visual` 后台任务（WSL）构建视觉包。浏览器实测：选 `visual_hybrid_package_v3` 显示契约与权重哈希、对 5R **兼容**；对 `cartesian_3dof` **不兼容**（关节名/顺序差异）。截图：`experiments/20261110-p2/page-deployment.png`。

## 续：三维场景编辑画布（本轮）

- `/scenes/:id/edit` 新增**三维画布**（`frontend/src/components/SceneCanvas.tsx`）：加载场景规范化 `visual.obj`，碰撞代理以线框显示；点选实例/代理后 `TransformControls` 平移并回写数值；网格开关；`OrbitControls` 与 gizmo 互斥。
- 编辑器支持：撤销/重做（实例与代理快照）、未保存标记、放弃修改、保存（`scene/instance` + `colliders` 写入项目配置新版本）、**两点测量 + 已知距离缩放**（按已知/测量比例更新 uniform_scale）；高斯/点云资产显示全景预览并说明“无 MJCF 网格视觉，渲染由 hybrid 负责”。
- 浏览器实跑：切到已绑定项目 `test_scene_edit` → 添加代理 → 保存 → 后端 `GET /projects/test_scene_edit/scene` 确认 colliders 变为 desk/post/proxy_3（配置 v14）。截图 `experiments/20261110-p2/page-scene-editor.png`。
- 第三轮已补：实例/代理的旋转 gizmo 与四元数数值字段、世界变换与圆柱轴向修正、单请求原子保存与 409 版本冲突、撤销/重做。仍限制：测量为数值点而非画布点击拾取；撤销覆盖结构/变换提交（不逐键）；自动化未模拟鼠标按住 gizmo 拖动（走同一提交路径注入非零初值，建议人工复测）；高斯资产仅全景预览，不称三维高斯编辑器。

## 续：Web 单体生成入口 + 阶段文档（本轮）

- **Lux3D 单体生成 Web 入口**：`GET /api/v1/lux3d/status`（configured 状态）、`POST /api/v1/lux3d/generate`（无 `AHOLO_API_KEY` 返回 409 明确提示；有 key 时创建 `lux3d_generate` 后台任务，复用 `asset_pipeline/generate_lux3d.py`，命令构造已验证）。资产中心新增“单体资产（Lux3D）”表单（名称/提示词/提交），无 key 显示等待配置。真实在线生成未验证（无凭据）。
- **阶段文档**：`docs/RENDERING_PIPELINE.zh-CN.md`、`docs/SCENE_EDITOR.zh-CN.md`、`docs/PHASE2_VALIDATION.zh-CN.md`（T01–T18 逐项）、`docs/FEATURE_STATUS.zh-CN.md`。

## 本轮新增（已实现 + 已验证）

1. **真实高斯 PLY 解析**（`vlab/gaussian.py`）：读取 `world_586c79650960/1/source/scene.ply`（binary_little_endian，199747 顶点）。正确解码均值、`exp(scale)`、`sigmoid(opacity)`、SH DC→RGB（0.2820948·dc+0.5）、四元数归一化（rot_0=w）。**不降级为点云**。
   - 实测：count=199747；scale>0；opacity∈[0,1]；rgb∈[0,1]；四元数范数偏差<1e-3；bbox min[-4.711,-3.042,-1.640] / max[4.407,4.196,1.944]（米，Z-up）。
2. **采集时刻不可变快照**（`vlab/rendering/snapshot.py`）：在相机采样时刻从当前 MuJoCo data 复制 qpos、连杆/任务物 xpos/xmat、相机世界位姿（`cam_xpos/cam_xmat`，含腕部挂载）与内参/分辨率；数组为拷贝，后续物理步不得改变。
   - 接入 `engine._push_frame`：每帧带 `camera_world_pos`/`camera_world_mat`，使帧与捕获时刻相机位姿可对应（为异步渲染器铺路）。
   - 实测：快照相机位姿==`data.cam_xpos`；腕部相机随 eef 移动；步进后快照不变（确为拷贝）；帧含相机世界位姿；`capture≤available≤decision`。
   - 测试：`tests/test_gaussian_snapshot.py` 全过。

## 未完成/限制（截至第三轮）

- **完整遮挡与独立离屏服务**：splat 光栅化、CPU 渲染与解析盒遮挡均已验证，但 hybrid 采用受限模式 `static_background_overlay`——splat 不遮挡 MuJoCo 机器人/任务物；独立 ray_box 测试不代表真实场景遮挡。独立离屏服务（浏览器 Spark 路径已验证）未做成自动化服务。
- **腕部/运动相机 hybrid**：明确拒绝（`UnsupportedMode`），仅支持世界固定 front 相机。
- **高斯场景标定**：地面平移后方差仅 +6%，水平/尺度仍未精化，背景与工作区存在对齐差。
- **Web 单体/场景生成**：Lux3D Web 入口与命令构造已验证，真实在线生成无 `AHOLO_API_KEY`，未验证。
- **部署**：视觉模型包已自包含并隔离验证；通用 PolicyAdapter 仍只覆盖 VisualBC，其他目标平台未接。
- **实机/sim2real**：无硬件，未验证；不宣称数字孪生。

历史（第二轮末）状态“hybrid 未实现、编辑器未实现、导出包未覆盖视觉模型”已被后续三轮完成，见上。

## 与旧配置的兼容

- 高斯资产解析为只读能力，不影响现有 mesh/默认环境与采集/训练回归；`MujocoSensorRenderer` 路径未变。
- 快照为附加元数据，不改变 observation 既有字段与时间语义。

## 建议下一步

1. 固定一个 3DGS 渲染方案（先做 front 单相机离屏小样：加载真实 scene.ply、动态机械臂/mesh 同屏、指定内外参读回 RGB、基础遮挡），记录版本/显存/耗时；通过后再接 wrist/延迟/吞吐。
2. 场景编辑器（大画布 + 校准 + 代理绘制 + 版本保存），复用现有 TransformControls 与 `scene.colliders` 语义（frame=world 默认）。
3. 高斯场景离线采集 → LeRobot（保留逐帧 capture/available/frame_id）→ VisualBC 闭环，分组报告真实成功数。
