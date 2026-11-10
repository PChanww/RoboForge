# P1 报告：三维主视图布局 + 多页面路由 + 网页任务闭环

日期：2026-11-10 · 截图：本目录 `layout-*.png`、`page-datasets.png`

## 布局（仿真工作台）

- 三栏：左场景树（默认 200px，可折叠/拖宽）、中央弹性三维主视图、右属性（默认 280px，可折叠/拖宽）。
- 中央下方为受控高度停靠区（默认 148px，可拖高/收起），标签「相机 / 日志 / 曲线」，默认相机；相机卡片固定 180px 宽、固定高容器 + object-contain，横向滚动，不随屏幕宽度膨胀成大正方形。
- 相机卡片显示名称/原始分辨率/配置 FPS/实际采集 FPS/图像年龄/等待/掉线状态；点击选中相机；⤢ 打开可关闭放大浮层。
- 视图工具栏：聚焦/复位/俯/正/侧/全屏 + 网格/关节轴/视锥/碰撞开关；TransformControls 与 OrbitControls 互斥（拖拽时禁用轨道）。
- 面板折叠/宽度/停靠高度存 localStorage；<1280 自动收右栏、<1024 收左栏；窄屏无横向溢出（见 layout-cameras.png 窄屏响应式）。
- 相机视锥由真实内参（fy/分辨率）计算并考虑挂载连杆世界变换。

## 多页面路由（React Router）

全局导航：项目 / 资产 / 数据集 / 训练 / 评测 / 部署 / 设置。路由：
`/projects`、`/projects/:id/sim`、`/library`、`/library/:assetId`、`/robots/:id/edit`、`/datasets`、`/datasets/:id`、`/training`、`/training/:jobId`、`/evaluations`、`/evaluations/:runId`、`/deployments`、`/deployments/:packageId`、`/settings`。深链接/刷新/后退可用（dev 由 Vite SPA 回退）；切换路由不销毁后台会话（store 会话保持，返回恢复）。

## 网页任务闭环（后端 jobs）

`vlab/jobs.py`：持久化任务（queued/running/succeeded/failed/cancelled/interrupted），subprocess 参数数组白名单（collect/train/eval/export/convert/checks），日志落盘、停止 terminate、重启识别 interrupted。训练/转换/检查走 WSL 学习环境，采集/评测/导出走仿真环境。

- 实测：网页「质量检查」→ checks 任务 succeeded；网页「采集（伺服专家）」→ collect 任务 **succeeded**（21 episode 重采）。任务表实时轮询显示状态（page-datasets.png）。
- API：`/api/v1/jobs`(POST/GET)、`/{id}`、`/{id}/logs`、`/{id}/stop`、`/api/v1/projects`、`/api/v1/datasets`、`/api/v1/datasets/{id}`、`/api/v1/asset-library`、`/api/v1/deployments`、`/api/v1/generation-jobs`。

## 渲染可靠性

- 后台无桌面进程 GLFW 不稳（gladLoadGL/段错误）。引擎渲染失败优雅降级（camera_status=render_unavailable，物理仍运行）；**推荐用交互式桌面窗口启动后端**（`Start-Process -WindowStyle Minimized` 或普通终端），实测相机 front/wrist 均 ok。

## 验证

- 前端 `vite build` 通过；P0 回归（camera latency / kinematics / control）全过。
- 截图：layout-default.png（窄屏响应式）、layout-cameras.png（相机预览+主视图）、page-datasets.png（多页面+任务表）。

## 未完成 / 依赖（如实）

- 宽屏 1440/1920 三栏占比截图未能在内置浏览器（窄视口）采集；布局逻辑按宽度默认展开，需宽视口复核。
- 资产真实生成（Aholo）需 `AHOLO_API_KEY`；当前为等待配置；网页本地 GLB 导入上传入口未接（CLI 复用 import_asset.py）。
- 视觉策略（图像 BC / SmolVLA 新契约）未训练；评测/训练页已接状态/特权 BC。
- 评测/部署详情页为列表级占位（`/:runId`、`/:packageId` 回落到列表），机器人编辑页指向仿真属性面板。
- SPA 生产回退需在静态服务器配置（dev 已验证）。