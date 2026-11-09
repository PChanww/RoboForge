# 阶段 C 报告：三维编辑器前端

日期：2026-11-09 · 代码：`frontend/`（React 18 + TypeScript + Vite 6 + React Three Fiber 8 + Drei + Tailwind + shadcn 风格组件 + Zustand + ECharts）

## 实现内容

- **布局**：顶部工具栏；左侧场景树（机器人/连杆/关节/相机/任务）；中间可交互三维场景 + 实时相机画面；右侧所选对象属性编辑；底部日志/时间调度/曲线（ECharts）。中文界面。
- **三维场景**（`Scene.tsx`）：R3F Canvas，OBJ 网格按**后端权威 body 位姿**（`observation.state.body_poses`）放置，杯子按 `object_pose`+视觉偏移放置；关节轴 gizmo、相机 gizmo 可点选；轨道控制。MuJoCo z-up 经根组旋转映射到 three y-up。渲染器与物理共用同一场景状态，几何/姿态/时间由后端同步（独立渲染器差异：仅视觉材质简化，无几何偏差）。
- **编辑**（`PropertiesPanel.tsx`）：关节类型/轴/限位/初始/阻尼/电枢；连杆原点/质量/重力补偿；相机父坐标系/位置/xyaxes/内参 fx fy cx cy/分辨率/FPS/延迟；任务资产/指令/目标/时限。修改写入本地配置（dirty 标记）。
- **状态**（`store.ts`）：Zustand 管理配置、会话、观测、选择、运行循环、日志。保存=校验→递增版本→重载会话（明确的重置边界应用编辑）。
- **运行**：单步（保持动作）、运行/暂停脚本专家（60ms 间隔调用后端 expert_step）、重置（可选 seed）。实时相机 JPEG、图像年龄、采样间隔、末端高度/夹爪曲线。

## 验证结果（真实执行）

- TypeScript 类型检查通过；`vite build` 生产构建成功（1221 模块）。
- 浏览器 DOM 快照：工具栏、场景树（全部连杆/关节/相机）、相机画面（front/wrist 256×256@30FPS，capture 时间戳）、属性面板、日志均正确渲染；控制台无运行时错误。
- **关节编辑闭环**：肘关节阻尼 0.8→1.5，保存→校验通过→版本 v2→v3→会话重载→后端 `GET /api/v1/config` 确认 damping=1.5。
- **相机外参编辑闭环**：front 位置 x .36→.40，保存→v4→后端确认 [0.4,-0.70,0.46]。（验收 #2 两项均满足）
- **运行抓放**：前端驱动 expert_step 至 tick 157，`success/lifted/grasped=true`，前视相机 capture_time 推进到 7.834s（画面实时更新），末端到达托盘高度。
- 编辑后已恢复验证基线（阻尼 0.8、front [.36,-.70,.46]，v5）。

## 启动

```powershell
# 后端（端口 8767）
..\..\work\venv\Scripts\python.exe -m uvicorn vlab.api:app --host 127.0.0.1 --port 8767
# 前端（端口 5173，代理 /api 与 /assets 到 8767）
cd frontend; npm install; npm run dev
```

打开 http://localhost:5173。

## 截图证据（可见浏览器面板下实拍）

- `editor-initial.png`：初始页面（配置 v5、tick 0、场景树、实时相机、日志）。
- `editor-framed.png`：三维场景完整取景（机械臂、杯子、托盘、关节轴与相机 gizmo）。
- `editor-grasp.png`：运行专家至 tick 80（4.00s），机械臂夹持杯子；肘关节属性面板（轴 0,1,0 / 限位 0.05–3.08 / 初始 1.5 / 阻尼 0.8 / 实时值 1.6172）；front/wrist 画面 capture 4.000s。
- `editor-success.png`：tick 157（7.85s）工具栏「✓ 抓放成功」，杯子已在托盘内，相机画面 capture 7.834s、图像年龄 16ms，日志记录成功。

## 未解决限制

- 连续实时墙钟 pacing 仍为步进/定时驱动；推理超时处理转入阶段 D/E。
- 三维编辑目前通过属性面板数值输入；三维变换 gizmo 拖拽尚未实现（数值输入已满足"真正写入配置并影响后台"）。
- 独立渲染器（R3F）与 MuJoCo 共用后端权威位姿，几何/姿态/时间同步；仅材质简化为近似，无几何偏差。