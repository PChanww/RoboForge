# RoboForge · 可配置机器人仿真与学习平台

在三维交互界面中，从提示词生成或导入机器人与场景资产，配置关节、物理参数、相机和控制器，快速验证不同机器人设计与控制方法；并支持数据采集、模型训练与评测，以及可迁移部署包导出。物理后端为 MuJoCo，前端为 React + React Three Fiber，训练与数据格式对接 LeRobot / PyTorch。

**核心用户流程**

> 提示词生成资产 → 绑定关节与传感器 → 插入控制器 → 运行任务 → 采数据 / 训练 → 评测 / 导出

## 主要能力

- **配置驱动**：带 `schema_version` 的持久化项目配置（场景 / 机器人连杆树 / 关节 / 执行器 / 传感器 / 控制器 / 任务 / 策略）。编辑产生新版本，在明确的重置或重载边界应用；候选配置与已验证配置分开保存，参数可记录来源。
- **多机器人**：首批为 Lux3D 生成的五旋转关节固定基座机械臂，另含 3 自由度直角坐标龙门验证扩展性。接口按名称与配置维数寻址，代码无固定五/七关节长度假设。
- **多频率调度**：物理步长、底层控制周期、相机采样周期、策略周期独立配置，由仿真时钟驱动；记录每个观测的 capture/available 时间、图像年龄、丢帧与实际采样间隔，支持非整数周期比。
- **相机**：可固定在世界或挂载在连杆上；分辨率、针孔内参、帧率、延迟可配；网页自由观察相机与模型传感器相机分离，三维网格位置来自后端权威物理状态。
- **控制器与模型**：`joint_position` 与 `cartesian_delta_ik` 两种模式并声明动作契约；Python 控制器插件协议；外部策略经 HTTP / WebSocket 完成观测-动作循环；每会话独立环境、单一控制权，冲突返回可理解的 409。
- **数据与训练**：LeRobot v3 数据集（多相机 + 状态 + 动作 + 语言指令 + 时间戳 + episode 边界 + 结果），数据检查与图像/状态/动作对齐回放；PyTorch 行为克隆训练、checkpoint 保存/重载、留出 episode 评测。
- **部署**：自包含部署包（配置快照 + 资产 + 权重 + 观测/动作契约 + 独立载入示例），在干净进程复建并运行；MJCF 导出并报告坐标/碰撞/传感器/执行器转换情况。

## 已验证结果（真实物理判定，非脚本自评）

| 项 | 结果 |
| --- | --- |
| 机械臂抓放（4 杯型 × 10 种子，脚本专家） | **40/40** 成功 |
| 相同路径开爪阴性对照 | **0/40** |
| 任务工作区自碰撞 / IK 运动学误差 | 0 / 0.0 mm |
| 30 FPS 相机 @ 20 Hz 控制 | 271 帧 / 9 s，平均间隔 0.03333 s |
| LeRobot 数据集 | 21 episode / 3297 帧，官方 loader 回读通过 |
| 部署包独立进程复建 | 专家控制器重建场景后成功抓放 |

> **诚实说明**：行为克隆策略的闭环成功率目前为 0（开环定时专家的可模仿性局限，已诊断并记录）；SmolVLA 新契约适配、微调与实机验证为待办。脚本成绩与学习策略成绩分别报告，详见各阶段报告与文末验收对照。

## 快速开始

```powershell
# 后端（端口 8767，含旧版 /api/* 兼容）
..\..\work\venv\Scripts\python.exe -m uvicorn vlab.api:app --host 127.0.0.1 --port 8767

# 前端（端口 5173，代理 /api 与 /assets 到 8767）
cd frontend
npm install
npm run dev
```

打开 <http://localhost:5173>：在场景树选择关节或相机 → 修改参数 → 「保存并应用」；「运行专家」观看真实物理抓放。另一台机器可用 Python 3.13 虚拟环境 `pip install -r requirements.txt` 后运行上述命令。

## 历史版本与兼容

当前机构为五旋转关节机械臂。下文的旧版接口与历史成绩（30/30、10/10）属于**三轴龙门夹爪**，仅作兼容与追溯，不适用于当前机械臂；若 8765 端口进程启动早于代码更新，需重启。旧版 `server.py`（8765）接口如下。

| HTTP 接口 | 输入和结果 |
| --- | --- |
| GET /api/assets | 返回可选杯型。 |
| POST /api/reset | JSON 对象包含 asset 和 seed，返回新 episode 和 tick。 |
| GET /api/observation?images=1 | 返回前视和腕部 JPEG 的 Base64、夹爪位置（米）、开度及任务指令。 |
| POST /api/step | 发送 episode、tick 和四维 action。前三维为归一化 XYZ 目标增量（最大每步 12 mm），末维由 −1 合爪到 +1 开爪。 |
| GET /api/result | 返回双指接触、提起、托盘接触、放稳及物体位置等评测信息。 |

每个控制步推进 0.05 秒仿真时间。服务运行一个共享场景；当其他控制器推进或重置环境，旧请求返回 HTTP 409，外部策略需重新读取观测。

**已接入的参赛资产**为 `imported/lux3d-ivory`。它来自 Lux3D G1-Turbo 文生 3D 任务 5018305。[原始 GLB](assets/imported/lux3d-ivory/original.glb) 和 [来源、提示词及尺寸记录](assets/imported/lux3d-ivory/manifest.json) 均已保存。使用三轴夹爪和脚本控制时，种子 0 至 9 的十次运行全部成功。[逐次结果](evidence/lux3d-physics.json) · [种子 0 的抓放画面](evidence/lux3d-grasp.jpg)

同一个 Lux3D 杯子在种子 0 上也由外部 SmolVLA 实际运行过 180 步。模型完成 36 次推理并控制夹爪移动，但在本轮中未碰到杯子。[模型动作与评分](evidence/lux3d-smolvla.json)

运行 `python evaluate.py --no-video` 可复核三个程序杯型的 60 次配对试验。[对照报告](evidence/baseline-results.json) 和 [视频](evidence/baseline-comparison.mp4) 使用脚本控制。其他杯型见 [Lux3D 资产生成与导入说明](asset_pipeline/README.md)。杯身碰撞用显式半径和高度建立圆柱近似，生成网格用于视觉观测。

## 可配置平台后端（阶段 B，2026-11-09）

`vlab/` 是配置驱动的新后端，与旧版 `server.py` 并存。它用带 `schema_version` 的持久化配置（机器人连杆树/关节、相机内外参、控制器、任务）构建 MuJoCo 场景，按仿真时钟做多频率调度，并提供会话隔离与控制权管理。关节和动作维数都来自配置，代码无固定五/七关节长度假设。

在项目目录运行（端口 8767，避开旧版 8765）：

```powershell
..\..\work\venv\Scripts\python.exe -m uvicorn vlab.api:app --host 127.0.0.1 --port 8767
```

| 接口 | 说明 |
| --- | --- |
| GET /api/v1/capabilities | 引擎版本、机器人、支持的控制模式、控制器/传感器/任务描述。 |
| GET/PUT /api/v1/config | 读取/保存配置；保存先做静态+物理构建校验并递增版本号。 |
| POST /api/v1/config/validate | 返回结构化校验问题（含 schema 与物理构建错误）。 |
| GET /api/v1/config/versions[/{n}] | 不可变版本快照列表/读取。 |
| POST /api/v1/sessions | 从配置快照创建独立会话（独立物理线程与环境）。 |
| POST /api/v1/sessions/{id}/control/acquire | 获取控制权，返回 token；冲突返回 409。 |
| POST /api/v1/sessions/{id}/action | 带 token 执行一步；错误维数/越界返回 400，过期 tick 返回 409。 |
| POST /api/v1/sessions/{id}/action_block | 执行动作块。 |
| GET /api/v1/sessions/{id}/observation | 版本化观测：按名称的关节状态、相机帧（capture/available_time）、时间调度统计。 |
| GET /api/v1/sessions/{id}/result | 观测 + 任务判据（terminated/truncated 区分）。 |
| WS /api/v1/ws/{id} | WebSocket 观测-动作循环（acquire/action/observation）。 |
| /api/assets、/api/reset、/api/step 等 | 旧版接口原样保留。 |

Gymnasium 封装见 `vlab/gym_env.py`（`VLabGymEnv`，区分 terminated 与 truncated）。验收测试：`python tests/test_engine.py`、`python tests/test_api.py`。详见 [阶段 B 报告](experiments/20261109-platform-core/REPORT.md)。

## 三维编辑器前端（阶段 C，2026-11-09）

`frontend/` 是 React + TypeScript + Vite + React Three Fiber 的中文编辑器：左侧场景树、中间三维场景（网格按后端权威 body 位姿放置）+ 实时相机画面、右侧关节/连杆/相机/任务属性编辑、底部日志/时间调度/曲线。编辑保存后递增配置版本并在重载会话时应用。

先启动上面的后端（8767），再在项目目录运行：

```powershell
cd frontend
npm install
npm run dev
```

打开 <http://localhost:5173>（Vite 将 `/api` 与 `/assets` 代理到 8767）。选择关节或相机→修改参数→「保存并应用」；「运行专家」可观看脚本抓放。详见 [阶段 C 报告](experiments/20261109-frontend/REPORT.md)。

## 数据、训练与部署（阶段 D/E，2026-11-10）

- **数据采集/LeRobot**：`scripts/collect_expert.py` 采集 21 个专家 episode（3297 帧，双相机+状态+动作+语言指令+成功标签）；`scripts/lerobot_writer.py` 用官方 lerobot 0.4.4 写出并回读校验；`scripts/data_checks.py` 检查时间戳/范围/图像/动作-位移对齐（余弦 0.911）；`scripts/replay_episode.py` 生成对齐回放 MP4。详见 [阶段 D 报告](experiments/20261110-data-training/REPORT.md)。
- **训练闭环**：`scripts/train_bc.py`（WSL torch）真实训练 BC MLP、保存 checkpoint、重载验证（0 误差）、导出 numpy 权重；`scripts/eval_policy.py` 在与训练分开的 episode/摆位上闭环评测。诚实结果：模仿策略闭环成功率 0（原因与下一步见报告）。
- **部署包**：`scripts/export_deploy.py` 导出 `deploy/lux3d_5r-v4/`（配置快照+资产+权重+契约+run.py），独立进程复建并运行（专家成功、策略运行记录）。详见 [阶段 E 报告](experiments/20261110-deploy-export/REPORT.md)。
- **第二种机器人**：`configs/second_robot.json`（3 自由度直角坐标龙门）经通用接口描述并运行，验证无固定关节数假设。
- **辅助模型/插件/导出**：`vlab/aux_model.py`（无凭据时等待配置，结构化导入校验）、`vlab/plugins.py`（控制器插件协议+示例）、`scripts/export_mjcf.py`（MJCF 导出）。

## 验收对照（12 项）

| # | 验收项 | 状态 | 证据 |
| --- | --- | --- | --- |
| 1 | 机械臂真实抓放 + 阴性对照 | ✅ 40/40，开爪 0/40 | [A 报告](experiments/20261109-arm-audit/REPORT.md) |
| 2 | 修改关节参数+相机外参，保存/重开/确认 | ✅ v2→v5 生效 | [C 报告](experiments/20261109-frontend/REPORT.md) |
| 3 | 30FPS/20Hz 多周期按仿真时间运行 | ✅ 271 帧/9s，间隔 0.03333s | `tests/test_engine.py` |
| 4 | 外部控制器/模型客户端观测-动作循环 | ✅ HTTP+WS+插件 | `tests/test_api.py`、`datasets/reports/aux_plugin.json` |
| 5 | 过期动作/错误维数/控制权冲突 | ✅ 409/400 | `tests/test_api.py` |
| 6 | 采集并回读数据集、回放对齐 | ✅ 21 集/3297 帧 | [D 报告](experiments/20261110-data-training/REPORT.md)、`datasets/replay/` |
| 7 | 策略训练/保存/重载/独立评测 | ✅ 管线真实；闭环 0/11（已诊断） | `datasets/reports/train_bc.json`、`eval_policy.json` |
| 8 | VLA 与微调分别记录、成绩分开报告 | ◐ 脚本 40/40；学习 0/11；VLA 待适配 | D 报告 |
| 9 | 部署包独立进程复建运行 | ✅ 专家成功 | `datasets/reports/deploy_check.json` |
| 10 | 新终端按文档启动全流程 | ✅ README 命令 | 本文件 |
| 11 | 两类机器人通用描述，无固定关节长度 | ✅ 5R + 3DOF 龙门 | `datasets/reports/second_robot.json` |
| 12 | 已完成有入口/未通过说明原因 | ✅ 见各报告"未解决限制" | experiments/ |

**未解决限制**：BC 闭环成功率 0（定时专家可模仿性差）；SmolVLA 新契约待适配、微调与实机未验证；URDF/USD/Isaac/ROS/ONNX 待适配；远程插件进程隔离未启用；实时墙钟 pacing 未实现。
