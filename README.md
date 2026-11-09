# 桌面 VLA 抓放实验台

> **版本说明（2026-11-09）**：当前代码已切换为 Lux3D 生成的五旋转关节固定基座机械臂（`arm.py`），不再是下文历史段落描述的三轴龙门夹爪。机械臂抓放已通过真实物理验证：4 种杯子 × 10 种子脚本专家 40/40 成功，开爪阴性对照 0/40，见 [阶段 A 验证报告](experiments/20261109-arm-audit/REPORT.md)。下文接口行为不变，但历史成绩（30/30、10/10）属于旧龙门机构，不适用于当前机械臂。若 8765 端口服务进程启动早于代码更新，需重启后才是当前版本。

仿真任务是用桌面机械臂把杯子放稳在托盘内。浏览器显示 MuJoCo 相机和实时接触结果。外部控制器可观察图像、发送动作，再读取新的图像。

在本机工作区根目录运行以下命令，然后打开 <http://127.0.0.1:8765>。本机已安装运行环境。

```powershell
.\work\venv\Scripts\python.exe .\outputs\vla-lab\server.py
```

在另一台机器上，可在项目目录中用 Python 3.13 的虚拟环境安装 `python -m pip install -r requirements.txt`，然后运行 `python server.py`。

选择杯型和初始位置种子，点「应用设置并重置」。点击「运行脚本基线」可验证物理抓放。真实 SmolVLA 的推理和动作投影见 [外部策略使用说明](integrations/README.md)。界面显示的成功条件包含双指接触中抬高超过 31 mm、释放后接触托盘并连续放稳 0.30 秒。脚本能够读取初始物体位置。VLA 的输入只有图像、夹爪状态和指令。

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
