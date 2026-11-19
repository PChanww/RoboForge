# RoboForge

[English](README.md) | **简体中文**

可配置的机器人仿真与学习工作台。通过提示词生成或导入三维资产，配置机器人关节、物理参数、相机和控制器，随后采集示范、评测策略、训练基线模型并导出部署材料。

> 生成资产 → 配置关节与传感器 → 接入控制器 → 运行任务 → 采数据 / 训练 → 评测 / 导出

当前物理后端为 MuJoCo，编辑器采用 React + React Three Fiber，数据转换与学习脚本对接 LeRobot 和 PyTorch。

## 当前能力

- **项目版本管理。** 使用带 `schema_version` 的配置保存连杆树、关节、执行器、相机、控制器和任务。保存产生新版本，在重建会话时应用。
- **两种机器人配置。** Lux3D 生成的五旋转关节固定基座机械臂，以及用于检验不同机器人结构的直角坐标龙门配置。
- **交互编辑。** 在场景树选择对象，编辑关节、连杆、相机和任务属性，保存并应用；三维网格位姿来自后端物理状态。
- **传感器与时间设置。** 相机可固定在世界或挂载到连杆，支持针孔内参、分辨率、帧率和延迟配置。物理、控制、相机采集和策略时间由仿真时钟调度，记录采集与可用时间戳。
- **控制器与策略接口。** 按机器人能力提供 `joint_position`、`cartesian_delta_ik` 模式，支持 HTTP/WebSocket 观测—动作循环和本地 Python 控制器插件。各会话具有独立环境与明确的控制权。
- **数据与学习脚本。** 专家示范采集、LeRobot 数据转换与回读、数据检查、对齐回放、状态行为克隆（BC）、checkpoint 重载和闭环评测。
- **导出。** 导出配置、资产、权重、输入输出契约及运行示例，并支持 MJCF 导出。部署运行器目前需要项目的 `vlab` 包和 Python 依赖。

网页自由观察相机与策略使用的传感器相机分别配置。当前界面主要使用中文，文档提供中英文版本。

实验结果、验证证据与实现限制见[审计报告](docs/AUDIT.zh-CN.md)。

## 快速开始

本地 C 盘移动搬运的完整流程、CUDA 启动、停止保存和卡顿排查见 [VLA 仿真工作台操作手册](frontend/VLA_WORKBENCH.md)。

以下命令从项目目录 `vla-lab` 执行。已准备的主机使用 Windows/PowerShell，仿真环境为 Python 3.13。前端需要 Node.js/npm，相机渲染需要可用的 OpenGL 上下文。

### 使用当前工作区环境

在一个终端启动后端：

```powershell
..\..\work\venv\Scripts\python.exe -m uvicorn vlab.api:app --host 127.0.0.1 --port 8767
```

在另一个终端启动前端：

```powershell
cd frontend
npm ci
npm run dev
```

打开 [http://localhost:5173](http://localhost:5173)。Vite 将 `/api` 和 `/assets` 代理到 8767 端口的后端。

在场景树选择关节或相机，修改属性后点击「保存并应用」。点击「运行专家」可以运行脚本控制器。

### 新建仿真环境

在 `vla-lab` 中创建并激活 Python 3.13 环境：

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn vlab.api:app --host 127.0.0.1 --port 8767
```

然后按上面的步骤启动前端。修改后端端口时，同步修改 `frontend/vite.config.ts`。

核心依赖覆盖仿真、API 和 API 测试。LeRobot/PyTorch 训练、资产生成和 MP4 回放有下文说明的额外依赖。

## 工程结构

```text
vlab/                配置驱动物理环境、API、会话、调度器、Gymnasium 封装
frontend/            React、TypeScript、Vite、React Three Fiber 编辑器
configs/             默认项目与第二种机器人配置
projects/            持久化项目版本
assets/              机器人网格、生成物体与来源记录
asset_pipeline/      Aholo 生成、GLB/ZIP 导入、机械臂网格绑定
plugins/             本地控制器插件示例
scripts/             采集、转换、训练、评测与导出工具
tests/               物理环境和 API 检查
datasets/            本地暂存数据、模型权重、回放和报告
deploy/              生成的部署包和 MJCF
experiments/         按日期保存的实验报告与验证证据
integrations/        历史 SmolVLA 客户端与运行说明
sim_env.py, arm.py    原型使用的机械臂与任务实现
server.py            旧版 HTTP 服务，默认端口 8765
index.html           旧版前端
evidence/            历史龙门试验材料
```

新克隆的仓库可能没有生成的数据集和部署包。请参照 `.gitignore`，需要时使用脚本重新生成。

## 平台 API

当前后端使用 **8767** 端口。启动后可在 [http://127.0.0.1:8767/docs](http://127.0.0.1:8767/docs) 查看交互式接口定义。

| 方法与路径 | 用途 |
| --- | --- |
| `GET /api/v1/capabilities` | 引擎版本、机器人能力和控制器/传感器/任务描述 |
| `GET /api/v1/config` | 读取当前项目配置 |
| `POST /api/v1/config/validate` | 校验结构与物理构建 |
| `PUT /api/v1/config` | 校验并保存配置版本 |
| `GET /api/v1/config/versions` | 列出不可变版本 |
| `GET /api/v1/config/versions/{version}` | 读取指定版本 |
| `POST /api/v1/sessions` | 创建独立环境会话 |
| `POST /api/v1/sessions/{sid}/reset` | 重置会话 |
| `POST /api/v1/sessions/{sid}/control/acquire` | 获取控制 token |
| `POST /api/v1/sessions/{sid}/control/release` | 释放控制权 |
| `POST /api/v1/sessions/{sid}/action` | 携带 token 和当前 tick 执行一步动作 |
| `POST /api/v1/sessions/{sid}/action_block` | 执行动作块 |
| `GET /api/v1/sessions/{sid}/observation` | 读取按名称组织的关节状态、相机帧与时间信息 |
| `GET /api/v1/sessions/{sid}/result` | 读取观测、任务与终止状态 |
| `GET /api/v1/sessions/{sid}/frame.jpg` | 获取相机画面 |
| `DELETE /api/v1/sessions/{sid}` | 关闭会话 |
| `WS /api/v1/ws/{sid}` | 通过 WebSocket 获取控制权、执行动作和读取观测 |

过期动作与控制权冲突返回 HTTP 409，动作维数错误或数值越界返回 HTTP 400。支持的动作模式由机器人配置决定。

Gymnasium 接口位于 `vlab/gym_env.py`，类名为 `VLabGymEnv`，分别返回 `terminated` 和 `truncated`。

## 采集、训练与评测

使用仿真 Python 环境，在 `vla-lab` 执行：

```powershell
python scripts/collect_expert.py
python scripts/replay_episode.py --episode 19
```

采集器将预定义的 21 个 episode 写入 `datasets/staging`。回放需要 PATH 中可用的 `ffmpeg`，输出 MP4 到 `datasets/replay`。

转换与训练使用独立学习环境。已记录的 WSL 环境为 Python 3.11、LeRobot 0.4.4、PyTorch 2.6.0+cu124 和 Transformers 4.57.3。这些依赖独立于核心 `requirements.txt`，脚本还使用 pandas 和 pyarrow 等 Parquet 引擎。

在该环境的项目目录执行：

```bash
python scripts/lerobot_writer.py --staging datasets/staging --out datasets/lerobot/expert21
python scripts/data_checks.py --lab .
python scripts/train_bc.py --lab .
python scripts/eval_policy.py --group all
```

`eval_policy.py` 也可使用导出的 NumPy 权重在仿真环境中运行。可选的特权状态诊断流程为：

```bash
python scripts/train_priv.py --lab .
python scripts/eval_policy.py --priv --group heldout_spawn
```

`train_bc.py` 使用 episode 0–14 训练，15–18 验证留出摆位，19–20 评测留出资产。PyTorch checkpoint 和 NumPy 权重写入 `datasets/models`。如果已有 `datasets/dagger/*.npz`，训练会将其加入；比较实验时需要记录这些额外输入。

上述命令会生成或更新本地数据、权重与报告。当前 BC 模型是状态输入的 MLP。SmolVLA 推理步骤和历史动作投影的限制见 [integrations/README.md](integrations/README.md)。

## 导出与扩展

```powershell
python scripts/export_deploy.py
python scripts/export_mjcf.py
python scripts/test_second_robot.py
python scripts/test_aux_plugin.py
```

部署导出依赖已有的 BC 权重、训练报告与评测报告，输出到 `deploy/lux3d_5r-v4`，包含配置、资产、权重、输入输出 manifest 和 `run.py`，随后在新进程中分别验证专家与策略执行。

从 `vla-lab` 手动运行导出包时，使用仿真 Python 环境并指定项目包位置：

```powershell
$env:VLAB_LAB = (Get-Location).Path
python deploy/lux3d_5r-v4/run.py --controller expert --ticks 200 --report deploy/lux3d_5r-v4/manual-expert.json
```

迁移到其他机器时，还需要提供 `vlab` 包及兼容的 Python 依赖。MJCF 导出按文件名引用网格，需要将相应网格放到导出模型可访问的位置。

- **资产。** 参见[资产生成与导入](asset_pipeline/README.md)。现有 Lux3D 来源记录保存在 `assets/robot/source.json` 和 `assets/imported/lux3d-ivory/manifest.json`。API 生成使用可选 Aholo SDK 和 `AHOLO_API_KEY`。
- **本地控制器。** 插件暴露 `create_controller(config)`，并实现 `reset(obs)`、`act(obs)`。参见 `vlab/plugins.py` 和 `plugins/example_openloop.py`。
- **辅助模型。** `vlab/aux_model.py` 读取 `AUX_MODEL_BASE_URL`、`AUX_MODEL_API_KEY`。未配置时返回 `waiting_for_configuration`，可进行结构化候选校验。模型名称通过适配器配置。

凭据保存在本机环境中，避免写入源码、前端、日志和截图。

## 开发检查

```powershell
python tests/test_engine.py
python tests/test_api.py
cd frontend
npm run build
```

## 场景与视觉训练

可选 World SDK：`python -m pip install -r requirements-world.txt`，在后端环境配置 `AHOLO_API_KEY`。资产中心支持场景导入、生成和显式查询任务；成功下载后登记场景版本。高斯资产仍仅供预览，尚不支持训练渲染。

网格导入会实际应用单位比例；物理使用需确认尺寸，并保留默认桌面或提供碰撞代理。仿真页按路由加载对应项目。视觉采集、转换、训练和评估可选择项目、数据集与模型，作业保存配置快照；转换不会删除已有数据集。新模型记录输入/动作契约，旧模型需通过 CLI 明确选择历史契约假设。

## 对话式生成闭环

把自然语言需求变成可训练场景的连通闭环：对话（可选）或直接表单/JSON → 版本化 `AssetSpec`/`PhysicsSpec`
（字段级 provenance）→ 确定性编译预览 → 显式确认 → Lux3D 单体 / World 背景生成（幂等、节点级恢复）→
导入校验 → `SceneRecipe` 组装 → 真实 MuJoCo 校验 → 原子应用到新项目版本 → 仿真 / 采集。

- 语言模型只理解与建议，程序校验、编排、落地。估计的质量/摩擦/尺度保持 `model_inferred/unverified`，
  不会变成"已测量"。未知尺寸 → 未标定视觉资产，处理前不绑定物理。
- 未配置 `AUX_MODEL_BASE_URL`/`AUX_MODEL_API_KEY` 时对话显示 `waiting_for_configuration`；表单/JSON/本地导入
  仍可用，不用模板冒充模型。凭据仅后端环境读取。
- 两种场景模式：`composed_scene`（参数化工作台+显式静态代理+可操作对象，无需 World/标定）与
  `background_scene`（World 外观+显式代理；高斯需受限固定前视 `hybrid`）。远程生成成功 ≠ 已导入 ≠ 可训练。
- API 位于 `/api/v1/generation/*` 与 `/api/v1/scene-recipes/*`；页面：生成工作台、任务详情、RecipePanel
  （校验后原子应用到项目版本并读回）。
- 采集 manifest 记录任务资产 mesh/manifest 哈希、生成来源、recipe/spec revision、物理 basis、相机与 `renderer_version`。

详见 [生成闭环工作流](docs/GENERATION_WORKFLOW.zh-CN.md) 与
[第四轮测试/能力报告](docs/ROUND4_GENERATION_REPORT.zh-CN.md)。

## 历史原型

`server.py` 默认端口为 **8765**。8767 端口的平台后端也提供旧版 `/api/*` 兼容路由。

| 路由 | 旧版契约 |
| --- | --- |
| `GET /api/assets` | 可用杯子资产 |
| `POST /api/reset` | 输入 `asset`、`seed`，返回新 episode |
| `GET /api/observation?images=1` | 末端 XYZ、夹爪开度、指令、两张 Base64 编码的 256×256 JPEG |
| `POST /api/step` | 输入 `episode`、`tick`、`action: [dx, dy, dz, grip]` |
| `GET /api/result` | 接触、抬升、放置与任务指标 |

旧版四维动作均在 `[-1, 1]` 内。每步平移目标增量最大 0.012 米，并进行工作区裁剪。grip −1 合爪，+1 开爪。每步前进 0.05 秒仿真时间，过期 episode/tick 返回 HTTP 409。

`python evaluate.py --no-video` 会使用当前 `sim_env.py` 运行旧评测器，结果随检出的实现而变化，应保存为新的试验记录。
