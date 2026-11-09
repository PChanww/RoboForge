# 阶段 A 验证报告：五旋转关节机械臂抓放基线

日期：2026-11-09 · 实验目录：`experiments/20261109-arm-audit/`

## 环境实测

| 项 | 实测值 |
| --- | --- |
| 仿真 Python | work/venv，3.13.17（Windows） |
| 物理引擎 | MuJoCo 4.0.0（mujoco-core/studio 同版） |
| 依赖 | numpy 2.5.4、pillow 13.0.0、trimesh 5.2.0、websockets 17.4 |
| VLA 推理 | WSL2 Ubuntu-22.04，miniconda env `voxcpm`，Python 3.11.15 |
| VLA 依赖 | torch 2.6.0+cu124（CUDA 可用）、lerobot 0.4.4、transformers 4.57.3，PYTHONPATH=work/policy/deps |
| GPU | RTX 4070 Laptop 8 GB |
| 端口 | 8765 有旧进程（PID 32336，今日 19:30 启动）；8767 空闲 |

**进程与磁盘不一致**：8765 进程的 observation 缺少 `robot`/`joints_rad` 字段，装载的是中间版代码。需重启该进程才能加载本次修复。本阶段验证全部在独立进程完成，未受影响。

## 定位的问题与修复（arm.py）

1. **腕部永久穿透**：tool 球体（contype=1）与前臂碰撞胶囊端部重叠 20 mm（胶囊端点距关节仅 .015 m，球心就在关节上）。修复：关节座球体（turret/forearm/wrist hub、tool 球）保留质量但设为非碰撞（contype=0 conaffinity=0），连杆碰撞由胶囊承担。
2. **掌部与前臂擦碰**：碰撞胶囊端部缩短至距关节 .07 m；掌部盒缩小为 .016×.040×.008 m 并下移（TOOL .067→.075 m，IK 同步使用同一常数）。
3. 修复前：专家基线无法提起杯子，最大跟踪误差 67.7 mm。修复后：平均跟踪误差 2.5 mm，最大 9.5 mm（起步伺服暂态），固定姿态 IK 姿态偏差 ≤0.39°，运动学 IK 误差 0.0 mm。

碰撞位掩码说明：连杆胶囊 contype=2/conaffinity=1，与世界/杯子/夹爪碰撞，连杆之间不碰撞（仅两根胶囊且为父子关系，本就被引擎过滤）；非相邻约束（如肘部全折时夹爪撞大臂）保持有效。

## 验证结果（validation.json）

- **脚本专家基线 40/40 成功**：standard/tall/wide/imported-lux3d-ivory 各 10 种子。成功判据为双指接触抬起 >31 mm、释放后托盘内直立静置 0.3 s（物理接触判定，非脚本自评）。
- **开爪阴性对照 0/40 成功、0 次抬起**：相同 XYZ 路径全程开爪。
- **任务工作区包络**（7×6×6 网格）：零自碰撞、零不可达、零限位越界、IK 运动学误差 0.0 mm。
- **机械停止有效**：肘部全折（3.08 rad）时夹爪与大臂碰撞（-12.5 mm），非相邻连杆约束未被禁用。
- **腕部-前臂间隙**：wrist_pitch 全范围 40 点扫描零接触（设计目标）。
- 全动作箱扫描：仅基座正前方（x≈0, y≈0.17）近肩折叠死区有 5 处接触，属真实机械限制，已记录于 `full_box_dead_zones`。

证据文件：`diagnosis.json`（修复前后对比）、`validation.json`（含代码/资产 SHA-256、物理参数、逐种子结果）、`grasp-tick*.jpg`（抓取/搬运/放置画面）、`rest.jpg`。

## 历史证据与当前代码的区分

- `evidence/baseline-results.json`、`lux3d-physics.json` 的 30/30、10/10 为**旧龙门夹爪**成绩，不适用于当前机械臂。
- `evidence/lux3d-smolvla.json`、`integrations/rollout-seed0.json` 为真实 SmolVLA 推理记录，未完成抓放；脚本成绩不能归于 VLA。
- 本目录 40/40 为**当前五旋转关节机械臂 + 脚本专家**成绩，VLA 尚未在新契约下复测（阶段 D）。

## 未解决限制

- 固定末端朝向解析 IK，不支持任意六维位姿（5 自由度机构约束）。
- 近肩死区由碰撞机械停止兜底，未做可达性预检接口（阶段 B 能力发现中声明）。
- 8765 旧进程需重启；README 历史段落已加版本说明。