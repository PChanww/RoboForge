# 阶段 D 报告：数据采集、LeRobot 数据集、训练与评测闭环

日期：2026-11-10 · 代码：`vlab/collect.py`、`scripts/*.py` · 数据：`datasets/`

## 采集与数据集

- **采集入口**：`scripts/collect_expert.py` → 脚本专家在配置场景中逐控制步记录：按名称关节状态/速度、末端位姿、夹爪开度、施加动作、双相机帧（含 capture/available 时间）、成功标签、控制来源。21 个 episode＝15 训练（standard/tall/wide 种子 0–4）+ 4 留出摆位（种子 30/31）+ 2 留出资产（Lux3D 杯）。全部 21/21 成功。
- **动作/状态配对齐**：初期把"产生当前状态的动作"错配为"当前状态应执行的动作"，导致闭环失败且离线 MSE 异常低（反向映射）。已修正为"状态 S_t ↔ 在 S_t 执行的动作 a_t"，并新增数据检查：动作平移方向与末端实际位移余弦 **0.911**（对齐良好）。
- **LeRobot v3**：`scripts/lerobot_writer.py` 用官方 lerobot 0.4.4 `LeRobotDataset.create/add_frame/save_episode/finalize` 写出到 `datasets/lerobot/expert21`，再用 `LeRobotDataset(root=...)` **回读校验**：21 episode / 3297 帧 / 20 FPS / observation.state 18 维 / action 4 维 / front+wrist 图像 / 语言指令 `Place the cup in the tray.`。
- **数据检查**（`scripts/data_checks.py`，0 问题）：时间戳单调、动作在 [-1,1]、关节在限位内、图像 256×256、图像年龄均值 7.9 ms（最大 16 ms）、动作-位移方向余弦 0.911。
- **回放**：`scripts/replay_episode.py` 将双相机帧与 frame/时间/末端高度/夹爪/动作叠加成 MP4（`datasets/replay/ep019_replay.mp4`），显示图像、状态、动作对齐。

## 训练与重载（真实优化器）

| 模型 | 输入 | 数据 | 优化步 | val MSE | gen MSE | 重载一致性 |
| --- | --- | --- | --- | --- | --- | --- |
| BC MLP（状态，可部署观测） | 18+4 | 3297 帧（训练 2355） | 1200 | 0.0387 | 0.0300 | 最大绝对差 0.0 |
| BC MLP（特权，含物体位姿） | 21+4 | 同上 | 1000 | 0.0208 | 0.0214 | 0.0 |

均真实执行 Adam 更新、写出 `.pt` checkpoint、保存 numpy 权重、重新加载并验证输出一致。

## 闭环评测（与训练分开的 episode/摆位）

| 组 | 状态策略 | 特权策略 |
| --- | --- | --- |
| 训练内摆位 | 0/5 | — |
| 留出摆位（种子 30/31） | 0/4 | 0/4 |
| 留出资产（Lux3D） | 0/2 | — |

**诊断**：策略在接触前就把夹爪闭合并把夹爪阶段整体"压缩"——例如特权策略在 tick≈25 闭合夹爪（专家在 tick 60），随后提前飞向托盘。原因是脚本专家是**开环定时**轨迹（阶段由 tick 决定），而纯状态/特权模仿策略无法观测"阶段时间"，其每步动作略饱和使轨迹加速，导致阶段提前与夹爪时机错位。这是行为克隆在定时专家上的已知脆弱性，不是平台缺陷：数据、训练、重载、独立评测链路均已真实贯通。

**结论与下一步**：小数据训练贯通实验通过（真实优化器、权重写出、重载、留出评测、模型类型与数据量记录）。要提升闭环成功率，需要（a）视觉策略（SmolVLA / 图像 BC），或（b）改用闭环位置伺服专家（动作由测量位置决定而非 tick 定时），或（c）DAgger 多轮 + 阶段显式输入。当前提供的是诚实基线。

## SmolVLA 状态

历史记录（旧龙门夹爪）：真实推理运行、未完成抓放（见 `evidence/lux3d-smolvla.json`）。当前 5R 机械臂使用同一 4D cartesian 契约，SmolVLA 的 z+0.60 与七维裁剪适配尚未按新契约重新标定，本轮未重跑；标记为**待适配**（阶段 E 逐项验证）。脚本成绩与学习策略成绩分别报告。

## 复现命令

```powershell
python scripts/collect_expert.py                       # 采集 21 episode
python scripts/replay_episode.py --episode 19          # 对齐回放 MP4
# WSL（lerobot 0.4.4 + torch）：
python scripts/lerobot_writer.py --staging datasets/staging --out datasets/lerobot/expert21
python scripts/data_checks.py --lab <vla-lab>
python scripts/train_bc.py --lab <vla-lab>
python scripts/train_priv.py --lab <vla-lab>
# Windows：
python scripts/eval_policy.py --group all
python scripts/eval_policy.py --priv --group all --out datasets/reports/eval_priv.json
```

## 改进（续）：闭环位置伺服专家 → 非零闭环成功率

上文 0/11 的根因是**开环定时专家不可模仿**。新增 `PlatformEnv.servo_expert_action()`：动作＝实测末端到航点的伺服增量，阶段按**位置/接触**切换（接近→下降→闭合→提起→搬运→下放→张开→撤回），不再依赖 tick 定时。该专家自身抓放 **11/11** 成功且更快（约 93 tick）。

用闭环专家重采（21 episode / 1952 帧，`control_source=closed_loop_servo_expert`）、重写 LeRobot、重训后闭环评测：

| 组 | 状态策略（可部署观测，18+4 维） | 特权策略（含物体位姿，21+4 维） |
| --- | --- | --- |
| 训练内摆位 | 1/5 | 4/5 |
| 留出摆位 | 2/4 | **4/4** |
| 留出资产（Lux3D） | 0/2 | **2/2** |
| 合计 | 3/11 | **10/11** |

- 特权策略 val MSE 0.0036、重载 0 误差；状态策略 val MSE 0.0060。
- **结论更新**：训练/保存/重载/独立评测闭环可产出** working 策略**（特权 10/11，含留出摆位与留出资产）。状态策略 3/11 的差距来自**缺少物体位姿可观测性**——可部署的完整性能需视觉策略（SmolVLA / 图像 BC）。此前"0/11"为开环专家基线，已被本闭环结果取代。
- 产物：`datasets/reports/eval_policy.json`（状态）、`eval_priv.json`（特权）、`train_bc.json`、`train_priv.json`。