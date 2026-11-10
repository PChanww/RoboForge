# RoboForge — 审计与实验结果

[English](AUDIT.md) | **简体中文** | [返回 README](../README.zh-CN.md)

本文集中保存原 README 中的实验结果、验证证据、实现限制与历史成绩。内容依据已有报告整理，本次文档调整没有重新运行实验。各项结果对应原报告记录的配置与代码版本。

## 结果与限制

下表引用已保存的实验报告。脚本专家和学习策略的成绩分别列出。

| 实验 | 已记录结果 | 证据 |
| --- | --- | --- |
| 当前 5R 机械臂，脚本抓放专家，4 杯型 × 10 种子 | **40/40 成功** | [机械臂审计](../experiments/20261109-arm-audit/REPORT.md) |
| 相同运动路径、夹爪始终打开 | **0/40 成功** | [机械臂审计](../experiments/20261109-arm-audit/REPORT.md) |
| 采样任务工作区内的自碰撞 / 解析 IK 误差 | 该测试中未检测到自碰撞 / 运动学误差 0.0 mm | [机械臂审计](../experiments/20261109-arm-audit/REPORT.md) |
| 30 FPS 相机、20 Hz 控制 | 9 秒内 271 帧，平均间隔 0.03333 秒 | [核心报告](../experiments/20261109-platform-core/REPORT.md)、`tests/test_engine.py` |
| 闭环位置伺服专家（阶段按位置/接触切换） | **11/11 成功**，约 93 tick/episode | [数据与训练报告](../experiments/20261110-data-training/REPORT.md) |
| 专家数据集（伺服示范） | 21 episode / 1952 帧，官方 LeRobot loader 回读通过 | [数据与训练报告](../experiments/20261110-data-training/REPORT.md) |
| 状态 BC 策略（可部署观测，无物体位姿） | 已完成训练、重载与评测；**闭环 3/11** | `datasets/reports/eval_policy.json` |
| 特权 BC 策略（含物体位姿） | **闭环 10/11**（留出摆位 4/4、留出资产 2/2） | `datasets/reports/eval_priv.json` |
| 导出包在新进程中运行 | 专家成功，学习策略执行完成但未完成任务 | [部署报告](../experiments/20261110-deploy-export/REPORT.md) |

当前限制：

- 5R 机械臂采用固定末端朝向的解析 IK，支持的任务空间受到机构自由度约束。
- 模仿质量取决于专家设计与可观测性：原开环定时专家闭环 0/11；改用闭环位置伺服专家后特权策略达 10/11。仅状态观测的策略为 3/11，差距来自缺少物体位姿——可部署的完整性能仍需视觉策略（SmolVLA / 图像 BC）。
- SmolVLA 曾在历史龙门版本真实推理，未完成抓放；新机械臂的观测动作适配和 SmolVLA 微调仍待完成。
- 实机迁移尚未验证。URDF/USD 导出、Isaac Lab、ROS 2/厂商 SDK 和 ONNX 适配仍待实现。
- 辅助模型的结构化导入已经检查，真实辅助模型 API 调用尚未验证。
- 控制器插件运行可信本地代码，远程代码进程隔离与资源限制尚未实现。
- 调度采用仿真时间，实时墙钟 pacing 尚未实现。
- 平台接口按配置中的关节集合运行，部分采集与训练脚本仍面向当前 5R 数据集。

## 验证证据索引

| 阶段 | 范围 | 报告 |
| --- | --- | --- |
| A | 机械臂碰撞/IK 审计、脚本抓放与阴性对照 | [机械臂审计](../experiments/20261109-arm-audit/REPORT.md) |
| B | 配置、时间调度、会话、控制权与 Gymnasium | [平台核心](../experiments/20261109-platform-core/REPORT.md) |
| C | 场景编辑器、版本保存与相机/关节参数应用 | [前端](../experiments/20261109-frontend/REPORT.md) |
| D | 数据回读、对齐检查、训练/重载与策略评测 | [数据与训练](../experiments/20261110-data-training/REPORT.md) |
| E | 部署包、第二种机器人、MJCF 与适配状态 | [部署与导出](../experiments/20261110-deploy-export/REPORT.md) |

报告记录功能覆盖情况和仍存在的失败。管线检查通过后，学习策略的任务成功率与实机迁移效果仍需要分别验证。

## 历史龙门版本结果

历史程序杯专家 **30/30**、Lux3D 杯专家 **10/10** 成绩属于旧龙门版本。杯子来自 Lux3D 任务 **5018305**，机械臂来自任务 **5025807**。历史 SmolVLA 执行过真实模型动作，未完成抓取。参见[基线报告](../evidence/baseline-results.json)、[对照视频](../evidence/baseline-comparison.mp4)、[Lux3D 杯结果](../evidence/lux3d-physics.json)和 [SmolVLA 动作记录](../evidence/lux3d-smolvla.json)。
