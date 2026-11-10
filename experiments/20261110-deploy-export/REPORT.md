# 阶段 E 报告：部署包、第二种机器人、导出与适配点

日期：2026-11-10 · 代码：`vlab/deploy.py`、`vlab/aux_model.py`、`vlab/plugins.py`、`scripts/*.py`

## 部署包导出与独立进程复建（验收 #9 ✅）

`scripts/export_deploy.py` 导出 `deploy/lux3d_5r-v4/`，自包含：

- `config.json`：不可变配置快照；`manifest.json`：观测/动作契约（状态 18 维名称、图像键与 FPS、动作名/维数/范围/坐标系、物理步长/控制频率）、策略类型与归一化、数据版本、评测摘要、资产清单与权重 SHA-256。
- `assets/robot/`：网格与物理参数；`model.npz`：策略权重；`run.py`：独立载入示例；`README.md`。
- **独立进程验证**（`datasets/reports/deploy_check.json`）：新进程 `python run.py` 仅凭包内文件重建 MuJoCo 场景——专家控制器 **成功**（tick 157）；导出的策略控制器正常执行观测-动作循环并记录结果（未成功，见阶段 D 诊断）。两步返回码均为 0。

## 第二种机器人（验收 #11 ✅）

`configs/second_robot.json`：3 自由度直角坐标龙门（prismatic x/y/z + 平行夹爪），与 5R 机械臂自由度与运动方式不同。`scripts/test_second_robot.py` 验证：

- 通用配置校验通过，环境可构建。
- 观测关节按名称给出 `["x","y","z","left","right"]`，动作维数 4 来自配置，`supported_modes==["joint_position"]`（该机构无解析 IK，cartesian 模式返回明确错误）。
- 脚本 joint_position 序列运行 145 tick；接口无固定五/七关节长度假设。

## 导出格式（验收 #9 部分）

`scripts/export_mjcf.py`：用 `mj_saveLastXML` 导出 `deploy/mjcf/lux3d_5r.xml`（11 body / 34 geom / 8 关节 / 7 执行器 / 2 相机 / 5 网格）。坐标（z-up）、碰撞（contype/conaffinity/friction/solref）、相机针孔内参、位置执行器增益与限幅、夹爪滑轨均保留；网格按文件名引用，需一并复制 `assets/robot`。报告：`datasets/reports/mjcf_export.json`。
URDF/USD、ONNX、Isaac Lab 配置：**待适配**（未实现，未声称通过）。

## 辅助模型适配器

`vlab/aux_model.py`：OpenAI-compatible 适配器，凭据来自 `AUX_MODEL_BASE_URL` / `AUX_MODEL_API_KEY`。

- 未配置时返回 `waiting_for_configuration` 并给出期望的结构化 schema（links/joints/assumptions/units/needs_confirmation）。
- 结构化导入 `validate_suggestion`：将候选合并进配置副本、尝试构建环境、逐关节检查单位轴与限位；返回 valid/issues/candidate_sha256/applied_fields，供界面逐项应用/撤销。
- 实测（`datasets/reports/aux_plugin.json`）：状态 `waiting_for_configuration`；结构化导入 `valid=true`（用当前机械臂配置回灌）。真实 API 调用未测（无凭据）。

## 外部控制器插件协议

`vlab/plugins.py`：插件模块暴露 `create_controller(config)`，实现 `reset(obs)` / `act(obs)->action`；加载时记录源码 SHA-256，动作经统一校验。示例 `plugins/example_openloop.py` 实测在会话中完成 40 步观测-动作循环、无异常。安全：此加载器仅用于本地已审计代码；远程上传代码的**进程隔离与资源限制未启用**，为已知待办。

## SmolVLA 与实机

- SmolVLA 推理：历史在旧龙门上真实运行但未完成抓放（`evidence/lux3d-smolvla.json`）。新 5R 契约下 z+0.60 / 七维裁剪适配需重新标定，**本轮未重跑，标记待适配**。
- 微调：未执行（需 SmolVLA 微调管线与匹配动作定义），标记待办。
- 实机：无硬件。接口侧提供 simulate/录制适配点；**实机未验证**，硬件侧限位/限速/超时/急停需在部署时实现。

## 目标平台适配点状态

| 目标 | 状态 |
| --- | --- |
| MuJoCo 独立进程重建 | ✅ 已测试（专家成功，策略运行） |
| LeRobot 数据 | ✅ 已测试（写出+回读） |
| MJCF 导出 | ✅ 已测试 |
| URDF / USD / Isaac Lab / ROS 2 / 厂商 SDK / ONNX | 待适配 |