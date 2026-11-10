# P2 报告：视觉策略、多机器人、部署兼容检查、辅助模型

日期：2026-11-10 · 代码：`scripts/train_image_bc.py`、`scripts/eval_image_policy.py`、`vlab/compat.py`、`vlab/jobs.py`、`vlab/aux_model.py`、前端 pages/SimWorkspace/PropertiesPanel/Scene/TreePanel

## 可复现视觉策略流程（图像 BC，无特权物体位姿）

- 输入仅 front 相机图(64×64) + 关节状态(18) + 上一步动作(4)；CNN+MLP→4 维 cartesian 动作。
- 训练（WSL PyTorch，LeRobot expert21）：1952 帧、440 优化步、val MSE 0.0135、gen MSE 0.0146；checkpoint 保存/重载最大绝对差 0.0（`datasets/reports/train_visual.json`、`models/visual_bc.pt`）。
- 闭环评测（WSL 客户端经 HTTP 驱动 Windows 仿真，按资产建评测项目）：**训练内 1/2、留出摆位 1/2、留出资产 1/2，共 3/6**（`datasets/reports/eval_visual.json`）。这是不含物体位姿的可部署视觉基线，真实成功率含失败如实记录。
- 网页入口：训练中心「训练视觉 BC」、评测中心「评测视觉 BC」（后台 job：train_visual / eval_visual）。

## 多机器人选择

- 默认配置并入第二机器人（cartesian_3dof 龙门），仿真页工具栏新增活动机器人下拉；切换重建会话并按机器人选控制器（5R→cartesian_delta_ik，龙门→joint_position）。
- 移除前端 `robots[0]` 假设：TreePanel/PropertiesPanel/Scene 改用 `activeRobot` selector；编辑回写定位到活动机器人；切换时清空选择避免同名关节歧义。
- 实测：龙门会话 `robot=cartesian_3dof, mode=joint_position, joints=[x,y,z,left,right]`，与 5R 的 7 关节/4 维 cartesian 正确区分。

## 部署契约兼容检查

- `vlab/compat.py` + `GET /api/v1/compat?robot_id=&model=state|priv|visual`：比对关节名/顺序、动作模式/维数/名称、相机键、频率、特权标记。
- 实测：state 契约 vs lux3d_5r → compatible；vs cartesian_3dof → **不兼容**（关节名/顺序差异），阻止误运行；priv 标记特权不可部署。
- 部署中心新增兼容检查面板（选模型+机器人→显示兼容/不兼容与差异）。

## 辅助模型

- `AuxModel.suggest` 现发送文本+图像（views→image_url/data URL），解析结构化 JSON 候选，返回 assumptions/needs_confirmation；非 JSON 返回 unparsed 提示结构化导入；无凭据仍 waiting_for_configuration。凭据仅后端环境。

## 验证

- 前端 tsc + vite build 通过；兼容/多机器人/视觉评测均真实运行。

## 未完成 / 依赖（如实）

- 视觉 BC 成功率 3/6 为小样本基线；提升需更多数据/更大模型/SmolVLA 微调。
- 资产真实生成仍需 AHOLO_API_KEY；网页 GLB 导入上传端点已设计（ImportBody）但前端上传 UI 未接。
- 宽屏 1440/1920 三栏占比截图仍受内置浏览器窄视口限制。
- 评测/部署 `/:runId`、`/:packageId` 详情仍为列表级占位。