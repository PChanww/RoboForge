# 场景链路代码审查

审查日期：2026-11-10。基线：`4affcac`，主要比较 `a8a506b..4affcac`，并检查新功能依赖的已有页面、作业和评估入口。没有修改业务源码；本文件独立于 README 和原作者的验证报告。

## 结论

已验证一个受限的本地网格场景能够进入 MuJoCo RGB，并运行默认任务。当前实现尚不能通过 Web 完成场景生成、编辑、采集、训练的完整流程。发现以下可复现问题；现有场景测试通过不能排除这些问题。

## 需要修复的问题

### 1. [P1] World 提供方调用了实际 SDK 不存在的接口

位置：`vlab/world_provider.py:42-57`。

生成调用 `c.text_to_world.create()`，重建调用 `c.reconstruction.create()`，查询调用 `c.tasks.get()`。安装并检查官方 `manycore-aholo-sdk-world==1.4.0` 后，这些接口不属于 WorldClient。真实客户端上执行当前 `submit_generation()`，在发出网络请求前即得到：

```text
AttributeError: 'WorldClient' object has no attribute 'text_to_world'
```

正确接口为 `world.generations.create(name=..., prompt=..., resources=...)`、`world.reconstructions.create(resources=...)`、`world.retrieve(world_id)`；创建返回值需要提取 `worldId`，不能对整个结果直接 `str()`。结果文件来自 `assets.splats.urls`，不是 Lux3D 的 `outputs[].content`。

修复：按 World 的真实结构统一创建、状态、错误与结果转换，并使用与官方结构一致的测试替身。

来源：[官方 Python SDK 文档](https://labs.aholo.ai/api-docs/sdk/python)。此次使用官方 SDK 另行提交的真实任务见后面的在线验证记录；不能把该独立脚本的成功当成项目适配器通过。

### 2. [P1] 生成任务显示成功后，不会下载或注册场景

位置：`vlab/world_provider.py:145-156`。

`GenerationJobStore.poll()` 只保存 state/error，丢弃 outputs；没有调用 `download_assets()`、场景导入器或资产登记。因此即使修复问题 1，任务仍会停留在“成功但资产中心没有场景”的状态，返回结果也没有可供用户下载的资产引用。

复现：测试替身返回 succeeded 和一个有效结构的 outputs；poll 返回 succeeded，而下载方法调用次数为 0，作业中也没有结果字段。

修复：区分远端成功、下载、处理、资产可用；持久化产物和 scene asset revision，下载失败可重试且不再次提交远端生成。

### 3. [P1] 场景单位换算参数只保存，没有应用

位置：`vlab/scene_import.py:72-88`，关联 `vlab/scene_builder.py:50-54`。

`unit_scale` 只写入 manifest，既没有作用于导入网格，也没有参与环境实例变换。以毫米计量的资产可以被标记 calibrated/ready，但实际仍以相同数值作为米使用，导致相机、机器人和碰撞代理的尺度失配。

复现：导入边长 1000 的立方体，指定 `unit_scale=0.001, calibrated=True`，预期规范尺寸为 1 米；实际 bbox 三轴跨度均为 1000，processed OBJ 也未缩放。

修复：选定唯一的规范化步骤，实际应用单位比例并更新 bbox/hash；保留原始单位信息，避免实例缩放再次乘同一比例。

### 4. [P1] 大角度旋转的矩阵转四元数公式错误

位置：`vlab/transforms.py:53-61`。

`trace <= 0` 分支中，当前 `t=sqrt(...)` 与后面的除数/主分量公式不配套，缺少对应的 2 倍因子，输出的四元数分量比例错误。MuJoCo 即使归一化也无法恢复正确姿态。`scene_builder` 在生成环境位姿时实际调用此函数，因此会改变场景朝向。

复现：绕归一化轴 `[1,2,3]` 旋转 150°，计算矩阵→四元数→矩阵，最大元素误差为 `0.8570585613641981`。更简单的绕 Z 轴 150°，最终实际角度约为 86.0305°。现有测试仅检查 90° 绕 Z 的四元数转矩阵，没有覆盖该逆转换。

修复：采用正确、完整的旋转矩阵转换，并覆盖多个旋转轴、120°/150°/180°、单位缩放和随机旋转的往返验证。

### 5. [P1] 导入资产 ID 可以逃逸资产目录

位置：`vlab/scene_import.py:48`；外部入口为 `vlab/api.py` 的 SceneImportBody 和场景导入接口。

未校验 `asset_id` 就直接拼接输出路径。后端可收到绝对路径或 `..`，前端文件名清理不构成后端校验。Windows 上绝对路径会丢弃之前拼接的目录前缀，使 API 将文件和 manifest 写入场景资产根目录之外的新目录。

复现：仅在临时目录内，把 asset_id 指定为另一个绝对目录；导入成功，manifest 实际位于该指定目录下，而非 lab/assets/scenes 内。

修复：服务端限制 ID 为普通单段标识，拒绝路径分隔符、绝对路径及点段；解析后的目标必须位于已确认的资产根目录内。读取场景和项目实例引用时也应用同一规则。

### 6. [P1] Web 后台作业没有传递新增脚本参数，仍使用旧项目/数据集/模型

位置：`vlab/jobs.py:64-95`；新增参数位于 `scripts/train_image_bc.py:60-67`、`scripts/eval_image_policy.py:89-106`。

脚本虽新增 `--dataset`、`--model-out`、`--base-project`，但 jobs.py 未同步。collect 只传 --out；train_visual 和 eval_visual 只传 --lab。即使 POST /jobs 提交场景项目、数据集和模型参数，也会被忽略，继续处理 default、expert21、visual_bc，并复用固定采集目录。

复现：对 `_command()` 传入 scene_demo/visual_scene 等参数，实际命令中没有这些输入。没有启动新训练或覆盖已有数据。

修复：将项目不可变快照、dataset/model ID、采集计划逐层传到命令；独立作业输出目录，前端可选择对应输入。该问题涉及本次新功能与已有作业入口的集成，原命令结构在上一版本已存在。

### 7. [P1] 绑定场景后，仿真主视图仍加载默认项目并绘制固定桌面

位置：`frontend/src/SimWorkspace.tsx:42`、`frontend/src/store.ts:58-63`、`frontend/src/components/Scene.tsx:137-145,238-256`；新入口为 `frontend/src/components/pages.tsx:125-128`。

新增绑定功能默认写入 scene_demo，但仿真页没有读取路由 projectId，store.init() 使用无参数 getConfig()，加载 default。即使在后端把场景绑定到 default，Scene.tsx 也没有渲染 scene.environment 或 scene.colliders，仍显示硬编码 StaticScene。于是主 3D 视图与物理/传感器画面不同，用户无法靠主视图判断障碍物或场景位置。

证据：检查完整前端读取链路；本次 Scene.tsx、store.ts、SimWorkspace.tsx 均未为新场景接入修改。前端类型的 scene 也仍只有 physics_dt/seed/gravity。此条是新绑定功能暴露的既有路由缺口和缺失的场景显示集成，未进行 UI 自动点击验证。

修复：按路由加载/切换项目并重建对应会话；增加环境资产与代理显示组件，使用后端一致的变换；移除重复的固定桌面绘制。

### 8. [P2] 绑定接口接受不存在或不可仿真的场景，先写入损坏配置

位置：`vlab/api.py:503-511`，关联 `vlab/scene_builder.py:48-51`。

bind-scene 不检查资产/修订是否存在、不检查 visual_asset/status、不调用配置构建验证。导入高斯 PLY 后页面允许“用于项目”，但此类 manifest 的 visual_asset=null，下一次构建得到 TypeError。不存在的资产同样会先被保存，后续打开会话才失败。

复现：临时 app 中绑定 does_not_exist/revision=999，接口返回 200，配置确实保存；对仓库自带 demo_splat 构建环境，得到 `unsupported operand type(s) for /: 'WindowsPath' and 'NoneType'`。

修复：保存前验证所选修订的可用性和配置构建；未支持的格式/未标定物理使用需明确拒绝，返回清楚的 4xx，失败时不修改当前有效配置。

### 9. [P2] 新的 base-project 评估参数没有进入兼容性检查

位置：`scripts/eval_image_policy.py:101-106`；关联 `vlab/api.py:513-525`。

评估从 BASE_PROJECT 取 robot_id，但请求的 /compat 仍只会加载 default，而且比较的是 API 内硬编码的视觉契约，不是本次加载的 checkpoint 契约。目标项目如果改变 front 相机、控制频率或动作定义，检查可能通过，实际评估才报错或以不匹配的观测运行。

修复：兼容性检查传入本次评估的不可变项目配置和实际模型契约，返回状态/动作/相机/频率等差异，执行前阻止不兼容组合。

## 功能范围缺口（与上述具体缺陷分开）

- 3DGS hybrid 渲染未实现，配置 hybrid 会显式报错；代码没有静默回退，这是合理的失败方式，但不能将第三步的 3DGS 训练输入视为已完成。
- SceneEditPage 目前仅展示元数据和绑定按钮，没有三维校准、碰撞体编辑、保存新修订的操作；资产列表不同 revision 也进入同一“最新版本”页面。
- work_surface_z、manifest.calibration 等字段未参与任务坐标或环境变换；不能视为已实现任意工作表面调整。
- 原验证报告中的“Pydantic+TS 场景类型”“固定配置快照的 Web 采集”等表述与当前完整链路不符，应在修复前收窄描述。
- 现有场景 RGB 差异测试同时更换默认桌子和碰撞配置，不能单独证明视觉环境网格的影响；应固定其他条件只切换视觉网格，再验证真实 observation/采集帧。

## 此次实际执行的检查

- `tests/test_scene_pipeline.py`：通过。包含默认任务回归及受限网格场景测试。
- `frontend: npm run build`：通过。产生大体积 bundle 提示，非此次主要阻塞；构建生成的未跟踪 tsbuildinfo 已清理。
- 补充临时目录验证：单位换算、路径逃逸、旋转往返、不可用资产绑定、作业参数传递、生成成功后的下载调用。
- 安装 World SDK 1.4.0，确认真实客户端的接口结构；生产提供方调用在网络之前失败。
- `tests/test_world_provider.py`：安装 World SDK 后运行失败，断言中的 unconfigured=False。当前实现只在 SDK 未安装时借助 ImportError 标记未配置，SDK 存在但无密钥时没有正确的显式配置检查。mock 成功部分不能覆盖真实适配器接口。
- 未修改业务源文件，未运行会覆盖现有 staging 数据的采集任务，未重训模型。

## 真实在线生成验证

使用官方 SDK 的独立验证脚本，凭据仅用于进程内鉴权，不写入项目或结果文件。

- Lux3D 杯子资产：任务 `5887561`。
- World 空工作区实验室场景：任务 `3FO4K4RMF54N`。
- 提交已成功，处理结果与文件记录位于仓库外的 `outputs/live-generation-validation-2026-11-10/`；最终状态以该目录中的 asset.json、scene.json 为准。
- 此独立在线验证不经过项目 WorldProvider，不代表问题 1/2 已修复。
- 杯子：GLB 下载成功，8,300,904 字节，17,794 三角形；已导入 `assets/imported/lux3d-blue-live-5887561/`。在默认场景用手工指定的 0.09 米高度、0.03 米圆柱半径运行，伺服专家 seed=0 的一回合成功，93 控制步。属于资产/物理链路验证，不是学习策略评估或实机迁移证明。
- 场景：官方状态最终为 SUCCEEDED，约 18 分钟；已下载 `generated-lab.ply`（13,583,212 字节，199,747 个高斯，上轴 Z）和全景图。实际文件头确认为高斯 PLY，没有三角面。
- 对真实生成场景在临时资产目录验证：导入器标记 preview_only，当前环境构建因 visual_asset=null 抛出 TypeError。没有把该场景绑定到用户现有项目。证明云端生成可用，同时确认当前平台尚未接通真实 World 产物的传感器渲染。
- 新资产及本审查文档尚未提交 git；业务源码未修改。
