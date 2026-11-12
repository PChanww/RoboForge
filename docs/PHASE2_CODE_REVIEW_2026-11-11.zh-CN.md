# 第二阶段增量代码审查

日期：2026-11-11（北京时间）。审查基线为当前工作区，HEAD 仍为 4affcac；保护全部未提交修改。本次仅审查和编写后续任务，没有修改业务源码。

## 结论

本轮已新增 CPU 高斯光栅化、front-only hybrid 数据链、编辑画布及视觉包导出，不能继续按“完全没有渲染实现”评价。但存在会改变输入图像、编辑结果与部署行为的错误，应先修正，再评价新模型的有效性。

本次运行 gaussian_snapshot、gaussian_covariance、camera_projection 共 9 项测试，通过。补充的小型用例复现了测试未覆盖的以下问题。未重新训练，也未重新运行报告中的 4/6 评估。

## 发现

### 1. [P1] MuJoCo 前景分割 ID 偏移一位

位置：vlab/rendering/hybrid.py:41；HybridRenderer.compose。

foreground_ids 返回 gid+1，但 mujoco.Renderer.enable_segmentation_rendering() 返回的是 objid/objtype，内部编码已解码，objid 就是 geom ID。当前匹配会丢掉本该保留的对象，并可能选中相邻的错误几何体；同时应检查第二通道确为 mjOBJ_GEOM。

实际复现：一个地面 geom=0、杯子球体 geom=1 的最小 MuJoCo 场景，分割中杯子有 168 像素；foreground_ids 返回 {2}，合成保留的前景像素为 0。

修复后必须重新采集并训练受影响的 hybrid 数据，旧结果保留为历史，不覆盖。

### 2. [P1] 投影协方差没有转到相机坐标

位置：vlab/splat_render.py:57-65。

中心已从世界转入相机，但 Sigma 仍为世界坐标协方差，直接与相机投影 Jacobian 相乘。cam-to-world 为 Rc 时，应先计算 Sigma_cam = Rc.T @ Sigma_world @ Rc，再做 J @ Sigma_cam @ J.T。

实际复现：相机看向 -Z，原点前方单个沿世界 X 拉长的高斯，尺度 [0.3,0.04,0.04]。相机 roll=0° 和 90° 的足迹宽高都为 48×6；正确情况下 90° 应交换长短轴。当前 T03 只旋转高斯/实例而不旋转相机，因此无法发现。

### 3. [P1] hybrid 使用显式碰撞代理时直接崩溃

位置：vlab/engine.py:94。

条件 cfg.scene.use_default_table or config.scene.colliders 引用了未定义变量 config。启用默认桌面时短路掩盖问题；关闭默认桌面并提供合法代理时触发。

实际复现：Gaussian 环境 + use_default_table=False + 合法 desk box → NameError: name 'config' is not defined。

### 4. [P1] 编辑接口绕过验证，允许持久化损坏配置

位置：vlab/api.py:588-628；frontend/src/components/pages.tsx:245-250。

set_scene_instance 使用 model_copy(update=...)，不会重新运行 Pydantic 验证；ConfigStore.save 也不补验证。实例和代理分两次保存，第二次失败时第一次已永久生效。没有 expected_version 冲突检查，旧页面还能覆盖较新编辑。

实际复现：临时 app 中为项目预置 SceneInstance，PUT scene/instance {uniform_scale:-1} 返回 200，current.json 的 uniform_scale 确为 -1。之后重新加载可失败。

需要一个原子场景提交入口：完整验证实例、代理、跨字段条件、资源与必要物理构建；任何失败不写 current 或版本；保存时比较版本。若允许无效草稿，必须单独存储并显式标记，不能充当可运行 current。

### 5. [P1] 编辑画布变换与保存配置不一致

位置：frontend/src/components/SceneCanvas.tsx:20,30,60-85。

SceneMesh/ColMesh 把位置放在子节点，TransformControls 却绑定初始位置为零的父 Group；拖动后将父 Group.position 当作绝对位置回写。非零原位置时会丢掉旧位置，还可能出现父子重复平移。

SceneMesh 与 ColMesh 都未使用 quaternion_xyzw；cylinderGeometry 默认沿 Y，但物理圆柱沿规范世界 Z，缺少对应转换。数值旋转并未在画布中真实反映。

此项为源码确认，未在本次做浏览器拖动复现。修复后必须真实验证非零起点拖动、四元数旋转、圆柱方向及刷新后的世界矩阵一致性。

### 6. [P1] 导出包没有实际使用包内权重，且丢失训练契约

位置：scripts/export_visual_package.py:RUN_HTTP、main；scripts/visual_policy_adapter.py:VisualBCAdapter.__init__/contract；vlab/api.py:deployment_compat。

导出会复制 model.pt，但 run_http.py 仍要求 --lab，并用 VisualBCAdapter(a.lab,a.model) 从原工程 datasets/models 加载权重。所谓重载验证也两次从原工程加载，未验证包内 model.pt。原模型缺失时包不能运行；原模型同名更新时包会静默运行别的权重。

adapter.contract() 返回硬编码简化契约，没有保留 checkpoint 的 camera_specs、controller_spec、state_joint_names 等；部署兼容路由再次只映射少量字段，可能显示兼容但相机/控制增量已经不匹配。训练模块也被当作运行依赖导入，带入 LeRobot/Pydantic/vlab，和列出的推理依赖不一致。

需要显式 from_package/from_checkpoint，包路径加载与哈希验证、保留完整训练契约；在移走原工程模型的隔离环境测试。HTTP runner 运行前检查真实目标配置，并在 finally 删除会话、释放控制权。

### 7. [P2] reset 后 hybrid 缓存未随相机变化失效

位置：vlab/engine.py:155-156,209-214；vlab/rendering/hybrid.py:_cache_key。

跨 reset 保留 self._hybrid，而本地 _hybrid_key 仅含 PLY 路径和实例变换，不含分辨率、相机位姿及内参。全局背景缓存有这些字段也不能补救：本地条件先阻止了重新查找。

同一个 env 修改相机后 reset，可继续返回旧背景，帧元数据却写入新位姿；改变分辨率还可能造成数组形状错误。缓存还应绑定资产哈希和渲染版本，设置大小上限并处理共享数组只读。

此项为源码确认，尚未执行带真实场景的长耗时复现。

### 8. [P2] 预览图和正式 observation 使用不同渲染后端

位置：vlab/engine.py:437-445。

_render_camera 支持 hybrid，但 render() 仍直接返回 MuJoCo RGB，frame.jpg/Gym render 等调用会遗漏高斯背景。应共用 renderer 选择；预览是否属于缓冲帧可以不同，但后端与场景来源不能静默不同。

## 不能由现有测试支持的验收结论

- T05 的独立 ray_box_hit 将单位射线的距离 t 与 splat 中心的 camera-Z 深度直接比较，离轴时单位不一致；nearest depth 还更新整个包围矩形，未按 alpha mask 更新。不能用这些图片证明完整 hybrid 遮挡正确。
- 正式 hybrid compose 明确把全部前景覆盖到背景，尚无 Gaussian→动态网格的遮挡；应记录为“受限静态背景模式”，并落实空工作区约束。
- T07 当前覆盖旧动作拒绝和同步缓冲清空，不等同于真实异步响应跨 reset 丢弃。
- T13 同时变化几何/桌面或仅回读已有帧，不能替代固定物理、只变化视觉的对照测试。
- 独立静态页面与 PNG 保存端点，不等同于可被后端调用的离屏渲染服务。
- 状态文档后半段仍保留“未实现 hybrid/编辑器”的过期段落，和前半段新结论相互矛盾，应整理为当前状态与历史记录两部分。

## 下一步

优先修复 1/2/3/4/5/6，增加针对性失败测试，再处理缓存和预览一致性。修正所有改变图像的错误后，新建数据集/模型 ID，重新采集、转换、训练和评估；旧 4/6 仅作为旧渲染器结果，不用作修复后模型成绩。GPU、wrist、通用 VLA 在此之后继续。
