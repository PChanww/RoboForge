# 场景编辑器（第二阶段）

## 入口与布局

- `/scenes/:sceneId/edit`：加载资产清单与项目场景配置；上方元数据（来源/格式/状态/revision/up_axis/unit_scale/calibrated/包围盒）、绑定按钮、校验；下方**三维画布** + 实例数值标定 + 两点测量 + 碰撞代理列表。
- 网格画布加载规范化 `visual.obj`（`/api/v1/scene-assets/{id}/revisions/{rev}/visual.obj`），显示碰撞代理线框和网格。实例和代理支持平移、旋转及数值编辑。高斯与点云使用预览入口，高斯可以另作 hybrid 的视觉背景。

## 操作

1. **绑定**：选择项目与是否使用默认桌面，`bind-scene` 写入项目配置（网格需 calibrated；高斯需 hybrid + 默认桌面或碰撞代理，否则拒绝）。
2. **实例标定**：position xyz + uniform_scale；画布拖动或数值输入，保存写 `scene/instance`（新配置版本）。
3. **测量缩放**：输入两点坐标得到测量距离，输入已知实际距离，点击按比例缩放（uniform_scale × 已知/测量）。
4. **碰撞代理**：添加/选择/删除 box/plane/cylinder；编辑名称、形状、全尺寸(full_size_m)、位置；保存写 `scene/colliders`。圆柱要求 x==y，非均匀缩放拒绝。
5. **撤销/重做**：对实例与代理的结构/变换快照（最多 50 步）。
6. **保存/放弃**：保存后配置版本递增；放弃从服务器重载。

## 版本与安全

- 资产原始文件不被编辑覆盖；项目绑定明确 revision（不随“最新版本”漂移）。
- 保存走配置存储（`projects/<id>/versions/vN.json`），刷新/重开一致。
- 绑定/校验在服务端构建环境中执行；网格需标定与哈希校验；高斯需 hybrid 且物理有默认桌面或代理。

## 实测

- 画布渲染演示工作台网格与线框（截图 `experiments/20261110-p2/page-scene-editor.png`）。
- 脚本化编辑：切到 `test_scene_edit` → 添加代理 → 保存 → 后端确认 colliders=desk/post/proxy_3（配置 v14）。
- API 测试 `tests/test_scene_editor.py`：实例/代理持久化、圆柱 x≠y 拒绝、编辑后杯子落在代理桌面。

## 支持范围与限制

- 本地场景导入接收 GLB 或 PLY，处理后的网格用 OBJ 显示。点云用于预览，3DGS 可按[渲染条件](RENDERING_PIPELINE.zh-CN.md)绑定为固定前视背景。
- 两点测量使用数值坐标。场景实例使用统一缩放，通用静态代理使用盒体、圆柱或平面。机器人和相机是否可在当前项目运行，见[平台支持范围](PLATFORM_SCOPE.zh-CN.md)。
