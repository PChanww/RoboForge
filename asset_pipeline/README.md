# 导入团队自己生成的 Lux3D 杯子

在项目根目录 `vla-lab` 下运行命令，使用运行仿真的 Python。训练和评测内置模型不需要 Aholo 账户。资产导入器在 trimesh 5.2.0 上验证。

本页的导入命令针对直立杯型，使用圆柱近似杯身。托盘等静态道具、机器人外观和环境网格的入口见[资产生成与场景配方](../docs/GENERATION_WORKFLOW.zh-CN.md)。选择与资产用途匹配的碰撞代理后，在实际任务里验证接触。

## 已有模型文件

在 [Lux3D 创作空间](https://lux3d.aholo3d.cn/lux3d/home) 用文字或图片生成杯子，选网格结果，下载 GLB 或包含 GLB 的 ZIP。保存任务页和原始提示词。下面的任务编号占位符要替换为自己的实际值。

```powershell
python asset_pipeline/import_asset.py "D:/Models/cup.glb" --slug my-cup --source lux3d --task-id "替换为实际任务或资产编号" --height-m 0.09 --radius-m 0.03 --mass-kg 0.12
```

ZIP 中有多个 GLB 时用 `--member folder/mesh.glb` 选择。GLB 必须自包含。默认采用 glTF 的 Y 向上约定；仅当导出器已将 Z 设为上轴时加 `--up z`。其他有权使用的模型可用 `--source other`。

## 通过 API 生成并直接注册

在本地设置 **AHOLO_API_KEY**。请从 [中国区开发者中心](https://labs.aholo3d.cn/api-keys) 管理凭据，并仅配置在本机环境中。PowerShell 7 可用下列输入方式临时设置变量，输入值不会显示。

```powershell
$env:AHOLO_API_KEY = Read-Host "Aholo API key" -MaskInput
python -m pip install manycore-aholo-sdk-lux3d==1.9.0
python asset_pipeline/generate_lux3d.py --slug lux3d-white --height-m 0.09 --radius-m 0.03
```

该命令会在当前账户创建一次可能消耗积分的 G1-Turbo 文生 3D 任务。默认生成无把手的白釉圆柱杯，可用 `--prompt "…"` 指定其他杯型。脚本会轮询任务，下载网格，随后运行同一导入器。任务编号保存在 `assets/generation_jobs/<slug>.json`。网络中断后用原 slug 重试，或加 `--task-id 实际编号` 下载账号已有任务。若创建请求超时且本地未记录编号，先查看 Aholo 任务历史，以免重复提交。[官方 Python SDK](https://labs.aholo.ai/api-docs/sdk/python)

## 如何核对结果

注册成功后，打开或刷新本地实验台并选择导入的杯子。API 中资产名为 `imported/<slug>`，每件模型保存在 `assets/imported/<slug>/`：

- `original.glb` 保留原始网格及内嵌材质，`manifest.json` 保存其 SHA-256、生成编号、来源、物理参数和提示词（如有）。
- `visual.obj` 是将场景节点变换烘焙、杯底居中置于 z=0、按指定高度缩放后的三角网格，以米为单位。仿真用单色渲染；如果 GLB 内有可读取的表面色，会按面积估计平均色，图案与完整 PBR 仍保留在原文件中。
- 抓取接触由圆柱近似。半径应量杯身、不含杯柄。未指定半径时，用缩放后横向较短跨度的一半估算。高度、质量、碰撞半径都是仿真参数，不能据此宣称复现了真实杯子的抓取难度。
- `source=lux3d` 且来自手动导入时，编号是使用者提供的溯源信息；API 下载路径另标为服务端任务产物。参赛报告可放原始生成记录和仿真实际运行截图供复查。
