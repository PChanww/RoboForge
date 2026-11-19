# RoboForge 后端效果 Demo

**http://127.0.0.1:5173/transport** 是[VLA仿真验证与数据采集工作台](VLA_WORKBENCH.md)：侧边导航分开任务、模型、仿真、数据与参数；生成模型经用户确认后执行真实[移动抓取搬运](MOBILE_TRANSPORT.md)，输出可行性、失败阶段和完整实验数据。当前为状态专家基线，VLA移动策略尚未接入。

**http://127.0.0.1:5173/drive** 是[四轮移动机器人](MOBILE_DEMO.md)：复用 Aholo 外观，真实轮地接触驱动，可手动驾驶、执行示例路线并记录运动数据。

打开 **http://127.0.0.1:5173/create** 是[需求生成工作台](SCENARIO_WORKBENCH.md)：输入环境、机器人、物品和工作状态，检查方案后生成或复用 Aholo 模型，自动组装真实场景并采集双相机数据。

**http://127.0.0.1:5173/** 是[塑料鸭传送带分拣](DUCK_SORTING.md)：传送带不停、机械臂跟随抓取并受控入箱，支持随机初始姿态、双相机和数据下载。首页可直接描述物体或底盘外形，提交 Aholo 生成，完成后用于场景；点击生成会使用配置的在线额度。旧版抓杯页面保留在 `/collect`，文字任务工作台在 `/studio`，完整平台在 `/projects`。以下抓杯说明只针对 `/collect`。

## 本轮约定

- 网页展示 RoboForge 后端已有的 Aholo 模型、MuJoCo 仿真和采集结果。
- 抓杯页面通过已有接口串联流程。用户随后明确授权扩展后端实现鸭子传送带的真实仿真与采集；具体变更和边界见上述任务文档。
- 当前使用已导入的蓝色杯和陶瓷杯，不提交新的在线生成任务。后续功能继续复用 RoboForge 与 Aholo。

## 操作流程

1. 打开页面，自动读取资产库并显示真实的初始场景。
2. 选择物品、采集数量（1～20）、起始随机种子。
3. 点击“开始采集”，切换前视／腕部相机，查看机械臂抓取和放置。
4. 完成后查看成功率、帧数和时长，下载 JSON 清单或单条 NPZ 数据。

页面记住最近一次作业，刷新可恢复状态和最后画面。更换物品或种子会回到新配置的初始场景。采集每条递增种子，每条的观测、动作、图像与时间戳一起保存。

## 接口与数据

前端适配逻辑在 `src/demo/backend.ts`，使用后端现有接口：

| 操作 | 现有接口 |
| --- | --- |
| 读取 Aholo 资产 | `GET /api/v1/asset-library` |
| 读取默认模板 | `GET /api/v1/config` |
| 保存并验证独立项目 | `PUT /api/v1/config?project=aholo_demo_<id>` |
| 获取初始相机观测 | `POST /api/v1/sessions`，随后 `DELETE /api/v1/sessions/{id}` |
| 执行后端专家采集 | `POST /api/v1/jobs`，`kind=collect` |
| 实时画面／状态／日志 | `GET /api/v1/jobs/{id}/preview`、`/{id}`、`/{id}/logs` |
| 样本与清单 | `GET /api/v1/datasets/collection__{dataset_id}` 与 `/episodes/{index}` |

独立项目采用后端默认机器人的配置，绑定真实的 `imported/lux3d-*` 资产，不覆盖默认项目。采集作业引用具体配置版本，由后端保存快照。初始预览通过短暂仿真会话读取两路图像后关闭；采集期间直接读取采集进程的预览，不在浏览器重做运动。

输出目录：`datasets/collections/aholo_demo_<id>/`。清单中的 `task_asset` 由后端记录模型来源、生成任务 ID、文件哈希和物理参数；`generation_provenance` 额外保留网页选择的资产信息。

当前显示生成模型的 OBJ 网格，颜色沿用后端的材质均值；物理碰撞仍使用后端的圆柱代理。模型高度、质量由导入规格设定，未做实物标定。本版只演示单物体抓放，不包含 World 背景生成。

## 启动

项目根目录：

```powershell
.\.venv\Scripts\python.exe -m uvicorn vlab.api:app --host 127.0.0.1 --port 8767
```

`frontend` 目录：

```powershell
npm ci
npm run dev -- --host 127.0.0.1 --port 5173
```

Aholo 密钥放在根目录 `.env`，仅由后端读取，不放进前端或提交到仓库。前端构建检查：`npm run build`。

## 本机验证

使用现有后端接口的浏览器验收已完成：两件模型切换、初始场景、双相机、真实抓放采集、JSON/NPZ 下载、来源记录、刷新恢复、手机布局、默认项目未被修改、临时预览会话关闭。

蓝色杯、种子 0 的网页采集：1/1 成功、176 帧。此次未创建在线生成任务。
