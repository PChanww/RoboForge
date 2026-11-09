# 阶段 B 报告：版本化配置与后端核心

日期：2026-11-09 · 代码：`vlab/` · 测试：`tests/test_engine.py`、`tests/test_api.py`

## 实现内容

| 模块 | 职责 |
| --- | --- |
| `schema.py` | schema_version=1.0 的 Pydantic 配置：Scene/Robot(连杆树+关节+执行器)/Sensor(相机内外参+分辨率+FPS+延迟)/Controller/Task/Policy。校验重复名、断链、闭环、零轴、非法惯量、限位、初始姿态、传感器父坐标系。参数来源 provenance。 |
| `mjcf.py` | 从连杆树递归生成 MJCF body/joint/geom/actuator/camera，按名称寻址，无固定 DOF。相机用 sensorsize+focalpixel+principalpixel+resolution 表达完整针孔内参（MuJoCo 4.0 实测支持）。 |
| `scheduler.py` | 仿真时钟驱动多频率调度：物理步长、控制周期、各相机周期独立；非整数周期比按"步进到首个≥计划时刻的物理 tick"处理，记录实际 capture_time/available_time、采样间隔、图像年龄。半开区间避免边界重复采样。 |
| `engine.py` | 配置→模型→重置→步进；版本化观测（按名称关节状态+相机帧+时间统计）；动作校验（维数/范围/有限性）；cartesian_delta_ik 与 joint_position 两模式；任务接触判定。 |
| `tasks.py` | 杯子抓放任务插件：场景几何、随机摆位、双指接触/抬起/托盘直立静置判据。复用 sim_env 资产加载。 |
| `gym_env.py` | Gymnasium 封装：Dict 观测空间（状态向量+各相机图像）、动作空间来自控制器配置、terminated/truncated 区分。 |
| `store.py` | 配置持久化：current.json + 不可变 versions/v{n}.json + 运行快照；静态+物理构建校验。 |
| `sessions.py` | 每会话独立单线程（保留 OpenGL 上下文绑定）+独立环境；单一控制权 owner+token，冲突 409。 |
| `api.py` | FastAPI：能力发现、配置版本化、会话/控制权/动作/观测/结果、WebSocket 观测-动作循环、旧版 /api/* 原样兼容。 |

## 验证结果（真实执行）

- `test_engine.py`：脚本专家在新引擎抓放**成功**（与旧版 parity）；30 FPS 相机在 20 Hz 控制下 9 s 共 **271 帧/相机**，平均采样间隔 **0.03333 s**（量化到 0.032/0.034）；错误维数/越界/NaN 动作均被拒绝。
- `test_api.py`：
  - 配置保存递增版本、不可变快照可读取、运行快照写出。
  - 零轴非法配置返回 `valid:false` 结构化问题。
  - 两个会话隔离：步进会话 1 后会话 2 tick 仍为 0。
  - 无控制权动作→409；异主抢占→409；错误维数→400；过期 tick→409。
  - 旧版 /api/reset + /api/step 行为不变；错误 episode→409。
  - Gymnasium 观测/动作空间合法，terminated/truncated 为布尔，info 含 success。
- 真实 uvicorn（端口 8767）端到端：capabilities→创建会话→acquire→action（tick 1, sim_time 0.05）→delete 全链路通过。
- WebSocket：连接收到 tick 0 观测→acquire→action→收到 tick 1 观测。

## 验收对照

- #3 多周期组合按仿真时间运行并报告时间戳/间隔：✅（271 帧、0.03333 s）。
- #4 外部控制器/模型客户端从普通入口完成观测-动作循环：✅（HTTP + WS）。
- #5 过期动作/错误维数/非法值/控制权冲突正确处理：✅。
- #11 通用接口无固定关节长度假设：✅（joint_position 与观测按名称/配置维数；解析 IK 为 lux3d_5r 专用，其他机器人 cartesian 模式返回明确 UnsupportedMode）。

## 依赖锁定

requirements.txt 追加：fastapi==0.162.2、uvicorn==0.54.0、pydantic==2.14.3、gymnasium==1.4.0、websockets==17.4、httpx==0.28.1（测试）。原有 mujoco/numpy/pillow/trimesh 版本未改动。

## 未解决限制（转入后续阶段）

- 连续实时运行模式（墙钟 pacing、推理超时/动作耗尽处理）尚未实现，当前为步进驱动；阶段 C 编辑器接入运行/暂停/单步时补齐。
- 数据导出端点、LeRobot 数据集、训练闭环：阶段 D。
- 辅助模型适配器、外部控制器插件协议、第二种机器人、部署包导出：阶段 E。
- cartesian_delta_ik 的解析 IK 仅适用于当前 5R 结构；任意六维位姿不支持（机构约束，已在控制器 task_space_note 声明）。