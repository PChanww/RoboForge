# RoboForge 模型接入与资产绑定指南

使用这份指南，把生成资产放进仿真，或让已有模型读取 RoboForge 的关节和图像并回传动作。

当前仿真任务是带双指夹爪的桌面抓放。五轴机械臂和三轴龙门可分别使用自己的控制器。资产类型、任务、相机和示范采集各有独立的适用条件，见[平台支持范围](PLATFORM_SCOPE.zh-CN.md)。

## 1. 先下载场景对应的绑定指南

- **资产已入库**。在生成工作台的任务列表或详情选择「下载绑定指南」。文件列出资产引用、尺寸与物理角色，并说明下一步怎样配置碰撞、场景或机器人部件。任务有关联项目版本时，文件会附上该版本的关节、相机和控制器供参考。
- **配方已应用**。在应用结果选择「下载项目绑定指南」。保存后的项目版本决定实际使用的机器人、传感器、控制器以及场景资产。已有项目可直接调用下面的项目接口。

| 下载项 | HTTP 路径 |
| --- | --- |
| 生成资产绑定指南 | `GET /api/v1/generation/sessions/{session_id}/tasks/{job_id}/binding-guide` |
| 项目绑定指南 | `GET /api/v1/projects/{project_id}/binding-guide`，可加 `?version=7` 下载指定版本 |

杯子和托盘通常通过场景配方添加。机械臂外观可先拆分部件，再将部件绑定到配置好的连杆。关节、驱动器和相机由项目配置决定。模型的张量形状、预处理和归一化则由模型适配器负责。

## 2. 定义并核对输入输出契约

读取 `GET /api/v1/config?project=default` 和 `GET /api/v1/capabilities?project=default`。先选择 `robot_id`，再从这个机器人的 `controller_ids` 和 `sensor_ids` 选择接入所需的控制器与相机。绑定指南也列出 `incompatible_controllers` 和 `unavailable_sensors` 的具体原因。与模型一起保存以下信息。

| 契约项 | 需要固定的内容 |
| --- | --- |
| 项目与机器人 | 项目 ID、配置版本、机器人 ID、控制器 ID。创建会话后核对版本。 |
| 关节输入 | 位置或速度、具名关节顺序、单位、方向和零位转换。 |
| 相机输入 | 模型输入名到传感器 ID 的映射，以及颜色顺序、图像裁剪和缩放、张量布局、数据类型与归一化。 |
| 时序 | 控制频率、允许的帧年龄、多相机的时间对齐要求、历史窗口和重置逻辑。 |
| 动作输出 | 顺序、维数、单位、坐标系、绝对值或增量、范围和夹爪的含义。 |

从所选机器人的 `actuators[].joint` 核对后端提供的关节状态，按模型训练顺序从 `obs["state"]["joint_pos"]` 和 `joint_vel` 取值。五轴机械臂提供 `waist, shoulder, elbow, wrist_pitch, wrist_yaw, left, right`，龙门提供 `x, y, z, left, right`。旋转关节位置用弧度，直线关节用米。可另取 `eef_position`（世界坐标 XYZ，单位米）和 `gripper`（归一化开度）。

相机通过 `sensors[].id` 绑定。模型的 `scene_rgb` 可以读取 `obs["images"]["front"]["jpeg_b64"]`，`hand_rgb` 可以读取 `wrist`。默认 `front` 挂在 `world`，`wrist` 挂在 `eef`。解码 JPEG 后按模型要求处理图像。通过 `camera_status`、`capture_time`、`available_time` 和 `timing.image_age_s` 判断帧能否送入模型。Hybrid 只提供固定前视，双相机示例使用 MuJoCo 渲染。有传感器延迟的项目可用确定的保持动作推进仿真，再读取可用帧。

| 示例机器人和控制器 | 输出顺序与含义 |
| --- | --- |
| 五轴机械臂 `joints` | `[waist, shoulder, elbow, wrist_pitch, wrist_yaw, gripper]`。前五项是绝对角度，最后一项是以米计的共享手指位置。 |
| 五轴机械臂 `cartesian` | `[dx, dy, dz, grip]`。各项在 `[-1, 1]`，默认 XYZ 乘以 `0.012 m` 得到每步世界坐标增量。此实现使用固定姿态 5R 逆运动学，`grip=-1` 合爪，`+1` 开爪。 |
| 龙门 `second_robot` 的 `joints` | `[x, y, z, gripper]`。三轴绝对位置和共享手指位置均以米计。控制器 ID 与默认机械臂中的同名项分别属于各自的项目。 |

实际动作语义以所选控制器的 `action_names`、`units`、`action_range` 和 `frequency_hz` 为准。模型输出增量而控制器接收绝对位置时，在适配器中转换。速度、力矩、独立手指位置或任意六维末端姿态需要相匹配的控制器实现。

`POST /api/v1/compat` 接收 `project`、`robot_id`、可选的 `controller_id` 和 `contract`。检查项包括控制关节顺序 `joint_names`、输入关节 `state_joint_names`、动作顺序和单位 `action_names` 与 `action_units`、相机 ID、频率以及可选的 `controller_spec` 和 `camera_specs`。省略控制器时按项目顺序选择该机器人的第一个兼容项。模型图像预处理和归一化仍须按训练约定配置。

## 3. 跑通最小 HTTP 客户端

先启动 FastAPI 后端。安装依赖可用 `python -m pip install httpx pillow`。把以下代码保存为 `connect_model.py`，用 `--project`、`--robot`、`--controller` 和可重复的 `--camera` 参数指定测试对象。示例会从项目读取驱动关节并执行一步保持动作。接入模型时，将取值顺序和动作转换替换为模型自己的契约。

```python
import argparse
import base64
import io

import httpx
from PIL import Image

def model_inputs(obs, joints, cameras):
    state = obs["state"]
    inputs = {
        "q": {name: state["joint_pos"][name] for name in joints},
        "dq": {name: state["joint_vel"][name] for name in joints},
        "gripper": state["gripper"],
        "images": {},
    }
    for sensor_id in cameras:
        frame = obs["images"].get(sensor_id)
        if not frame:
            status = obs["camera_status"].get(sensor_id, {}).get("state", "unknown")
            raise RuntimeError(f"{sensor_id}: {status}")
        raw = base64.b64decode(frame["jpeg_b64"])
        with Image.open(io.BytesIO(raw)) as image:
            inputs["images"][sensor_id] = image.convert("RGB")
    return inputs


def hold_action(inputs, controller, gripper):
    if controller["mode"] == "cartesian_delta_ik":
        return [0.0, 0.0, 0.0, 2.0 * inputs["gripper"] - 1.0]
    q = dict(inputs["q"])
    q[controller["action_names"][-1]] = (q[gripper["left_joint"]] + q[gripper["right_joint"]]) / 2
    return [
        min(high, max(low, q[name]))
        for name, (low, high) in zip(controller["action_names"], controller["action_range"])
    ]


def run_one_step(client, project, robot_id, controller_id=None, cameras=()):
    def call(method, path, **options):
        response = client.request(method, f"/api/v1{path}", **options)
        response.raise_for_status()
        return response.json()

    cfg = call("GET", "/config", params={"project": project})
    robot = next(r for r in cfg["robots"] if r["id"] == robot_id)
    caps = call("GET", "/capabilities", params={"project": project})
    binding = next(r for r in caps["robots"] if r["id"] == robot_id)
    if controller_id is None:
        controller_id = next(iter(binding["controller_ids"]), None)
    if controller_id not in binding["controller_ids"]:
        raise ValueError(f"控制器不可用: {binding['incompatible_controllers']}")
    if set(cameras) - set(binding["sensor_ids"]):
        raise ValueError(f"相机不可用: {binding['unavailable_sensors']}")
    controller = next(c for c in cfg["controllers"] if c["id"] == controller_id)
    joints = [actuator["joint"] for actuator in robot["actuators"]]
    session = call("POST", "/sessions", json={
        "project": project, "robot_id": robot_id, "controller_id": controller_id,
    })
    path = f"/sessions/{session['id']}"
    try:
        if session["config_version"] != cfg["version"]:
            raise RuntimeError("项目配置已更新，请重新核对契约并创建会话")
        token = call("POST", path + "/control/acquire", json={"owner": "model-client"})["token"]
        obs = call("GET", path + "/observation")
        inputs = model_inputs(obs, joints, cameras)
        action = hold_action(inputs, controller, robot["gripper"])
        after = call("POST", path + "/action", json={
            "token": token, "epoch": obs["epoch"], "tick": obs["tick"], "action": action,
        })
        print(f"{robot_id}/{controller_id}: tick {obs['tick']} -> {after['tick']}")
    finally:
        call("DELETE", path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="default")
    parser.add_argument("--robot", default="lux3d_5r")
    parser.add_argument("--controller")
    parser.add_argument("--camera", action="append", default=[])
    args = parser.parse_args()
    with httpx.Client(base_url="http://127.0.0.1:8767", timeout=30) as client:
        run_one_step(client, args.project, args.robot, args.controller, args.camera)
```

在项目根目录执行示例。龙门使用自带的控制器配置，先将它保存为独立项目。

```powershell
python connect_model.py --robot lux3d_5r --controller cartesian --camera front --camera wrist
Invoke-RestMethod -Uri 'http://127.0.0.1:8767/api/v1/config?project=second_robot' -Method Put -ContentType 'application/json' -InFile 'configs/second_robot.json'
python connect_model.py --project second_robot --robot cartesian_3dof --controller joints --camera front --camera wrist
```

开始新回合时，调用 `POST /api/v1/sessions/{sid}/reset`，并清空模型的历史帧、前一动作及动作块缓存。用 `GET .../observation` 读回新回合的 `epoch` 和当前 `tick`。遇到 HTTP 409，重新读取会话观测并检查控制权，按新观测重新决策。HTTP 400 通常需要检查动作维数、有限性和范围。

新架构可通过这个外部客户端接入。如果希望用平台的 `policy_step` 直接运行它，需要实现对应 runner 并接到后端分派。
