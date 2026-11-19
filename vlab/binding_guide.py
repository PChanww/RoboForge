"""Render binding instructions from registered assets and versioned project config."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import Response

from .compat import robot_bindings
from .scene_assets import contained_file, load_scene, valid_id
from .scene_builder import parse_asset_ref
from .schema import ProjectConfig
from .store import ConfigStore


def _json_block(data: dict) -> str:
    return "```json\n" + json.dumps(data, ensure_ascii=False, indent=2) + "\n```\n"


def guide_download(text: str, filename: str) -> Response:
    return Response(text, media_type="text/markdown",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


def load_guide_config(store: ConfigStore, project_id: str, version: int | None) -> ProjectConfig:
    if version is None:
        return store.load(project_id)
    try:
        return store.load_version(project_id, version)
    except FileNotFoundError:
        current = store.load(project_id)
        if current.version != version:
            raise FileNotFoundError(f"Project {project_id} version {version} is unavailable")
        return current


def _bindings(cfg: ProjectConfig) -> dict:
    robots = []
    for robot in cfg.robots:
        actuated = {actuator.joint for actuator in robot.actuators}
        robots.append({
            "robot_id": robot.id,
            **robot_bindings(cfg, robot),
            "joints": [
                {"name": joint.name, "type": joint.type, "axis": joint.axis,
                 "position_unit": "rad" if joint.type == "revolute" else "m",
                 "actuated": joint.name in actuated}
                for joint in robot.joints if joint.type != "fixed"
            ],
            "gripper": robot.gripper.model_dump(exclude={"pad_geoms"}) if robot.gripper else None,
        })
    return {
        "project": cfg.project_id,
        "config_version": cfg.version,
        "sensor_renderer": cfg.scene.sensor_renderer,
        "robots": robots,
        "sensors": [
            sensor.model_dump(include={"id", "type", "parent", "resolution", "fps", "latency_s"})
            for sensor in cfg.sensors
        ],
        "controllers": [
            ctrl.model_dump(include={"id", "mode", "action_names", "action_dim", "units",
                                     "action_range", "frame", "frequency_hz", "max_delta_m"})
            for ctrl in cfg.controllers
        ],
    }


def _project_instructions(cfg: ProjectConfig) -> str:
    project_id = valid_id(cfg.project_id)
    return (
        f"## 项目接线参考 {project_id} v{cfg.version}\n\n"
        f"读取配置：`GET /api/v1/config?project={project_id}`，"
        f"历史版本入口为 `GET /api/v1/config/versions/{cfg.version}?project={project_id}`。"
        "新建仿真会话时显式选择项目、机器人和控制器，并核对响应中的版本。\n\n"
        "- 当前内置任务是双指夹爪的桌面抓放。场景与夹爪接触结果需要通过仿真实跑确认。\n"
        "- 关节位置与速度：按模型输入顺序，从 `observation.state.joint_pos` 和 "
        "`observation.state.joint_vel` 按关节名取值。速度单位对应 rad/s 或 m/s。\n"
        "- RGB 相机：按传感器 ID 从 `observation.images[ID].jpeg_b64` 读图，"
        "通过 `camera_status` 和时间戳确认帧可用。Hybrid 只提供固定前视。\n"
        "- 先选机器人，再从它的 `controller_ids` 和 `sensor_ids` 选择配置。"
        "列表为空时可参考 `incompatible_controllers` 或 `unavailable_sensors`。\n"
        "- 动作：按所选控制器的 `action_names`、单位与范围排列输出。"
        "模型的张量形状、预处理和归一化由适配器按模型要求补齐。\n"
        "- 接入：创建会话并获取控制权，读取观测，再向 `/api/v1/sessions/{sid}/action` "
        "提交当前 `token`、`epoch`、`tick` 和 `action`。\n\n"
        + _json_block(_bindings(cfg))
    )


def project_binding_guide(cfg: ProjectConfig) -> str:
    scene = cfg.scene
    return (
        "# 项目绑定指南\n\n"
        + _project_instructions(cfg)
        + "\n## 场景资产引用\n\n"
        + _json_block({
            "task_id": cfg.task.id,
            "task_asset": cfg.task.asset,
            "environment": (scene.environment.model_dump(include={"asset_id", "revision"})
                            if scene.environment else None),
            "asset_nodes": [
                node.model_dump(include={"node_id", "asset_ref", "role", "collider_ref"})
                for node in scene.asset_nodes
            ],
        })
    )


def asset_binding_guide(lab_root: Path, job: dict, store: ConfigStore | None) -> str:
    asset_ref = job["asset_id"]
    if "/" not in asset_ref:
        asset_ref = f"scenes/{valid_id(asset_ref)}/{job['revision']}"
    namespace, asset_id, *revision = parse_asset_ref(asset_ref)
    if namespace == "scene":
        _, manifest = load_scene(lab_root, asset_id, revision[0])
        kind = "environment_mesh" if manifest.get("visual_asset") else "environment_splat"
    elif namespace == "imported":
        path = contained_file(lab_root / "assets" / "imported", f"{asset_id}/manifest.json")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        kind = manifest.get("kind", "task_object")
    else:
        raise ValueError("Expected a registered generation asset")

    fields = ("role", "calibrated", "physics_status", "status", "up_axis", "actual_extents_m",
              "height_m", "width_m", "depth_m", "radius_m", "mass_kg", "friction", "scale_basis",
              "collision_proxy_desc", "sha256_glb")
    asset = {"asset_ref": asset_ref, "kind": kind, "remote_id": job.get("remote_id")}
    asset.update({name: manifest[name] for name in fields if manifest.get(name) is not None})
    if kind == "robot_visual":
        next_step = ("按需拆分外观部件，在机器人编辑器中把部件绑定到连杆，配置关节轴、限位、执行器与相机，"
                     "保存后下载该项目版本的绑定指南。")
    elif kind == "environment_splat":
        next_step = ("作为空间背景使用。在背景场景中配置支持的固定前视 hybrid 渲染，"
                     "并保留默认桌面或设置独立碰撞代理。")
    elif manifest.get("calibrated") is not True:
        next_step = "先预览资产并确认尺寸。完成标定并配置碰撞代理后，再校验物理场景。"
    elif kind == "environment_mesh":
        next_step = "绑定到项目的环境场景，核对工作面和碰撞代理，再完成场景校验。"
    elif kind == "task_object" and manifest.get("role") == "dynamic":
        next_step = "在场景配方中选择该资产作为任务对象或动态节点，核对尺寸、质量和碰撞，再校验并应用。"
    else:
        next_step = "在场景配方中将资产添加为对应角色的节点，设置位姿与所需碰撞，再校验并应用。"

    text = ("# 资产绑定指南\n\n## 入库记录\n\n" + _json_block(asset)
            + "\n## 下一步\n\n" + next_step + "\n")
    params = job.get("params", {})
    project_id = job.get("project_id") or params.get("project_id")
    version = job.get("config_version")
    if version is None:
        version = params.get("config_version")
    if store is not None and project_id and version is not None:
        try:
            cfg = load_guide_config(store, project_id, version)
        except (FileNotFoundError, ValueError):
            return text + "\n生成时关联的项目版本已不可读取。请选择目标项目并下载其项目绑定指南。\n"
        return (text + "\n以下内容来自生成任务关联的项目版本。"
                "应用配方或装配机器人后，请下载保存后的版本以确认实际绑定。\n\n"
                + _project_instructions(cfg))
    return text + "\n选择目标项目并保存场景或机器人后，从项目绑定指南读取关节、相机与动作顺序。\n"
