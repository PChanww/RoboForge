"""Deployment contract compatibility checks.

Compares a policy's observation/action contract against a target robot config:
joint names/order, action mode/dim/units, camera keys/resolution, frequency and
normalization. Incompatible combinations are reported so the UI can block runs.
"""
from __future__ import annotations

import math


IK_JOINTS = ("waist", "shoulder", "elbow", "wrist_pitch", "wrist_yaw")


def supports_cartesian_ik(robot):
    rotary = {joint.name for joint in robot.joints if joint.type == "revolute"}
    actuated = {actuator.joint for actuator in robot.actuators}
    return set(IK_JOINTS) <= rotary & actuated


def controller_binding_issues(robot, controller):
    joints = {joint.name: joint for joint in robot.joints if joint.type != "fixed"}
    actuated = {actuator.joint for actuator in robot.actuators} & joints.keys()
    gripper = robot.gripper
    if gripper is None:
        return ["the tabletop grasping task requires a configured gripper"]
    fingers = {gripper.left_joint, gripper.right_joint}
    if len(fingers) != 2 or not fingers <= actuated:
        return ["the gripper must bind two actuated joints"]
    if not (math.isfinite(gripper.open_m) and math.isfinite(gripper.closed_m)
            and gripper.open_m > gripper.closed_m):
        return ["the gripper requires finite open and closed positions with open > closed"]
    pad_names = {geom.name for link in robot.links for geom in link.collisions}
    pads = set(gripper.pad_geoms)
    if len(pads) != 2 or not pads <= pad_names:
        return ["the tabletop grasping task requires two configured contact pad geoms"]
    if controller.mode == "cartesian_delta_ik":
        if not supports_cartesian_ik(robot):
            return ["Cartesian IK requires the actuated fixed-attitude 5R chain"]
        step = controller.max_delta_m
        if controller.action_dim != 4 or step is None or not math.isfinite(step) or step <= 0:
            return ["Cartesian IK requires four action channels and a positive finite max_delta_m"]
        return []

    if controller.action_dim < 1:
        return ["joint position requires a final shared gripper channel"]
    action_joints = controller.action_names[:-1]
    unknown = sorted(set(action_joints) - (actuated - fingers))
    if unknown:
        return ["joint-position channels do not bind to actuated non-gripper joints: " + ", ".join(unknown)]
    if len(action_joints) != len(set(action_joints)):
        return ["joint-position channels contain duplicate joint names"]
    return []


def select_controller(config, robot, controller_id=None):
    if controller_id is not None:
        controller = next((item for item in config.controllers if item.id == controller_id), None)
        if controller is None:
            raise ValueError(f"Unknown controller {controller_id!r}")
        issues = controller_binding_issues(robot, controller)
        if issues:
            raise ValueError(f"Controller {controller_id!r} is incompatible with {robot.id!r}: {issues[0]}")
        return controller
    return next((item for item in config.controllers if not controller_binding_issues(robot, item)), None)


def camera_binding_issue(robot, sensor, renderer):
    if sensor.parent != "world" and all(link.name != sensor.parent for link in robot.links):
        return f"camera {sensor.id!r} mount {sensor.parent!r} is absent on robot {robot.id!r}"
    if renderer == "hybrid" and (sensor.id != "front" or sensor.parent != "world"):
        return "static_background_overlay only supports the world-fixed front camera"
    return None


def robot_bindings(config, robot):
    controller_issues = {
        controller.id: issues for controller in config.controllers
        if (issues := controller_binding_issues(robot, controller))
    }
    sensor_issues = {
        sensor.id: issue for sensor in config.sensors
        if (issue := camera_binding_issue(robot, sensor, config.scene.sensor_renderer))
    }
    return {
        "controller_ids": [item.id for item in config.controllers if item.id not in controller_issues],
        "incompatible_controllers": controller_issues,
        "sensor_ids": [item.id for item in config.sensors if item.id not in sensor_issues],
        "unavailable_sensors": sensor_issues,
    }


def check_contract(contract: dict, config, robot_id: str | None = None, controller_id: str | None = None):
    robot = config.robots[0] if robot_id is None else next((r for r in config.robots if r.id == robot_id), None)
    if robot is None:
        return {"compatible": False, "mismatches": [f"robot {robot_id} not in config"]}
    mism = []
    actuated = [j.name for j in robot.joints if j.type != "fixed"]
    if robot.gripper:
        joints = [n for n in actuated if n not in (robot.gripper.left_joint, robot.gripper.right_joint)]
    else:
        joints = actuated
    cj = contract.get("joint_names") or []
    if cj and cj != joints:
        mism.append(f"joint names/order differ: policy {cj} vs robot {joints}")
    try:
        ctrl = select_controller(config, robot, controller_id)
    except ValueError as exc:
        return {"compatible": False, "mismatches": [str(exc)], "robot": robot.id}
    if ctrl is None:
        return {"compatible": False, "mismatches": [f"No compatible controller for robot {robot.id!r}"],
                "robot": robot.id}
    if contract.get("action_mode") and contract["action_mode"] != ctrl.mode:
        mism.append(f"action mode {contract['action_mode']} != robot controller {ctrl.mode}")
    if contract.get("action_dim") and contract["action_dim"] != ctrl.action_dim:
        mism.append(f"action dim {contract['action_dim']} != {ctrl.action_dim}")
    if contract.get("action_names") and contract["action_names"] != ctrl.action_names:
        mism.append(f"action names differ: {contract['action_names']} vs {ctrl.action_names}")
    if contract.get("action_units") and contract["action_units"] != ctrl.units:
        mism.append(f"action units differ: {contract['action_units']} vs {ctrl.units}")
    cams = [s.id for s in config.sensors if not camera_binding_issue(robot, s, config.scene.sensor_renderer)]
    ck = contract.get("camera_keys") or []
    if ck and not set(ck) <= set(cams):
        mism.append(f"camera keys {ck} not subset of available robot sensors {cams}")
    known_state_joints = {actuator.joint for actuator in robot.actuators}
    missing_state_joints = sorted(
        set(contract.get("state_joint_names") or [])
        - known_state_joints - {f"{name}_vel" for name in known_state_joints}
    )
    if missing_state_joints:
        mism.append(f"unknown state joints: {missing_state_joints}")
    if contract.get("fps") and abs(contract["fps"] - ctrl.frequency_hz) > 1e-6:
        mism.append(f"policy fps {contract['fps']} != controller {ctrl.frequency_hz}")
    if contract.get("privileged"):
        mism.append("policy requires privileged object pose (not deployable on sensor-only observation)")
    for field, value in contract.get("controller_spec", {}).items():
        if ctrl.model_dump().get(field) != value:
            mism.append(f"controller {field} differs from training")
    for name, expected in contract.get("camera_specs", {}).items():
        actual = next((s for s in config.sensors if s.id == name), None)
        if actual is None:
            mism.append(f"camera {name} missing")
            continue
        actual = actual.model_dump()
        for field, value in expected.items():
            if actual.get(field) != value:
                mism.append(f"camera {name}.{field} differs from training")
    return {"compatible": not mism, "mismatches": mism, "robot": robot.id, "controller": ctrl.id}


def visual_contract(config):
    """The current CNN evaluator supports one named 5R baseline, not arbitrary robots."""
    if config.scene.environment and not config.scene.use_default_table and not any(c.enabled for c in config.scene.colliders):
        raise ValueError("Scene has no table collision support; repair the scene and recollect data before training")
    expected = ["waist", "shoulder", "elbow", "wrist_pitch", "wrist_yaw"]
    contract = {"joint_names": expected, "action_mode": "cartesian_delta_ik", "action_dim": 4,
                "action_names": ["dx", "dy", "dz", "grip"], "fps": 20.0,
                "camera_keys": ["front"], "privileged": False}
    result = check_contract(contract, config)
    names = [j.name for j in config.robots[0].joints if j.type != "fixed"]
    if names != expected + ["left", "right"]:
        result["mismatches"].append("VisualBC requires the seven baseline state joints in order")
    if result["mismatches"]:
        raise ValueError("Unsupported VisualBC input: " + "; ".join(result["mismatches"]))
    sensor = next(s for s in config.sensors if s.id == "front")
    contract["camera_specs"] = {"front": {k: v for k, v in sensor.model_dump().items() if k != "id"}}
    contract["state_joint_names"] = names
    contract["controller_spec"] = {k: v for k, v in config.controllers[0].model_dump().items()
                                   if k in {"mode", "action_dim", "action_names", "frame", "max_delta_m", "frequency_hz", "action_range"}}
    return contract
