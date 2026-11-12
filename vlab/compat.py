"""Deployment contract compatibility checks.

Compares a policy's observation/action contract against a target robot config:
joint names/order, action mode/dim/units, camera keys/resolution, frequency and
normalization. Incompatible combinations are reported so the UI can block runs.
"""
from __future__ import annotations


def check_contract(contract: dict, config, robot_id: str | None = None):
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
    ctrl = config.controllers[0]
    if contract.get("action_mode") and contract["action_mode"] != ctrl.mode:
        mism.append(f"action mode {contract['action_mode']} != robot controller {ctrl.mode}")
    if contract.get("action_dim") and contract["action_dim"] != ctrl.action_dim:
        mism.append(f"action dim {contract['action_dim']} != {ctrl.action_dim}")
    if contract.get("action_names") and contract["action_names"] != ctrl.action_names:
        mism.append(f"action names differ: {contract['action_names']} vs {ctrl.action_names}")
    cams = [s.id for s in config.sensors]
    ck = contract.get("camera_keys") or []
    if ck and not set(ck) <= set(cams):
        mism.append(f"camera keys {ck} not subset of robot sensors {cams}")
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
    return {"compatible": not mism, "mismatches": mism, "robot": robot.id}


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
