"""Pure inference architecture/contract helpers for the visual BC policy.

Imports only torch/numpy so it can be shipped inside a deployment package without
LeRobot, Pydantic or the vlab training environment.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

STATE_DIM = 18
PREV_DIM = 4
IMG = 64
JOINTS = ["waist", "shoulder", "elbow", "wrist_pitch", "wrist_yaw", "left", "right"]
STATE_NAMES = JOINTS + [n + "_vel" for n in JOINTS] + ["eef_x", "eef_y", "eef_z", "gripper"]
ACTION_NAMES = ["dx", "dy", "dz", "grip"]


class VisualBC(nn.Module):
    def __init__(self, in_channels: int = 3, chunk_size: int = 1):
        super().__init__()
        self.in_channels = int(in_channels)
        self.chunk_size = max(1, int(chunk_size))
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channels, 16, 5, stride=2), nn.ReLU(),   # 30
            nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(),  # 14
            nn.Conv2d(32, 64, 3, stride=2), nn.ReLU(),  # 6
            nn.Flatten())
        self.img_fc = nn.Linear(64 * 6 * 6, 64)
        self.state_fc = nn.Linear(STATE_DIM + PREV_DIM, 32)
        self.head = nn.Sequential(nn.Linear(64 + 32, 128), nn.ReLU(),
                                  nn.Linear(128, 4 * self.chunk_size), nn.Tanh())

    def forward(self, img, state, prev):
        i = F.relu(self.img_fc(self.conv(img)))
        s = F.relu(self.state_fc(torch.cat([state, prev], dim=1)))
        out = self.head(torch.cat([i, s], dim=1))
        if self.chunk_size == 1:
            return out
        return out.view(-1, self.chunk_size, 4)


def fallback_contract(img_size=IMG):
    return {"arch": "visual_bc_cnn_v1", "image_key": "front", "image_size": img_size,
            "state_names": STATE_NAMES, "prev_action_dim": 4, "action_names": ACTION_NAMES,
            "action_mode": "cartesian_delta_ik", "fps": 20.0, "privileged": False,
            "normalization": "train-split state/prev mean-std; image /255"}


def check_target_config(contract, config):
    """Compare a package contract with a project config dict (no vlab/Pydantic)."""
    mism = []
    robots = config.get("robots") or []
    robot = robots[0] if robots else None
    if robot is None:
        return ["target config has no robots"]
    actuated = [j["name"] for j in robot.get("joints", []) if j.get("type") != "fixed"]
    grip = robot.get("gripper") or {}
    expected_joints = [n for n in actuated if n not in (grip.get("left_joint"), grip.get("right_joint"))]
    names = contract.get("state_joint_names")
    if names and names[:len(expected_joints)] != expected_joints:
        mism.append(f"state joint names/order differ: policy {names} vs robot {expected_joints}")
    ctrl = (config.get("controllers") or [{}])[0]
    spec = contract.get("controller_spec") or {}
    for field, value in spec.items():
        if ctrl.get(field) != value:
            mism.append(f"controller {field} differs from training")
    if contract.get("action_mode") is not None and ctrl.get("mode") != contract["action_mode"]:
        mism.append(f"action_mode {contract['action_mode']} != target {ctrl.get('mode')}")
    for field in ("action_dim", "action_names", "fps"):
        key = "frequency_hz" if field == "fps" else field
        if contract.get(field) is not None and ctrl.get(key) != contract[field]:
            mism.append(f"{field} {contract[field]} != target {ctrl.get(key)}")
    sensors = {s["id"]: s for s in config.get("sensors", [])}
    keys = contract.get("camera_keys") or ([contract["image_key"]] if contract.get("image_key") else [])
    for key in keys:
        if key not in sensors:
            mism.append(f"camera {key} missing in target")
    for name, expected in (contract.get("camera_specs") or {}).items():
        actual = sensors.get(name)
        if actual is None:
            mism.append(f"camera {name} missing in target")
            continue
        for field, value in expected.items():
            if actual.get(field) != value:
                mism.append(f"camera {name}.{field} differs from training")
    if contract.get("privileged"):
        mism.append("policy requires privileged object pose")
    return mism
