"""Config-driven kinematics for the fixed-attitude 5R arm.

Chain geometry (base offset, shoulder height, link lengths, tool offset) is read
from the robot configuration / MuJoCo model rather than hard-coded, so editing a
link origin or tool offset rebuilds consistent FK/IK. The analytic solution is
verified against MuJoCo forward kinematics; unreachable targets raise a
structured error.
"""
from __future__ import annotations

import numpy as np


class Unreachable(ValueError):
    def __init__(self, reason, target):
        super().__init__(reason)
        self.reason = reason
        self.target = list(target)


def chain_params(robot):
    links = {l.name: l for l in robot.links}
    joints = {j.name: j for j in robot.joints}
    base_y = links[robot.root_link].origin[1]
    shoulder_z = links[joints["waist"].child].origin[2]
    l1 = links[joints["elbow"].child].origin[2]
    l2 = links[joints["wrist_pitch"].child].origin[2]
    # The eef link origin is the model's actual tool transform (source of truth).
    tool = -links[robot.end_effector].origin[2]
    return {"base_y": base_y, "shoulder_z": shoulder_z, "l1": l1, "l2": l2, "tool": tool}


def analytic_ik(point, p):
    x, y, z = point
    radial = np.hypot(x, y - p["base_y"])
    h = z + p["tool"] - p["shoulder_z"]
    dist = np.hypot(radial, h)
    if dist > p["l1"] + p["l2"] - 1e-6:
        raise Unreachable(f"target distance {dist:.4f} m exceeds reach {p['l1']+p['l2']:.4f} m", point)
    if dist < abs(p["l1"] - p["l2"]) + 1e-6:
        raise Unreachable(f"target distance {dist:.4f} m inside inner dead zone", point)
    bend = np.arccos(np.clip((radial**2 + h**2 - p["l1"]**2 - p["l2"]**2) / (2 * p["l1"] * p["l2"]), -1, 1))
    yaw = np.arctan2(y - p["base_y"], x)
    pitch = np.arctan2(radial, h) - np.arctan2(p["l2"] * np.sin(bend), p["l1"] + p["l2"] * np.cos(bend))
    return [yaw, pitch, bend, -pitch - bend, -yaw]


def analytic_fk(q, p):
    yaw, pitch, bend, wp, wy = q
    radial = p["l1"] * np.sin(pitch) + p["l2"] * np.sin(pitch + bend)
    z = p["shoulder_z"] + p["l1"] * np.cos(pitch) + p["l2"] * np.cos(pitch + bend) - p["tool"]
    return np.array([radial * np.cos(yaw), p["base_y"] + radial * np.sin(yaw), z])


def validate(model, data, robot, eef_id, targets, qposadr):
    """Set IK joints, run MuJoCo FK, compare eef position to the requested target."""
    p = chain_params(robot)
    rows = []
    saved = data.qpos.copy()
    for t in targets:
        try:
            q = analytic_ik(t, p)
        except Unreachable as exc:
            rows.append({"target": list(t), "reachable": False, "reason": exc.reason})
            continue
        for name, v in zip(("waist", "shoulder", "elbow", "wrist_pitch", "wrist_yaw"), q):
            data.qpos[qposadr[name]] = v
        import mujoco
        mujoco.mj_forward(model, data)
        err = float(np.linalg.norm(data.xpos[eef_id] - np.asarray(t)))
        rows.append({"target": list(t), "reachable": True, "fk_error_mm": round(1000 * err, 4)})
    data.qpos[:] = saved
    import mujoco
    mujoco.mj_forward(model, data)
    ok = all(r.get("reachable") and r["fk_error_mm"] < 1.0 for r in rows if r.get("reachable"))
    return {"consistent": ok, "params": p, "samples": rows}