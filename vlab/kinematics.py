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


def clamp_to_reach(point, p, *, margin=5e-4):
    """Project a Cartesian target onto this arm's reachable annulus.

    The analytic IK enforces reach by raising, which turns a millimetre of
    overshoot into a dead episode. Callers that must keep running (a learned
    policy has no reach model) need the bounded form instead: keep the azimuth
    and the radial/vertical ratio, rescale the distance into
    ``[|l1-l2|+margin, l1+l2-margin]``.

    Returns ``(point, info)``. ``point`` is the caller's own array and ``info``
    is None when the request was already reachable; otherwise ``info`` carries
    the before/after evidence needed to record what was asked for versus what
    the robot could execute.

    ``margin`` defaults to 0.5 mm so a world->base->world round trip in front
    of the analytic solver cannot land exactly on the singular boundary.
    """
    q = np.asarray(point, dtype=float)
    if q.shape != (3,) or not np.isfinite(q).all():
        raise ValueError("reach projection needs a finite 3D point")
    x, y, z = (float(v) for v in q)
    radial = np.hypot(x, y - p["base_y"])
    h = z + p["tool"] - p["shoulder_z"]
    dist = float(np.hypot(radial, h))
    outer = p["l1"] + p["l2"] - margin
    inner = abs(p["l1"] - p["l2"]) + margin
    if inner <= dist <= outer:
        return q, None
    limit = outer if dist > outer else inner
    if dist <= 1e-12:
        unit_radial, unit_h = 1.0, 0.0
    else:
        unit_radial, unit_h = radial / dist, h / dist
    new_radial, new_h = unit_radial * limit, unit_h * limit
    yaw = np.arctan2(y - p["base_y"], x)
    projected = np.array([new_radial * np.cos(yaw),
                          p["base_y"] + new_radial * np.sin(yaw),
                          p["shoulder_z"] + new_h - p["tool"]])
    return projected, {
        "kind": "reach_clamped_outer" if dist > outer else "reach_clamped_inner",
        "requested_distance_m": round(dist, 6),
        "limit_m": round(float(limit), 6),
        "overshoot_m": round(dist - limit, 6),
        "requested_point": [round(v, 6) for v in q.tolist()],
        "projected_point": [round(float(v), 6) for v in projected],
    }


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