"""Immutable per-capture snapshot for sensor rendering.

Taken at the scheduler capture instant from the CURRENT MuJoCo data (not the
control-tick observation.state). Arrays are copied AND marked read-only; nested
metadata is deep-copied so no mutable state is shared with the simulation.
`to_wire`/`from_wire` give an explicit JSON-safe transport contract (arrays -> lists);
for high-frequency transport prefer a binary npz, never raw JSON of 200k gaussians.
"""
from __future__ import annotations

import copy
import uuid
from types import MappingProxyType

import numpy as np

from ..transforms import mat_to_pos_quat_xyzw


def _ro(a):
    a = np.array(a, dtype=np.float64, copy=True)
    a.setflags(write=False)
    return a


def capture_snapshot(env, capture_time, request_id=None, session_id=None):
    mdl, data = env.model, env.data
    bodies = {}
    for link in env.robot.links:
        bid = mdl.body(link.name).id
        bodies[link.name] = MappingProxyType({"xpos": _ro(data.xpos[bid]),
                                              "xmat": _ro(data.xmat[bid].reshape(3, 3))})
    cup = mdl.body("cup").id
    bodies["cup"] = MappingProxyType({"xpos": _ro(data.xpos[cup]),
                                      "xmat": _ro(data.xmat[cup].reshape(3, 3))})
    cams = {}
    for s in env.config.sensors:
        cid = mdl.camera(s.id).id
        pos = _ro(data.cam_xpos[cid])
        mat = _ro(data.cam_xmat[cid].reshape(3, 3))
        T = np.eye(4)
        T[:3, :3] = mat
        T[:3, 3] = pos
        _, quat_xyzw, _ = mat_to_pos_quat_xyzw(T)
        cams[s.id] = MappingProxyType({
            "world_pos": pos, "world_mat": mat, "quat_xyzw": tuple(quat_xyzw),
            "parent": s.parent, "fx": s.intrinsics.fx, "fy": s.intrinsics.fy,
            "cx": s.intrinsics.cx, "cy": s.intrinsics.cy, "resolution": tuple(s.resolution)})
    scene = copy.deepcopy(dict(getattr(env, "_env_info", {})))
    return MappingProxyType({
        "request_id": request_id or uuid.uuid4().hex,
        "session_id": session_id or getattr(env, "session_id", "standalone"),
        "epoch": int(env.epoch), "tick": int(env.scheduler.tick),
        "capture_time": float(capture_time),
        "scene_revision": env.config.version,
        "coordinate_convention": env.config.scene.coordinate_convention,
        "qpos": _ro(data.qpos), "bodies": MappingProxyType(bodies),
        "cameras": MappingProxyType(cams), "scene": MappingProxyType(scene),
    })


def to_wire(snap):
    """JSON-safe encoding (arrays -> nested lists). Not for 200k-gaussian payloads."""
    def enc(v):
        if isinstance(v, np.ndarray):
            return {"__array__": v.tolist()}
        return v
    return {
        "request_id": snap["request_id"], "session_id": snap["session_id"],
        "epoch": snap["epoch"], "tick": snap["tick"], "capture_time": snap["capture_time"],
        "scene_revision": snap["scene_revision"], "coordinate_convention": snap["coordinate_convention"],
        "qpos": enc(snap["qpos"]),
        "bodies": {k: {kk: enc(vv) for kk, vv in v.items()} for k, v in snap["bodies"].items()},
        "cameras": {k: {kk: (enc(vv) if isinstance(vv, np.ndarray) else vv) for kk, vv in v.items()}
                    for k, v in snap["cameras"].items()},
        "scene": dict(snap["scene"]),
    }


def from_wire(w):
    def dec(v):
        if isinstance(v, dict) and "__array__" in v:
            return _ro(v["__array__"])
        return v
    bodies = {k: MappingProxyType({kk: dec(vv) for kk, vv in v.items()}) for k, v in w["bodies"].items()}
    cams = {k: MappingProxyType({kk: dec(vv) for kk, vv in v.items()}) for k, v in w["cameras"].items()}
    return MappingProxyType({
        "request_id": w["request_id"], "session_id": w["session_id"], "epoch": w["epoch"],
        "tick": w["tick"], "capture_time": w["capture_time"], "scene_revision": w["scene_revision"],
        "coordinate_convention": w["coordinate_convention"], "qpos": dec(w["qpos"]),
        "bodies": MappingProxyType(bodies), "cameras": MappingProxyType(cams),
        "scene": MappingProxyType(w["scene"])})