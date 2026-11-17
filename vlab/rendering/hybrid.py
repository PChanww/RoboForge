"""Hybrid sensor renderer: 3DGS background + MuJoCo robot/task foreground.

Restricted mode `static_background_overlay`: the world-fixed camera's gaussian
background is rendered once (static), then every MuJoCo foreground pixel is composited
over it. Splats never occlude the robot; this is a visual-background experiment mode,
not a full occlusion/physical digital twin. Moving cameras (wrist) are rejected. The gaussian asset drives appearance only; MuJoCo remains
the physics source of truth. Assets that are uncalibrated may not align with the robot
table (reported, not hidden).
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict

import numpy as np

from ..gaussian import load_gaussians
from ..splat_render import render as splat_render

HYBRID_MODE = "static_background_overlay"
RENDERER_VERSION = 3
_BG_CACHE_MAX = 4


def find_gaussian_ply(lab_root, scene_instance):
    """Locate the gaussian .ply for a bound scene revision (from its manifest files)."""
    import json
    from pathlib import Path
    d = Path(lab_root) / "assets" / "scenes" / scene_instance.asset_id / str(scene_instance.revision)
    mf = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    if mf.get("ply_kind") != "gaussian":
        return None
    for name in mf.get("files", {}):
        if name.lower().endswith(".ply"):
            return d / name
    return None


def foreground_ids(env):
    """Segmentation object ids for robot links + cup + tray + scene asset nodes. MuJoCo Renderer
    returns objid/objtype directly, so the foreground id is the geom id (no +1)."""
    fg = set()
    robot_bodies = {l.name for l in env.robot.links} | {"cup"}
    tray_ref = getattr(env.config.task, "goal_tray_collider_ref", None)
    tray_prefix = (tray_ref[:-7] if tray_ref and tray_ref.endswith("_part_0") else tray_ref)
    for gid in range(env.model.ngeom):
        name = env.model.geom(gid).name
        body = env.model.body(env.model.geom_bodyid[gid]).name
        if (body in robot_bodies or body.startswith("node_")
                or name == "tray"
                or (tray_ref and (name == tray_ref or name.startswith(f"{tray_prefix}_part_")))
                or name.endswith("_pad") or name == "cup_contact" or name.startswith("cup_contact_")):
            fg.add(gid)
    return fg


def segmentation_mask(seg, ids):
    """True where the segmentation channel is a geom object with objid in ids.
    Channel 0 is objid (-1 background), channel 1 is objtype; require mjOBJ_GEOM."""
    import mujoco
    ids_set = np.asarray(sorted(ids), dtype=np.int32)
    obj = seg[:, :, 0]
    typ = seg[:, :, 1]
    return np.isin(obj, ids_set) & (typ == int(mujoco.mjtObj.mjOBJ_GEOM))


_BG_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
_SHA_CACHE: dict = {}


def asset_sha256(ply_path) -> str:
    p = str(ply_path)
    st = __import__("os").stat(p)
    key = (p, st.st_mtime_ns, st.st_size)
    sha = _SHA_CACHE.get(key)
    if sha is None:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        sha = h.hexdigest()
        _SHA_CACHE.clear()
        _SHA_CACHE[key] = sha
    return sha


def _cache_key(ply_path, cam, inst, width, height):
    key = (str(ply_path), asset_sha256(ply_path), RENDERER_VERSION, HYBRID_MODE,
           tuple(np.round(np.asarray(cam["world_pos"]).ravel(), 6)),
           tuple(np.round(np.asarray(cam["world_mat"]).ravel(), 6)),
           round(float(cam["fx"]), 6), round(float(cam["fy"]), 6),
           round(float(cam["cx"]), 6), round(float(cam["cy"]), 6), width, height)
    if inst:
        key += (tuple(inst.get("position", [0, 0, 0])), tuple(inst.get("quat_xyzw", [0, 0, 0, 1])),
                float(inst.get("uniform_scale", 1.0)))
    return key


def _cache_get(key):
    hit = _BG_CACHE.get(key)
    if hit is not None:
        _BG_CACHE.move_to_end(key)
    return hit


def _cache_put(key, value):
    _BG_CACHE[key] = value
    _BG_CACHE.move_to_end(key)
    while len(_BG_CACHE) > _BG_CACHE_MAX:
        _BG_CACHE.popitem(last=False)


class HybridRenderer:
    mode = HYBRID_MODE
    version = RENDERER_VERSION

    def __init__(self, ply_path, camera_snapshot=None, width=256, height=256, inst=None, camera=None):
        self.width, self.height = width, height
        cam = camera if camera is not None else camera_snapshot["cameras"]["front"]
        key = _cache_key(ply_path, cam, inst, width, height)
        self.key = key
        cached = _cache_get(key)
        if cached is not None:
            self._bg, self._bg_depth, self.stats = cached
            self.from_cache = True
            return
        self.g = load_gaussians(ply_path)
        pos = np.asarray(cam["world_pos"])
        R = np.asarray(cam["world_mat"])
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = pos
        bg, depth, stats = splat_render(
            self.g, M.T.flatten(), cam["fx"], cam["fy"], cam["cx"], cam["cy"], width, height, inst=inst)
        bg.setflags(write=False)
        depth.setflags(write=False)
        self._bg, self._bg_depth, self.stats = bg, depth, stats
        _cache_put(key, (bg, depth, stats))
        self.from_cache = False


    def compose(self, mujoco_rgb, seg, fg_ids):
        mask = segmentation_mask(seg, fg_ids)
        out = np.array(mujoco_rgb, copy=True)
        out[~mask] = (self._bg[~mask] * 255).astype(np.uint8)
        return out, mask

    def background(self):
        return (self._bg * 255).astype(np.uint8)


def clear_background_cache():
    _BG_CACHE.clear()
    _SHA_CACHE.clear()