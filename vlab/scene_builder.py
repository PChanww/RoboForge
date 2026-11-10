"""Environment MJCF builder: default tabletop OR imported scene visual + static colliders.

Separated from task objects (tray/cup) so a generated background never replaces the
dynamic task object. Static colliders have no freejoint (no mass dynamics).
"""
from __future__ import annotations

import json
from pathlib import Path

from .schema import ProjectConfig
from .transforms import quat_xyzw_to_wxyz, transform_matrix, mat_to_pos_quat_xyzw

LIGHTS = """
        <light pos="-.4 -.3 .85" dir=".25 .25 -1" diffuse=".69 .65 .59" castshadow="true"/>
        <light pos=".5 .4 .6" dir="-.6 -.3 -1" diffuse=".44 .48 .56" castshadow="false"/>"""
FLOOR = '<geom type="plane" size="2 2 .01" pos="0 0 -.061" rgba=".19 .23 .25 1"/>'
DEFAULT_TABLE = '<geom name="table" type="box" pos="0 0 -.028" size=".32 .215 .028" rgba=".29 .37 .39 1"/>'


def _fmt(v):
    return " ".join(f"{x:.6g}" for x in v)


def _collider_xml(c):
    q = _fmt(quat_xyzw_to_wxyz(c.quaternion_xyzw))
    common = (f'name="{c.name}" pos="{_fmt(c.position)}" quat="{q}" '
              f'friction="{_fmt(c.friction)}" contype="{c.contype}" conaffinity="{c.conaffinity}"')
    fx, fy, fz = c.full_size_m
    if c.shape == "box":
        return f'<geom type="box" size="{fx/2:.6g} {fy/2:.6g} {fz/2:.6g}" {common}/>'
    if c.shape == "cylinder":
        return f'<geom type="cylinder" size="{fx/2:.6g} {fz/2:.6g}" {common}/>'
    return f'<geom type="plane" size="{fx/2:.6g} {fy/2:.6g} .01" {common}/>'


def scene_asset_dir(lab_root: Path, asset_id: str, revision: int) -> Path:
    return Path(lab_root) / "assets" / "scenes" / asset_id / str(revision)


def environment_xml(config: ProjectConfig, lab_root: Path):
    sc = config.scene
    parts = [LIGHTS]
    assets = {}
    info = {"environment": None, "colliders": [c.name for c in sc.colliders if c.enabled],
            "default_table": False}
    if sc.environment:
        d = scene_asset_dir(lab_root, sc.environment.asset_id, sc.environment.revision)
        mf = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        visual = d / mf["visual_asset"]
        assets["scene_env.obj"] = visual.read_bytes()
        T = transform_matrix(sc.environment.position, sc.environment.quaternion_xyzw,
                             sc.environment.uniform_scale)
        pos, q, s = mat_to_pos_quat_xyzw(T)
        parts.append(f'<geom class="visual" type="mesh" mesh="scene_env" pos="{_fmt(pos)}" '
                     f'quat="{_fmt(quat_xyzw_to_wxyz(q))}" rgba=".62 .66 .70 1"/>')
        info["environment"] = {"asset_id": sc.environment.asset_id, "revision": sc.environment.revision,
                               "calibrated": mf.get("calibrated", False)}
        parts.append(FLOOR)
        if sc.use_default_table:
            parts.append(DEFAULT_TABLE)
            info["default_table"] = True
    else:
        parts.append(FLOOR)
        if sc.use_default_table:
            parts.append(DEFAULT_TABLE)
            info["default_table"] = True
    for c in sc.colliders:
        if c.enabled:
            parts.append(_collider_xml(c))
    return "".join(parts), assets, info