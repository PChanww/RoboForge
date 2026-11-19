"""Environment MJCF builder: default tabletop OR imported scene visual + static colliders.

Separated from task objects (tray/cup) so a generated background never replaces the
dynamic task object. Static colliders have no freejoint (no mass dynamics).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .schema import ProjectConfig
from .transforms import quat_xyzw_to_wxyz, transform_matrix, mat_to_pos_quat_xyzw
from .scene_assets import scene_asset_dir, load_scene, contained_file

LIGHTS = """
        <light pos="-.4 -.3 .85" dir=".25 .25 -1" diffuse=".69 .65 .59" castshadow="true"/>
        <light pos=".5 .4 .6" dir="-.6 -.3 -1" diffuse=".44 .48 .56" castshadow="false"/>"""
FLOOR = '<geom type="plane" size="2 2 .01" pos="0 0 -.061" rgba=".19 .23 .25 1"/>'
DEFAULT_TABLE = '<geom name="table" type="box" pos="0 0 -.028" size=".32 .215 .028" rgba=".29 .37 .39 1"/>'


def _fmt(v):
    return " ".join(f"{x:.6g}" for x in v)


def _collider_xml(c, *, visual_present=False):
    q = _fmt(quat_xyzw_to_wxyz(c.quaternion_xyzw))
    rgba = ' rgba=".5 .6 .65 0"' if visual_present else ''
    common = (f'name="{c.name}" pos="{_fmt(c.position)}" quat="{q}"{rgba} '
              f'friction="{_fmt(c.friction)}" contype="{c.contype}" conaffinity="{c.conaffinity}"')
    fx, fy, fz = c.full_size_m
    if c.shape == "box":
        return f'<geom type="box" size="{fx/2:.6g} {fy/2:.6g} {fz/2:.6g}" {common}/>'
    if c.shape == "cylinder":
        return f'<geom type="cylinder" size="{fx/2:.6g} {fz/2:.6g}" {common}/>'
    return f'<geom type="plane" size="{fx/2:.6g} {fy/2:.6g} .01" {common}/>'


def _prop_xml(prop, lab_root):
    imported_root = (Path(lab_root) / "assets" / "imported").resolve()
    directory = contained_file(imported_root, prop.asset.removeprefix("imported/"))
    manifest_path = contained_file(directory, "manifest.json")
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Scene prop {prop.name}: cannot read imported asset manifest") from exc
    if manifest.get("role") not in {"static", "visual_only"}:
        raise ValueError(f"Scene prop {prop.name}: only static or visual_only assets may be attached")
    if manifest.get("kind") not in {"static_prop", "robot_visual", "task_object"}:
        raise ValueError(f"Scene prop {prop.name}: unsupported imported asset kind")
    if prop.collision and not manifest.get("calibrated"):
        raise ValueError(f"Scene prop {prop.name}: collision needs a scale-calibrated imported asset")
    visual_path = contained_file(directory, manifest.get("obj", "visual.obj"))
    if visual_path.suffix.lower() != ".obj":
        raise ValueError(f"Scene prop {prop.name}: expected an imported OBJ mesh")
    try:
        original = visual_path.read_bytes()
        lines = original.decode("ascii").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Scene prop {prop.name}: cannot read imported OBJ mesh") from exc
    vertices, scaled_lines = [], []
    for line in lines:
        fields = line.split()
        if fields and fields[0] == "v":
            if len(fields) != 4:
                raise ValueError(f"Scene prop {prop.name}: mesh vertices must have three coordinates")
            vertex = np.asarray([float(v) for v in fields[1:]], dtype=float) * prop.uniform_scale
            if not np.isfinite(vertex).all():
                raise ValueError(f"Scene prop {prop.name}: mesh has non-finite scaled vertices")
            vertices.append(vertex)
            scaled_lines.append("v " + _fmt(vertex))
        else:
            scaled_lines.append(line)
    if len(vertices) < 4:
        raise ValueError(f"Scene prop {prop.name}: mesh has insufficient vertices")
    vertices = np.asarray(vertices)
    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    extent = upper - lower
    if not np.isfinite(extent).all() or np.any(extent <= 1e-8):
        raise ValueError(f"Scene prop {prop.name}: mesh bounds must have three positive finite dimensions")
    color = manifest.get("color_rgb")
    rgb = np.asarray([.62, .66, .70] if color is None else color, dtype=float)
    if rgb.shape != (3,) or not np.isfinite(rgb).all() or np.any(rgb < 0) or np.any(rgb > 1):
        raise ValueError(f"Scene prop {prop.name}: invalid asset color")
    prefix = f"scene_prop_{prop.name}"
    parts = [f'<body name="{prefix}" pos="{_fmt(prop.position)}" '
             f'quat="{_fmt(quat_xyzw_to_wxyz(prop.quaternion_xyzw))}">',
             f'<geom name="{prefix}_visual" class="visual" type="mesh" mesh="{prefix}" rgba="{_fmt(rgb)} 1"/>']
    collision = None
    if prop.collision:
        center = (lower + upper) / 2
        parts.append(f'<geom name="{prefix}_contact" type="box" size="{_fmt(extent/2)}" '
                     f'pos="{_fmt(center)}" rgba="0 0 0 0" group="3"/>')
        collision = {"shape": "box", "full_size_m": extent.tolist(), "position": center.tolist(),
                     "basis": "estimated axis-aligned bounds of the scaled imported mesh; not measured"}
    parts.append('</body>')
    info = {"name": prop.name, "asset": prop.asset, "source": manifest.get("source"),
            "generation_id": manifest.get("generation_id"), "spec_id": manifest.get("spec_id"),
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "mesh_sha256": hashlib.sha256(original).hexdigest(), "source_glb_sha256": manifest.get("sha256_glb"),
            "position": prop.position, "quaternion_xyzw": prop.quaternion_xyzw,
            "uniform_scale": prop.uniform_scale, "bounds_m": [lower.tolist(), upper.tolist()],
            "collision_proxy": collision, "scale_basis": manifest.get("scale_basis")}
    return "".join(parts), prefix + ".obj", ("\n".join(scaled_lines) + "\n").encode("ascii"), info


def environment_xml(config: ProjectConfig, lab_root: Path):
    sc = config.scene
    parts = [LIGHTS]
    assets = {}
    info = {"environment": None, "colliders": [c.name for c in sc.colliders if c.enabled],
            "default_table": False}
    if sc.environment:
        d, mf = load_scene(lab_root, sc.environment.asset_id, sc.environment.revision, physics=False)
        gaussian = mf.get("ply_kind") == "gaussian"
        if gaussian:
            if sc.sensor_renderer != "hybrid":
                raise ValueError("Gaussian scene requires sensor_renderer=hybrid (not a MJCF mesh)")
            for name, expected in mf.get("files", {}).items():
                f = contained_file(d, name)
                if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest() != expected:
                    raise ValueError(f"Scene asset missing or checksum mismatch: {name}")
            if not (sc.use_default_table or any(c.enabled for c in sc.colliders)):
                raise ValueError("Gaussian scene needs a default table or static colliders for physics")
            info["environment"] = {"asset_id": sc.environment.asset_id, "revision": sc.environment.revision,
                                   "calibrated": mf.get("calibrated", False), "renderer": "hybrid-gaussian"}
            parts.append(FLOOR)
            if sc.use_default_table:
                parts.append(DEFAULT_TABLE)
                info["default_table"] = True
        else:
            d, mf = load_scene(lab_root, sc.environment.asset_id, sc.environment.revision, physics=True)
            visual = contained_file(d, mf["visual_asset"])
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
            parts.append(_collider_xml(c, visual_present=sc.environment is not None))
    if sc.props:
        info["props"] = []
        for prop in sc.props:
            xml, filename, data, provenance = _prop_xml(prop, lab_root)
            parts.append(xml)
            assets[filename] = data
            info["props"].append(provenance)
    table_rgb = _fmt(sc.table_color_rgb)
    return "".join(parts).replace('rgba=".29 .37 .39 1"', f'rgba="{table_rgb} 1"'), assets, info
