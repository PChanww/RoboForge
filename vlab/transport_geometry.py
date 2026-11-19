"""Per-run object calibration and an open physical bin for mobile transport.

Imported source files are read-only. Parameter edits produce an in-memory OBJ;
the generated bin shell is visual only and never supplies a convex lid/contact.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from . import tasks
from .scene_assets import contained_file, valid_id

BIN_FLOOR = .009
WALL_THICKNESS = .006
OBJECT_LIMITS = {
    "height_m": (.02, .075), "width_m": (.015, .085),
    "depth_m": (.015, .085), "mass_kg": (.005, .15),
    "friction": (.2, 3.),
}


def _fmt(values):
    return " ".join(f"{float(v):.10g}" for v in values)


def _finite_vector(value, length, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f"{name}需要{length}个有限数值。")
    return result


def _package(lab_root, asset_ref):
    if not isinstance(asset_ref, str) or not asset_ref.startswith("imported/"):
        raise ValueError("请选择已导入的Aholo模型。")
    valid_id(asset_ref.removeprefix("imported/"))
    directory = contained_file((Path(lab_root) / "assets/imported").resolve(), asset_ref.removeprefix("imported/"))
    manifest_path = contained_file(directory, "manifest.json")
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw.decode("utf-8-sig"))
    if manifest.get("source") != "lux3d":
        raise ValueError("模型来源需要为Aholo/Lux3D。")
    visual_path = contained_file(directory, manifest.get("obj", "visual.obj"))
    if visual_path.suffix.lower() != ".obj":
        raise ValueError("模型需要为已导入的OBJ网格。")
    return manifest_path, manifest, visual_path, hashlib.sha256(raw).hexdigest()


def _mesh_data(mesh):
    try:
        lines = mesh.decode("ascii").splitlines()
    except UnicodeError as exc:
        raise ValueError("OBJ网格需要有效ASCII内容。") from exc
    vertices = []
    for line in lines:
        fields = line.split()
        if fields and fields[0] == "v":
            if len(fields) < 4:
                raise ValueError("OBJ顶点缺少三维坐标。")
            vertices.append([float(v) for v in fields[1:4]])
    points = np.asarray(vertices, dtype=float)
    if points.ndim != 2 or points.shape[0] < 4 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("OBJ网格需要至少4个有限三维顶点。")
    extent = np.ptp(points, axis=0)
    if np.any(extent <= 1e-8):
        raise ValueError("OBJ模型的三个方向尺寸都必须大于零。")
    return lines, points


def _transform_obj(lines, points, scale, origin, translation):
    changed = []
    vertex_index = 0
    for line in lines:
        fields = line.split()
        if fields and fields[0] == "v":
            transformed = (points[vertex_index]-origin)*scale+translation
            vertex_index += 1
            tail = " " + " ".join(fields[4:]) if len(fields) > 4 else ""
            changed.append("v " + _fmt(transformed) + tail)
        elif fields and fields[0] == "vn":
            normal = _finite_vector([float(v) for v in fields[1:]], 3, "OBJ法向") / scale
            norm = float(np.linalg.norm(normal))
            if norm <= 1e-12:
                raise ValueError("OBJ法向长度必须大于零。")
            changed.append("vn " + _fmt(normal/norm))
        else:
            changed.append(line)
    return ("\n".join(changed)+"\n").encode("ascii")


def _object_parameters(parameters):
    if parameters is None:
        return {}
    if not isinstance(parameters, dict):
        raise ValueError("物品参数需要为对象。")
    unknown = set(parameters)-set(OBJECT_LIMITS)
    if unknown:
        raise ValueError("不支持的物品参数：" + ", ".join(sorted(unknown)))
    result = {}
    for key, value in parameters.items():
        if value is None:
            continue
        low, high = OBJECT_LIMITS[key]
        if isinstance(value, bool):
            raise ValueError(f"{key}需要数值，不能为布尔值。")
        number = float(value)
        if not math.isfinite(number) or not low <= number <= high:
            raise ValueError(f"{key}需要在{low:g}～{high:g}之间。")
        result[key] = number
    return result


def prepare_object(lab_root, asset_ref, parameters=None):
    """Return r,h,mass,color,OBJ,offset and effective calibration metadata.

    Explicit width/depth/height map to X/Y/Z independently. Omitted values keep
    the imported mesh's actual dimensions, rather than its cylindrical proxy.
    An empty parameter dictionary preserves the legacy asset tuple exactly.
    """
    overrides = _object_parameters(parameters)
    manifest_path, manifest, visual_path, manifest_hash = _package(lab_root, asset_ref)
    if manifest.get("kind") != "task_object" or manifest.get("role") != "dynamic":
        raise ValueError("操作物品需要为Aholo动态单体模型。")
    mesh = visual_path.read_bytes()
    lines, points = _mesh_data(mesh)
    lower, upper = points.min(axis=0), points.max(axis=0)
    source_extent = upper-lower
    if manifest.get("radius_m") is None:
        # The general importer uses box proxies, whose manifests deliberately
        # contain radius_m=null. This task collides with the real mesh, so the
        # legacy cylinder helper radius is derived from that mesh in memory.
        r = float(max(source_extent[:2])/2)
        h, mass = float(manifest["height_m"]), float(manifest["mass_kg"])
        if not (0 < r < 1 and 0 < h < 1 and 0 < mass < 10):
            raise ValueError("资产尺寸和质量超出桌面仿真支持范围。")
        rgb = np.asarray(manifest.get("color_rgb", [.20, .60, .63]), dtype=float)
        if rgb.shape != (3,) or not np.isfinite(rgb).all() or np.any((rgb < 0) | (rgb > 1)):
            raise ValueError("资产颜色需要三个0～1之间的有限数值。")
        color = " ".join(f"{v:.6f}" for v in rgb)
        offset = -h/2
    else:
        r, h, mass, color, mesh, offset, _ = tasks.asset_params(str(manifest_path))
    source_mesh_hash = hashlib.sha256(mesh).hexdigest()
    friction = manifest.get("friction")
    # The original conveyor task uses a sliding coefficient of 1.2 when the
    # imported asset has no explicit estimate.
    if isinstance(friction, (list, tuple)):
        friction = friction[0] if friction else None
    friction = 1.2 if friction is None else float(friction)
    if not math.isfinite(friction) or not .2 <= friction <= 3:
        raise ValueError("资产摩擦估计超出支持范围0.2～3。")
    source_mass = mass
    dimension_keys = ("width_m", "depth_m", "height_m")
    final_extent = np.array([overrides.get(key, source_extent[i]) for i, key in enumerate(dimension_keys)])
    for i, key in enumerate(dimension_keys):
        low, high = OBJECT_LIMITS[key]
        if not low-1e-8 <= final_extent[i] <= high+1e-8:
            raise ValueError(f"物品实际{key}超出支持范围{low:g}～{high:g} m。")
    if any(key in overrides for key in dimension_keys):
        origin = np.r_[(lower[:2]+upper[:2])/2, lower[2]]
        mesh = _transform_obj(lines, points, final_extent/source_extent, origin, np.zeros(3))
        h = float(final_extent[2])
        r = float(max(final_extent[:2])/2)
        offset = -h/2
    mass = overrides.get("mass_kg", mass)
    if not math.isfinite(mass) or not .005 <= mass <= .15:
        raise ValueError("物品质量需要在0.005～0.15 kg之间。")
    friction = overrides.get("friction", friction)
    effective = {"width_m": float(final_extent[0]), "depth_m": float(final_extent[1]),
                 "height_m": float(final_extent[2]), "mass_kg": float(mass), "friction": float(friction)}
    provenance = {
        "asset": asset_ref, "source": "lux3d", "generation_id": manifest.get("generation_id"),
        "manifest_sha256": manifest_hash, "effective_mesh_sha256": hashlib.sha256(mesh).hexdigest(),
        "source_mesh_sha256": source_mesh_hash,
        "source_parameters": {"width_m": float(source_extent[0]), "depth_m": float(source_extent[1]),
                              "height_m": float(source_extent[2]), "mass_kg": float(source_mass)},
        "parameters": effective, "friction": float(friction),
        "parameter_sources": {key: "user_explicit" if key in overrides else "asset_or_task_estimate" for key in effective},
        "scaling": "independent explicit X/Y/Z dimensions; imported source files unchanged",
        "verification": "estimated; not real-world measured",
    }
    return r, h, mass, color, mesh, offset, provenance


def bin_geometry(goal_xy, size_m, color, lab_root=None, asset_ref=None):
    """Build a tray and four walls; size Z is wall height above the .009 m floor."""
    goal = _finite_vector(goal_xy, 2, "箱子位置")
    size = _finite_vector(size_m, 3, "箱子尺寸")
    if not (.16 <= size[0] <= .30 and .16 <= size[1] <= .30 and .04 <= size[2] <= .09):
        raise ValueError("箱子宽深支持0.16～0.30 m、墙高支持0.04～0.09 m。")
    if not isinstance(color, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
        raise ValueError("箱子颜色需要为#RRGGBB。")
    rgb = [int(color[i:i+2], 16)/255 for i in (1, 3, 5)]
    rgba = _fmt([*rgb, 1])
    hx, hy, height = size/2
    # The third element above is half the wall height, matching MuJoCo size.
    mid_z = BIN_FLOOR+height
    parts = [f'<geom name="tray" type="box" pos="{_fmt([*goal, BIN_FLOOR/2])}" size="{_fmt([hx,hy,BIN_FLOOR/2])}" rgba="{rgba}"/>']
    half_wall = WALL_THICKNESS/2
    walls = ((-hx+half_wall, 0, half_wall, hy), (hx-half_wall, 0, half_wall, hy),
             (0, -hy+half_wall, hx, half_wall), (0, hy-half_wall, hx, half_wall))
    for index, (dx,dy,sx,sy) in enumerate(walls):
        parts.append(f'<geom name="bin_wall_{index}" type="box" pos="{_fmt([goal[0]+dx,goal[1]+dy,mid_z])}" size="{_fmt([sx,sy,height])}" rgba="{rgba}" solref=".008 1"/>')
    assets = {}
    metadata = {"goal_xy": goal.tolist(), "size_m": size.tolist(), "color": color.lower(),
                "inner_half_m": (size[:2]/2-WALL_THICKNESS).tolist(), "floor_z_m": BIN_FLOOR,
                "rim_z_m": BIN_FLOOR+float(size[2]), "wall_thickness_m": WALL_THICKNESS,
                "collision": "open rectangular floor plus four wall boxes; no convex mesh lid",
                "visual_asset": None}
    if asset_ref is not None:
        if lab_root is None:
            raise ValueError("导入箱子外观需要lab_root。")
        _, manifest, visual_path, manifest_hash = _package(lab_root, asset_ref)
        if manifest.get("kind") != "static_prop" or manifest.get("role") not in {"static", "visual_only"}:
            raise ValueError("箱子外观需要Aholo静态道具模型。")
        original = visual_path.read_bytes()
        lines, points = _mesh_data(original)
        low, high = points.min(axis=0), points.max(axis=0)
        extent = high-low
        visual_size = np.r_[size[:2], BIN_FLOOR+size[2]]
        origin = np.r_[(low[:2]+high[:2])/2, low[2]]
        assets["bin_visual.obj"] = _transform_obj(lines, points, visual_size/extent, origin, np.zeros(3))
        parts.append(f'<body name="transport_bin_shell" pos="{_fmt([*goal,0])}"><geom name="transport_bin_visual" class="visual" type="mesh" mesh="bin_visual" contype="0" conaffinity="0" mass="0" rgba="{rgba}"/></body>')
        # Keep all contact proxies but make them transparent when the generated
        # shell is displayed. Their shape stays a hollow rectangular box.
        parts[:5] = [re.sub(r'rgba="[^"]+"', 'rgba="0 0 0 0" group="3"', part) for part in parts[:5]]
        metadata["visual_asset"] = {"id": asset_ref, "source": manifest.get("source"),
                                     "generation_id": manifest.get("generation_id"), "manifest_sha256": manifest_hash,
                                     "source_mesh_sha256": hashlib.sha256(original).hexdigest(),
                                     "effective_mesh_sha256": hashlib.sha256(assets["bin_visual.obj"]).hexdigest(),
                                     "collision": "visual_only; independent open rectangular contact proxy",
                                     "visual_size_m": visual_size.tolist(),
                                     "verification": "generated appearance; open top not inferred from convex mesh"}
    return "".join(parts), assets, metadata
