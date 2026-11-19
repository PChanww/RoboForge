"""Generic single-asset import (cup / tray-bin / static prop / robot visual).

Unlike the cup-specific ``import_asset.register``, this keeps the asset kind and
role, builds a box OR cylinder proxy from the spec, and records normalization.
Unknown dimensions produce an uncalibrated visual-only asset: no physics binding
until the scale is confirmed. Source GLB and processed OBJ hashes are preserved.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from .import_asset import MAX_FILE, parse_geometry, read_glb

REGISTRY = Path(__file__).resolve().parents[1] / "assets" / "imported"


def _check_dims(height, width, depth, radius, mass):
    for name, v in (("height", height), ("width", width), ("depth", depth), ("radius", radius)):
        if v is not None and (not np.isfinite(v) or not 0 < v < 1):
            raise ValueError(f"{name} must be in (0,1) m or null")
    if mass is not None and (not np.isfinite(mass) or not 0 < mass < 10):
        raise ValueError("mass must be in (0,10) kg or null")


def register_object(path: Path, output: Path, *, kind: str = "task_object", role: str = "dynamic",
                    height: float | None = None, width: float | None = None, depth: float | None = None,
                    radius: float | None = None, mass: float | None = None,
                    friction: list[float] | None = None, source: str = "other",
                    task_id: str | None = None, prompt: str | None = None, member: str | None = None,
                    up: str = "y", source_detail: str = "user-supplied export",
                    spec_id: str | None = None, spec_revision: int | None = None,
                    spec_hash: str | None = None, compiler_version: str | None = None) -> Path:
    if output.exists():
        raise ValueError(f"Asset already exists: {output}. Use a new slug.")
    if kind not in {"task_object", "static_prop", "robot_visual"}:
        raise ValueError("register_object handles task_object/static_prop/robot_visual; scenes use scene_import")
    if role not in {"visual_only", "static", "dynamic"}:
        raise ValueError("role must be visual_only/static/dynamic")
    if kind == "robot_visual" and role == "dynamic":
        raise ValueError("robot_visual is appearance only; cannot be dynamic")
    _check_dims(height, width, depth, radius, mass)
    data = read_glb(path, member)
    mesh, color = parse_geometry(data)
    if up == "y":
        mesh.apply_transform([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    low, high = mesh.bounds
    if np.any(high - low < 1e-9):
        raise ValueError("Mesh has a flat or zero-volume bounding box.")
    original_extents = (high - low).tolist()
    calibrated = height is not None
    if calibrated:
        mesh.apply_scale(height / (high[2] - low[2]))
        low, high = mesh.bounds
    # Center under the foot so a projecting handle does not offset the proxy.
    bottom = mesh.vertices[mesh.vertices[:, 2] <= low[2] + 0.12 * (high[2] - low[2]), :2]
    center = (bottom.min(axis=0) + bottom.max(axis=0)) / 2
    mesh.apply_translation([-center[0], -center[1], -low[2]])
    extents = mesh.extents
    if radius is not None:
        proxy = {"shape": "cylinder", "size_m": [radius * 2, radius * 2, float(extents[2])],
                 "position": [0, 0, float(extents[2]) / 2], "basis": "user_explicit"}
        proxy_desc = "upright cylinder, body only; no handle/rim/interior contacts"
    else:
        proxy = {"shape": "box", "size_m": [float(v) for v in extents],
                 "position": [0, 0, float(extents[2]) / 2],
                 "basis": "user_explicit" if (width or depth) else "proxy_estimate"}
        proxy_desc = "axis-aligned box from mesh extents; no concave/interior contacts"
    effective_role = role if calibrated else "visual_only"
    physics_status = "uncalibrated" if not calibrated else ("proxy_estimate" if mass is not None else "needs_physics")
    manifest = {
        "obj": "visual.obj", "kind": kind, "role": effective_role,
        "height_m": height, "width_m": width, "depth_m": depth, "radius_m": radius,
        "mass_kg": mass, "friction": friction,
        "bounds_m": mesh.bounds.tolist(), "up_axis": "Z", "triangles": int(len(mesh.faces)),
        "collision_proxy": proxy, "collision_proxy_desc": proxy_desc,
        "calibrated": calibrated, "physics_status": physics_status,
        "scale_basis": "user-specified height" if calibrated else "original units; uncalibrated, not a physical measurement",
        "normalization": {"source_up_axis": up, "scaled_to_height_m": height,
                          "centered_footprint": True, "original_extents_m": original_extents},
        "source": source, "source_detail": source_detail, "generation_id": task_id, "prompt": prompt,
        "source_mesh": "original.glb", "sha256_glb": hashlib.sha256(data).hexdigest(),
        "spec_id": spec_id, "spec_revision": spec_revision, "spec_hash": spec_hash,
        "compiler_version": compiler_version,
    }
    if color is not None:
        manifest["color_rgb"] = color
        manifest["color_basis"] = "area-weighted mean of material or indexed vertex colors; uniform simulator tint"
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".building-", dir=output.parent))
    try:
        (temp / "original.glb").write_bytes(data)
        with (temp / "visual.obj").open("w", encoding="ascii", newline="\n") as f:
            np.savetxt(f, mesh.vertices, fmt="v %.9g %.9g %.9g")
            np.savetxt(f, mesh.faces + 1, fmt="f %d %d %d")
        (temp / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.rename(output)
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    return output / "manifest.json"