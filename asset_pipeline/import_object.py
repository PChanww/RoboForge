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

from .collision_decomp import decompose_mesh_to_boxes
from .import_asset import MAX_FILE, parse_geometry, read_glb

REGISTRY = Path(__file__).resolve().parents[1] / "assets" / "imported"


def _maybe_level_flat_mesh(mesh) -> None:
    """Level shallow/planar meshes tilted in 3D generator output (e.g. 20-45 deg presentation pitch)."""
    if len(mesh.vertices) < 12:
        return
    pts = np.asarray(mesh.vertices, dtype=np.float64)
    cov = np.cov((pts - pts.mean(axis=0)).T)
    w_eig, u_eig = np.linalg.eigh(cov)
    az = u_eig[:, 0].copy()
    if w_eig[1] > 1e-12 and w_eig[0] < 0.15 * w_eig[1] and 0.40 < abs(az[2]) < 0.998:
        if az[2] < 0:
            az = -az
        ax = u_eig[:, 2].copy()
        ay = np.cross(az, ax)
        ay /= max(float(np.linalg.norm(ay)), 1e-12)
        ax = np.cross(ay, az)
        ax /= max(float(np.linalg.norm(ax)), 1e-12)
        rot_pca = np.eye(4, dtype=np.float64)
        rot_pca[:3, :3] = np.column_stack([ax, ay, az]).T
        mesh.apply_transform(rot_pca)


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
                    spec_hash: str | None = None, compiler_version: str | None = None,
                    job_id: str | None = None, scale_mode: str = "auto",
                    collision_mode: str = "auto") -> Path:
    if output.exists():
        raise ValueError(f"Asset already exists: {output}. Use a new slug.")
    if kind not in {"task_object", "static_prop", "robot_visual"}:
        raise ValueError("register_object handles task_object/static_prop/robot_visual; scenes use scene_import")
    if role not in {"visual_only", "static", "dynamic"}:
        raise ValueError("role must be visual_only/static/dynamic")
    if kind == "robot_visual" and role == "dynamic":
        raise ValueError("robot_visual is appearance only; cannot be dynamic")
    if scale_mode not in {"auto", "uniform", "bbox_anisotropic"}:
        raise ValueError("scale_mode must be auto, uniform, or bbox_anisotropic")
    _check_dims(height, width, depth, radius, mass)
    data = read_glb(path, member)
    mesh, color = parse_geometry(data)
    if up == "y":
        mesh.apply_transform([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    low, high = mesh.bounds
    if np.any(high - low < 1e-9):
        raise ValueError("Mesh has a flat or zero-volume bounding box.")
    is_shallow_flat = (height is not None and width is not None and depth is not None and height <= 0.25 * min(width, depth))
    if collision_mode == "compound" or is_shallow_flat:
        _maybe_level_flat_mesh(mesh)
        low, high = mesh.bounds
    orig_ext = high - low
    # Align dominant horizontal axis with requested (width, depth) aspect before scaling
    if width is not None and depth is not None and (width - depth) * (orig_ext[0] - orig_ext[1]) < -1e-4:
        mesh.apply_transform([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        low, high = mesh.bounds
        orig_ext = high - low
    original_extents = orig_ext.tolist()

    linear_count = sum(v is not None for v in (height, width, depth))
    if height is None and width is None and depth is None and radius is None:
        effective_scale_mode = "none"
        scale_factors = [1.0, 1.0, 1.0]
    elif scale_mode == "bbox_anisotropic" or (scale_mode == "auto" and radius is None and linear_count >= 2):
        effective_scale_mode = "bbox_anisotropic"
        fallback = (
            (height / orig_ext[2]) if height is not None
            else (width / orig_ext[0]) if width is not None
            else (depth / orig_ext[1]) if depth is not None
            else (2.0 * radius / min(orig_ext[0], orig_ext[1]))
        )
        sx = (width / orig_ext[0]) if width is not None else fallback
        sy = (depth / orig_ext[1]) if depth is not None else fallback
        sz = (height / orig_ext[2]) if height is not None else fallback
        if radius is not None and width is None and depth is None:
            s_xy = (2.0 * radius) / min(orig_ext[0], orig_ext[1])
            sx = sy = s_xy
        scale_factors = [float(sx), float(sy), float(sz)]
        mesh.apply_transform(np.diag([sx, sy, sz, 1.0]))
    else:
        effective_scale_mode = "uniform"
        if height is not None:
            s = float(height / orig_ext[2])
        elif width is not None:
            s = float(width / orig_ext[0])
        elif depth is not None:
            s = float(depth / orig_ext[1])
        else:
            s = float((2.0 * radius) / min(orig_ext[0], orig_ext[1]))
        scale_factors = [s, s, s]
        mesh.apply_scale(s)

    low, high = mesh.bounds
    # Center under the foot so a projecting handle does not offset the proxy.
    bottom = mesh.vertices[mesh.vertices[:, 2] <= low[2] + 0.12 * (high[2] - low[2]), :2]
    center = (bottom.min(axis=0) + bottom.max(axis=0)) / 2
    mesh.apply_translation([-center[0], -center[1], -low[2]])
    extents = mesh.extents

    unsatisfied: dict[str, dict[str, float]] = {}
    for key, req, actual in (
        ("width_m", width, float(extents[0])),
        ("depth_m", depth, float(extents[1])),
        ("height_m", height, float(extents[2])),
    ):
        if req is not None:
            tol = max(0.002, 0.05 * req)
            if abs(actual - req) > tol:
                unsatisfied[key] = {"requested_m": float(req), "actual_m": actual}
    if radius is not None and width is None and depth is None:
        # Compare body diameter (shorter horizontal footprint so a cup handle can project on X)
        body_diam = float(min(extents[0], extents[1]))
        req_diam = float(2.0 * radius)
        if abs(body_diam - req_diam) > max(0.015, 0.35 * req_diam):
            unsatisfied["radius_m"] = {"requested_m": float(radius), "actual_body_radius_m": body_diam / 2.0}

    calibrated = height is not None and not unsatisfied

    if radius is not None:
        proxy_basis = "user_explicit" if not unsatisfied else "proxy_estimate"
        proxy = {"shape": "cylinder", "size_m": [radius * 2, radius * 2, float(extents[2])],
                 "position": [0, 0, float(extents[2]) / 2], "basis": proxy_basis}
        proxy_desc = "upright cylinder, body only; no handle/rim/interior contacts"
    else:
        all_box_axes_explicit = (width is not None and depth is not None and height is not None and not unsatisfied)
        proxy_basis = "user_explicit" if all_box_axes_explicit else "proxy_estimate"
        proxy = {"shape": "box", "size_m": [float(v) for v in extents],
                 "position": [0, 0, float(extents[2]) / 2],
                 "basis": proxy_basis}
        proxy_desc = "axis-aligned box from mesh extents; no concave/interior contacts"

    decomp = decompose_mesh_to_boxes(
        mesh, resolution=16, max_boxes=20, force_compound=(collision_mode == "compound")
    )
    compound_proxy = decomp if (collision_mode != "primitive" and decomp["box_count"] >= 2) else None
    if compound_proxy is not None:
        proxy_desc = f"concave compound ({decomp['box_count']} boxes, fill_ratio={decomp['fill_ratio']}); preserves hollow/rim contacts"

    effective_role = role if calibrated else "visual_only"
    physics_status = "uncalibrated" if not calibrated else ("proxy_estimate" if mass is not None else "needs_physics")
    if not calibrated:
        if unsatisfied:
            scale_basis = f"{effective_scale_mode} scaling; requested dimension constraints unsatisfied"
        elif effective_scale_mode != "none":
            scale_basis = f"{effective_scale_mode} scaling; height unknown, uncalibrated"
        else:
            scale_basis = "original units; uncalibrated, not a physical measurement"
    elif effective_scale_mode == "bbox_anisotropic":
        scale_basis = "anisotropic bbox scaling to user-specified dimensions"
    else:
        scale_basis = "uniform scaling by user-specified height"

    manifest = {
        "obj": "visual.obj", "kind": kind, "role": effective_role,
        "height_m": height, "width_m": width, "depth_m": depth, "radius_m": radius,
        "mass_kg": mass, "friction": friction,
        "bounds_m": mesh.bounds.tolist(), "actual_extents_m": [float(v) for v in extents],
        "up_axis": "Z", "triangles": int(len(mesh.faces)),
        "collision_proxy": proxy, "collision_proxy_desc": proxy_desc,
        "compound_proxy": compound_proxy, "concave_detected": bool(decomp["concave_detected"]),
        "collision_mode": collision_mode,
        "calibrated": calibrated, "physics_status": physics_status,
        "scale_mode": effective_scale_mode, "scale_factors": scale_factors,
        "unsatisfied_dimensions": unsatisfied, "scale_basis": scale_basis,
        "normalization": {"source_up_axis": up, "scaled_to_height_m": height,
                          "scale_mode": effective_scale_mode, "scale_factors": scale_factors,
                          "centered_footprint": True, "original_extents_m": original_extents},
        "source": source, "source_detail": source_detail, "generation_id": task_id, "prompt": prompt,
        "source_mesh": "original.glb", "sha256_glb": hashlib.sha256(data).hexdigest(),
        "spec_id": spec_id, "spec_revision": spec_revision, "spec_hash": spec_hash,
        "compiler_version": compiler_version, "job_id": job_id,
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


def register_environment_mesh(
    path: Path,
    lab_root: Path,
    asset_id: str,
    *,
    revision: int = 1,
    height: float | None = None,
    width: float | None = None,
    depth: float | None = None,
    up: str = "y",
    source: str = "lux3d",
    task_id: str | None = None,
    prompt: str | None = None,
    member: str | None = None,
    spec_id: str | None = None,
    spec_revision: int | None = None,
    spec_hash: str | None = None,
    job_id: str | None = None,
) -> Path:
    """Register an environment triangle mesh under assets/scenes/<asset_id>/<revision>/
    with Z=0 aligned floor and compound box colliders so environment_mesh works online."""
    from vlab.scene_assets import scene_asset_dir
    for name, v in (("height", height), ("width", width), ("depth", depth)):
        if v is not None and (not np.isfinite(v) or not 0 < v <= 10.0):
            raise ValueError(f"environment {name} must be in (0, 10] m or null")
    data = read_glb(path, member)
    mesh, color = parse_geometry(data)
    if up == "y":
        mesh.apply_transform([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    low, high = mesh.bounds
    if np.any(high - low < 1e-9):
        raise ValueError("Environment mesh has zero-volume bounds")
    if height is not None and width is not None and depth is not None and height <= 0.25 * min(width, depth):
        _maybe_level_flat_mesh(mesh)
        low, high = mesh.bounds
    orig_ext = high - low
    if width is not None and depth is not None and (width - depth) * (orig_ext[0] - orig_ext[1]) < -1e-4:
        mesh.apply_transform([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        low, high = mesh.bounds
        orig_ext = high - low
    linear = sum(v is not None for v in (height, width, depth))
    if linear >= 2:
        fallback = (height / orig_ext[2]) if height is not None else ((width / orig_ext[0]) if width is not None else (depth / orig_ext[1]))
        sx = (width / orig_ext[0]) if width is not None else fallback
        sy = (depth / orig_ext[1]) if depth is not None else fallback
        sz = (height / orig_ext[2]) if height is not None else fallback
        mesh.apply_transform(np.diag([sx, sy, sz, 1.0]))
    elif height is not None:
        mesh.apply_scale(float(height / orig_ext[2]))
    elif width is not None:
        mesh.apply_scale(float(width / orig_ext[0]))
    elif depth is not None:
        mesh.apply_scale(float(depth / orig_ext[1]))

    low, high = mesh.bounds
    center_xy = 0.5 * (low[:2] + high[:2])
    mesh.apply_translation([-center_xy[0], -center_xy[1], -low[2]])
    calibrated = (height is not None) or (width is not None) or (depth is not None)

    decomp = decompose_mesh_to_boxes(mesh, resolution=18, max_boxes=24, force_compound=True)
    colliders = [
        {
            "name": f"{asset_id}_col_{i}",
            "shape": "box",
            "full_size_m": [round(float(v), 6) for v in b["size_m"]],
            "position": [round(float(v), 6) for v in b["position"]],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            "tag": "environment_mesh_decomp",
        }
        for i, b in enumerate(decomp["boxes"])
    ]

    out_dir = scene_asset_dir(lab_root, asset_id, revision)
    if out_dir.exists():
        raise ValueError(f"Scene revision already exists: {out_dir}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".scene-building-", dir=out_dir.parent))
    try:
        (temp / "original.glb").write_bytes(data)
        with (temp / "visual.obj").open("w", encoding="ascii", newline="\n") as f:
            np.savetxt(f, mesh.vertices, fmt="v %.9g %.9g %.9g")
            np.savetxt(f, mesh.faces + 1, fmt="f %d %d %d")
        obj_sha = hashlib.sha256((temp / "visual.obj").read_bytes()).hexdigest()
        manifest = {
            "asset_id": asset_id,
            "revision": revision,
            "visual_asset": "visual.obj",
            "source_mesh": "original.glb",
            "sha256_glb": hashlib.sha256(data).hexdigest(),
            "ply_kind": "mesh",
            "up_axis": "Z",
            "unit_scale": 1.0,
            "normalized_units": True,
            "calibrated": calibrated,
            "physics_status": "compound_boxes" if calibrated else "uncalibrated",
            "bounds_m": mesh.bounds.tolist(),
            "actual_extents_m": [float(v) for v in mesh.extents],
            "triangles": int(len(mesh.faces)),
            "colliders": colliders,
            "compound_proxy": decomp,
            "files": {"visual.obj": obj_sha},
            "source": source,
            "generation_id": task_id,
            "prompt": prompt,
            "spec_id": spec_id,
            "spec_revision": spec_revision,
            "spec_hash": spec_hash,
            "job_id": job_id,
        }
        if color is not None:
            manifest["color_rgb"] = color
        (temp / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.rename(out_dir)
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    return out_dir / "manifest.json"