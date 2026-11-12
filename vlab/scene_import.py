"""Scene-specific local import (GLB / PLY). Unlike cup import, no cup scaling/centering.

PLY is classified by header/properties (triangle mesh vs point cloud vs 3D Gaussian),
not by extension. Unknown units are marked uncalibrated (preview only, not a verified
physical scene). Source files are preserved; a processed OBJ is emitted for meshes.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import trimesh

from .scene_assets import scene_asset_dir

MAX_SCENE = 256 * 1024**2


def detect_ply(path: Path) -> str:
    with open(path, "rb") as f:
        head = f.read(4096).decode("utf-8", errors="replace")
    if not head.startswith("ply"):
        raise ValueError("Not a PLY file (missing magic).")
    props = head.lower()
    if "f_dc_0" in props or ("scale_0" in props and "rot_0" in props):
        return "gaussian"
    if "element face" in props:
        return "mesh"
    if "element vertex" in props:
        return "pointcloud"
    raise ValueError("Unrecognized PLY structure.")


def _write_obj(mesh: trimesh.Trimesh, path: Path):
    with path.open("w", encoding="ascii", newline="\n") as f:
        np.savetxt(f, mesh.vertices, fmt="v %.9g %.9g %.9g")
        np.savetxt(f, mesh.faces + 1, fmt="f %d %d %d")


def import_scene(src: Path, lab_root: Path, asset_id: str, revision: int = 1,
                 source_kind: str = "import", up_axis: str = "Y", unit_scale: float = 1.0,
                 calibrated: bool = False, display_name: str = "") -> Path:
    src = Path(src)
    if src.stat().st_size > MAX_SCENE:
        raise ValueError("Scene exceeds 256 MiB.")
    out = scene_asset_dir(lab_root, asset_id, revision)
    if up_axis not in {"Y", "Z"}:
        raise ValueError("up_axis must be Y or Z")
    if not np.isfinite(unit_scale) or unit_scale <= 0:
        raise ValueError("unit_scale must be positive and finite")
    if out.exists():
        raise ValueError(f"Scene revision already exists: {out}")
    suffix = src.suffix.lower()
    fmt, kind = "", ""
    if suffix == ".ply":
        kind = detect_ply(src)
        fmt = "ply"
        if kind == "mesh":
            mesh = trimesh.load(src, file_type="ply", process=False)
        else:
            mesh = None
    elif suffix == ".glb":
        from asset_pipeline.import_asset import parse_geometry
        mesh, _ = parse_geometry(src.read_bytes())
        fmt, kind = "glb", "mesh"
    else:
        raise ValueError("Scene import supports .glb or .ply.")
    if mesh is not None:
        if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
            raise ValueError("Scene must contain a finite triangle mesh")
        mesh = mesh.copy()
        mesh.apply_scale(unit_scale)
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".import-", dir=out.parent))
    try:
        return _finish_import(src, out, staging, mesh, asset_id, revision, display_name,
                              source_kind, fmt, kind, up_axis, unit_scale, calibrated)
    finally:
        if staging.exists() and staging.resolve().is_relative_to(out.parent.resolve()):
            shutil.rmtree(staging)


def _finish_import(src, target, out, mesh, asset_id, revision, display_name,
                   source_kind, fmt, kind, up_axis, unit_scale, calibrated):
    (out / "source").mkdir()
    (out / "processed").mkdir()
    shutil.copy(src, out / "source" / src.name)
    files = {f"source/{src.name}": hashlib.sha256(src.read_bytes()).hexdigest()}
    bbox = None
    if mesh is not None:
        if up_axis == "Y":
            mesh = mesh.copy()
            mesh.apply_transform([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        _write_obj(mesh, out / "processed" / "visual.obj")
        files["processed/visual.obj"] = hashlib.sha256((out / "processed" / "visual.obj").read_bytes()).hexdigest()
        bbox = mesh.bounds.tolist()
        status = "ready" if calibrated else "uncalibrated"
    else:
        status = "preview_only"  # gaussian/pointcloud: not a physics scene
    manifest = {
        "asset_id": asset_id, "revision": revision, "display_name": display_name or asset_id,
        "category": "scene", "provider": "local", "source_kind": source_kind,
        "created": datetime.now(timezone.utc).isoformat(), "files": files, "format": fmt,
        "ply_kind": kind, "up_axis": "Z" if mesh is not None else up_axis,
        "source_up_axis": up_axis,
        "unit_scale": unit_scale, "calibrated": calibrated,
        "normalized_units": mesh is not None,
        "visual_asset": "processed/visual.obj" if mesh is not None else None,
        "bbox": bbox, "status": status,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    out.rename(target)
    return target / "manifest.json"
