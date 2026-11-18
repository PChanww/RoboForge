"""Validated identifiers and immutable scene-resource lookup."""
import hashlib
import json
import re
from pathlib import Path


def valid_id(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_-]{0,127}", value):
        raise ValueError("ID must contain 1-128 letters, digits, underscores or hyphens")
    return value


def contained_file(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Asset file must be a relative path")
    root = root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Asset file escapes its asset directory")
    return path


def scene_asset_dir(lab_root: Path, asset_id: str, revision: int) -> Path:
    valid_id(asset_id)
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        raise ValueError("Scene revision must be a positive integer")
    root = (Path(lab_root) / "assets" / "scenes").resolve()
    path = (root / asset_id / str(revision)).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Scene directory escapes its asset root")
    return path


def load_scene(lab_root: Path, asset_id: str, revision: int, *, physics=False):
    directory = scene_asset_dir(lab_root, asset_id, revision)
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Scene {asset_id} revision {revision} does not exist") from exc
    if manifest.get("asset_id") != asset_id or manifest.get("revision") != revision:
        raise ValueError("Scene manifest identity differs from its directory")
    if physics:
        if not manifest.get("visual_asset") or manifest.get("status") == "preview_only":
            raise ValueError("This scene is preview-only; Gaussian/point-cloud sensor rendering is not supported yet")
        if not manifest.get("calibrated"):
            raise ValueError("Scene units must be calibrated before use in physics")
        if manifest.get("unit_scale", 1) != 1 and not manifest.get("normalized_units"):
            raise ValueError("Legacy scene units were not normalized; reimport this scene")
        visual = contained_file(directory, manifest["visual_asset"])
        if not visual.is_file():
            raise ValueError("Scene visual asset is missing")
        for name, expected in manifest.get("files", {}).items():
            file = contained_file(directory, name)
            if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Scene asset missing or checksum mismatch: {name}")
    return directory, manifest
