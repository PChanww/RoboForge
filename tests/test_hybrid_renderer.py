"""Hybrid renderer wired into engine observation.images (front only, clear errors)."""
import base64
import io
import sys
from pathlib import Path

import pytest
from PIL import Image

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
from vlab.engine import PlatformEnv, UnsupportedMode
from vlab.schema import ProjectConfig, SceneInstance


def cfg(renderer="hybrid", table=True, asset="world_586c79650960"):
    c = ProjectConfig.model_validate_json((LAB / "configs" / "default.json").read_text(encoding="utf-8"))
    c.scene.sensor_renderer = renderer
    c.scene.environment = SceneInstance(asset_id=asset, revision=1)
    c.scene.use_default_table = table
    return c


def test_hybrid_front_and_wrist_error():
    env = PlatformEnv(cfg(), LAB)
    try:
        obs = env.reset(0)
        st = obs["camera_status"]
        assert st["front"]["state"] == "ok", st["front"]
        assert obs["images"]["front"] is not None
        img = Image.open(io.BytesIO(base64.b64decode(obs["images"]["front"]["jpeg_b64"])))
        assert img.size == (256, 256)
        assert st["wrist"]["state"] == "render_unavailable"
        assert "front camera" in (st["wrist"]["render_error"] or "")
    finally:
        env.close()


def test_hybrid_requires_gaussian_and_physics():
    with pytest.raises(UnsupportedMode):
        PlatformEnv(cfg(asset="demo_workspace"), LAB)  # mesh scene under hybrid
    with pytest.raises(UnsupportedMode):
        PlatformEnv(cfg(table=False), LAB)  # no table and no colliders -> physics unsupported


def test_mujoco_mode_unaffected():
    c = ProjectConfig.model_validate_json((LAB / "configs" / "default.json").read_text(encoding="utf-8"))
    c.scene.sensor_renderer = "mujoco"  # no environment: legacy default scene
    env = PlatformEnv(c, LAB)
    try:
        obs = env.reset(0)
        assert obs["camera_status"]["front"]["state"] == "ok"
        assert obs["camera_status"]["wrist"]["state"] == "ok"
    finally:
        env.close()