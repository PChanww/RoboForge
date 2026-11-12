"""P0-3: hybrid build with explicit colliders, cache invalidation and preview parity."""
import sys
from pathlib import Path

import numpy as np
import pytest

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
from vlab.engine import PlatformEnv, UnsupportedMode  # noqa: E402
from vlab.schema import ProjectConfig, SceneCollider, SceneInstance  # noqa: E402
from vlab.rendering import hybrid  # noqa: E402


def cfg(resolution=(128, 128), table=True, colliders=None):
    c = ProjectConfig.model_validate_json((LAB / "configs" / "default.json").read_text(encoding="utf-8"))
    c.scene.sensor_renderer = "hybrid"
    c.scene.environment = SceneInstance(asset_id="world_586c79650960", revision=1)
    c.scene.use_default_table = table
    c.scene.colliders = colliders or []
    c.sensors[0].resolution = list(resolution)
    return c


def desk():
    return SceneCollider(name="desk", shape="box", full_size_m=[0.8, 0.6, 0.05], position=[0, 0, -0.025])


def test_custom_collider_builds_and_rejects_missing_collision():
    env = PlatformEnv(cfg(table=False, colliders=[desk()]), LAB)  # was NameError: config
    try:
        env.reset(0)
        img = env.render("front", 128, 128)
        assert img.shape == (128, 128, 3)
        assert env.render_mode == hybrid.HYBRID_MODE
    finally:
        env.close()
    for bad in (cfg(table=False), cfg(table=False, colliders=[desk().model_copy(update={"enabled": False})])):
        with pytest.raises(UnsupportedMode):
            PlatformEnv(bad, LAB)


def test_cache_key_and_bounds():
    hybrid.clear_background_cache()
    cam = {"world_pos": [0, 1, 0], "world_mat": np.eye(3).ravel(), "fx": 300, "fy": 300, "cx": 64, "cy": 64}
    ply = LAB / "assets" / "scenes" / "world_586c79650960" / "1" / "source" / "scene.ply"
    k1 = hybrid._cache_key(ply, cam, None, 128, 128)
    cam2 = dict(cam, fx=301)
    assert hybrid._cache_key(ply, cam2, None, 128, 128) != k1
    assert hybrid._cache_key(ply, cam, None, 160, 120) != k1
    assert hybrid._cache_key(ply, cam, {"position": [0.01, 0, 0], "quat_xyzw": [0, 0, 0, 1],
                                        "uniform_scale": 1.0}, 128, 128) != k1
    other = LAB / "assets" / "scenes" / "world_586c79650960" / "1" / "source" / "scene.ply"
    assert hybrid._cache_key(other, cam, None, 128, 128) == k1  # same file -> same hash
    for i in range(6):
        hybrid._cache_put((i,), (np.zeros(2), np.zeros(2), {}))
    assert len(hybrid._BG_CACHE) <= hybrid._BG_CACHE_MAX
    hybrid.clear_background_cache()


def test_camera_and_resolution_invalidate_background():
    c = cfg(resolution=(96, 96))
    env = PlatformEnv(c, LAB)
    try:
        env.reset(0)
        env._render_camera(c.sensors[0])
        k1, bg1 = env._hybrid_key, env._hybrid._bg
        assert bg1.shape == (96, 96, 3) and not bg1.flags.writeable
        c.sensors[0].position = list(np.asarray(c.sensors[0].position) + [0.01, 0, 0])
        env.reset(0)
        img = env._render_camera(c.sensors[0])
        assert env._hybrid_key != k1 and env._hybrid._bg is not bg1
        assert img.shape == (96, 96, 3)

        k2 = env._hybrid_key
        c.sensors[0].intrinsics.fx *= 1.05
        env.reset(0)
        env._render_camera(c.sensors[0])
        k3 = env._hybrid_key
        assert k3 != k2
        c.sensors[0].resolution = [160, 120]
        env.reset(0)
        img2 = env._render_camera(c.sensors[0])
        assert img2.shape == (120, 160, 3) and env._hybrid._bg.shape == (120, 160, 3)
        assert env._hybrid_key != k3
    finally:
        env.close()
    hybrid.clear_background_cache()


def test_preview_uses_hybrid_backend():
    c = cfg(resolution=(96, 96))
    env = PlatformEnv(c, LAB)
    try:
        env.reset(0)
        preview = env.render("front", 96, 96)
        obs_frame = env._render_camera(c.sensors[0], 96, 96)
        assert np.array_equal(preview, obs_frame)
        mj = np.zeros((96, 96, 3), dtype=np.uint8)
        r = env._renderers[(96, 96)]
        r.update_scene(env.data, camera="front")
        mj = r.render()
        diff = float(np.mean(preview.astype(int) - mj.astype(int) != 0))
        assert diff > 0.05, f"preview silently dropped gaussian background (diff={diff})"
        assert env.render_mode == "static_background_overlay"
        with pytest.raises(UnsupportedMode):
            env.render("wrist", 96, 96)
    finally:
        env.close()
    hybrid.clear_background_cache()


def main():
    test_custom_collider_builds_and_rejects_missing_collision()
    test_cache_key_and_bounds()
    test_camera_and_resolution_invalidate_background()
    test_preview_uses_hybrid_backend()
    print({"hybrid_build": "ok", "cache_bounds": "ok", "preview_parity": "ok"})


if __name__ == "__main__":
    main()
