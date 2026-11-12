"""Visual-only control: changing only the rendered background must not change physics.

Runs the same seed/controller with (a) hybrid gaussian background and (b) plain MuJoCo
on the identical table/task; trajectories and success must match, images must differ.
"""
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
from vlab.engine import PlatformEnv  # noqa: E402
from vlab.schema import ProjectConfig, SceneInstance  # noqa: E402


def cfg(hybrid: bool):
    c = ProjectConfig.model_validate_json((LAB / "configs" / "default.json").read_text(encoding="utf-8"))
    c.scene.use_default_table = True
    if hybrid:
        c.scene.sensor_renderer = "hybrid"
        c.scene.environment = SceneInstance(asset_id="world_586c79650960", revision=1)
    else:
        c.scene.sensor_renderer = "mujoco"
        c.scene.environment = None
    c.sensors[0].resolution = [160, 160]
    return c


def rollout(hybrid: bool, ticks=120):
    env = PlatformEnv(cfg(hybrid), LAB)
    try:
        obs = env.reset(0)
        frames, traj, success = [], [], False
        for t in range(ticks):
            env.step(env.servo_expert_action())
            m = env.metrics()
            traj.append([m["object_xyz"], env.data.xpos[env._eef].tolist()])
            if t in (0, 60, ticks - 1):
                f = env.observation()["images"]["front"]
                frames.append(np.asarray(Image.open(io.BytesIO(base64.b64decode(f["jpeg_b64"]))).convert("RGB")))
            if m["success"]:
                success = True
                break
        return {"traj": np.asarray(traj, dtype=float), "success": success,
                "frames": frames, "render_mode": env.render_mode}
    finally:
        env.close()


def main():
    a = rollout(True)
    b = rollout(False)
    assert a["render_mode"] == "static_background_overlay" and b["render_mode"] == "mujoco"
    assert a["success"] == b["success"] is True
    n = min(len(a["traj"]), len(b["traj"]))
    diff = float(np.abs(a["traj"][:n] - b["traj"][:n]).max())
    assert n > 10 and diff < 1e-6, f"physics changed when only visuals changed: {diff}"
    img_diff = float(np.abs(a["frames"][0].astype(int) - b["frames"][0].astype(int)).mean())
    assert img_diff > 5, f"background change too small to be meaningful: {img_diff}"
    out = {"success_hybrid": a["success"], "success_mujoco": b["success"],
           "ticks_compared": n, "max_physics_diff": diff, "mean_image_diff": round(img_diff, 2)}
    ev = LAB / "outputs" / "phase2-render-validation" / "visual_only_control"
    ev.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a["frames"][0]).save(ev / "hybrid.png")
    Image.fromarray(b["frames"][0]).save(ev / "mujoco.png")
    (ev / "control.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(out)
    return out


def test_visual_only_control():
    assert main()


if __name__ == "__main__":
    main()
