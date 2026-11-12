"""Short hybrid CLI collection: gaussian background + MuJoCo robot/task foreground.

Renders the world-fixed front-camera gaussian background once, then per servo step
composites the MuJoCo robot/cup/tray foreground (segmentation) over it, recording
images + state + action + timestamps to an npz. Decodes one stored frame back to PNG
as evidence that real gaussian RGB reached the raw data.
"""
import io
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
from vlab.engine import PlatformEnv
from vlab.schema import ProjectConfig
from vlab.rendering.snapshot import capture_snapshot
from vlab.rendering.hybrid import HybridRenderer, foreground_ids

PLY = LAB / "assets" / "scenes" / "world_586c79650960" / "1" / "source" / "scene.ply"
OUT = LAB / "datasets" / "collections" / "hybrid_check"
W = H = 256


def main():
    project = sys.argv[1] if len(sys.argv) > 1 else "default"
    ticks = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    cfg = ProjectConfig.model_validate_json(
        (LAB / "projects" / project / "current.json").read_text(encoding="utf-8"))
    env = PlatformEnv(cfg, LAB)
    try:
        env.reset(0)
        snap = capture_snapshot(env, env.episode_time())
        t0 = time.time()
        hybrid = HybridRenderer(PLY, snap, W, H)
        bg_ms = int((time.time() - t0) * 1000)
        fg = foreground_ids(env)
        r_rgb = mujoco.Renderer(env.model, H, W)
        r_seg = mujoco.Renderer(env.model, H, W)
        r_seg.enable_segmentation_rendering()
        OUT.mkdir(parents=True, exist_ok=True)
        frames, states, actions, ctimes = [], [], [], []
        for t in range(ticks):
            env.step(env.servo_expert_action())
            r_rgb.update_scene(env.data, camera="front")
            rgb = r_rgb.render()
            r_seg.update_scene(env.data, camera="front")
            seg = r_seg.render()
            comp, mask = hybrid.compose(rgb, seg, fg)
            b = io.BytesIO()
            Image.fromarray(comp).save(b, format="JPEG", quality=88)
            frames.append(np.frombuffer(b.getvalue(), dtype=np.uint8))
            states.append([env.observation()["state"]["joint_pos"][n] for n in env._joints])
            actions.append(env._last_action)
            ctimes.append(env.episode_time())
            if t == ticks - 1:
                Image.fromarray(comp).save(OUT / "hybrid-front.png")
                Image.fromarray(hybrid.background()).save(OUT / "hybrid-background.png")
        # persist as npz (ragged jpeg -> object array) and decode one back
        np.savez_compressed(OUT / "episode.npz",
                            images=np.array(frames, dtype=object),
                            states=np.asarray(states), actions=np.asarray(actions),
                            capture_times=np.asarray(ctimes), tune=np.asarray(["front"]))
        z = np.load(OUT / "episode.npz", allow_pickle=True)
        decoded = Image.open(io.BytesIO(bytes(z["images"][-1]))).convert("RGB")
        decoded.save(OUT / "decoded-hybrid.png")
        report = {"project": project, "ticks": ticks, "resolution": [W, H],
                  "gaussian_background_ms": bg_ms, "bg_stats": hybrid.stats,
                  "foreground_seg_ids": len(fg), "frames": int(len(frames)),
                  "npz_bytes": int((OUT / "episode.npz").stat().st_size),
                  "note": "uncalibrated asset: robot/table may not align with the room"}
        (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps(report))
    finally:
        env.close()


if __name__ == "__main__":
    main()