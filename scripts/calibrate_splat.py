"""Estimate a scene-level calibration for the World gaussian room (floor -> z=0) and
render the MuJoCo front camera before/after to show the effect.

Asset units are unconfirmed; this only aligns the gaussian floor plane to the simulator
floor. Horizontal placement/scale remain unverified (reported).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
from vlab.gaussian import load_gaussians
from vlab.splat_render import render

OUT = LAB / "outputs" / "phase2-render-validation"
PLY = LAB / "assets" / "scenes" / "world_586c79650960" / "1" / "source" / "scene.ply"


def main():
    g = load_gaussians(PLY)
    cam = json.loads((OUT / "camera.json").read_text())
    W, H = cam["width"], cam["height"]
    fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
    z = g["means"][:, 2]
    floor_z = float(np.percentile(z, 1.0))
    inst = {"position": [0, 0, -floor_z], "quat_xyzw": [0, 0, 0, 1], "uniform_scale": 1.0}
    report = {"floor_z_estimate": round(floor_z, 4), "instance": inst,
              "note": "floor aligned to z=0; horizontal position and scale unverified (uncalibrated asset)"}

    t = time.time()
    img0, _, _ = render(g, cam["M"], fx, fy, cx, cy, W, H)
    Image.fromarray((img0 * 255).astype(np.uint8)).save(OUT / "front-uncalibrated.png")
    report["uncalibrated_std"] = round(float(img0.std()), 4)
    report["uncalibrated_ms"] = int((time.time() - t) * 1000)

    t = time.time()
    img1, _, _ = render(g, cam["M"], fx, fy, cx, cy, W, H, inst=inst)
    Image.fromarray((img1 * 255).astype(np.uint8)).save(OUT / "front-calibrated.png")
    report["calibrated_std"] = round(float(img1.std()), 4)
    report["calibrated_ms"] = int((time.time() - t) * 1000)

    report["front_view_improved"] = report["calibrated_std"] > report["uncalibrated_std"] * 1.05
    (OUT / "calibration.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()