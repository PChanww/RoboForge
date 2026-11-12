"""Render real 3DGS scene.ply to PNG with a specified camera (CPU splat rasterizer).

Produces: splat-front.png (MuJoCo front camera from camera.json), splat-auto.png
(auto-framed at the gaussian center), splat-with-box.png (opaque box occlusion test).
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
from vlab.splat_render import render, ray_box_hit

OUT = LAB / "outputs" / "phase2-render-validation"
PLY = LAB / "assets" / "scenes" / "world_586c79650960" / "1" / "source" / "scene.ply"


def save(img, name):
    Image.fromarray((img * 255).astype(np.uint8)).save(OUT / name)


def auto_matrix(bbox_min, bbox_max, dist=7.0, lift=1.5):
    c = (np.asarray(bbox_min) + np.asarray(bbox_max)) / 2
    pos = c + np.array([0, -dist, lift])
    f = (c - pos) / np.linalg.norm(c - pos)
    upw = np.array([0, 0, 1.0])
    right = np.cross(f, upw); right /= np.linalg.norm(right)
    up = np.cross(right, f)
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2] = right, up, -f
    M[:3, 3] = pos
    return M, c, pos


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    g = load_gaussians(PLY)
    load_ms = int((time.time() - t0) * 1000)
    bmin, bmax = g["means_bbox_min"], g["means_bbox_max"]
    cam = json.loads((OUT / "camera.json").read_text())
    W, H = cam["width"], cam["height"]
    fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
    report = {"load_ms": load_ms, "count": g["count"], "bbox_min": bmin, "bbox_max": bmax}

    # 1) MuJoCo front camera (uncalibrated asset -> may point at empty area)
    t = time.time()
    img, depth, st = render(g, cam["M"], fx, fy, cx, cy, W, H)
    report["front"] = {**st, "ms": int((time.time() - t) * 1000)}
    save(img, "splat-front.png")

    # 2) auto-framed camera at gaussian center (proves real splat rasterization)
    M, center, pos = auto_matrix(bmin, bmax)
    t = time.time()
    img2, depth2, st2 = render(g, M, fx, fy, cx, cy, W, H)
    report["auto"] = {**st2, "ms": int((time.time() - t) * 1000), "center": center.tolist()}
    save(img2, "splat-auto.png")

    # 3) opaque box occlusion via analytic ray/box depth vs nearest splat depth
    fwd = (center - pos) / np.linalg.norm(center - pos)
    right = M[:3, 0]
    box_center = pos + fwd * 3.5 + right * 0.9; box_half = np.array([0.3, 0.3, 0.3])
    t = time.time()
    box_d = ray_box_hit(M, box_center, box_half, W, H, fx, fy, cx, cy)
    img3 = img2.copy()
    hit = box_d < depth2
    img3[hit] = np.array([1.0, 0.33, 0.2]) * 0.9
    report["with_box"] = {"ms": int((time.time() - t) * 1000), "box_pixels": int(hit.sum()),
                          "box_center": box_center.tolist()}
    save(img3, "splat-with-box.png")

    (OUT / "render_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("load_ms", "count", "front", "auto", "with_box")}))


if __name__ == "__main__":
    main()