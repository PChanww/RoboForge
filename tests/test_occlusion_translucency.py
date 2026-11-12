"""T05: occlusion (front/behind/edge) and translucency for the CPU splat rasterizer.

Synthetic gaussians give exact ground truth: an opaque analytic box is composited by
per-pixel depth vs the splat depth map, and translucent splats must alpha-blend.
Writes images + a numeric report under outputs/phase2-render-validation/occlusion/.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
from vlab.splat_render import ray_box_hit, render  # noqa: E402

OUT = LAB / "outputs" / "phase2-render-validation" / "occlusion"
W = H = 128
FX = FY = 120.0
CX = CY = 64.0
M = np.eye(4)  # camera at origin, right +X, up +Y, looks down -Z
BOX_RGB = np.array([1.0, 0.55, 0.1])


def gaussians(entries):
    means = np.array([e["m"] for e in entries], dtype=float)
    scales = np.array([e["s"] for e in entries], dtype=float)
    opa = np.array([e.get("o", 1.0) for e in entries], dtype=float)
    rgb = np.array([e["c"] for e in entries], dtype=float)
    return {"means": means, "scales": scales, "quat_xyzw": np.tile([0, 0, 0, 1.0], (len(entries), 1)),
            "opacity": opa, "rgb": rgb, "count": len(entries)}


def save(img, name):
    OUT.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(OUT / name)


def composite(img, depth, center, half):
    bd = ray_box_hit(M, center, half, W, H, FX, FY, CX, CY)
    hit = bd < depth
    out = img.copy()
    out[hit] = BOX_RGB
    return out, bd, hit


def main():
    center = (H // 2, W // 2)
    wide = gaussians([{"m": [0, 0, -2.0], "s": [0.35, 0.35, 0.05], "c": [0.85, 0.1, 0.1]}])
    img, depth, _ = render(wide, M, FX, FY, CX, CY, W, H)
    r = {}

    # front: box at z=-1 -> strictly nearer than the gaussian (z=-2)
    front, bd, hit = composite(img, depth, [0, 0, -1.0], [0.2, 0.2, 0.05])
    r["front"] = {"box_pixels_in_front": int(hit.sum()), "center_is_box": bool(hit[center]),
                  "corner_keeps_splat": bool(not hit[4, 4] and not np.allclose(front[4, 4], BOX_RGB))}
    save(front, "box-front.png")

    # behind: box at z=-3, entirely behind the gaussian surface at z=-2
    back, bd, hit = composite(img, depth, [0, 0, -3.0], [0.1, 0.1, 0.05])
    r["behind"] = {"box_pixels_visible": int(hit.sum()), "center_keeps_splat": bool(not hit[center])}
    save(back, "box-behind.png")

    # edge: two gaussians at different depths; box at z=-2 splits them into
    # "box in front" (left, splat at -2.5) and "box behind" (right, splat at -1.6).
    pair = gaussians([{"m": [-0.45, 0, -2.5], "s": [0.15, 0.15, 0.05], "c": [0.1, 0.2, 0.9]},
                      {"m": [0.45, 0, -1.6], "s": [0.15, 0.15, 0.05], "c": [0.1, 0.8, 0.2]}])
    img2, depth2, _ = render(pair, M, FX, FY, CX, CY, W, H)
    edge, bd2, hit2 = composite(img2, depth2, [0, 0, -2.0], [0.5, 0.18, 0.03])
    left = hit2[:, :int(CX)]
    right = hit2[:, 80:95]
    r["edge"] = {"box_pixels_left_in_front": int(left.sum()), "box_pixels_right_kept_splat": int(right.sum()),
                 "left_is_box": bool(hit2[center[0], 42]), "right_keeps_splat": bool(not hit2[center[0], 86])}
    save(edge, "box-edge.png")

    # translucency: front red at 50% opacity over back blue at 100%
    trans = gaussians([{"m": [0, 0, -2.2], "s": [0.30, 0.30, 0.05], "c": [0.0, 0.0, 1.0]},
                       {"m": [0, 0, -1.8], "s": [0.30, 0.30, 0.05], "c": [1.0, 0.0, 0.0], "o": 0.5}])
    t50, _, _ = render(trans, M, FX, FY, CX, CY, W, H)
    expected = 0.5 * np.array([1.0, 0.0, 0.0]) + 0.5 * np.array([0.0, 0.0, 1.0])
    patch = t50[center[0] - 1:center[0] + 2, center[1] - 1:center[1] + 2].reshape(-1, 3).mean(0)
    trans["opacity"][1] = 0.95
    t95, _, _ = render(trans, M, FX, FY, CX, CY, W, H)
    patch95 = t95[center]
    trans["opacity"][1] = 0.01  # below opacity_min -> culled
    t0, _, _ = render(trans, M, FX, FY, CX, CY, W, H)
    patch0 = t0[center]
    r["translucency"] = {"blend_mean": [round(float(v), 4) for v in patch],
                         "expected_50_50": [round(float(v), 4) for v in expected],
                         "blend_err": round(float(np.abs(patch - expected).max()), 4),
                         "opaque95_red": round(float(patch95[0]), 4),
                         "culled01_blue": round(float(patch0[2]), 4)}
    save(t50, "translucent-50.png")
    save(t95, "translucent-95.png")

    ok = (r["front"]["center_is_box"] and r["front"]["corner_keeps_splat"]
          and r["front"]["box_pixels_in_front"] > 100
          and r["behind"]["box_pixels_visible"] == 0 and r["behind"]["center_keeps_splat"]
          and r["edge"]["left_is_box"] and r["edge"]["right_keeps_splat"]
          and r["edge"]["box_pixels_left_in_front"] > 50 and r["edge"]["box_pixels_right_kept_splat"] == 0
          and r["translucency"]["blend_err"] < 0.06
          and r["translucency"]["opaque95_red"] > 0.85 and r["translucency"]["culled01_blue"] > 0.9)
    report = {"resolution": [W, H], "intrinsics": [FX, FY, CX, CY], "results": r,
              "all_checks": ok,
              "limitation": "hybrid compose overlays MuJoCo foreground without splat-depth occlusion"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "occlusion_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(r))
    print({"all_checks": ok, "report": str(OUT / "occlusion_report.json")})
    return ok


def test_occlusion_and_translucency():
    assert main()


if __name__ == "__main__":
    main()
