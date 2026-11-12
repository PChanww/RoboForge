"""T03: gaussian covariance orientation (scale + quaternion, including inst transform).

Renders an elongated gaussian and checks that the 2D footprint's principal axis follows
the quaternion rotation (90 deg about Z swaps the major axis), both via g["quat_xyzw"]
and via the asset instance transform.
"""
import json
import sys
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
from vlab.splat_render import render  # noqa: E402

W = H = 160
FX = FY = 140.0
CX = CY = 80.0
M = np.eye(4)


def one_gauss(quat=(0, 0, 0, 1)):
    return {"means": np.array([[0.0, 0.0, -2.0]]), "scales": np.array([[0.035, 0.22, 0.035]]),
            "quat_xyzw": np.array([quat], dtype=float), "opacity": np.array([1.0]),
            "rgb": np.array([[0.9, 0.1, 0.1]]), "count": 1}


def footprint(img, thresh=0.08):
    ys, xs = np.nonzero(img[:, :, 0] > thresh)
    pts = np.stack([xs - xs.mean(), ys - ys.mean()])
    cov = pts @ pts.T / max(len(xs), 1)
    vals, vecs = np.linalg.eigh(cov)
    major = vecs[:, np.argmax(vals)]
    angle = float(np.degrees(np.arctan2(major[1], major[0]))) % 180
    return len(xs), angle, float(np.sqrt(vals.max() / max(vals.min(), 1e-9)))


def main():
    q90 = [0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)]  # 90 deg about Z, xyzw
    base = footprint(render(one_gauss(), M, FX, FY, CX, CY, W, H)[0])
    rotated = footprint(render(one_gauss(q90), M, FX, FY, CX, CY, W, H)[0])
    inst = footprint(render(one_gauss(), M, FX, FY, CX, CY, W, H,
                            inst={"position": [0, 0, 0], "quat_xyzw": q90, "uniform_scale": 1.0})[0])
    scale2 = footprint(render({"means": np.array([[0.0, 0.0, -2.0]]),
                               "scales": np.array([[0.035, 0.44, 0.035]]),
                               "quat_xyzw": np.array([[0, 0, 0, 1.0]]), "opacity": np.array([1.0]),
                               "rgb": np.array([[0.9, 0.1, 0.1]]), "count": 1},
                              M, FX, FY, CX, CY, W, H)[0])

    def delta(a, b):
        d = abs(a - b) % 180
        return min(d, 180 - d)

    r = {"base": {"pixels": base[0], "angle_deg": round(base[1], 2), "elongation": round(base[2], 2)},
         "quat90": {"pixels": rotated[0], "angle_deg": round(rotated[1], 2), "elongation": round(rotated[2], 2)},
         "inst90": {"pixels": inst[0], "angle_deg": round(inst[1], 2), "elongation": round(inst[2], 2)},
         "scale2x": {"pixels": scale2[0], "angle_deg": round(scale2[1], 2), "elongation": round(scale2[2], 2)},
         "angle_shift_quat_deg": round(delta(base[1], rotated[1]), 2),
         "angle_shift_inst_deg": round(delta(base[1], inst[1]), 2),
         "pixels_ratio_scale2": round(scale2[0] / max(base[0], 1), 2)}
    ok = (base[2] > 2.0 and rotated[2] > 2.0
          and abs(r["angle_shift_quat_deg"] - 90) < 6
          and abs(r["angle_shift_inst_deg"] - 90) < 6
          and r["pixels_ratio_scale2"] > 1.6)
    out = LAB / "outputs" / "phase2-render-validation" / "gaussian_covariance.json"
    out.write_text(json.dumps({**r, "all_checks": ok}, indent=1), encoding="utf-8")
    print(r)
    print({"all_checks": ok, "report": str(out)})
    return ok


def test_covariance_follows_quaternion():
    assert main()


if __name__ == "__main__":
    main()
