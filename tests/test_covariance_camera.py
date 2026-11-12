"""P0-2: world->camera covariance, depth units and depth-mask coverage.

An elongated gaussian along world X must rotate with camera roll. An independent
sample-based projection (200k points) is compared against the analytic 2x2 covariance
for roll/pitch/yaw, fx!=fy and off-center principal point.
"""
import sys
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
from vlab.splat_render import ray_box_hit, render  # noqa: E402

W = H = 256
FX, FY = 220.0, 200.0
CX, CY = 128.0, 128.0


def rot(axis, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def quat_to_mat(q):
    x, y, z, w = q
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def gauss(scale, quat=(0, 0, 0, 1), mu=(0, 0, -3.0)):
    return {"means": np.array([mu], float), "scales": np.array([scale], float),
            "quat_xyzw": np.array([quat], float), "opacity": np.array([1.0]),
            "rgb": np.array([[0.9, 0.1, 0.1]]), "count": 1}


def footprint(img):
    ys, xs = np.nonzero(img[:, :, 0] > 0.08)
    if len(xs) < 5:
        return None
    pts = np.stack([xs - xs.mean(), ys - ys.mean()])
    cov = pts @ pts.T / len(xs)
    vals, vecs = np.linalg.eigh(cov)
    major = vecs[:, np.argmax(vals)]
    return {"n": len(xs), "w": xs.max() - xs.min() + 1, "h": ys.max() - ys.min() + 1,
            "angle": float(np.degrees(np.arctan2(major[1], major[0]))) % 180,
            "ratio": float(np.sqrt(vals.max() / max(vals.min(), 1e-9)))}


def empirical_cov(scale, quat, Rc, pos, mu=(0, 0, -3.0), n=200000):
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(n, 3)) * np.asarray(scale) + np.asarray(mu)
    if quat != (0, 0, 0, 1):
        pts = pts @ quat_to_mat(np.asarray(quat)).T
    cam = (pts - pos) @ Rc
    zp = -cam[:, 2]
    u = CX + FX * cam[:, 0] / zp
    v = CY - FY * cam[:, 1] / zp
    d = np.stack([u - u.mean(), v - v.mean()])
    vals, vecs = np.linalg.eigh(d @ d.T / n)
    major = vecs[:, np.argmax(vals)]
    return {"angle": float(np.degrees(np.arctan2(major[1], major[0]))) % 180,
            "ratio": float(np.sqrt(vals.max() / max(vals.min(), 1e-9)))}


def analytic_cov(scale, quat, Rc, pos):
    Rig = quat_to_mat(np.asarray(quat))
    Sig = Rig @ np.diag(np.asarray(scale) ** 2) @ Rig.T
    Sig_c = Rc.T @ Sig @ Rc
    x = Rc.T @ (np.array([0.0, 0.0, -3.0]) - pos)
    z = -x[2]
    J = np.array([[FX / z, 0, FX * x[0] / z ** 2], [0, -FY / z, -FY * x[1] / z ** 2]])
    return J @ Sig_c @ J.T


def angle_diff(a, b):
    d = abs(a - b) % 180
    return min(d, 180 - d)


def main():
    scale = [0.3, 0.04, 0.04]
    cases = {
        "roll0": np.eye(3),
        "roll90": rot("z", 90),
        "pitch12": rot("x", 12),
        "yaw12": rot("y", 12),
    }
    out = {}
    for name, Rc in cases.items():
        M = np.eye(4)
        M[:3, :3] = Rc
        img, _, _ = render(gauss(scale), M, FX, FY, CX, CY, W, H)
        fp = footprint(img)
        small = [0.08, 0.012, 0.012]  # keep perspective nonlinearity negligible for the independent check
        emp = empirical_cov(small, (0, 0, 0, 1), Rc, np.zeros(3))
        ana = analytic_cov(small, (0, 0, 0, 1), Rc, np.zeros(3))
        vals, vecs = np.linalg.eigh(ana)
        major = vecs[:, np.argmax(vals)]
        ana_angle = float(np.degrees(np.arctan2(major[1], major[0]))) % 180
        ana_ratio = float(np.sqrt(vals.max() / vals.min()))
        out[name] = {"footprint": {k: round(float(v), 2) for k, v in fp.items()},
                     "empirical": {"angle": round(emp["angle"], 2), "ratio": round(emp["ratio"], 2)},
                     "analytic": {"angle": round(ana_angle, 2), "ratio": round(ana_ratio, 2)},
                     "render_vs_analytic_angle": round(angle_diff(fp["angle"], ana_angle), 2),
                     "empirical_vs_analytic_angle": round(angle_diff(emp["angle"], ana_angle), 2)}
        assert out[name]["render_vs_analytic_angle"] < 8, out[name]
        assert out[name]["empirical_vs_analytic_angle"] < 4, out[name]
        assert abs(fp["ratio"] / ana_ratio - 1) < 0.25, out[name]

    # roll must swap the major axis of a world-X elongated gaussian
    d_roll = angle_diff(out["roll0"]["footprint"]["angle"], out["roll90"]["footprint"]["angle"])
    assert abs(d_roll - 90) < 8, d_roll
    # old implementation produced identical footprints (48x6) for both rolls
    assert (out["roll0"]["footprint"]["w"], out["roll0"]["footprint"]["h"]) != \
           (out["roll90"]["footprint"]["w"], out["roll90"]["footprint"]["h"])

    # instance rotation and scale go through the same correct path
    img90, _, _ = render(gauss(scale), np.eye(4), FX, FY, CX, CY, W, H,
                         inst={"position": [0, 0, 0], "quat_xyzw": [0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)],
                               "uniform_scale": 1.0})
    fp90 = footprint(img90)
    assert abs(angle_diff(fp90["angle"], out["roll0"]["footprint"]["angle"]) - 90) < 8
    img2, _, _ = render(gauss([2 * s for s in scale]), np.eye(4), FX, FY, CX, CY, W, H)
    fp2 = footprint(img2)
    assert fp2["ratio"] > 3.5 and fp2["n"] / out["roll0"]["footprint"]["n"] > 2.5

    # off-center principal point: an on-axis gaussian stays centered at (cx, cy)
    imgc, _, _ = render(gauss([0.06, 0.06, 0.06], mu=(0, 0, -2.0)), np.eye(4), FX, FY, CX, CY, W, H)
    ys, xs = np.nonzero(imgc[:, :, 0] > 0.3)
    out["offcenter"] = {"centroid": [round(float(xs.mean()), 2), round(float(ys.mean()), 2)], "target": [CX, CY]}
    assert abs(xs.mean() - CX) < 2 and abs(ys.mean() - CY) < 2

    # depth units: ray_box_hit returns camera-Z, not euclidean distance
    box_c, box_h = np.array([0.0, 0.0, -2.0]), np.array([0.5, 0.5, 0.05])
    bd = ray_box_hit(np.eye(4), box_c, box_h, W, H, FX, FY, CX, CY)
    off = bd[H // 2, int(CX) + 40]
    out["box_depth"] = {"offaxis_pixel_depth": round(float(off), 6), "near_face_z": 1.95,
                        "euclidean_if_t": round(float(1.95 * np.sqrt(1 + (40 / FX) ** 2)), 6)}
    assert abs(off - 1.95) < 1e-9, off

    # nearest depth only covers pixels the splat actually writes (alpha mask)
    imgd, depd, _ = render(gauss([0.05, 0.05, 0.05], mu=(0, 0, -2.0)), np.eye(4), FX, FY, CX, CY, W, H)
    covered = imgd[:, :, 0] > 0
    finite = np.isfinite(depd)
    out["depth_mask"] = {"finite": int(finite.sum()), "covered": int(covered.sum()),
                         "finite_outside": int((finite & ~covered).sum())}
    assert np.array_equal(finite, covered), out["depth_mask"]

    print(out)
    return out


def test_projection_covariance_and_depth():
    assert main()


if __name__ == "__main__":
    main()
