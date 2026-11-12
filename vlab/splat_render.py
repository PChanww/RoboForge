"""CPU 3D Gaussian Splatting rasterizer (numpy), consuming the real decoded PLY.

Projects each Gaussian with its 3D covariance (from scale+rotation), sorts far->near,
and alpha-blends 2D elliptical footprints into an RGB image for a specified pinhole
camera (world matrix + fx/fy/cx/cy). This is a real splat render (not a point cloud
or panorama), used where a browser/3DGS library is unavailable.
"""
from __future__ import annotations

import numpy as np


def _quat_to_mat(q):
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R = np.empty((q.shape[0], 3, 3))
    R[:, 0, 0] = 1 - 2 * (yy + zz); R[:, 0, 1] = 2 * (xy - wz); R[:, 0, 2] = 2 * (xz + wy)
    R[:, 1, 0] = 2 * (xy + wz); R[:, 1, 1] = 1 - 2 * (xx + zz); R[:, 1, 2] = 2 * (yz - wx)
    R[:, 2, 0] = 2 * (xz - wy); R[:, 2, 1] = 2 * (yz + wx); R[:, 2, 2] = 1 - 2 * (xx + yy)
    return R


def render(g, M, fx, fy, cx, cy, W, H, near=0.05, opacity_min=0.02, max_gauss=200000, inst=None):
    """inst = {'position':[3], 'quat_xyzw':[4], 'uniform_scale':s} applied to gaussians
    before the camera transform (mean: s*R*mu+t; covariance: s^2*R*Sigma*R^T)."""
    M = np.asarray(M, dtype=np.float64).reshape(4, 4, order="F")
    Rc = M[:3, :3]          # columns are camera axes in world
    pos = M[:3, 3]
    idx = np.argsort(-g["opacity"])[:max_gauss]
    means = g["means"][idx]
    scales = g["scales"][idx]
    ok = g["opacity"][idx] > opacity_min
    means, scales = means[ok], scales[ok]
    quat = g["quat_xyzw"][idx][ok]
    rgb = g["rgb"][idx][ok]
    opa = g["opacity"][idx][ok]

    Ri = si = None
    if inst:
        Ri = _quat_to_mat(np.asarray(inst.get("quat_xyzw", [0, 0, 0, 1]), dtype=float)[None])[0]
        si = float(inst.get("uniform_scale", 1.0))
        ti = np.asarray(inst.get("position", [0, 0, 0]), dtype=float)
        means = si * (means @ Ri.T) + ti  # asset -> scene/world

    cam = (means - pos) @ Rc           # world -> camera (R columns are axes)
    zp = -cam[:, 2]                    # forward is -Z
    vis = zp > near
    cam, zp, means, scales, quat, rgb, opa = cam[vis], zp[vis], means[vis], scales[vis], quat[vis], rgb[vis], opa[vis]
    x, y = cam[:, 0], cam[:, 1]
    u = cx + fx * x / zp
    v = cy - fy * y / zp

    Rq = _quat_to_mat(quat)
    s2 = (scales ** 2)[:, :, None] * np.eye(3)[None]
    Sigma = np.einsum("nij,njk,nlk->nil", Rq, s2, Rq)  # R diag(s^2) R^T (asset/world)
    if inst:
        Sigma = si * si * np.einsum("ij,njk,lk->nil", Ri, Sigma, Ri)
    # World-space covariance must be rotated into camera coordinates before the
    # projection Jacobian: Sigma_cam = Rc^T Sigma_world Rc (Rc columns are cam axes).
    Sigma_cam = np.einsum("ji,njk,kl->nil", Rc, Sigma, Rc)
    J = np.zeros((len(zp), 2, 3))
    J[:, 0, 0] = fx / zp
    J[:, 0, 2] = fx * x / zp ** 2
    J[:, 1, 1] = -fy / zp
    J[:, 1, 2] = -fy * y / zp ** 2
    S2 = np.einsum("nij,njk,nlk->nil", J, Sigma_cam, J) + 0.3 * np.eye(2)[None]

    a = S2[:, 0, 0]; b = S2[:, 0, 1]; c = S2[:, 1, 1]
    det = a * c - b * b
    good = det > 1e-9
    a, b, c, det, u, v, zp, rgb, opa = a[good], b[good], c[good], det[good], u[good], v[good], zp[good], rgb[good], opa[good]
    ia, ib, ic = c / det, -b / det, a / det             # inverse covariance entries

    img = np.zeros((H, W, 3))
    depth = np.full((H, W), np.inf)                     # nearest covered splat center, camera-space Z
    order = np.argsort(-zp)                             # far first for back-to-front blend
    rad = np.sqrt(2.0 * np.maximum(a, c))               # ~1 sigma pixel radius
    for i in order:
        sx, sy = rad[i] * 3.0, rad[i] * 3.0
        x0, x1 = int(max(0, np.floor(u[i] - sx))), int(min(W - 1, np.ceil(u[i] + sx)))
        y0, y1 = int(max(0, np.floor(v[i] - sy))), int(min(H - 1, np.ceil(v[i] + sy)))
        if x0 > x1 or y0 > y1:
            continue
        xs = np.arange(x0, x1 + 1) - u[i]
        ys = np.arange(y0, y1 + 1) - v[i]
        X, Y = np.meshgrid(xs, ys)
        power = -0.5 * (ia[i] * X * X + 2 * ib[i] * X * Y + ic[i] * Y * Y)
        alpha = opa[i] * np.exp(np.clip(power, -20, 0))
        m = alpha > 1e-3
        if not m.any():
            continue
        sub = img[y0:y1 + 1, x0:x1 + 1]
        al = alpha[m][:, None]
        sub[m] = al * rgb[i] + (1 - al) * sub[m]
        img[y0:y1 + 1, x0:x1 + 1] = sub
        dp = depth[y0:y1 + 1, x0:x1 + 1]
        dp[m] = np.minimum(dp[m], zp[i])  # only pixels this splat actually covers

    return np.clip(img, 0, 1), depth, {"input": int(g["count"]), "rendered_splats": int(len(order)),
                                       "visible_splats": int(len(zp)), "resolution": [W, H]}


def ray_box_hit(M, box_center, box_half, W, H, fx, fy, cx, cy):
    """Analytic per-pixel ray/box intersection (world AABB) -> camera-space Z depth
    (inf = miss), the same unit as render()'s depth map. Off-axis rays are converted
    from distance-along-unit-ray to camera-plane depth (divide by the ray norm)."""
    M = np.asarray(M).reshape(4, 4, order="F")
    Rc = M[:3, :3]; pos = M[:3, 3]
    jj, ii = np.mgrid[0:H, 0:W]
    dx = (ii - cx) / fx; dy = -(jj - cy) / fy
    dirs = np.stack([dx, dy, -np.ones_like(dx)], axis=-1) @ Rc.T   # cam->world
    n = np.linalg.norm(dirs, axis=-1, keepdims=True)
    dirs = dirs / n
    c = np.asarray(box_center); h = np.asarray(box_half)
    inv = 1.0 / np.where(np.abs(dirs) < 1e-9, 1e-9, dirs)
    t1 = (c - h - pos) * inv; t2 = (c + h - pos) * inv
    tmin = np.max(np.minimum(t1, t2), axis=-1); tmax = np.min(np.maximum(t1, t2), axis=-1)
    hit = (tmax >= np.maximum(tmin, 0)) & (tmin > 0)
    return np.where(hit, tmin / n[:, :, 0], np.inf)