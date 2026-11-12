"""Scene/asset coordinate transforms (sensor cameras are built through MJCF).

World is metres, right-handed Z-up (MuJoCo). External quaternions are xyzw;
MuJoCo expects wxyz (conversion explicit here). Layered transform:
    T_world_from_asset = T_world_from_scene @ T_scene_from_asset
Uniform scale only (non-uniform would break contact geometry).
"""
from __future__ import annotations

import numpy as np


def quat_xyzw_to_wxyz(q):
    x, y, z, w = q
    return [w, x, y, z]


def quat_xyzw_to_mat(q):
    x, y, z, w = np.asarray(q, dtype=float)
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def transform_matrix(position, quat_xyzw, uniform_scale=1.0):
    T = np.eye(4)
    T[:3, :3] = quat_xyzw_to_mat(quat_xyzw) * uniform_scale
    T[:3, 3] = position
    return T


def compose(T_world_from_scene, T_scene_from_asset):
    return T_world_from_scene @ T_scene_from_asset


def mat_to_pos_quat_xyzw(T):
    R = T[:3, :3]
    s = np.cbrt(max(np.linalg.det(R), 1e-12))
    R = R / s
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        w = np.sqrt(tr + 1) / 2
        x = (R[2, 1] - R[1, 2]) / (4 * w)
        y = (R[0, 2] - R[2, 0]) / (4 * w)
        z = (R[1, 0] - R[0, 1]) / (4 * w)
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        if i == 0:
            t = 2 * np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
            x, y, z, w = t / 4, (R[0, 1] + R[1, 0]) / t, (R[0, 2] + R[2, 0]) / t, (R[2, 1] - R[1, 2]) / t
        elif i == 1:
            t = 2 * np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2])
            x, y, z, w = (R[0, 1] + R[1, 0]) / t, t / 4, (R[1, 2] + R[2, 1]) / t, (R[0, 2] - R[2, 0]) / t
        else:
            t = 2 * np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1])
            x, y, z, w = (R[0, 2] + R[2, 0]) / t, (R[1, 2] + R[2, 1]) / t, t / 4, (R[1, 0] - R[0, 1]) / t
    return T[:3, 3].tolist(), [float(x), float(y), float(z), float(w)], float(s)
