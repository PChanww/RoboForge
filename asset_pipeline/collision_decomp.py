"""Pure-NumPy + Trimesh deterministic concave mesh decomposition into tight box compounds.

Produces MuJoCo-ready multi-box static or dynamic collision proxies without needing
optional C++ packages (coacd, rtree, scipy), so concave trays, slots, L-brackets,
and open containers preserve real interior/wall contacts.
"""
from __future__ import annotations

import numpy as np
import trimesh


def _deterministic_surface_points(mesh: trimesh.Trimesh, voxel_size: np.ndarray) -> np.ndarray:
    """Sample every triangle deterministically denser than voxel pitch so thin walls never pinhole."""
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(faces) == 0:
        return verts
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    min_pitch = max(float(np.min(voxel_size)), 1e-5)
    e01 = np.linalg.norm(v1 - v0, axis=1)
    e12 = np.linalg.norm(v2 - v1, axis=1)
    e20 = np.linalg.norm(v0 - v2, axis=1)
    max_edge = np.maximum(e01, np.maximum(e12, e20))
    need_k = np.ceil(max_edge / (0.45 * min_pitch)).astype(int)
    extra: list[np.ndarray] = [verts]
    for bucket, lo, hi in ((2, 1, 2), (4, 3, 4), (8, 5, 8), (16, 9, 16), (24, 17, 10**9)):
        mask = (need_k >= lo) & (need_k <= hi)
        if not np.any(mask):
            continue
        grid = np.linspace(0.0, 1.0, bucket + 1)
        bu, bv = np.meshgrid(grid, grid, indexing="ij")
        tri_uv = np.stack([bu, bv], axis=-1).reshape(-1, 2)
        tri_uv = tri_uv[tri_uv[:, 0] + tri_uv[:, 1] <= 1.0 + 1e-9]
        w1 = tri_uv[:, 0][None, :]
        w2 = tri_uv[:, 1][None, :]
        w0 = 1.0 - w1 - w2
        pts = (w0[:, :, None] * v0[mask][:, None, :] + w1[:, :, None] * v1[mask][:, None, :] + w2[:, :, None] * v2[mask][:, None, :]).reshape(-1, 3)
        extra.append(pts)
    return np.vstack(extra)


def decompose_mesh_to_boxes(
    mesh: trimesh.Trimesh,
    *,
    resolution: int = 16,
    max_boxes: int = 24,
    force_compound: bool = False,
) -> dict:
    """Decompose an axis-aligned-normalized mesh into merged tight box colliders.

    Expects mesh already aligned to Z-up with bounds near its target frame.
    Returns dict with keys: shape ('compound'|'box'), boxes, box_count, fill_ratio,
    concave_detected, grid_shape, floor_box_index, basis.
    """
    low, high = np.asarray(mesh.bounds[0], dtype=np.float64), np.asarray(mesh.bounds[1], dtype=np.float64)
    ext = np.maximum(high - low, 1e-6)
    longest = float(ext.max())
    pitch = max(longest / max(4, int(resolution)), 1e-5)
    shape = np.round(ext / pitch).astype(int)
    # Give thin vertical features (like tray rims/floors) enough Z slices to separate bottom from cavity.
    min_z = 4 if ext[2] >= 0.008 else 2
    shape = np.array([
        int(np.clip(shape[0], 2, 24)),
        int(np.clip(shape[1], 2, 24)),
        int(np.clip(max(shape[2], min_z), 2, 24)),
    ], dtype=int)
    nx, ny, nz = map(int, shape)
    dx, dy, dz = ext / shape

    pts = _deterministic_surface_points(mesh, np.array([dx, dy, dz], dtype=np.float64))
    idx = np.floor((pts - low) / np.array([dx, dy, dz])).astype(int)
    idx[:, 0] = np.clip(idx[:, 0], 0, nx - 1)
    idx[:, 1] = np.clip(idx[:, 1], 0, ny - 1)
    idx[:, 2] = np.clip(idx[:, 2], 0, nz - 1)
    occ = np.zeros((nx, ny, nz), dtype=bool)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True

    # Fill strictly-enclosed interior voxels bounded along all 6 cardinal directions, so solid
    # closed bricks collapse to 1 box while open cavities (trays, bowls, cups, L-slots) stay open.
    hx_pos = np.maximum.accumulate(occ, axis=0)
    hx_neg = np.maximum.accumulate(occ[::-1, :, :], axis=0)[::-1, :, :]
    hy_pos = np.maximum.accumulate(occ, axis=1)
    hy_neg = np.maximum.accumulate(occ[:, ::-1, :], axis=1)[:, ::-1, :]
    hz_pos = np.maximum.accumulate(occ, axis=2)
    hz_neg = np.maximum.accumulate(occ[:, :, ::-1], axis=2)[:, :, ::-1]
    occ = occ | (hx_pos & hx_neg & hy_pos & hy_neg & hz_pos & hz_neg)
    # By convexity, a 1D line cannot cross [solid, air, solid] on a convex object.
    # Remaining empty voxels pinched on >=1 cardinal axis identify genuine concave cavities/slots.
    cavity_voxels = (~occ) & ((hx_pos & hx_neg) | (hy_pos & hy_neg) | (hz_pos & hz_neg))
    cavity_count = int(cavity_voxels.sum())

    wx = max(1, int(round(0.12 * nx)))
    wy = max(1, int(round(0.12 * ny)))
    mid_z = max(1, nz // 2)
    all_4_sides_high = (
        nx >= 6 and ny >= 6 and nz >= 4
        and bool(occ[:wx, :, mid_z:].any())
        and bool(occ[nx - wx :, :, mid_z:].any())
        and bool(occ[:, :wy, mid_z:].any())
        and bool(occ[:, ny - wy :, mid_z:].any())
    )
    is_shallow_tray = force_compound and (ext[2] <= 0.25 * min(ext[0], ext[1])) and nx >= 6 and ny >= 6 and nz >= 4
    if is_shallow_tray or (all_4_sides_high and (force_compound or cavity_count >= max(2, int(0.02 * occ.size)))):
        # Regularize open-top 4-walled trays into a flat interior floor slab + 4 perimeter rim walls.
        fz = max(1, min(nz // 3, 2))
        occ = np.zeros((nx, ny, nz), dtype=bool)
        occ[:, :, :fz] = True
        occ[:wx, :, fz:] = True
        occ[nx - wx :, :, fz:] = True
        occ[:, :wy, fz:] = True
        occ[:, ny - wy :, fz:] = True
        cavity_count = int((nx - 2 * wx) * (ny - 2 * wy) * (nz - fz))

    raw_boxes: list[tuple[float, list[float], list[float], int]] = []
    # Z-layer-first greedy maximal cuboid merging keeps flat floor/base layers intact.
    while occ.any() and len(raw_boxes) < max_boxes * 4:
        iz, ix, iy = map(int, np.argwhere(occ.transpose(2, 0, 1))[0])
        x2 = ix + 1
        while x2 < nx and occ[x2, iy, iz]:
            x2 += 1
        y2 = iy + 1
        while y2 < ny and occ[ix:x2, y2, iz].all():
            y2 += 1
        z2 = iz + 1
        while z2 < nz and occ[ix:x2, iy:y2, z2].all():
            z2 += 1
        occ[ix:x2, iy:y2, iz:z2] = False
        center = [
            float(low[0] + (ix + x2) * 0.5 * dx),
            float(low[1] + (iy + y2) * 0.5 * dy),
            float(low[2] + (iz + z2) * 0.5 * dz),
        ]
        size = [float((x2 - ix) * dx), float((y2 - iy) * dy), float((z2 - iz) * dz)]
        raw_boxes.append((size[0] * size[1] * size[2], center, size, iz))

    raw_boxes.sort(key=lambda item: (item[3], -item[0]))
    selected = raw_boxes[:max_boxes]
    boxes = [
        {
            "shape": "box",
            "size_m": size,
            "position": center,
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        }
        for _, center, size, _ in selected
    ]
    aabb_vol = float(ext[0] * ext[1] * ext[2])
    comp_vol = sum(b["size_m"][0] * b["size_m"][1] * b["size_m"][2] for b in boxes)
    fill_ratio = float(min(1.0, comp_vol / max(aabb_vol, 1e-12)))
    concave_detected = len(boxes) >= 2 and cavity_count >= max(2, int(0.02 * occ.size))

    if not concave_detected and not force_compound:
        boxes = [{
            "shape": "box",
            "size_m": [float(ext[0]), float(ext[1]), float(ext[2])],
            "position": [float(0.5 * (low[0] + high[0])), float(0.5 * (low[1] + high[1])), float(0.5 * (low[2] + high[2]))],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        }]
        fill_ratio = 1.0

    # Identify widest box touching the bottom plane as the primary support/receptacle floor.
    floor_idx = 0
    best_score = -1.0
    for i, b in enumerate(boxes):
        z_bot = b["position"][2] - 0.5 * b["size_m"][2]
        if z_bot <= low[2] + dz * 1.15:
            area_xy = b["size_m"][0] * b["size_m"][1]
            if area_xy > best_score:
                best_score = area_xy
                floor_idx = i

    return {
        "shape": "compound" if len(boxes) > 1 else "box",
        "boxes": boxes,
        "box_count": len(boxes),
        "fill_ratio": round(fill_ratio, 4),
        "concave_detected": bool(concave_detected),
        "grid_shape": [nx, ny, nz],
        "floor_box_index": int(floor_idx),
        "basis": "voxel_box_decomposition",
    }
