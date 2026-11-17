"""Pure-NumPy + Trimesh multi-part mesh splitting for GLB/OBJ assets.

Supports three real split paths without optional C++/scipy libraries:
1. glTF Scene nodes: preserves distinct sub-meshes present in a multi-node GLB (e.g.
   outputs from Lux3D part_split or multi-part CAD exports).
2. Topological connected components: Union-Find over mesh face_adjacency.
3. Kinematic axis partitioning: splits monolithic meshes into N ordered segments
   along their principal axis (or explicit axis) for articulated link assembly.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import trimesh


def _safe_name(name: str, fallback: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", (name or "").strip()).strip("_").lower()
    return cleaned[:40] or fallback


def _connected_face_groups(mesh: trimesh.Trimesh, min_faces: int = 8) -> list[np.ndarray]:
    n_faces = len(mesh.faces)
    if n_faces == 0:
        return []
    parent = list(range(n_faces))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in mesh.face_adjacency:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    groups: dict[int, list[int]] = {}
    for i in range(n_faces):
        groups.setdefault(find(i), []).append(i)
    filtered = [np.asarray(g, dtype=np.int64) for g in groups.values() if len(g) >= min_faces]
    if not filtered:
        return [np.arange(n_faces, dtype=np.int64)]
    filtered.sort(key=len, reverse=True)
    return filtered


def _submesh_from_faces(mesh: trimesh.Trimesh, face_indices: np.ndarray) -> trimesh.Trimesh:
    faces = mesh.faces[face_indices]
    uniq, inverse = np.unique(faces.reshape(-1), return_inverse=True)
    new_verts = mesh.vertices[uniq]
    new_faces = inverse.reshape(-1, 3)
    return trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=False)


def _export_obj(mesh: trimesh.Trimesh, path: Path) -> None:
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    with path.open("w", encoding="ascii", newline="\n") as f:
        np.savetxt(f, verts, fmt="v %.9g %.9g %.9g")
        np.savetxt(f, faces + 1, fmt="f %d %d %d")


def split_mesh_file(
    src_path: Path,
    output_dir: Path,
    *,
    up: str = "y",
    segments: int = 0,
    axis: str = "auto",
) -> dict:
    """Split a GLB/OBJ into individual part OBJ files + parts_manifest.json.

    If the source is a multi-node glTF scene, preserves scene parts.
    Otherwise checks topological connected components; if monolithic and
    segments > 1, partitions sequentially along the requested/principal axis.
    """
    src_path = Path(src_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loaded = trimesh.load(src_path, force=None)
    rot = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float64) if up == "y" else np.eye(4)

    part_meshes: list[tuple[str, trimesh.Trimesh]] = []
    method = "single"
    if isinstance(loaded, trimesh.Scene):
        # Collect transformed meshes preserving node/geometry names when multiple nodes exist
        node_meshes = loaded.dump(concatenate=False)
        if len(node_meshes) > 1:
            method = "gltf_nodes"
            for i, pm in enumerate(node_meshes):
                if isinstance(pm, trimesh.Trimesh) and len(pm.faces) > 0:
                    m_copy = pm.copy()
                    m_copy.apply_transform(rot)
                    raw_name = getattr(pm, "metadata", {}).get("name") or f"node_{i}"
                    part_meshes.append((_safe_name(str(raw_name), f"part_{i}"), m_copy))
        elif len(node_meshes) == 1:
            loaded = node_meshes[0]

    if not part_meshes:
        if isinstance(loaded, trimesh.Scene):
            mesh = loaded.dump(concatenate=True)
        else:
            mesh = loaded
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            raise ValueError(f"Mesh {src_path} contains no triangle faces")
        mesh = mesh.copy()
        mesh.apply_transform(rot)
        min_faces = max(8, int(len(mesh.faces) * 0.004))
        comp_groups = _connected_face_groups(mesh, min_faces=min_faces)
        if len(comp_groups) > 1 and (segments <= 1 or len(comp_groups) >= segments):
            method = "connected_components"
            for i, grp in enumerate(comp_groups):
                part_meshes.append((f"component_{i}", _submesh_from_faces(mesh, grp)))
        elif segments > 1:
            method = "axis_segments"
            work = mesh
            while len(work.faces) < max(300, segments * 60):
                work = work.subdivide()
            centroids = work.vertices[work.faces].mean(axis=1)
            if axis == "x":
                coord = centroids[:, 0]
            elif axis == "y":
                coord = centroids[:, 1]
            elif axis == "z":
                coord = centroids[:, 2]
            else:
                ext = work.extents
                coord = centroids[:, int(np.argmax(ext))]
            lo, hi = float(coord.min()), float(coord.max())
            edges = np.linspace(lo, hi + 1e-9, segments + 1)
            slot_names = ["base", "upper", "fore", "wrist"] if segments == 4 else [f"seg_{i}" for i in range(segments)]
            for i in range(segments):
                mask = (coord >= edges[i]) & (coord < edges[i + 1] if i < segments - 1 else coord <= edges[i + 1])
                idx = np.nonzero(mask)[0]
                if len(idx) == 0:
                    continue
                part_meshes.append((slot_names[i], _submesh_from_faces(work, idx)))
        else:
            part_meshes.append(("whole", mesh))

    # Normalize parts onto Z>=0 global floor and write part OBJs centered around each part's min-Z/XY centroid
    all_min_z = min(float(pm.bounds[0][2]) for _, pm in part_meshes)
    parts_meta = []
    for i, (name, pm) in enumerate(part_meshes):
        pm_norm = pm.copy()
        pm_norm.apply_translation([0.0, 0.0, -all_min_z])
        fname = f"part_{i:02d}_{name}.obj"
        _export_obj(pm_norm, output_dir / fname)
        low, high = pm_norm.bounds
        parts_meta.append({
            "index": i,
            "name": name,
            "obj_file": fname,
            "triangles": int(len(pm_norm.faces)),
            "bounds_m": pm_norm.bounds.tolist(),
            "extents_m": [float(v) for v in pm_norm.extents],
            "centroid_m": [float(v) for v in pm_norm.centroid],
            "volume_m3": float(abs(pm_norm.volume)) if pm_norm.is_watertight else None,
            "pivot_bottom_m": [float(0.5 * (low[0] + high[0])), float(0.5 * (low[1] + high[1])), float(low[2])],
        })

    manifest = {
        "source": src_path.name,
        "method": method,
        "part_count": len(parts_meta),
        "parts": parts_meta,
    }
    (output_dir / "parts_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest
