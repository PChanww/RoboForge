"""Create a small local demo scene (empty workspace + environment geometry) and a tiny
3DGS PLY fixture, then run them through the real scene import pipeline. No online API.
"""
import sys
from pathlib import Path

import numpy as np
import trimesh

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
from vlab.scene_import import import_scene, detect_ply


def box(center, size):
    b = trimesh.creation.box(extents=size)
    b.apply_translation(center)
    return b


def main():
    tmp = LAB / "datasets" / "scene_fixtures"
    tmp.mkdir(parents=True, exist_ok=True)
    # table slab (top at z=0) + back wall + left wall, metres, Z-up
    table = box([0, 0, -0.025], [0.8, 0.6, 0.05])
    # walls placed outside the arm workspace so the task area stays unobstructed
    back = box([0, 0.42, 0.2], [0.9, 0.02, 0.4])
    left = box([-0.52, 0, 0.2], [0.02, 0.8, 0.4])
    scene = trimesh.util.concatenate([table, back, left])
    ply = tmp / "demo_workspace.ply"
    scene.export(ply)
    print("ply kind:", detect_ply(ply))
    mf = import_scene(ply, LAB, "demo_workspace", 2, source_kind="import", up_axis="Z",
                      unit_scale=1.0, calibrated=True, display_name="空工作台演示场景")
    print("imported:", mf)

    # tiny 3D Gaussian PLY fixture (header/properties identify it as gaussian, not mesh)
    g = tmp / "demo_splat.ply"
    n = 8
    lines = ["ply", "format ascii 1.0", f"element vertex {n}",
             "property float x", "property float y", "property float z",
             "property float f_dc_0", "property float f_dc_1", "property float f_dc_2",
             "property float opacity", "property float scale_0", "property float scale_1",
             "property float scale_2", "property float rot_0", "property float rot_1",
             "property float rot_2", "property float rot_3", "end_header"]
    rng = np.random.default_rng(0)
    for i in range(n):
        p = rng.uniform(-0.3, 0.3, 3)
        lines.append(f"{p[0]} {p[1]} {p[2]} 0.5 0.5 0.5 1 -2 -2 -2 1 0 0 0")
    g.write_text("\n".join(lines) + "\n", encoding="ascii")
    print("splat kind:", detect_ply(g))
    mf2 = import_scene(g, LAB, "demo_splat", 2, source_kind="import", up_axis="Z", calibrated=False)
    print("imported splat:", mf2)


if __name__ == "__main__":
    main()