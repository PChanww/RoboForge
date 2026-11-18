"""Fit longitudinal regions of the generated calibration mesh to local rigid-link frames."""
from pathlib import Path
import numpy as np
import trimesh

try:
    from import_asset import parse_geometry
except ImportError:
    from asset_pipeline.import_asset import parse_geometry


def _cyl_x90(radius: float, height: float, pos: list[float]) -> trimesh.Trimesh:
    c = trimesh.creation.cylinder(radius=radius, height=height, sections=24)
    c.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2.0, [1.0, 0.0, 0.0]))
    c.apply_translation(pos)
    return c


def _cyl_z(radius: float, height: float, pos: list[float]) -> trimesh.Trimesh:
    c = trimesh.creation.cylinder(radius=radius, height=height, sections=24)
    c.apply_translation(pos)
    return c


def main():
    root = Path(__file__).resolve().parents[1] / "assets" / "robot"
    mesh, _ = parse_geometry((root / "original.glb").read_bytes())
    # glTF +Y is up. Express vertices in simulator coordinates.
    v = mesh.vertices[:, [0, 2, 1]].copy()
    v[:, 1] *= -1
    z = (v[:, 2] - v[:, 2].min()) / np.ptp(v[:, 2])
    specs = [
        ("base", 0.0, 0.18, 0.108, 0.08),
        ("upper", 0.18, 0.51, 0.062, 0.23),
        ("fore", 0.51, 0.82, 0.056, 0.23),
        ("wrist", 0.82, 1.001, 0.041, 0.045),
    ]
    for name, lo, hi, width, length in specs:
        center_z = z[mesh.faces].mean(axis=1)
        faces = mesh.faces[(center_z >= lo) & (center_z < hi)]
        ids, idx = np.unique(faces, return_inverse=True)
        pts = v[ids].copy()
        pts -= pts.min(axis=0)
        pts[:, :2] -= np.ptp(pts[:, :2], axis=0) / 2.0
        pts[:, :2] *= width / np.ptp(pts[:, :2], axis=0).max()
        pts[:, 2] *= length / np.ptp(pts[:, 2])
        if name in ("upper", "fore"):
            t = np.clip(np.minimum(pts[:, 2], length - pts[:, 2]) / 0.026, 0.0, 1.0)
            pts[:, 0] *= 0.58 + 0.42 * t
            pts[:, 1] *= 0.85 + 0.15 * t
        if name == "wrist":
            pts[:, 2] *= -1

        seg = trimesh.Trimesh(vertices=pts, faces=idx.reshape(-1, 3), process=False)
        parts = [seg]
        if name == "base":
            parts.append(_cyl_z(0.052, 0.006, [0.0, 0.0, 0.003]))
            parts.append(_cyl_z(0.033, 0.016, [0.0, 0.0, 0.072]))
        elif name == "upper":
            parts.append(_cyl_x90(0.025, 0.056, [0.0, 0.0, 0.0]))
            parts.append(_cyl_x90(0.025, 0.056, [0.0, 0.0, 0.23]))
        elif name == "fore":
            parts.append(_cyl_x90(0.023, 0.050, [0.0, 0.0, 0.0]))
            parts.append(_cyl_x90(0.023, 0.050, [0.0, 0.0, 0.23]))
        elif name == "wrist":
            parts.append(_cyl_x90(0.020, 0.040, [0.0, 0.0, 0.0]))
            parts.append(_cyl_z(0.020, 0.016, [0.0, 0.0, -0.045]))

        combined = trimesh.util.concatenate(parts)
        with (root / f"{name}.obj").open("w", encoding="ascii", newline="\n") as f:
            np.savetxt(f, combined.vertices, fmt="v %.8f %.8f %.8f")
            np.savetxt(f, combined.faces + 1, fmt="f %d %d %d")


if __name__ == "__main__":
    main()
