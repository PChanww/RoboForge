"""Gaussian reader (narrow dialect) + immutable snapshot tests.

Real-asset checks are skipped (with an explicit message) if the asset is absent, so a
skip is never reported as a pass. Synthetic binary PLYs exercise decode formulas and
error paths.
"""
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
from vlab.gaussian import load_gaussians
from vlab.engine import PlatformEnv
from vlab.schema import ProjectConfig
from vlab.sessions import Session
from vlab.rendering.snapshot import capture_snapshot, to_wire, from_wire
from vlab.transforms import quat_xyzw_to_mat

REAL = LAB / "assets" / "scenes" / "world_586c79650960" / "1" / "source" / "scene.ply"

PROPS = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
         "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]


def write_ply(path, rows, props=PROPS, fmt="binary_little_endian", extra_header=""):
    header = [f"ply", f"format {fmt} 1.0", f"element vertex {len(rows)}"]
    header += [f"property float {p}" for p in props]
    header.append("end_header")
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n" + extra_header).encode("ascii"))
        if fmt == "binary_little_endian":
            for r in rows:
                f.write(struct.pack("<" + "f" * len(props), *[float(r.get(p, 0.0)) for p in props]))


def cfg():
    return ProjectConfig.model_validate_json((LAB / "configs" / "default.json").read_text(encoding="utf-8"))


def test_decode_formulas(tmp_path):
    rows = [dict(x=1, y=2, z=3, scale_0=np.log(2), scale_1=np.log(3), scale_2=np.log(4),
                 opacity=0.0, f_dc_0=0, f_dc_1=0, f_dc_2=0, rot_0=1.0)]
    p = tmp_path / "a.ply"
    write_ply(p, rows)
    g = load_gaussians(p)
    assert np.allclose(g["means"][0], [1, 2, 3])
    assert np.allclose(g["scales"][0], [2, 3, 4], rtol=1e-5)  # G1
    assert abs(g["opacity"][0] - 0.5) < 1e-6  # G2
    assert np.allclose(g["rgb"][0], [0.5, 0.5, 0.5], atol=1e-6)  # G2 SH DC=0


def test_quaternion_direction(tmp_path):
    ang = np.radians(150)
    rows = [dict(rot_0=np.cos(ang / 2), rot_3=np.sin(ang / 2))]  # 150deg about Z, wxyz->xyzw in file
    p = tmp_path / "q.ply"
    write_ply(p, rows)
    g = load_gaussians(p)
    R = quat_xyzw_to_mat(g["quat_xyzw"][0])
    assert np.allclose(R @ [1, 0, 0], [np.cos(ang), np.sin(ang), 0], atol=1e-5)  # G3


def test_rejects(tmp_path):
    good = [dict(x=1, rot_0=1.0)]
    # ASCII dialect
    p = tmp_path / "ascii.ply"
    write_ply(p, good, fmt="ascii")
    with pytest.raises(ValueError):
        load_gaussians(p)
    # high-order SH
    p = tmp_path / "sh.ply"
    write_ply(p, [dict(x=1, rot_0=1.0)], props=PROPS + ["f_rest_0"])
    with pytest.raises(ValueError):
        load_gaussians(p)
    # missing property
    p = tmp_path / "miss.ply"
    write_ply(p, [dict(x=1)], props=[q for q in PROPS if q != "opacity"])
    with pytest.raises(ValueError):
        load_gaussians(p)
    # zero quaternion
    p = tmp_path / "zero.ply"
    write_ply(p, [dict(x=1, rot_0=0, rot_1=0, rot_2=0, rot_3=0)])
    with pytest.raises(ValueError):
        load_gaussians(p)
    # truncated
    p = tmp_path / "trunc.ply"
    write_ply(p, [dict(x=1, rot_0=1.0)])
    data = p.read_bytes()
    p.write_bytes(data[:-8])
    with pytest.raises(ValueError):
        load_gaussians(p)
    # non-finite
    p = tmp_path / "nan.ply"
    write_ply(p, [dict(x=float("nan"), rot_0=1.0)])
    with pytest.raises(ValueError):
        load_gaussians(p)


def test_comment_header_binary(tmp_path):
    p = tmp_path / "c.ply"
    rows = [dict(x=1, rot_0=1.0)]
    header = ["ply", "format binary_little_endian 1.0", "comment made by test",
              f"element vertex {len(rows)}"] + [f"property float {q}" for q in PROPS] + ["end_header"]
    with open(p, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        for r in rows:
            f.write(struct.pack("<" + "f" * len(PROPS), *[float(r.get(q, 0.0)) for q in PROPS]))
    assert load_gaussians(p)["count"] == 1


def test_real_asset():
    if not REAL.is_file():
        pytest.skip(f"real gaussian asset absent at {REAL}: NOT validated against real asset")
    g = load_gaussians(REAL)
    assert g["count"] == 199747
    assert g["scale_range"][0] > 0
    assert 0 <= g["opacity_range"][0] <= g["opacity_range"][1] <= 1
    assert np.allclose(np.linalg.norm(g["quat_xyzw"], axis=1), 1, atol=1e-6)  # decoded unit quats


def test_snapshot_epoch_and_immutability():
    s = Session(cfg(), LAB, None, None)
    try:
        r = s.reset(0)
        snap = s.submit(lambda: capture_snapshot(s.env, s.env.episode_time()))
        assert snap["epoch"] == r["epoch"] == s.epoch  # S1
        assert snap["session_id"] == s.id
        assert snap["qpos"].flags.writeable is False  # immutable value object
        pos = np.array(snap["cameras"]["front"]["world_pos"], copy=True)
        q = np.array(snap["qpos"], copy=True)
        s.submit(lambda: [s.env.step([0, 0, 0, 1]) for _ in range(5)])
        s.reset(1)
        assert np.array_equal(np.asarray(snap["cameras"]["front"]["world_pos"]), pos)  # S3
        assert np.array_equal(np.asarray(snap["qpos"]), q)
        w = to_wire(snap)
        back = from_wire(w)  # S4
        assert np.allclose(np.asarray(back["qpos"]), q, atol=1e-9)
        assert np.allclose(np.asarray(back["cameras"]["front"]["world_pos"]), pos, atol=1e-9)
    finally:
        s.close()


def test_frame_uses_snapshot():
    env = PlatformEnv(cfg(), LAB)
    try:
        env.reset(0)
        obs = env.step([0, 0, 0, 1])
        f = obs["images"]["front"]
        assert f is not None
        assert f["request_id"] and f["snapshot_epoch"] == env.epoch  # S5
        assert "camera_world_pos" in f
        assert f["capture_time"] <= f["available_time"] <= obs["sim_time"] + 1e-6
    finally:
        env.close()