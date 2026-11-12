"""3D Gaussian Splatting PLY reader (narrow, verified dialect).

Supported dialect (matches the real World asset): binary_little_endian, a single
`vertex` element of float32 properties x,y,z,nx,ny,nz,f_dc_0..2,opacity,
scale_0..2,rot_0..3. Any other dialect (ascii, big endian, extra elements, extra
SH bands f_rest_*, missing required props) is rejected with a clear ValueError
rather than guessed. Decodes log-scale->exp, opacity logit->sigmoid (stable),
SH DC->RGB (0.2820948*dc+0.5), quaternion (rot_0=w) normalized.
"""
from __future__ import annotations

import numpy as np

SH_C0 = 0.28209479177387814
REQUIRED = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2",
            "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]


def _read_header(f):
    if f.readline().strip() != b"ply":
        raise ValueError("not a PLY file")
    fmt = None
    elements = []  # (name, count, [(type, prop)])
    while True:
        line = f.readline()
        if not line:
            raise ValueError("unterminated PLY header (no end_header)")
        s = line.strip()
        if not s or s.startswith(b"comment") or s.startswith(b"obj_info"):
            continue
        if s == b"end_header":
            break
        parts = s.decode("ascii", "replace").split()
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element":
            elements.append((parts[1], int(parts[2]), []))
        elif parts[0] == "property":
            if not elements:
                raise ValueError("property before element")
            if parts[1] == "list":
                raise ValueError("list properties are not supported for gaussian PLY")
            elements[-1][2].append((parts[1], parts[2]))
    return fmt, elements


def load_gaussians(path):
    with open(path, "rb") as f:
        fmt, elements = _read_header(f)
        body = f.read()
    if fmt != "binary_little_endian":
        raise ValueError(f"unsupported PLY dialect '{fmt}'; only binary_little_endian is verified")
    names = [e[0] for e in elements]
    if names != ["vertex"]:
        raise ValueError(f"expected a single vertex element, got {names}")
    _, count, props = elements[0]
    pnames = [p[1] for p in props]
    if any(n.startswith("f_rest_") for n in pnames):
        raise ValueError("higher-order SH (f_rest_*) is not supported by this reader")
    missing = [n for n in REQUIRED if n not in pnames]
    if missing:
        raise ValueError(f"missing gaussian properties: {missing}")
    if any(t != "float" for t, _ in props):
        raise ValueError("only float32 gaussian attributes are supported")
    if count <= 0:
        raise ValueError("zero vertices")
    dt = np.dtype([(n, "<f4") for n in pnames])
    if len(body) < count * dt.itemsize:
        raise ValueError("PLY body truncated")
    arr = np.frombuffer(body[:count * dt.itemsize], dtype=dt)
    c = {n: arr[n].astype(np.float64) for n in pnames}
    return _decode(c)


def _decode(c):
    means = np.stack([c["x"], c["y"], c["z"]], axis=1)
    log_scale = np.stack([c["scale_0"], c["scale_1"], c["scale_2"]], axis=1)
    scales = np.exp(np.clip(log_scale, -30, 30))  # avoid inf/0
    logit = np.clip(c["opacity"], -60, 60)
    opacity = 1.0 / (1.0 + np.exp(-logit))
    dc = np.stack([c["f_dc_0"], c["f_dc_1"], c["f_dc_2"]], axis=1)
    rgb = np.clip(SH_C0 * dc + 0.5, 0, 1)  # display RGB; raw DC retained separately
    quat = np.stack([c["rot_1"], c["rot_2"], c["rot_3"], c["rot_0"]], axis=1)  # xyzw
    norms = np.linalg.norm(quat, axis=1)
    if not np.isfinite(norms).all() or (norms < 1e-8).any():
        raise ValueError("degenerate (zero) quaternion in gaussian data")
    quat = quat / norms[:, None]
    for name, a in (("means", means), ("scales", scales), ("opacity", opacity),
                    ("dc", dc), ("quat", quat)):
        if not np.isfinite(a).all():
            raise ValueError(f"non-finite values in gaussian {name}")
    return {
        "count": int(len(means)), "means": means, "scales": scales, "opacity": opacity,
        "rgb": rgb, "sh_dc": dc, "quat_xyzw": quat,
        "means_bbox_min": means.min(0).tolist(), "means_bbox_max": means.max(0).tolist(),
        "scale_range": [float(scales.min()), float(scales.max())],
        "opacity_range": [float(opacity.min()), float(opacity.max())],
        "quat_raw_norm_dev": float(np.abs(norms - 1).max()),
        "quat_norm_dev_after_normalize": float(np.abs(np.linalg.norm(quat, axis=1) - 1).max()),
        "note": "bbox is gaussian centers only; not surface extent. Units require calibration.",
    }