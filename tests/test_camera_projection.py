"""T04: numeric camera projection validation through the production camera XML.

Markers sit at known WORLD positions; the camera is built by vlab.mjcf._camera_xml
(the tag actually used by the engine), on an optional hinged mount. Renders are
segmentation-based (unbiased centroids) and compared to the analytic pinhole
projection from MuJoCo's camera world pose + corner-based pixel intrinsics.
Covers fx!=fy, off-center principal point, non-square resolution and camera motion.
"""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
from vlab.mjcf import _camera_xml  # noqa: E402
from vlab.schema import CameraIntrinsics, SensorConfig  # noqa: E402

CAM_XYAXES = [-1, 0, 0, 0, 0, 1]  # pixel right = -X world, pixel up = +Z, looks toward -Y
PIVOT = np.array([0.0, 1.2, 0.35])


def _fmt(v):
    return " ".join(f"{x:.6g}" for x in v)


def build(marker, W, H, fx, fy, cx, cy, hinged=False, qpos=0.0):
    sensor = SensorConfig(id="cam", parent="mount", position=[0, 0, 0], xyaxes=CAM_XYAXES,
                          intrinsics=CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy),
                          resolution=[W, H])
    hinge = '' if not hinged else '<joint name="j" type="hinge" axis="0 0 1" range="-3 3"/>'
    xml = f'''<mujoco><compiler angle="radian"/><option timestep=".002"/>
    <visual><global offwidth="{W}" offheight="{H}"/></visual>
    <worldbody>
      <light pos="0 1.3 3" dir="0 0 -1" diffuse="1 1 1"/>
      <geom type="plane" size="4 4 .001" pos="0 0 -.001" rgba="0 0 0 1"/>
      <geom name="mk" type="sphere" size=".02" pos="{_fmt(marker)}" rgba="1 0 0 1"/>
      <body name="mount" pos="{_fmt(PIVOT)}">{hinge}
        <inertial pos="0 0 0" mass="0.1" diaginertia="1e-4 1e-4 1e-4"/>
        {_camera_xml(sensor)}
      </body>
    </worldbody></mujoco>'''
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    if hinged:
        d.qpos[0] = qpos
    mujoco.mj_forward(m, d)
    return m, d


def centroid(m, d, W, H):
    r = mujoco.Renderer(m, height=H, width=W)
    r.enable_segmentation_rendering()
    r.update_scene(d, camera="cam")
    img = r.render()
    r.close()
    ys, xs = np.nonzero(img[:, :, 0] == m.geom("mk").id)
    return (float(xs.mean()), float(ys.mean())) if len(xs) else None


def project(d, cam_id, p_world, fx, fy, cx, cy):
    pos = np.asarray(d.cam_xpos[cam_id])
    R = np.asarray(d.cam_xmat[cam_id]).reshape(3, 3)  # rows are camera X/Y/Z axes in world
    pc = R.T @ (np.asarray(p_world) - pos)
    depth = -pc[2]
    return cx + fx * pc[0] / depth, cy - fy * pc[1] / depth, depth


def view_frame(q):
    c, s = np.cos(q), np.sin(q)
    forward = np.array([s, -c, 0.0])
    right = np.array([-c, -s, 0.0])
    return forward, right, np.array([0.0, 0.0, 1.0])


def run_case(name, W, H, fx, fy, cx, cy, q=0.0, hinged=False):
    forward, right, up = view_frame(q)
    offsets = [(0.0, 0.0), (0.02, 0.03), (-0.03, 0.02), (0.01, -0.035)]
    errs, axis_err = [], None
    for k, (a, b) in enumerate(offsets):
        mk = PIVOT + 0.3 * forward + a * right + b * up
        m, d = build(mk, W, H, fx, fy, cx, cy, hinged, q)
        c = centroid(m, d, W, H)
        if c is None:
            return {"case": name, "error": "marker not visible", "marker": mk.tolist()}
        u, v, depth = project(d, m.camera("cam").id, mk, fx, fy, cx, cy)
        if depth <= 0:
            return {"case": name, "error": "marker behind camera"}
        err = float(np.hypot(c[0] - u, c[1] - v))
        if k == 0:
            axis_err = err
        errs.append(err)
    # Model roundtrip: MuJoCo stores principal as a center-relative offset that the
    # renderer applies with inverted sign (see docs/PHASE2_VALIDATION T04).
    intr = np.asarray(m.cam_intrinsic[0]).ravel()
    ss = np.asarray(m.cam_sensorsize[0]).ravel()
    cx_model = W / 2 - intr[2] * W / ss[0]
    cy_model = H / 2 - intr[3] * H / ss[1]
    fx_model = intr[0] * W / ss[0]
    fy_model = intr[1] * H / ss[1]
    return {"case": name, "centroid": [round(v, 2) for v in c], "axis_marker_err_px": round(axis_err, 3),
            "max_err_px": round(max(errs), 3),
            "intrinsic_roundtrip_px": {"fx": round(float(fx_model - fx), 4),
                                       "fy": round(float(fy_model - fy), 4),
                                       "cx": round(float(cx_model - cx), 4),
                                       "cy": round(float(cy_model - cy), 4)}}


def main():
    cases = [
        run_case("A_center_square_fx240_fy300", 256, 256, 240, 300, 128, 128),
        run_case("B_offset_nonsquare", 320, 240, 280, 240, 150, 100),
        run_case("C_hinge_q0", 256, 256, 260, 260, 128, 128, hinged=True, q=0.0),
        run_case("D_hinge_q0.6", 256, 256, 260, 260, 128, 128, hinged=True, q=0.6),
        run_case("E_hinge_q1.2_offcenter", 320, 240, 300, 250, 175, 95, hinged=True, q=1.2),
    ]
    for r in cases:
        print(r)
    ok = all(r.get("max_err_px", 999) < 2.5 for r in cases)
    out = LAB / "outputs" / "phase2-render-validation" / "camera_projection.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"mujoco": mujoco.__version__, "convention": "corner-based pixel intrinsics; "
         "MuJoCo principalpixel = resolution/2 - pixel principal",
         "threshold_px": 2.5, "cases": cases, "all_cases_under_threshold": ok}, indent=1),
        encoding="utf-8")
    print({"all_cases_under_2.5px": ok, "report": str(out)})
    return ok


def test_camera_projection_under_threshold():
    assert main()


if __name__ == "__main__":
    main()
