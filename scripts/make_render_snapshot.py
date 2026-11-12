"""Write a render-request JSON (front camera world matrix + pinhole projection) from a
real MuJoCo state, plus optional dynamic body boxes for the offscreen splat renderer.
"""
import json
import sys
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
from vlab.engine import PlatformEnv
from vlab.schema import ProjectConfig

OUT = LAB / "outputs" / "phase2-render-validation"


def pinhole(fx, fy, cx, cy, W, H, near, far):
    P = np.zeros((4, 4))
    P[0, 0] = 2 * fx / W
    P[0, 2] = 2 * cx / W - 1
    P[1, 1] = 2 * fy / H
    P[1, 2] = 2 * cy / H - 1
    P[2, 2] = -(far + near) / (far - near)
    P[2, 3] = -2 * far * near / (far - near)
    P[3, 2] = -1
    return P


def world_matrix(pos, R):
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = pos
    return M


def main():
    project = sys.argv[1] if len(sys.argv) > 1 else "default"
    cfg = ProjectConfig.model_validate_json(
        (LAB / "projects" / project / "current.json").read_text(encoding="utf-8"))
    env = PlatformEnv(cfg, LAB)
    try:
        env.reset(0)
        for _ in range(5):
            env.step([0, 0, 0, 1])
        cid = env.model.camera("front").id
        pos = env.data.cam_xpos[cid].copy()
        R = env.data.cam_xmat[cid].reshape(3, 3).copy()
        s = cfg.sensors[0]
        W, H = s.resolution
        P = pinhole(s.intrinsics.fx, s.intrinsics.fy, s.intrinsics.cx, s.intrinsics.cy, W, H, 0.01, 100.0)
        out = {
            "project": project, "camera": "front", "width": W, "height": H,
            # THREE Matrix4.fromArray is column-major
            "M": world_matrix(pos, R).T.flatten().tolist(),
            "P": P.T.flatten().tolist(),
            "camera_pos": pos.tolist(),
            "fx": s.intrinsics.fx, "fy": s.intrinsics.fy, "cx": s.intrinsics.cx, "cy": s.intrinsics.cy,
            "bodies": {name: {"pos": env.data.xpos[env.model.body(name).id].tolist(),
                              "mat": env.data.xmat[env.model.body(name).id].reshape(3, 3).tolist()}
                       for name in ("eef", "cup")},
        }
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "camera.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        print("wrote", OUT / "camera.json")
    finally:
        env.close()


if __name__ == "__main__":
    main()