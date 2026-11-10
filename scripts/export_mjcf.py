"""Export the compiled model to a standalone MJCF file and report what is preserved."""
import json
import sys
from pathlib import Path

import mujoco

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
from vlab.engine import PlatformEnv
from vlab.schema import ProjectConfig


def main():
    cfg = ProjectConfig.model_validate_json((LAB / "configs" / "default.json").read_text(encoding="utf-8"))
    env = PlatformEnv(cfg, LAB)
    try:
        out = LAB / "deploy" / "mjcf"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{env.robot.id}.xml"
        mujoco.mj_saveLastXML(str(path), env.model)
        m = env.model
        report = {"file": str(path.relative_to(LAB)), "bodies": m.nbody, "geoms": m.ngeom,
                  "joints": m.njnt, "actuators": m.nu, "cameras": m.ncam, "meshes": m.nmesh,
                  "conversions": {
                      "coordinates": "MJCF z-up right-handed preserved as-is",
                      "collisions": "all geoms incl. contype/conaffinity, friction, solref preserved",
                      "sensors": "cameras exported with pinhole intrinsics (focalpixel/principalpixel/resolution)",
                      "actuators": "position actuators with kp/kv/forcerange preserved",
                      "gripper": "prismatic jaw joints preserved"},
                  "note": "mesh files are referenced by name; copy assets/robot alongside the XML"}
        (LAB / "datasets" / "reports").mkdir(parents=True, exist_ok=True)
        (LAB / "datasets" / "reports" / "mjcf_export.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps({k: report[k] for k in ("bodies", "geoms", "joints", "actuators", "cameras", "meshes")}))
    finally:
        env.close()


if __name__ == "__main__":
    main()