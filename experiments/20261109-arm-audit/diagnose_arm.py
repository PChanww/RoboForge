"""Diagnose the 5R arm: self contacts at rest, IK tracking error, expert rollout outcome.

Run from the vla-lab directory with the workspace venv Python.
Writes diagnosis.json and sample frames next to this script.
"""
import hashlib
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import arm
from sim_env import TabletopEnv

OUT = Path(__file__).resolve().parent


def geom_name(model, gid):
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom{gid}"


def body_name(model, bid):
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or f"body{bid}"


def contact_report(env):
    rows = []
    for c in env.data.contact:
        if c.dist >= 0:
            continue
        g1, g2 = int(c.geom1), int(c.geom2)
        b1 = body_name(env.model, env.model.geom_bodyid[g1])
        b2 = body_name(env.model, env.model.geom_bodyid[g2])
        rows.append({"geom1": geom_name(env.model, g1), "geom2": geom_name(env.model, g2),
                     "body1": b1, "body2": b2, "dist_mm": round(1000 * float(c.dist), 3)})
    return rows


def main():
    env = TabletopEnv("standard", 0)
    report = {"asset": "standard", "seed": 0,
              "code_sha256": {n: hashlib.sha256((Path(arm.__file__).parent / n).read_bytes()).hexdigest()
                              for n in ("arm.py", "sim_env.py")},
              "physics": env.spec}
    report["rest_self_contacts"] = contact_report(env)
    report["rest_eef"] = env.data.xpos[env._eef].tolist()
    report["rest_target"] = env._target.tolist()
    Image.fromarray(env.render(800, 540)).save(OUT / "rest.jpg")

    ticks = []
    for tick in range(180):
        env.step(env.expert_action())
        eef = env.data.xpos[env._eef]
        err = np.linalg.norm(eef - env._target)
        xmat = env.data.xmat[env._eef].reshape(3, 3)
        att_err = float(np.degrees(np.arccos(np.clip((np.trace(xmat) - 1) / 2, -1, 1))))
        m = env.metrics()
        row = {"tick": tick + 1, "track_err_mm": round(1000 * float(err), 2),
               "att_err_deg": round(att_err, 3),
               "contacts": m["contacts"], "lifted": m["lifted"], "success": m["success"],
               "cup_xyz": [round(float(v), 4) for v in m["object_xyz"]]}
        if tick in (43, 59, 85, 125, 150, 164):
            row["contact_pairs"] = contact_report(env)
            Image.fromarray(env.render(800, 540)).save(OUT / f"tick{tick + 1:03d}.jpg")
        ticks.append(row)
    report["final_metrics"] = env.metrics()
    report["max_track_err_mm"] = max(t["track_err_mm"] for t in ticks)
    report["mean_track_err_mm"] = round(float(np.mean([t["track_err_mm"] for t in ticks])), 2)
    report["ticks"] = ticks
    env.close()
    (OUT / "diagnosis.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"success": report["final_metrics"]["success"],
                      "lifted": report["final_metrics"]["lifted"],
                      "max_track_err_mm": report["max_track_err_mm"],
                      "rest_self_contacts": report["rest_self_contacts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()