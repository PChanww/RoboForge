"""Phase A acceptance: paired grasp controls on 4 assets x 10 seeds + self-collision checks.

Run from vla-lab with the workspace venv Python. Writes validation.json here.
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
from sim_env import BUILTINS, TabletopEnv

OUT = Path(__file__).resolve().parent
LAB = Path(arm.__file__).resolve().parent
TICKS = 180
SEEDS = range(10)
ASSETS = [*BUILTINS, "imported/lux3d-ivory"]
ASSET_PATHS = {**{k: k for k in BUILTINS},
               "imported/lux3d-ivory": str(LAB / "assets" / "imported" / "lux3d-ivory" / "manifest.json")}


def rollout(asset, seed, open_only):
    env = TabletopEnv(ASSET_PATHS[asset], seed)
    try:
        z0 = env.metrics()["object_xyz"][2]
        peak, success_at = 0.0, None
        for tick in range(1, TICKS + 1):
            action = env.expert_action()
            if open_only:
                action[-1] = 1
            env.step(action)
            m = env.metrics()
            peak = max(peak, m["object_xyz"][2] - z0)
            if success_at is None and m["success"]:
                success_at = tick
        return {"success_tick": success_at, "max_rise_mm": round(1000 * peak, 1),
                "lifted": bool(m["lifted"]), "grasped": bool(m["grasped"]),
                "final_object_xyz": m["object_xyz"]}
    finally:
        env.close()


ARM_BODIES = {"pedestal", "turret", "upper_arm", "forearm", "wrist", "tool", "eef", "left", "right"}


def arm_self_contacts(env):
    hits = []
    for c in env.data.contact:
        if c.dist >= 0:
            continue
        b1 = env.model.geom_bodyid[int(c.geom1)]
        b2 = env.model.geom_bodyid[int(c.geom2)]
        n1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, b1)
        n2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, b2)
        if n1 in ARM_BODIES and n2 in ARM_BODIES:
            hits.append((n1, n2, round(1000 * float(c.dist), 3)))
    return hits


def self_collision_checks():
    env = TabletopEnv("standard", 0)
    try:
        rest = env.data.qpos[:5].copy()
        ranges = env.model.jnt_range[:5]

        def sweep(xs, ys, zs):
            contacts, unreachable, violations, errors = [], [], [], []
            for x in xs:
                for y in ys:
                    for z in zs:
                        q = np.asarray(arm.ik([x, y, z]))
                        radial, h = np.hypot(x, y - arm.BASE_Y), z + arm.TOOL - arm.SHOULDER_Z
                        if np.hypot(radial, h) > arm.L1 + arm.L2 - .005:
                            unreachable.append([round(float(x), 3), round(float(y), 3), round(float(z), 3)])
                            continue
                        if np.any(q < ranges[:, 0] - 1e-9) or np.any(q > ranges[:, 1] + 1e-9):
                            violations.append([round(float(x), 3), round(float(y), 3), round(float(z), 3)])
                            continue
                        env.data.qpos[:5] = q
                        env.data.qvel[:] = 0
                        mujoco.mj_forward(env.model, env.data)
                        errors.append(float(np.linalg.norm(env.data.xpos[env._eef] - [x, y, z])))
                        for pair in arm_self_contacts(env):
                            contacts.append({"target": [round(float(x), 3), round(float(y), 3), round(float(z), 3)],
                                             "pair": pair})
            return {"self_contacts": contacts, "unreachable": unreachable,
                    "joint_limit_violations": violations,
                    "max_kinematic_ik_error_mm": round(1000 * max(errors), 4) if errors else None}

        # 1) Task operating region (spawn area, tray, transit corridor): must be clean.
        task_region = sweep(np.linspace(-.16, .14, 7), np.linspace(-.09, .06, 6), np.linspace(.02, .18, 6))
        # 2) Full action clamp box: near-base fold dead zones are legitimate mechanical
        #    stops; record them as documentation, not failures.
        full_box = sweep(np.linspace(-.28, .28, 9), np.linspace(-.17, .17, 7), np.linspace(.022, .184, 6))
        # 3) Mechanical stops: extreme poses outside the IK envelope MUST collide
        #    (non-adjacent link constraints stay active).
        env.data.qpos[:5] = [rest[0], rest[1], 3.08, rest[3], rest[4]]
        env.data.qvel[:] = 0
        mujoco.mj_forward(env.model, env.data)
        elbow_stop = arm_self_contacts(env)
        wrist_sweep_contacts = []
        for wp in np.linspace(-3.4, 1.6, 40):
            env.data.qpos[:5] = [rest[0], rest[1], rest[2], wp, rest[4]]
            mujoco.mj_forward(env.model, env.data)
            for pair in arm_self_contacts(env):
                wrist_sweep_contacts.append({"wrist_pitch": round(float(wp), 3), "pair": pair})
        return {"task_region": task_region, "full_box_dead_zones": full_box,
                "mechanical_stops": {"elbow_full_fold": elbow_stop},
                "mechanical_stops_active": bool(elbow_stop),
                "wrist_forearm_clear": not wrist_sweep_contacts,
                "wrist_pitch_sweep_contacts": wrist_sweep_contacts}
    finally:
        env.close()


def main():
    report = {"phase": "A", "date": "2026-11-09",
              "code_sha256": {n: hashlib.sha256((LAB / n).read_bytes()).hexdigest()
                              for n in ("arm.py", "sim_env.py", "server.py")},
              "asset_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted((LAB / "assets" / "robot").glob("*")) if p.is_file()},
              "physics": json.loads((LAB / "assets" / "robot" / "physics.json").read_text()),
              "mujoco_version": mujoco.__version__, "ticks_per_run": TICKS, "seeds": list(SEEDS),
              "controls": {"baseline": "privileged scripted expert (reads initial object pose)",
                           "open_control": "identical XYZ path, jaws commanded open every tick"},
              "results": {}}
    for asset in ASSETS:
        base = [rollout(asset, s, False) for s in SEEDS]
        neg = [rollout(asset, s, True) for s in SEEDS]
        report["results"][asset] = {
            "baseline_successes": sum(r["success_tick"] is not None for r in base),
            "baseline_by_seed": base,
            "open_control_successes": sum(r["success_tick"] is not None for r in neg),
            "open_control_lifts": sum(r["lifted"] for r in neg),
            "open_control_by_seed": neg}
        print(asset, "baseline", report["results"][asset]["baseline_successes"], "/10, open",
              report["results"][asset]["open_control_successes"], "/10", flush=True)
    report["self_collision"] = self_collision_checks()
    env = TabletopEnv("standard", 0)
    try:
        for tick in range(TICKS):
            env.step(env.expert_action())
            if tick + 1 in (60, 86, 151):
                Image.fromarray(env.render(800, 540)).save(OUT / f"grasp-tick{tick + 1:03d}.jpg")
    finally:
        env.close()
    (OUT / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    total = sum(r["baseline_successes"] for r in report["results"].values())
    neg_total = sum(r["open_control_successes"] for r in report["results"].values())
    print(json.dumps({"baseline_total": f"{total}/40", "open_control_total": f"{neg_total}/40",
                      "task_region_clean": not report["self_collision"]["task_region"]["self_contacts"],
                      "mechanical_stops_active": report["self_collision"]["mechanical_stops_active"]}))


if __name__ == "__main__":
    main()