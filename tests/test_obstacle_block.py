"""T12: obstacle blocking probe.

A static wall collider is placed between the starting gripper and the cup. The
closed-loop servo expert keeps commanding the grasp; the wall must make real
contact with the robot while the task success stays false and the cup stays put.
"""
import json
import sys
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
from vlab.engine import PlatformEnv  # noqa: E402
from vlab.schema import ProjectConfig, SceneCollider  # noqa: E402

ROBOT_GEOMS = {"left_pad", "right_pad", "wrist", "forearm", "upper_arm", "tool", "eef", "turret"}


def blocked_config():
    c = ProjectConfig.model_validate_json((LAB / "configs" / "default.json").read_text(encoding="utf-8"))
    c.scene.colliders = [SceneCollider(name="block_wall", shape="box",
                                       full_size_m=[0.27, 0.012, 0.30],
                                       position=[-0.165, -0.015, 0.17])]
    return c


def run(ticks=220):
    env = PlatformEnv(blocked_config(), LAB)
    try:
        env.reset(0)
        cup0 = np.asarray(env.data.xpos[env._judge.cup]).copy()
        pairs, success, first_hit = set(), False, None
        for t in range(ticks):
            env.step(env.servo_expert_action())
            if env.metrics()["success"]:
                success = True
            for ct in env.data.contact:
                n1, n2 = env.model.geom(int(ct.geom1)).name, env.model.geom(int(ct.geom2)).name
                if "block_wall" in (n1, n2):
                    other = n2 if n1 == "block_wall" else n1
                    if other:
                        pairs.add(tuple(sorted(("block_wall", other))))
                        if other in ROBOT_GEOMS and first_hit is None:
                            first_hit = {"tick": t, "geom": other}
        cup = np.asarray(env.data.xpos[env._judge.cup])
        m = env.metrics()
        return {"robot_contacts": sorted(pairs), "first_robot_hit": first_hit,
                "success": bool(success), "cup_disp_m": round(float(np.linalg.norm(cup - cup0)), 4),
                "cup_lift_m": round(float(cup[2] - cup0[2]), 4),
                "final": {"tick": env.scheduler.tick, "eef": [round(v, 4) for v in env.data.xpos[env._eef]],
                          "object_xyz": [round(v, 4) for v in m["object_xyz"]],
                          "time_s": round(env.episode_time(), 3)}}
    finally:
        env.close()


def main():
    r = run()
    print(r)
    out = LAB / "outputs" / "phase2-render-validation" / "obstacle_block.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=1), encoding="utf-8")
    robot_hit = any(g in ROBOT_GEOMS for _, g in r["robot_contacts"])
    ok = robot_hit and r["first_robot_hit"] is not None and not r["success"] and abs(r["cup_lift_m"]) < 0.02
    print({"blocked_with_robot_contact": ok, "report": str(out)})
    raise SystemExit(0 if ok else 1)


def test_obstacle_blocks_expert():
    r = run()
    assert r["first_robot_hit"] is not None, r
    assert not r["success"], r
    assert abs(r["cup_lift_m"]) < 0.02, r


if __name__ == "__main__":
    main()
