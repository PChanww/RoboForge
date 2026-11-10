"""Verify the second robot (3-DOF prismatic gantry) through the generic interface.

Asserts the config describes it, the env builds, observation/action dims come from
config (no fixed 5/7 assumptions), and a scripted joint_position pick runs.
"""
import json
import sys
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
from vlab.engine import PlatformEnv
from vlab.schema import ProjectConfig
from vlab.store import validate_config


def main():
    cfg = ProjectConfig.model_validate_json((LAB / "configs" / "second_robot.json").read_text(encoding="utf-8"))
    issues = validate_config(cfg, LAB)
    assert not issues, issues
    env = PlatformEnv(cfg, LAB)
    obs = env.observation()
    joints = list(obs["state"]["joint_pos"].keys())
    assert joints == ["x", "y", "z", "left", "right"], joints
    assert env.controller.action_dim == 4 and env.controller.mode == "joint_position"
    assert env.supported_modes() == ["joint_position"], env.supported_modes()
    cup = np.asarray(env._judge.start_xyz)
    def act(x, y, z, grip):
        return [x, y, z, grip]
    # Move above the cup, descend, close, lift, carry to tray, lower, open.
    above = (cup[0], cup[1] - env.robot.links[0].origin[1], cup[2] - .09 + .06)
    grab = (cup[0], cup[1] - env.robot.links[0].origin[1], cup[2] - .09)
    goal = cfg.task.goal_xy
    carry = (goal[0], goal[1] - env.robot.links[0].origin[1], cup[2] - .09 + .06)
    drop = (goal[0], goal[1] - env.robot.links[0].origin[1], cfg.task.goal_surface_m + .043 - .09)
    phases = [(25, above, .058), (50, grab, .058), (70, grab, .006), (95, carry, .006),
              (125, drop, .006), (145, drop, .058)]
    for end, (x, y, z), grip in phases:
        while env.scheduler.tick < end:
            env.step(act(x, y, z, grip))
    m = env.metrics()
    env.close()
    result = {"robot": cfg.robots[0].id, "joints": joints, "action_dim": env.controller.action_dim,
              "supported_modes": ["joint_position"], "grasped": m["grasped"], "lifted": m["lifted"],
              "success": m["success"], "tick": m["tick"]}
    (LAB / "datasets" / "reports").mkdir(parents=True, exist_ok=True)
    (LAB / "datasets" / "reports" / "second_robot.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()