"""P0-2 regression: config-driven FK/IK consistency + grasp regression."""
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
from vlab.engine import PlatformEnv
from vlab.schema import ProjectConfig


def cfg():
    return ProjectConfig.model_validate_json((LAB / "configs" / "default.json").read_text(encoding="utf-8"))


def main():
    out = {}
    env = PlatformEnv(cfg(), LAB)
    rep = env.validate_kinematics()
    out["default_consistent"] = rep["consistent"]
    out["default_max_fk_err_mm"] = max(r["fk_error_mm"] for r in rep["samples"] if r.get("reachable"))
    out["unreachable_reported"] = any(not r["reachable"] for r in rep["samples"])
    env.close()
    assert out["default_consistent"] and out["unreachable_reported"]

    # lengthen forearm by 50mm -> IK must follow the NEW model
    c2 = cfg()
    next(l for l in c2.robots[0].links if l.name == "forearm").origin[2] += .05
    env2 = PlatformEnv(c2, LAB)
    rep2 = env2.validate_kinematics()
    out["forearm_plus50mm_consistent"] = rep2["consistent"]
    out["forearm_params_l1"] = rep2["params"]["l1"]
    env2.close()
    assert rep2["consistent"] and abs(rep2["params"]["l1"] - .28) < 1e-6

    # tool offset change via the eef link origin (the model transform)
    c3 = cfg()
    next(l for l in c3.robots[0].links if l.name == c3.robots[0].end_effector).origin[2] -= .02
    env3 = PlatformEnv(c3, LAB)
    rep3 = env3.validate_kinematics()
    out["tool_offset_consistent"] = rep3["consistent"]
    env3.close()
    assert rep3["consistent"]

    # grasp regression with default config
    env = PlatformEnv(cfg(), LAB)
    env.reset(0)
    succ = False
    for _ in range(200):
        env.step(env.servo_expert_action())
        if env.metrics()["success"]:
            succ = True
            break
    out["servo_grasp_regression"] = succ
    env.reset(0)
    neg = False
    for _ in range(200):
        a = env.servo_expert_action()
        a[-1] = 1  # jaws always open
        env.step(a)
        if env.metrics()["success"]:
            neg = True
            break
    out["open_jaw_negative"] = neg
    env.close()
    assert succ and not neg
    print(out)


if __name__ == "__main__":
    main()