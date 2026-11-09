"""Engine parity + multi-rate timing checks. Run: python -m tests.test_engine"""
import sys
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
from vlab.engine import PlatformEnv
from vlab.schema import ProjectConfig


def expert_rollout(env, cfg, ticks=180):
    MAX = .012
    src = env._judge.start_xyz
    h = env._asset_height
    goal = np.array(cfg.task.goal_xy)
    hover = min(.176, h / 2 + .09)
    grab = h / 2 + .002
    gs = cfg.task.goal_surface_m
    phases = ((24, src[:2], hover, 1), (44, src[:2], grab, 1), (60, src[:2], grab, -1),
              (86, src[:2], hover, -1), (126, goal, hover, -1), (151, goal, grab + gs, -1),
              (165, goal, grab + gs, 1))
    for tick in range(ticks):
        for end, xy, z, grip in phases:
            if tick < end:
                break
        else:
            xy, z, grip = goal, hover, 1
        delta = np.clip((np.r_[xy, z] - env._target) / MAX, -1, 1)
        env.step([*delta.tolist(), grip])
    return env.metrics()


def main():
    cfg = ProjectConfig.model_validate_json((LAB / "configs" / "default.json").read_text(encoding="utf-8"))
    env = PlatformEnv(cfg, LAB)
    obs = env.observation()
    assert list(obs["state"]["joint_pos"]) == ["waist", "shoulder", "elbow", "wrist_pitch", "wrist_yaw", "left", "right"]
    assert set(obs["images"]) == {"front", "wrist"}
    assert obs["schema_version"] == "1.0"
    m = expert_rollout(env, cfg)
    assert m["success"] and m["lifted"], m
    # 30 FPS cameras over 9 s of 20 Hz control => ~270 captures each, interval ~0.0333 s
    counts = env.scheduler.capture_count
    assert counts["front"] >= 260, counts
    intervals = env.scheduler.actual_intervals["front"]
    assert all(abs(iv - 1 / 30) < 0.004 for iv in intervals), intervals[:5]
    # action validation errors
    for bad in ([0, 0, 0], [0, 0, 0, 2], [float("nan"), 0, 0, 0]):
        try:
            env.validate_action(bad)
            raise AssertionError(f"accepted bad action {bad}")
        except ValueError:
            pass
    env.close()
    print(json_summary(m, counts, intervals))


def json_summary(m, counts, intervals):
    import json
    return json.dumps({"success": m["success"], "captures": counts,
                       "mean_interval_s": round(float(np.mean(intervals)), 5)})


if __name__ == "__main__":
    main()