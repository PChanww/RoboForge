"""Physical expert regression: smooth motion, supported release, real grasp.

Run directly with python tests/test_servo_motion.py. Cameras are disabled here;
the same MuJoCo dynamics and actuator commands run without rendering overhead.
"""
import json
import sys
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))

from vlab.engine import PlatformEnv
from vlab.schema import ProjectConfig


def rollout(asset, seed, always_open=False):
    cfg = ProjectConfig.model_validate_json((LAB / "configs/default.json").read_text(encoding="utf-8"))
    cfg.sensors = []
    cfg.task.asset = asset
    env = PlatformEnv(cfg, LAB)
    try:
        env.reset(seed)
        dt = env.scheduler.control_period
        positions, jaw_positions, jaw_targets = [], [], []
        supported_release, contact_ticks = False, 0
        for _ in range(200):
            phase = env._servo_phase
            contact_ticks = contact_ticks + 1 if env._judge.contacts == 2 else 0
            action = env.servo_expert_action()
            if phase == 2 and env._servo_phase == 3:
                assert contact_ticks * dt >= .25 - 1e-9, "lift before sustained bilateral contact"
            if phase == 5 and env._servo_phase == 6:
                assert env._judge.tray_contact, "release before the tray supports the cup"
                supported_release = True
            if always_open:
                action[-1] = 1
            env.step(action)
            positions.append(env.data.xpos[env._eef].copy())
            jaw_positions.append(env.data.qpos[env._qposadr[env.robot.gripper.left_joint]])
            jaw_targets.append(env._grip_target)
            if env.metrics()["success"]:
                break
        if always_open:
            assert not env.metrics()["success"] and not env.metrics()["lifted"]
            return {"always_open": "no lift or success"}
        assert env.metrics()["success"] and supported_release, (asset, seed, env.metrics())
        assert env.episode_time() <= cfg.task.time_limit_s, "expert exceeded the task budget"
        assert env._servo_hold_width > env._closed + .005, "grasp drives jaws to the closed stop"
        velocity = np.diff(positions, axis=0) / dt
        peak_acceleration = float(np.max(np.linalg.norm(np.diff(velocity, axis=0) / dt, axis=1)))
        peak_jaw_speed = float(np.max(np.abs(np.diff(jaw_positions)) / dt))
        assert peak_acceleration < 1.0, peak_acceleration
        assert peak_jaw_speed < .08, peak_jaw_speed
        assert np.max(np.abs(np.diff(jaw_targets))) <= .065 * dt + 1e-9
        return {"asset": asset, "seed": seed, "ticks": env.scheduler.tick,
                "peak_acceleration_m_s2": round(peak_acceleration, 3),
                "peak_jaw_speed_m_s": round(peak_jaw_speed, 3)}
    finally:
        env.close()


def main():
    results = [rollout(asset, seed) for asset in ("standard", "tall", "wide") for seed in (0, 1, 42)]
    results.append(rollout("standard", 0, always_open=True))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
