"""Regress the two real generated assets that lost grip at a rear-left box.

The original failures were caused by a 2*pi IK command discontinuity after
navigation, rather than failed generation or an undersized gripping force.
"""
from pathlib import Path

import mujoco
import numpy as np
import pytest

from test_mobile_transport import PhysicsAudit
from vlab.store import ConfigStore
from vlab.transport import MobileTransportEnv


LAB = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("asset", ["transport-object-675da3", "transport-object-a32b14"])
@pytest.mark.parametrize("seed", [3, 4, 5])
def test_generated_object_survives_dock_alignment_and_releases_in_bin(asset, seed, monkeypatch):
    spec = {
        "object_asset": f"imported/{asset}", "seed": seed,
        "belt_speed_m_s": .029, "speed_scale": 1.8, "retreat_distance_m": .55,
        "random_pose": True, "target_xy": [-.34, .28],
        "object_parameters": {"height_m": .06, "width_m": .05, "depth_m": .055,
                              "mass_kg": .035, "friction": 1.2},
    }
    env = MobileTransportEnv(ConfigStore(LAB / "projects").load(), LAB, spec, render=False)
    try:
        assert env.model.neq == 0
        assert sum(env.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE) == 2
        audit = PhysicsAudit(env, monkeypatch)
        previous = np.array([env.data.ctrl[env._ctrl_index[n]] for n in env._joints[:5]])
        peak_joint_target_step = 0.
        phase8_ticks = 0
        for tick in range(1800):
            before = env.phase
            env.step(env.expert_action())
            current = np.array([env.data.ctrl[env._ctrl_index[n]] for n in env._joints[:5]])
            peak_joint_target_step = max(peak_joint_target_step, float(np.max(np.abs(current-previous))))
            assert np.max(np.abs(current-previous)) < .35, "IK demanded an abrupt joint rotation"
            previous = current
            if 6 <= env.phase <= 10:
                assert env._judge.contacts == 2, "Both fingers must retain contact while carrying and aligning"
            if env.phase == 8:
                phase8_ticks += 1
            if before == 8 and env.phase == 9:
                delta = np.asarray(env.config.task.goal_xy)-env.state()["base_pose"][:2]
                heading = np.arctan2(delta[1], delta[0])+np.pi/2
                error = np.arctan2(np.sin(heading-env._yaw()), np.cos(heading-env._yaw()))
                assert abs(error) < .03, "The robot extended behind its shoulder before facing the box"
                assert abs(env.state()["base_velocity"][5]) < .04
            assert 1-2*np.sum(env.state()["base_pose"][[4,5]]**2) > .98, "Base tipped while placing"
            metrics = env.metrics()
            if metrics["success"] or metrics["failed"]:
                break
        assert metrics["success"], metrics
        assert metrics["controlled_release"] and metrics["tray_contact"]
        assert metrics["stable_ticks"] >= 6 and metrics["contacts"] == 0
        assert not metrics["delivery_failed"]
        assert phase8_ticks > 20 and peak_joint_target_step < .35
        assert audit.physics_ticks == (tick+1)*25
    finally:
        env.close()
