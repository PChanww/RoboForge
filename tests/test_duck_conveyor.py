"""Real contact, varied pose, negative controls and collection data contract."""
from pathlib import Path
import sys

import numpy as np
import mujoco
import pytest

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
from vlab.collect import EpisodeRecorder
from vlab.duck_demo import make_config
from vlab.engine import PlatformEnv
from vlab.store import ConfigStore


def config(seed=0):
    base = ConfigStore(LAB / "projects").load()
    original = base.model_dump_json()
    cfg = make_config(base, LAB, seed)
    assert base.model_dump_json() == original
    cfg.sensors = []
    return cfg


@pytest.mark.parametrize("seed", range(10))
def test_contact_pick_and_stable_release(seed, monkeypatch):
    env = PlatformEnv(config(seed), LAB)
    try:
        start = env.data.xpos[env._judge.cup].copy()
        walls = {env.model.geom(f"bin_wall_{i}").id for i in range(4)}
        original_step = mujoco.mj_step
        adr = env.model.joint("cup_free").qposadr[0]
        last = env.data.qpos[adr:adr+3].copy()
        def audited_step(model, data, *args, **kwargs):
            nonlocal last
            original_step(model, data, *args, **kwargs)
            current = data.qpos[adr:adr+3].copy()
            assert np.linalg.norm(current-last) < .002, "discontinuous object motion within a 2 ms physics step"
            last = current
            for contact in data.contact:
                pair = {int(contact.geom1), int(contact.geom2)}
                assert not (pair & walls and pair & env._judge.pads), "finger struck box wall"
                if env._judge.geom in pair and pair & walls:
                    assert contact.dist > -.001, "object penetrated the bin wall"
        monkeypatch.setattr(mujoco, "mj_step", audited_step)
        held = 0
        for _ in range(500):
            phase = env._duck_task.phase
            action = env.servo_expert_action()
            if phase == 4 and env._duck_task.phase == 5:
                assert held >= 5, "lift must follow sustained bilateral contact"
            if phase == 7 and env._duck_task.phase == 8:
                assert env._judge.controlled_release and env._judge.inside()
                assert env._judge.bounds()[0][2] > .008
            env.step(action)
            if env.scheduler.tick > 3:
                assert env._duck_task.speed > .02, "belt paused during tracking, grasp or delivery"
            held = held + 1 if env._judge.contacts == 2 else 0
            if env.metrics()["success"]:
                break
        metrics = env.metrics()
        assert metrics["success"] and metrics["lifted"] and metrics["grasped"], metrics
        assert metrics["contacts"] == 0 and metrics["tray_contact"]
        assert metrics["controlled_release"] and not metrics["delivery_failed"]
        assert metrics["stable_ticks"] >= 6
        assert metrics["belt_travel_m"] > .04
        assert np.linalg.norm(env.data.xpos[env._judge.cup][:2] - start[:2]) > .25
        assert env.model.neq == 0, "no rigid weld or grasp attachment"
        rec = EpisodeRecorder(env)
        rec.capture()
        rec.set_action([0, 0, 0, 1])
        arrays = rec.arrays(True, env.config.task.asset, seed, "test")
        assert arrays["object_mat"].shape == (1, 3, 3)
        assert arrays["belt_speed_m_s"].shape == arrays["task_phase"].shape == (1,)
    finally:
        env.close()


def test_disabled_gripper_contacts_cannot_grasp_or_succeed():
    env = PlatformEnv(config(), LAB)
    try:
        for name in ("left_pad", "right_pad"):
            gid = env.model.geom(name).id
            env.model.geom_contype[gid] = env.model.geom_conaffinity[gid] = 0
        for _ in range(500):
            env.step(env.servo_expert_action())
        assert not any(env.metrics()[key] for key in ("grasped", "lifted", "success"))
    finally:
        env.close()


def test_roller_drive_transports_duck_not_visual_animation():
    positions = []
    for disabled in (False, True):
        env = PlatformEnv(config(), LAB)
        try:
            start = env.data.xpos[env._judge.cup, 0]
            if disabled:
                env.model.actuator_gainprm[env._duck_task.drives, 0] = 0
            for _ in range(20):
                env.step([0, 0, 0, 1])
            positions.append(env.data.xpos[env._judge.cup, 0] - start)
            assert env._duck_task.distance > .02  # visuals move in both cases
        finally:
            env.close()
    assert positions[0] > .015 and abs(positions[1]) < .005, positions
