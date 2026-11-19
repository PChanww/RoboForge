"""Wheel-contact locomotion, physical stops and negative controls for the mobile demo."""
from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))

from vlab.duck_demo import CHASSIS
from vlab.store import ConfigStore


@pytest.fixture
def mobile():
    from vlab.mobile import MobileEnv

    base = ConfigStore(LAB / "projects").load()
    before = base.model_dump_json()
    env = MobileEnv(base, LAB, CHASSIS)
    assert base.model_dump_json() == before, "The mobile demo must not mutate the default project"
    try:
        yield env
    finally:
        env.close()


def free_pose(env):
    indices = np.flatnonzero(env.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    assert len(indices) == 1
    adr = int(env.model.jnt_qposadr[indices[0]])
    return env.data.qpos[adr:adr + 7].copy()


def yaw(env):
    qw, qx, qy, qz = free_pose(env)[3:]
    return float(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))


def drive(env, v, w, seconds):
    for _ in range(round(seconds / .05)):
        env.step(v, w, dt=.05)
        assert np.isfinite(env.data.qpos).all()
        assert np.isfinite(env.data.qvel).all()


def wheel_geom_ids(env):
    wheel_bodies = []
    for jid in range(env.model.njnt):
        name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
        if "wheel" in name and env.model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE:
            wheel_bodies.append(int(env.model.jnt_bodyid[jid]))
    assert len(wheel_bodies) == 4, "Four separately driven wheel joints are required"
    return np.flatnonzero(np.isin(env.model.geom_bodyid, wheel_bodies))


def test_mobile_is_a_free_body_with_four_physical_wheels(mobile):
    pose = free_pose(mobile)
    assert np.linalg.norm(pose[:2]) < .02
    assert mobile.model.neq == 0, "No kinematic attachment or weld may carry the chassis"
    ids = wheel_geom_ids(mobile)
    assert np.any(mobile.model.geom_contype[ids] != 0)
    state = mobile.state()
    assert np.asarray(state["base_pose"]).shape == (7,)
    assert np.asarray(state["wheel_qpos"]).shape == (4,)
    assert np.asarray(state["wheel_qvel"]).shape == (4,)
    assert all(np.isfinite(state[key]).all() for key in ("base_pose", "wheel_qpos", "wheel_qvel"))


@pytest.mark.parametrize("direction", [1, -1])
def test_mobile_forward_and_reverse_are_real_translation(mobile, direction):
    start = free_pose(mobile)
    drive(mobile, direction * .15, 0, 2)
    delta = free_pose(mobile)[:2] - start[:2]
    assert direction * delta[0] > .12, delta
    assert abs(delta[1]) < .06, delta
    assert abs(yaw(mobile)) < .3
    assert np.max(np.abs(mobile.state()["wheel_qpos"])) > 1


def test_disabling_wheel_contact_prevents_motor_translation(mobile):
    ids = wheel_geom_ids(mobile)
    mobile.model.geom_contype[ids] = 0
    mobile.model.geom_conaffinity[ids] = 0
    # Let the now unsupported chassis settle before measuring motor translation.
    drive(mobile, 0, 0, 1)
    start = free_pose(mobile)
    drive(mobile, .15, 0, 2)
    delta = free_pose(mobile)[:2] - start[:2]
    assert np.linalg.norm(delta) < .025, "Wheel rotation alone must not teleport the base"
    assert np.max(np.abs(mobile.state()["wheel_qpos"])) > 1, "Negative control still runs the wheel motors"


@pytest.mark.parametrize("direction", [1, -1])
def test_mobile_turns_both_directions(mobile, direction):
    start = free_pose(mobile)
    drive(mobile, 0, direction * .9, 1.5)
    assert direction * yaw(mobile) > .3, yaw(mobile)
    assert np.linalg.norm(free_pose(mobile)[:2] - start[:2]) < .1


def test_zero_command_brakes_without_instant_position_change(mobile):
    drive(mobile, .18, 0, 1.5)
    start = free_pose(mobile)
    before = np.linalg.norm(mobile.data.qvel[:2])
    assert before > .08
    mobile.step(0, 0, dt=.05)
    after_one = free_pose(mobile)
    assert 0 < after_one[0] - start[0] < .02, "A moving body must brake over physical steps"
    drive(mobile, 0, 0, 1.5)
    assert np.linalg.norm(mobile.data.qvel[:2]) < .015
    assert .003 < free_pose(mobile)[0] - start[0] < .15


def test_wall_blocks_continuously_driven_chassis(mobile):
    # A motor command held for much longer than the crossing time must end in
    # contact with a physical wall rather than taking the chassis outside it.
    static_boxes = []
    for gid in range(mobile.model.ngeom):
        name = mujoco.mj_id2name(mobile.model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if ("wall" in name or "boundary" in name) and mobile.model.geom_bodyid[gid] == 0:
            if mobile.model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX:
                static_boxes.append(gid)
    forward_walls = [gid for gid in static_boxes if mobile.model.geom_pos[gid, 0] > .3]
    assert forward_walls, "The mobile arena needs a real positive-X boundary"
    wall = min(forward_walls, key=lambda gid: mobile.model.geom_pos[gid, 0])
    interior_edge = float(mobile.model.geom_pos[wall, 0] - mobile.model.geom_size[wall, 0])
    hit_wall = False
    for _ in range(round((interior_edge / .15 + 4) / .05)):
        mobile.step(.22, 0, dt=.05)
        assert np.isfinite(mobile.data.qpos).all()
        assert free_pose(mobile)[0] < interior_edge, "The chassis crossed the physical boundary"
        for c in mobile.data.contact:
            if wall in (int(c.geom1), int(c.geom2)):
                hit_wall = True
                assert c.dist > -.012, "The wall allowed deep penetration"
    assert hit_wall, "The test must actually reach and contact the wall"


def test_long_mixed_motion_keeps_finite_upright_pose(mobile):
    for _ in range(3):
        for v, w, duration in ((.12, .35, 1.5), (-.1, -.5, 1), (0, .8, 1), (0, 0, .5)):
            drive(mobile, v, w, duration)
            p = free_pose(mobile)
            assert np.linalg.norm(p[3:]) == pytest.approx(1, abs=1e-5)
            qw, qx, qy, qz = p[3:]
            upright = 1 - 2 * (qx * qx + qy * qy)
            assert upright > .85, "The mobile chassis tipped instead of driving"
