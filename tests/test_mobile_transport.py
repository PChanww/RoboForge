"""Contact-only conveyor pickup, wheel-driven retreat and physical bin delivery."""
from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))

from vlab.duck_demo import CHASSIS, DUCK
from vlab.store import ConfigStore


RETREAT_M = .4
MAX_TICKS = 1200


def make_env(seed=3):
    from vlab.transport import MobileTransportEnv

    base = ConfigStore(LAB / "projects").load()
    before = base.model_dump_json()
    spec = {
        "instruction": "从不停的传送带上抓取塑料小鸭，向后移动40厘米，放进后方的箱子。",
        "object_asset": DUCK,
        "chassis_asset": CHASSIS,
        "retreat_distance_m": RETREAT_M,
        "belt_speed_m_s": .025,
        "random_pose": True,
        "seed": seed,
    }
    env = MobileTransportEnv(base, LAB, spec, render=False)
    assert base.model_dump_json() == before, "The task must not modify the default project"
    return env


def free_pose(env, name):
    address = int(env.model.joint(name).qposadr[0])
    return env.data.qpos[address:address + 7].copy()


def wheel_geom_ids(env):
    bodies = []
    for jid in range(env.model.njnt):
        name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
        if "wheel" in name and env.model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE:
            bodies.append(int(env.model.jnt_bodyid[jid]))
    assert len(bodies) == 4, "Retreat needs four physical wheel joints"
    return np.flatnonzero(np.isin(env.model.geom_bodyid, bodies))


class PhysicsAudit:
    """Observe every physics tick, including calls batched with nstep.

    Checking the input against the previous output catches pose writes between
    mj_step calls as well as impossible jumps during a physics tick.
    """

    def __init__(self, env, monkeypatch):
        self.env = env
        self.original_step = mujoco.mj_step
        self.addresses = [int(env.model.joint(name).qposadr[0])
                          for name in ("cup_free", "transport_free")]
        self.previous = self.poses()
        self.walls = {env.model.geom(f"bin_wall_{i}").id for i in range(4)}
        self.pads = {env.model.geom(name).id for name in ("left_pad", "right_pad")}
        self.object_geom = env.model.geom("cup_contact").id
        root = int(env.model.jnt_bodyid[env.model.joint("transport_free").id])
        base_bodies = {root}
        # Direct chassis and wheel bodies are the mobile collision envelope;
        # arm links are allowed to pass above the bin while placing an object.
        for bid in range(env.model.nbody):
            name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if int(env.model.body_parentid[bid]) == root and ("wheel" in name or "chassis" in name):
                base_bodies.add(bid)
        self.base_geoms = set(np.flatnonzero(np.isin(env.model.geom_bodyid, list(base_bodies))))
        self.physics_ticks = 0
        monkeypatch.setattr(mujoco, "mj_step", self.step)

    def poses(self):
        return np.array([self.env.data.qpos[address:address + 7].copy()
                         for address in self.addresses])

    def step(self, model, data, *args, **kwargs):
        assert model is self.env.model and data is self.env.data
        np.testing.assert_array_equal(
            self.poses(), self.previous,
            err_msg="A free-body pose was changed outside MuJoCo integration",
        )
        count = int(kwargs.pop("nstep", args[0] if args else 1))
        assert not kwargs and len(args) <= 1
        for _ in range(count):
            self.original_step(model, data)
            current = self.poses()
            displacement = np.linalg.norm(current[:, :3] - self.previous[:, :3], axis=1)
            assert np.max(displacement) < .005, "A free body jumped within a 2 ms physics tick"
            assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
            for contact in data.contact:
                pair = {int(contact.geom1), int(contact.geom2)}
                if pair & self.walls:
                    assert not pair & self.base_geoms, "The chassis collided with the destination bin"
                    assert not pair & self.pads, "A finger struck the destination bin wall"
                    if self.object_geom in pair:
                        assert contact.dist > -.001, "The object penetrated the bin wall"
            self.previous = current
            self.physics_ticks += 1


@pytest.mark.parametrize("seed", [3, 4, 5])
def test_real_pick_retreat_and_stable_bin_release(seed, monkeypatch):
    env = make_env(seed)
    try:
        initial = free_pose(env, "transport_free")
        assert env.model.neq == 0, "No weld or grasp attachment may carry the object"
        free_joints = np.flatnonzero(env.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        assert len(free_joints) == 2, "Both the object and chassis must remain dynamic free bodies"
        wheel_geom_ids(env)
        audit = PhysicsAudit(env, monkeypatch)
        carried_ticks, retreat_ticks, missing_contact_ticks = 0, 0, 0
        for tick in range(MAX_TICKS):
            action = env.expert_action()
            assert isinstance(action, dict)
            env.step(action)
            metrics = env.metrics()
            phase = int(metrics["phase"] if "phase" in metrics else metrics["task_phase"])
            if tick > 3:
                assert metrics["belt_speed_m_s"] > .02, "The belt stopped during the task"
            if 6 <= phase <= 10:
                carried_ticks += 1
                missing_contact_ticks = missing_contact_ticks + 1 if metrics["contacts"] < 2 else 0
                assert missing_contact_ticks < 3, "The object was carried after sustained loss of grip"
                assert not metrics["delivery_failed"], metrics
            if phase == 7:
                retreat_ticks += 1
                assert metrics["contacts"] == 2, "Rearward travel requires bilateral finger contact"
            if metrics["success"] or metrics["delivery_failed"]:
                break
        metrics = env.metrics()
        assert metrics["success"] and metrics["grasped"] and metrics["lifted"], metrics
        assert metrics["controlled_release"] and not metrics["delivery_failed"], metrics
        assert metrics["contacts"] == 0 and metrics["tray_contact"], metrics
        assert metrics["stable_ticks"] >= 6
        actual_retreat = free_pose(env, "transport_free")[1] - initial[1]
        assert actual_retreat >= RETREAT_M - .03, (actual_retreat, metrics)
        assert metrics["retreated_distance_m"] >= RETREAT_M - .03, metrics
        assert carried_ticks > 20 and retreat_ticks > 20
        assert metrics["belt_travel_m"] > .08
        assert audit.physics_ticks > 1000
        state = env.state()
        for key, shape in (("base_pose", (7,)), ("wheel_qpos", (4,)), ("wheel_qvel", (4,))):
            assert np.asarray(state[key]).shape == shape
            assert np.isfinite(state[key]).all()
        base = free_pose(env, "transport_free")
        assert np.linalg.norm(base[3:]) == pytest.approx(1, abs=1e-5)
        assert 1 - 2 * (base[4] ** 2 + base[5] ** 2) > .85
        snapshot = env.snapshot(render=False)
        assert isinstance(snapshot, dict)
    finally:
        env.close()


def test_disabled_finger_contacts_prevent_pick_and_retreat():
    env = make_env()
    try:
        start = free_pose(env, "transport_free")
        for name in ("left_pad", "right_pad"):
            gid = env.model.geom(name).id
            env.model.geom_contype[gid] = env.model.geom_conaffinity[gid] = 0
        for _ in range(MAX_TICKS):
            env.step(env.expert_action())
            assert np.isfinite(env.data.qpos).all() and np.isfinite(env.data.qvel).all()
            metrics = env.metrics()
            assert not any(metrics[key] for key in ("grasped", "lifted", "success")), metrics
            if metrics["delivery_failed"]:
                break
        assert np.linalg.norm(free_pose(env, "transport_free")[:2] - start[:2]) < .03
    finally:
        env.close()


def test_disabled_wheel_contacts_cannot_deliver_to_rear_bin():
    env = make_env()
    try:
        start = free_pose(env, "transport_free")
        ids = wheel_geom_ids(env)
        env.model.geom_contype[ids] = env.model.geom_conaffinity[ids] = 0
        for _ in range(MAX_TICKS):
            env.step(env.expert_action())
            assert np.isfinite(env.data.qpos).all() and np.isfinite(env.data.qvel).all()
            metrics = env.metrics()
            assert not metrics["success"], "Wheel motors cannot transport a chassis without ground contact"
            if metrics["delivery_failed"]:
                break
        actual_retreat = free_pose(env, "transport_free")[1] - start[1]
        assert actual_retreat < RETREAT_M - .1, actual_retreat
    finally:
        env.close()


def test_random_pose_seeds_give_finite_different_objects():
    poses = []
    for seed in (3, 4, 5):
        env = make_env(seed)
        try:
            pose = free_pose(env, "cup_free")
            assert np.isfinite(pose).all()
            assert np.linalg.norm(pose[3:]) == pytest.approx(1, abs=1e-5)
            poses.append(pose)
        finally:
            env.close()
    assert np.max(np.ptp(np.asarray(poses)[:, :2], axis=0)) > .003
    assert np.max(np.ptp(np.asarray(poses)[:, 3:], axis=0)) > .03
