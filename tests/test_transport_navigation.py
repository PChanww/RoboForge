"""Arbitrary workcell delivery, contact clearance and effective user parameters."""
from pathlib import Path

import mujoco
import numpy as np
import pytest

from test_mobile_transport import PhysicsAudit
from vlab.store import ConfigStore
from vlab.transport import MobileTransportEnv
from vlab.transport_navigation import BOUNDS, BELT, CLEARANCE, START, plan_navigation, rect_distance


LAB = Path(__file__).resolve().parents[1]


def make_env(**spec):
    return MobileTransportEnv(ConfigStore(LAB / "projects").load(), LAB,
                              {"seed": 3, "speed_scale": 1.6, **spec}, render=False)


def test_builtin_bin_ignores_stale_imported_appearance_reference():
    env = make_env(target_xy=[.75, .65], bin_mode="builtin", bin_asset="imported/no-longer-present-bin")
    try:
        assert env.bin_metadata["visual_asset"] is None
        assert mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "transport_bin_visual") == -1
        assert all(env.model.geom(f"bin_wall_{index}").contype[0] != 0 for index in range(4))
    finally:
        env.close()


def test_transport_uses_20_hz_without_changing_saved_project_frequency():
    base = ConfigStore(LAB / "projects").load()
    base.controllers[0].frequency_hz = 10.
    env = MobileTransportEnv(base, LAB, {"seed": 3}, render=False)
    try:
        assert base.controllers[0].frequency_hz == 10.
        assert env.config.controllers[0].frequency_hz == 20.
        assert env.scheduler.control_period == pytest.approx(.05)
        before = env.episode_time()
        env.step(env.expert_action())
        assert env.episode_time()-before == pytest.approx(.05, abs=1e-10)
    finally:
        env.close()


class NavigationAudit(PhysicsAudit):
    """Keep the prior free-body/bin audit and also inspect the conveyor frame."""
    def __init__(self, env, monkeypatch):
        super().__init__(env, monkeypatch)
        self.frame = env.model.geom("conveyor_frame_contact").id
        self.integrate = self.original_step
        self.original_step = self.contact_step

    def contact_step(self, model, data):
        self.integrate(model, data)
        for contact in data.contact:
            pair = {int(contact.geom1), int(contact.geom2)}
            if self.frame in pair:
                assert not pair & self.base_geoms, "The base/wheels struck the conveyor frame"


def rollout(env, limit=1800):
    peak_speed = 0.
    phases = set()
    nav_ticks = 0
    for tick in range(limit):
        env.step(env.expert_action())
        metrics = env.metrics()
        phase = metrics["phase"]
        phases.add(phase)
        if phase == 7:
            nav_ticks += 1
            assert metrics["contacts"] == 2, "The robot travelled after losing bilateral object contact"
            peak_speed = max(peak_speed, float(np.linalg.norm(env.state()["base_velocity"][:2])))
        if tick >= 4:
            assert metrics["belt_speed_m_s"] > .02, "The conveyor stopped during delivery"
        if metrics["success"] or metrics["failed"]:
            break
    metrics = env.metrics()
    assert metrics["success"], metrics
    assert metrics["controlled_release"] and not metrics["delivery_failed"], metrics
    assert metrics["contacts"] == 0 and metrics["tray_contact"] and metrics["stable_ticks"] >= 6
    assert nav_ticks > 20
    return tick+1, peak_speed, phases


@pytest.mark.parametrize("target", [[.75, .65], [-.8, .7], [.7, -.35]])
def test_contact_only_navigation_reaches_multiple_directions(target, monkeypatch):
    env = make_env(target_xy=target)
    try:
        audit = NavigationAudit(env, monkeypatch)
        ticks, peak, phases = rollout(env)
        assert audit.physics_ticks == ticks*25
        assert {5, 6, 7, 8, 9, 10, 11, 12}.issubset(phases)
        assert env.metrics()["goal_distance_m"] < .06
        low, high = env._judge.bounds()
        half = np.asarray(env.bin_metadata["inner_half_m"])
        assert np.all(low[:2] > np.asarray(target)-half)
        assert np.all(high[:2] < np.asarray(target)+half)
        assert env.metrics()["travelled_distance_m"] > .4
        assert peak > .03
        snapshot = env.snapshot(False)
        assert snapshot["target_xy"] == target
        assert len(snapshot["planned_route"]) >= 2
        assert snapshot["trajectory"]
    finally:
        env.close()


def test_double_speed_completes_sooner_with_real_contact_transport():
    results = []
    for speed in (1., 2.):
        env = make_env(speed_scale=speed)
        try:
            ticks, peak, _ = rollout(env)
            results.append((ticks, peak))
            assert env.metrics()["retreated_distance_m"] > .37
            assert env.metrics()["travelled_distance_m"] > .37
        finally:
            env.close()
    assert results[1][0] < results[0][0]*.9, results
    assert results[1][1] > results[0][1]*1.3, results


def test_user_object_dimensions_and_mass_are_used_during_navigation(monkeypatch):
    parameters = {"height_m": .05, "width_m": .045, "depth_m": .05, "mass_kg": .04}
    env = make_env(target_xy=[.75, .65], object_parameters=parameters)
    try:
        audit = NavigationAudit(env, monkeypatch)
        rollout(env)
        for key, value in parameters.items():
            assert env.object_parameters["parameters"][key] == pytest.approx(value)
        assert env.model.body("cup").mass[0] == pytest.approx(.04)
        assert env._asset_height == pytest.approx(.05)
        assert audit.physics_ticks > 5000
    finally:
        env.close()


def test_custom_bin_size_changes_inside_judgement_without_wall_strikes(monkeypatch):
    target = [-.8, .7]
    env = make_env(target_xy=target, bin_size_m=[.26, .24, .07], bin_color="#CC6633")
    try:
        audit = NavigationAudit(env, monkeypatch)
        rollout(env)
        np.testing.assert_allclose(env.bin_metadata["inner_half_m"], [.124, .114])
        assert env.bin_metadata["rim_z_m"] == pytest.approx(.079)
        assert env._judge.inside()
        assert audit.physics_ticks > 5000
    finally:
        env.close()


def test_front_destination_path_goes_around_conveyor_with_full_clearance():
    target = [.7, -.35]
    plan = plan_navigation(target)
    assert plan["ready"], plan
    route = np.asarray(plan["planned_route"])
    np.testing.assert_allclose(route[0], START)
    assert len(route) >= 3, "A straight route would cut through the conveyor clearance"
    # The raw line is unsafe even though the necessary detour is short.
    direct = np.linspace(START, route[-1], 1000)
    assert min(rect_distance(point, BELT) for point in direct) < CLEARANCE
    assert plan["length_m"] > np.linalg.norm(route[-1]-START)
    for a, b in zip(route, route[1:]):
        count = max(2, int(np.linalg.norm(b-a)/.002)+1)
        for point in np.linspace(a, b, count):
            assert all(rect_distance(point, obstacle) >= CLEARANCE-.001 for obstacle in plan["obstacles"])
            assert BOUNDS[0]+CLEARANCE <= point[0] <= BOUNDS[1]-CLEARANCE
            assert BOUNDS[2]+CLEARANCE <= point[1] <= BOUNDS[3]-CLEARANCE


def test_navigation_final_heading_faces_box_with_negative_y_robot_forward():
    plan = plan_navigation([-.34, .28])
    assert plan["ready"]
    assert plan["heading_policy"] == "face_bin_before_placement"
    yaw = plan["final_yaw"]
    forward = np.array([np.sin(yaw), -np.cos(yaw)])
    to_box = np.asarray(plan["target_xy"])-plan["dock_xy"]
    np.testing.assert_allclose(forward, to_box/np.linalg.norm(to_box), atol=1e-12)


@pytest.mark.parametrize("target,size", [
    ([0., -.07], [.21, .22, .056]),  # Conveyor overlap.
    ([0., .21], [.21, .22, .056]),  # Initial robot workspace.
    ([1.15, 1.], [.21, .22, .056]),  # Bin protrudes outside arena.
    ([.7, .7], [.31, .22, .056]),
    ([float("nan"), .7], [.21, .22, .056]),
])
def test_unusable_bin_positions_or_dimensions_have_no_route(target, size):
    plan = plan_navigation(target, size)
    assert not plan["ready"]
    assert plan["issues"]
    assert not plan.get("planned_route")
