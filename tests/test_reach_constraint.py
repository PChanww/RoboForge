"""Explicit reach handling: a millimetre overshoot must not kill an episode.

Regression for the mobile VLA failure where the analytic IK raised
``Unreachable`` inside ``engine._apply``'s fallback branch, because the
*previous* target had itself become unreachable once the wheels moved.
"""
from pathlib import Path
import sys

import numpy as np
import pytest

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))

from vlab import kinematics  # noqa: E402

PARAMS = {"base_y": 0.0, "shoulder_z": 0.205, "l1": 0.28, "l2": 0.18, "tool": 0.0}
REACH = PARAMS["l1"] + PARAMS["l2"]


def _local_distance(point):
    """Distance of a chassis-frame point from the shoulder, as the IK measures it."""
    x, y, z = point
    radial = np.hypot(x, y - PARAMS["base_y"])
    return float(np.hypot(radial, z + PARAMS["tool"] - PARAMS["shoulder_z"]))


def test_reachable_point_is_returned_untouched():
    point = np.array([0.30, 0.0, PARAMS["shoulder_z"] - 0.10])
    assert abs(PARAMS["l1"] - PARAMS["l2"]) < _local_distance(point) <= REACH
    projected, info = kinematics.clamp_to_reach(point, PARAMS)
    assert info is None
    np.testing.assert_allclose(projected, point)


def test_outer_overshoot_is_clamped_not_raised():
    # 0.4 mm past the singular boundary: this is the size of the failure that
    # aborted the recorded mobile VLA run.
    point = np.array([REACH + 0.0004, 0.0, PARAMS["shoulder_z"]])
    with pytest.raises(kinematics.Unreachable):
        kinematics.analytic_ik(point, PARAMS)

    projected, info = kinematics.clamp_to_reach(point, PARAMS)
    assert info is not None and info["kind"] == "reach_clamped_outer"
    assert _local_distance(projected) <= REACH
    assert info["overshoot_m"] > 0
    # Direction is preserved: the clamp only shortens the reach.
    assert np.isclose(projected[0], REACH - 5e-4, atol=1e-6)
    assert np.isclose(projected[1], 0.0, atol=1e-9)


def test_inner_dead_zone_is_pushed_out():
    point = np.array([0.001, 0.0, PARAMS["shoulder_z"]])
    with pytest.raises(kinematics.Unreachable):
        kinematics.analytic_ik(point, PARAMS)
    projected, info = kinematics.clamp_to_reach(point, PARAMS)
    assert info is not None and info["kind"] == "reach_clamped_inner"
    assert _local_distance(projected) >= abs(PARAMS["l1"] - PARAMS["l2"])


def test_projection_preserves_azimuth():
    point = np.array([0.30, 0.30, PARAMS["shoulder_z"] + 0.20])
    yaw_before = np.arctan2(point[1] - PARAMS["base_y"], point[0])
    projected, info = kinematics.clamp_to_reach(point, PARAMS)
    assert info is not None
    yaw_after = np.arctan2(projected[1] - PARAMS["base_y"], projected[0])
    assert np.isclose(yaw_before, yaw_after, atol=1e-9)


def test_clamped_point_is_solvable_by_the_same_ik():
    point = np.array([0.60, 0.0, 0.60])
    projected, info = kinematics.clamp_to_reach(point, PARAMS)
    assert info is not None
    q = kinematics.analytic_ik(projected, PARAMS)          # must not raise
    assert np.isfinite(q).all()


@pytest.mark.parametrize("point", [[np.nan, 0, 0], [1, 2], []])
def test_malformed_points_are_rejected(point):
    with pytest.raises(ValueError):
        kinematics.clamp_to_reach(np.asarray(point, dtype=float), PARAMS)


def test_mobile_base_leaving_a_fixed_world_target_is_projected():
    """The real defect: reachability changes while the world target does not."""
    from vlab.duck_demo import CHASSIS, DUCK
    from vlab.store import ConfigStore
    from vlab.transport import MobileTransportEnv

    base = ConfigStore(LAB / "projects").load()
    env = MobileTransportEnv(base, LAB, {
        "object_asset": DUCK, "chassis_asset": CHASSIS,
        "retreat_distance_m": .4, "belt_speed_m_s": .025, "seed": 3,
    }, render=False)
    try:
        # A target the arm can already hold, then drive the chassis away so the
        # same world point is out of reach for the local solver.
        env._target = np.array([-.16, .095, .148])
        env._ik(env._target)                      # reachable before the move
        env.data.qpos[env._base_q + 1] += .35     # chassis retreats, pose only

        params = kinematics.chain_params(env.robot)
        local = env._frame(env._target)[2]
        with pytest.raises(kinematics.Unreachable):
            kinematics.analytic_ik(local, params)

        env._apply(np.array([0., 0., 0., 1.]))    # used to raise here
        assert env._constraint["limited"] is True
        assert env._constraint["detail"]["kind"] == "reach_clamped_outer"
        assert env._reach_limited == 1
        assert env.metrics()["reach_limited_steps"] == 1
        assert env._ik_error is None

        projected_local = env._frame(env._constraint["target"])[2]
        assert np.hypot(np.hypot(projected_local[0], projected_local[1]),
                        projected_local[2] + params["tool"] - params["shoulder_z"]) <= params["l1"] + params["l2"]
    finally:
        env.close()


def test_reach_limited_steps_do_not_change_success_judgement():
    """Constraint evidence is annotation only; the physics observer still decides."""
    from vlab.duck_demo import CHASSIS, DUCK
    from vlab.store import ConfigStore
    from vlab.transport import MobileTransportEnv

    base = ConfigStore(LAB / "projects").load()
    env = MobileTransportEnv(base, LAB, {
        "object_asset": DUCK, "chassis_asset": CHASSIS,
        "retreat_distance_m": .4, "belt_speed_m_s": .025, "seed": 3,
    }, render=False)
    try:
        before = env.metrics()["success"]
        env._target = np.array([-.16, .095, .148])
        env.data.qpos[env._base_q + 1] += .35
        env._apply(np.array([0., 0., 0., 1.]))
        assert env.metrics()["success"] == before is False
        assert env.metrics()["failed"] is False
    finally:
        env.close()
