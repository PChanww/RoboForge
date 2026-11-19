"""Entire physical programs must finish, including release and retrieval."""
from pathlib import Path

import numpy as np
import pytest

from test_mobile_transport import PhysicsAudit
from vlab.store import ConfigStore
from vlab.transport import MobileTransportEnv


LAB = Path(__file__).resolve().parents[1]


def make_env(kind, seed=3, **extra):
    spec = {"task_kind": kind, "target_xy": [-.34, .28], "speed_scale": 1.8,
            "belt_speed_m_s": .029, "seed": seed, **extra}
    return MobileTransportEnv(ConfigStore(LAB / "projects").load(), LAB, spec, render=False)


@pytest.mark.parametrize("kind", ["pick_return", "bin_return"])
@pytest.mark.parametrize("seed", [3, 4, 5])
def test_multistage_physical_return_finishes_only_after_real_belt_support(kind, seed, monkeypatch):
    e = make_env(kind, seed)
    try:
        audit = PhysicsAudit(e, monkeypatch) if seed == 3 else None
        assert e.model.neq == 0
        saw_bin_success_before_program = False
        for tick in range(1800):
            e.step(e.expert_action())
            m = e.metrics()
            assert np.isfinite(e.data.qpos).all() and np.isfinite(e.data.qvel).all()
            if e._program.delivered and not e._program.complete:
                saw_bin_success_before_program = True
                assert not m["success"]
            if m["terminated"]:
                break
        assert m["success"] and m["program_complete"], m
        assert m["returned_to_belt"] and m["return_stable_ticks"] >= 6
        assert m["return_release_error_m"] < .012
        stages = [event["stage"] for event in m["task_events"]]
        assert stages[0] == "pick_from_belt" and stages[-1] == "return_to_belt"
        final = m["task_events"][-1]
        assert final["roller_contact"] and final["contacts"] == 0 and final["controlled_release"]
        assert e.data.xpos[e._eef][2] > .12
        if kind == "bin_return":
            assert saw_bin_success_before_program
            assert m["delivered_to_bin"] and m["regrasped_from_bin"]
            for a, b in zip(["place_in_bin", "regrasp_from_bin", "lifted_from_bin", "return_along_route"],
                            ["regrasp_from_bin", "lifted_from_bin", "return_along_route", "return_to_belt"]):
                assert stages.index(a) < stages.index(b)
            assert m["return_path_rmse_m"] < .04
            assert np.linalg.norm(e.state()["base_pose"][:2]-e._base_start[:2]) < .06
        else:
            assert not m["delivered_to_bin"] and not m["regrasped_from_bin"]
            assert np.linalg.norm(e.state()["base_pose"][:2]-e._base_start[:2]) < .01
        if audit:
            assert audit.physics_ticks == (tick+1)*25
    finally:
        e.close()


def test_bin_result_is_not_whole_program_success_when_regrasp_contact_is_removed():
    e = make_env("bin_return")
    try:
        removed = False
        for _ in range(1200):
            if e.phase >= 14 and not removed:
                for name in ("left_pad", "right_pad"):
                    gid = e.model.geom(name).id
                    e.model.geom_contype[gid] = e.model.geom_conaffinity[gid] = 0
                removed = True
            e.step(e.expert_action())
            assert not e.metrics()["success"]
        assert removed and e._program.delivered
        assert not e._program.regrasped and not e._program.complete
    finally:
        e.close()


@pytest.mark.parametrize("kind", ["pick_return", "bin_return", "bin_retrieve"])
def test_policy_observer_scores_physical_action_replay_without_expert_fallback(kind, monkeypatch):
    # This trace exercises observation-only scoring, not learned-policy quality.
    e = make_env(kind)
    actions = []
    try:
        for _ in range(1800):
            action = e.expert_action()
            actions.append({k: list(v) if k == "arm" else v for k, v in action.items()})
            e.step(action)
            if e.metrics()["terminated"]:
                break
        assert e.metrics()["success"]
    finally:
        e.close()
    e = make_env(kind, controller="vla")
    try:
        monkeypatch.setattr(e, "expert_action", lambda: pytest.fail("Policy observer called the expert"))
        for action in actions:
            e.step(action)
        m = e.metrics()
        assert m["success"] and m["program_complete"], m
        stages = [event["stage"] for event in m["task_events"]]
        assert stages[0] == "pick_from_belt"
        assert stages[-1] == ("lifted_from_bin" if kind == "bin_retrieve" else "return_to_belt")
        if kind == "bin_return":
            assert "place_in_bin" in stages and "regrasp_from_bin" in stages and "return_along_route" in stages
        if kind == "bin_retrieve":
            assert m["contacts"] == 2 and m["retrieved_from_bin"]
            assert not m["returned_to_belt"]
    finally:
        e.close()


@pytest.mark.parametrize("seed", [3, 4, 5])
def test_bin_retrieve_finishes_while_stably_holding_object_above_box(seed, monkeypatch):
    e = make_env("bin_retrieve", seed)
    try:
        audit = PhysicsAudit(e, monkeypatch) if seed == 3 else None
        for tick in range(1400):
            e.step(e.expert_action())
            m = e.metrics()
            if e._program.delivered and not e._program.retrieved:
                assert not m["success"]
            if m["terminated"]:
                break
        assert m["success"] and m["program_complete"]
        assert m["delivered_to_bin"] and m["regrasped_from_bin"] and m["retrieved_from_bin"]
        assert m["contacts"] == 2 and not m["returned_to_belt"]
        assert m["retrieve_stable_ticks"] >= 5
        stages = [event["stage"] for event in m["task_events"]]
        assert stages.index("place_in_bin") < stages.index("regrasp_from_bin") < stages.index("lifted_from_bin")
        assert "return_along_route" not in stages and "return_to_belt" not in stages
        evidence = m["task_events"][-1]
        assert evidence["stage"] == "lifted_from_bin" and evidence["contacts"] == 2
        assert evidence["stable_ticks"] >= 5
        assert evidence["object_low_z"] > evidence["rim_z_m"]+.015
        # Keeping the completed program on screen must retain the object.
        for _ in range(8):
            e.step(e.expert_action())
            assert e._judge.contacts == 2
        if audit:
            assert audit.physics_ticks == (tick+9)*25
    finally:
        e.close()
