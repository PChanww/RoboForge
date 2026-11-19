"""A connected VLA must own its actions and never receive object truth."""
from types import SimpleNamespace
import threading
import time

import numpy as np
import pytest

from vlab import transport_policy as policy


IDENTITY = {"action_schema": "mobile7_v1", "action_dim": 7, "learned_policy": True,
            "weights_sha256": "a"*64, "label": "Test learned policy", "adapter": "test"}


def measured_scene():
    env = SimpleNamespace(data=SimpleNamespace(xpos=np.array([[.1, .2, .15]]), qpos=np.array([.02, .02])),
        _eef=0, robot=SimpleNamespace(gripper=SimpleNamespace(left_joint="left", right_joint="right")),
        _qposadr={"left": 0, "right": 1}, _closed=.006, _open=.04)
    env.expert_action = lambda: pytest.fail("A VLA must never request an expert action")
    state = {"time": .5, "base_pose": [0, 0, -.033, 1, 0, 0, 0], "base_velocity": [0]*6,
             "arm_qpos": [0]*7, "wheel_qpos": [0]*4, "wheel_qvel": [0]*4,
             "object_pose": [9]*7, "phase": 4, "contacts": 2}
    preview = {"tick": 10, "sim_time": .5, "images": {"front": "front_jpeg", "wrist": "wrist_jpeg"},
               "camera_status": {name: {"state": "ok", "render_error": None,
                    "capture_time": .5, "available_time": .5, "age_s": 0}
                    for name in ("front", "wrist")},
               "metrics": {"object_xyz": [9, 9, 9]}}
    return env, state, preview


def controller():
    instance = policy.VLAController("http://localhost:1234", dict(IDENTITY), {"vla_replan_steps": 2})
    instance.reset(3, "Pick up the object and put it back.")
    return instance


def response(packet, actions=None):
    return {"request_id": packet["request_id"], "session_id": packet["session_id"],
            "tick": packet["tick"], "action_schema": "mobile7_v1", "weights_sha256": "a"*64,
            "actions": [[.2, -.1, .3, -1, 0, 0, 0], [.1, 0, 0, -1, 0, 0, 0]] if actions is None else actions}


def test_vla_uses_language_cameras_and_robot_state_without_privileged_truth(monkeypatch):
    calls = []
    def infer(url, packet, **_):
        calls.append(packet)
        return response(packet)
    monkeypatch.setattr(policy, "_json_call", infer)
    agent = controller()
    env, state, preview = measured_scene()
    assert agent.action(env, state, preview)["arm"] == [.2, -.1, .3, -1]
    assert agent.diagnostics["executed_model_actions"] == 0
    agent.mark_applied()
    assert agent.action(env, state, preview)["arm"] == [.1, 0, 0, -1]
    agent.mark_applied()
    assert len(calls) == 1
    packet = calls[0]
    assert set(packet["observation"]) == {"state", "eef_position_base", "gripper_open_fraction", "images"}
    assert len(packet["observation"]["state"]) == 28
    assert packet["instruction"] == "Pick up the object and put it back."
    assert agent.diagnostics["inference_count"] == 1 and agent.diagnostics["executed_model_actions"] == 2
    assert agent.diagnostics["chunks"][0]["adapted_actions"][0][0] == .2
    agent.reset(4, "Next episode")
    assert not agent.queue


@pytest.mark.parametrize("wrong", ["tick", "session_id", "weights_sha256", "action_schema"])
def test_wrong_run_time_weights_or_schema_never_produce_an_action(monkeypatch, wrong):
    def infer(url, packet, **_):
        result = response(packet)
        result[wrong] = "incorrect"
        return result
    monkeypatch.setattr(policy, "_json_call", infer)
    with pytest.raises(ValueError, match="不匹配"):
        controller().action(*measured_scene())


@pytest.mark.parametrize("actions", [[], [[0]*6], [[float("nan")]*7], [[2, 0, 0, 0, 0, 0, 0]],
    [[0, 0, 0, 0, .5, 0, 0]], [[0, 0, 0, 0, 0, 1, 0]], [[0, 0, 0, 0, 0, 0, .5]]])
def test_malformed_or_out_of_range_model_actions_fail_closed(monkeypatch, actions):
    monkeypatch.setattr(policy, "_json_call", lambda url, packet, **_: response(packet, actions))
    with pytest.raises(ValueError):
        controller().action(*measured_scene())


def test_unavailable_service_cannot_start_vla_but_expert_remains_explicit(tmp_path, monkeypatch):
    def unavailable(*_, **__):
        raise ValueError("offline")
    monkeypatch.setattr(policy, "_json_call", unavailable)
    policy._HEALTH_CACHE.clear()
    registry = policy.PolicyRegistry(tmp_path)
    assert registry.status()["vla_connected"] is False
    with pytest.raises(ValueError, match="尚未就绪"):
        registry.create({"controller": "vla"})
    assert registry.create({"controller": "expert"}).identity["learned_policy"] is False


def test_identity_and_health_require_actual_weights_and_action_semantics(tmp_path, monkeypatch):
    policy._HEALTH_CACHE.clear()
    monkeypatch.setattr(policy, "_json_call", lambda *_, **__: {"ready": True, "identity": dict(IDENTITY)})
    registry = policy.PolicyRegistry(tmp_path)
    assert registry.status()["vla_connected"]
    assert registry.create({"controller": "vla"}).identity["weights_sha256"] == "a"*64
    monkeypatch.setattr(policy, "_json_call", lambda *_, **__: {"ready": True, "identity": {**IDENTITY, "action_dim": 4}})
    with pytest.raises(ValueError, match="尚未就绪"):
        registry.validate({"controller": "vla"})


def test_cancelled_http_wait_discards_late_response_and_clears_pending(monkeypatch):
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()
    agent = controller()
    cancel = threading.Event()
    progress = []
    agent.bind_runtime(cancel, lambda diagnostics: progress.append(dict(diagnostics)))
    def blocked_infer(url, packet, **_):
        entered.set()
        release.wait(timeout=3)
        returned.set()
        return response(packet)
    monkeypatch.setattr(policy, "_json_call", blocked_infer)
    errors = []
    def act():
        try:
            agent.action(*measured_scene())
        except Exception as exc:
            errors.append(exc)
    owner = threading.Thread(target=act)
    try:
        owner.start()
        assert entered.wait(timeout=1)
        assert agent.diagnostics["inference_pending"]
        started = time.monotonic()
        cancel.set()
        owner.join(timeout=1)
        assert not owner.is_alive() and time.monotonic()-started < 1
        assert len(errors) == 1 and isinstance(errors[0], policy.PolicyCancelled)
        assert agent.diagnostics["executed_model_actions"] == agent.diagnostics["inference_count"] == 0
        assert not agent.diagnostics["inference_pending"] and not agent.queue
        assert [value["inference_pending"] for value in progress] == [True, False]
        release.set()
        assert returned.wait(timeout=1)
        assert not agent.queue
    finally:
        release.set()
        owner.join(timeout=1)


def test_http_failure_clears_pending_without_any_executed_action(monkeypatch):
    agent = controller()
    progress = []
    agent.bind_runtime(threading.Event(), lambda diagnostics: progress.append(dict(diagnostics)))
    def failed_infer(*_, **__):
        raise ValueError("inference failed")
    monkeypatch.setattr(policy, "_json_call", failed_infer)
    with pytest.raises(ValueError, match="inference failed"):
        agent.action(*measured_scene())
    assert [value["inference_pending"] for value in progress] == [True, False]
    assert agent.diagnostics["executed_model_actions"] == 0 and not agent.queue


@pytest.mark.parametrize("fault", ["missing", "stale", "render_error", "future_available", "wrong_time"])
def test_vla_rejects_uncertain_or_expired_camera_frames_before_inference(monkeypatch, fault):
    env, state, preview = measured_scene()
    if fault == "missing":
        preview.pop("camera_status")
    elif fault == "stale":
        preview["camera_status"]["wrist"].update(capture_time=.45, age_s=.05)
    elif fault == "render_error":
        preview["camera_status"]["front"]["render_error"] = "GL unavailable"
    elif fault == "future_available":
        preview["camera_status"]["front"]["available_time"] = .55
    else:
        preview["sim_time"] = .45
    monkeypatch.setattr(policy, "_json_call", lambda *_args, **_kwargs: pytest.fail("Invalid RGB must not reach a VLA"))
    with pytest.raises(ValueError, match="相机"):
        controller().action(env, state, preview)
