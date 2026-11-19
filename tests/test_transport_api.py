"""Persistent combined-task execution and truthful paired data collection."""
import base64
import errno
import io
import hashlib
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

import vlab.transport_api as api
from vlab.duck_demo import CHASSIS, DUCK
from vlab.transport_api import TransportSpec, TransportStore, plan_transport, register_transport


def eventually(fn, timeout=3):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        result = fn()
        if result:
            return result
        time.sleep(.005)
    raise AssertionError("Timed out waiting for transport result")


def frame():
    stream = io.BytesIO()
    Image.fromarray(np.full((12, 16, 3), 87, np.uint8)).save(stream, format="JPEG")
    return "data:image/jpeg;base64,"+base64.b64encode(stream.getvalue()).decode()


class Plant:
    instances = []
    fail = False
    block = None
    entered = None

    def __init__(self, config, lab, spec):
        self.owner = threading.get_ident()
        self.spec = spec
        self.config = SimpleNamespace(model_dump=lambda **_: {"schema_version": "test", "sensors": [
            {"id": name, "parent": "world" if name == "front" else "gripper", "resolution": [16, 12],
             "intrinsics": {"fx": 20, "fy": 20, "cx": 8, "cy": 6}, "latency_s": 0}
            for name in ("front", "wrist")], "provenance": {"controller": "deterministic_test"}})
        self.closed = False
        self.seed = spec["seed"]
        self.image = frame()
        self.instances.append(self)
        self.reset(self.seed)

    def check(self):
        assert threading.get_ident() == self.owner, "GL and physics must share one owner"

    def reset(self, seed):
        self.check()
        self.seed, self.tick = seed, 0
        self.x, self.v = 0., 0.

    def state(self):
        self.check()
        return {"time": self.tick*.05, "base_pose": np.array([0, self.x, .03, 1, 0, 0, 0]),
                "base_velocity": np.array([0, self.v, 0, 0, 0, 0]),
                "wheel_qpos": np.full(4, self.x), "wheel_qvel": np.full(4, self.v),
                "arm_qpos": np.zeros(7), "object_pose": np.array([0, self.x, .2, 1, 0, 0, 0]),
                "object_velocity": np.zeros(6), "belt_speed_m_s": .025,
                "phase": self.tick, "contacts": 2}

    def expert_action(self):
        self.check()
        return {"arm": [0., 0., 0., -.5], "base_linear_m_s": .1, "hold_arm": True}

    def step(self, action):
        self.check()
        if self.entered is not None:
            self.entered.set()
        if self.block is not None:
            self.block.wait(timeout=3)
        self.v = min(self.v+.006, action["base_linear_m_s"])
        self.x += self.v*.05
        self.tick += 1
        return {"applied_action": [0, 0, 0, -.5, self.v, 0, 1]}

    def snapshot(self, render=True):
        self.check()
        return {"tick": self.tick, "sim_time": self.tick*.05, "phase": self.tick,
                "phase_label": "后退搬运", "pose": {"x": 0, "y": self.x, "z": .03, "yaw": 0},
                "images": {"front": self.image, "wrist": self.image},
                "camera_status": {name: {"state": "ok", "render_error": None,
                    "capture_time": self.tick*.05, "available_time": self.tick*.05, "age_s": 0}
                    for name in ("front", "wrist")},
                "metrics": {"success": self.tick >= 4 and not self.fail,
                            "failed": self.tick >= 3 and self.fail, "contacts": 2},
                "belt_speed_m_s": .025, "transport_distance_m": self.x}

    def close(self):
        self.check()
        self.closed = True


@pytest.fixture
def store(tmp_path, monkeypatch):
    Plant.instances, Plant.fail, Plant.block, Plant.entered = [], False, None, None
    def asset(lab, ref, name):
        return {"id": ref, "name": name, "source": "lux3d", "generation_id": "known",
                "kind": "task_object" if ref == DUCK else "robot_visual", "role": "dynamic" if ref == DUCK else "visual_only",
                "height_m": .06, "bounds_m": [[-.02, -.02, 0], [.02, .02, .06]], "mesh_sha256": "hash"}
    monkeypatch.setattr(api, "asset_info", asset)
    monkeypatch.setattr(api, "prepare_object", lambda *_: None)
    instance = TransportStore(SimpleNamespace(load=lambda: "unchanged-config"), tmp_path,
                              env_factory=Plant, realtime=False)
    monkeypatch.setattr(instance.catalog, "is_reusable", lambda *_: True)
    yield instance
    if Plant.block is not None:
        Plant.block.set()
    instance.close()


def finished(store, rid):
    return eventually(lambda: store.get(rid) if store.get(rid)["status"] in api.TERMINAL else None)


def test_plan_supports_requested_chain_and_parses_bounded_parameters():
    plan = plan_transport(TransportSpec(instruction="从不停带的传送带抓取物品，向后移动45厘米放入后方箱子，采集两次，种子7，带速2 cm/s。"))
    assert plan["ready"]
    assert plan["generation_count"] == 0
    assert plan["spec"]["episodes"] == 2
    assert plan["spec"]["seed"] == 7
    assert plan["spec"]["retreat_distance_m"] == .45
    assert plan["spec"]["belt_speed_m_s"] == .02
    for words in ("向后移动20厘米", "向后移动90厘米", "向后移动，倒水", "向后移动到坐标x=1"):
        rejected = plan_transport(TransportSpec(instruction=f"从传送带抓取物品，{words}，放入箱子。"))
        assert not rejected["ready"], words
    placed = plan_transport(TransportSpec(instruction="从传送带抓取物品，导航转向后放入箱子坐标(0.6,0.8)。"))
    assert placed["ready"], placed["issues"]
    assert placed["spec"]["target_xy"] == [.6, .8]
    assert placed["navigation"]["planned_route"]
    overlap = plan_transport(TransportSpec(target_xy=[-.12, -.07]))
    assert not overlap["ready"]
    for instruction in (
        "从传送带抓取物品，放到后方箱子里，然后在拿出来放回传送带。",
        "从传送带抓取物品，放入箱子，接着取出送回传送带。",
    ):
        supported = plan_transport(TransportSpec(instruction=instruction))
        assert supported["ready"], supported["issues"]
        assert supported["task_kind"] == "bin_return"
        assert supported["task_sequence"] == api.TASK_SEQUENCES["bin_return"]


@pytest.mark.parametrize("instruction", ["把传送带上的物品拿起，然后放回原位。", "拿起后原路放回", "pick from conveyor then put it back"])
def test_pick_return_program_preserves_canonical_stage_order(instruction):
    plan = plan_transport(TransportSpec(instruction=instruction))
    assert plan["ready"], plan["issues"]
    assert plan["task_kind"] == "pick_return"
    assert plan["spec"]["task_sequence"] == plan["task_sequence"] == api.TASK_SEQUENCES["pick_return"]
    assert any("放回" in step for step in plan["steps"])


@pytest.mark.parametrize("instruction", [
    "放到箱子里面再拿出来",
    "从传送带抓取物品，放到箱子里面再拿出来。",
    "从不停的传送带上抓取小鸭，放入后方箱内，然后取出来保持夹持。",
])
def test_bin_retrieval_does_not_invent_a_return_destination(instruction):
    plan = plan_transport(TransportSpec(instruction=instruction))
    assert plan["ready"], plan["issues"]
    assert plan["task_kind"] == "bin_retrieve"
    assert plan["task_sequence"] == api.TASK_SEQUENCES["bin_retrieve"]
    assert "return_to_belt" not in plan["task_sequence"]
    assert "保持稳定夹持" in plan["summary"]


@pytest.mark.parametrize("instruction", [
    "从传送带抓取物品，放入箱子，再取出放到桌子上。",
    "从传送带抓取物品，放入箱子，再取出放入第二个箱。",
    "从传送带抓取物品，放入箱子，再取出放回传送带，重复执行。",
    "从传送带抓取物品，和箱子一起放回传送带。",
])
def test_unsupported_extra_destinations_and_ambiguous_programs_are_rejected(instruction):
    assert not plan_transport(TransportSpec(instruction=instruction))["ready"]


def test_persistent_idempotent_run_and_paired_images_actions(store):
    initial = store.create(TransportSpec(), "stable-request")
    row = finished(store, initial["id"])
    assert row["status"] == "succeeded", row
    assert store.create(TransportSpec(), "stable-request")["id"] == row["id"]
    with pytest.raises(ValueError, match="其他要求"):
        store.create(TransportSpec(seed=4), "stable-request")
    assert len(Plant.instances) == 1
    assert Plant.instances[0].closed
    directory = store._directory(row["id"])
    arrays = np.load(directory / "episodes/ep000.npz")
    assert arrays["base_pose"].shape == (4, 7)
    assert arrays["image_front"].shape == arrays["image_wrist"].shape == (4, 12, 16, 3)
    assert arrays["action"].shape == arrays["requested_action"].shape == (4, 7)
    np.testing.assert_allclose(np.diff(arrays["base_pose"][:, 1]), arrays["action"][:-1, 4]*.05)
    assert np.all(arrays["action"][:, 4] < arrays["requested_action"][:, 4])
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["images_recorded"] is True
    assert manifest["summary"] == {"episodes": 1, "successes": 1, "frames": 4}
    assert "privileged" in manifest["physics"]["controller"]
    config_bytes = (directory / "config.json").read_bytes()
    assert manifest["configuration"]["sha256"] == hashlib.sha256(config_bytes).hexdigest()
    assert manifest["configuration"]["snapshot"] == json.loads(config_bytes)
    assert manifest["camera_configs"]["front"]["resolution"] == [16, 12]
    assert manifest["camera_configs"]["wrist"]["parent"] == "gripper"
    assert manifest["runtime"]["mujoco"]


def test_failed_grasp_saves_truth_and_blocks_remaining_demonstrations(store):
    Plant.fail = True
    row = finished(store, store.create(TransportSpec(episodes=3), "failure")["id"])
    assert row["status"] == "failed"
    assert row["summary"] == {"episodes": 1, "successes": 0, "frames": 3}
    assert not row["episodes"][0]["success"]
    assert Plant.instances[0].seed == 3, "A failed run may not begin the next specimen"


def test_stop_is_immediate_during_step_and_worker_blocks_new_run(store):
    Plant.block = threading.Event()
    Plant.entered = threading.Event()
    first = store.create(TransportSpec(), "blocked-step")
    assert Plant.entered.wait(timeout=3)
    before = time.monotonic()
    stopped = store.stop(first["id"])
    assert time.monotonic()-before < .2
    assert stopped["status"] == "stopping"
    assert stopped["stop_requested"] is True
    assert store.stop(first["id"])["status"] == "stopping"
    with pytest.raises(ValueError, match="正在运行"):
        store.create(TransportSpec(), "new-during-cancellation")
    Plant.block.set()
    row = finished(store, first["id"])
    assert Plant.instances[0].closed
    assert row["status"] == "cancelled"
    assert row["episodes"][0]["interrupted"]
    assert row["episodes"][0]["frames"] == 1
    assert (store._directory(first["id"]) / "episodes/ep000.npz").is_file()


@pytest.mark.parametrize("dead_worker", [False, True])
def test_stop_without_live_worker_finishes_and_releases_slot(store, monkeypatch, dead_worker):
    launch = store._launch
    monkeypatch.setattr(store, "_launch", lambda row: None)
    row = store.create(TransportSpec(), "orphaned-request")
    if dead_worker:
        worker = threading.Thread(target=lambda: None)
        worker.start()
        worker.join()
        store._workers[row["id"]] = worker
    stopped = store.stop(row["id"])
    assert stopped["status"] == "cancelled"
    assert stopped["stop_requested"]
    assert not stopped["episodes"]
    run_file = store._directory(row["id"]) / "run.json"
    assert json.loads(run_file.read_text(encoding="utf-8"))["status"] == "cancelled"
    assert store.stop(row["id"])["status"] == "cancelled"
    monkeypatch.setattr(store, "_launch", launch)
    assert finished(store, store.create(TransportSpec(), "next-request")["id"])["status"] == "succeeded"


def test_full_disk_during_stop_still_cancels_and_keeps_saved_episode(store, monkeypatch):
    Plant.block, Plant.entered = threading.Event(), threading.Event()
    initial = store.create(TransportSpec(), "disk-fails-during-stop")
    assert Plant.entered.wait(timeout=3)
    original_atomic = api._atomic
    original_write = Path.write_text

    def fail_run_write(path, row):
        if path.name == "run.json" and row.get("stop_requested"):
            raise OSError(errno.ENOSPC, "disk full during metadata write")
        return original_atomic(path, row)

    def fail_cancel_write(path, *args, **kwargs):
        if path.name == "cancel":
            raise OSError(errno.ENOSPC, "disk full during marker write")
        return original_write(path, *args, **kwargs)

    monkeypatch.setattr(api, "_atomic", fail_run_write)
    monkeypatch.setattr(Path, "write_text", fail_cancel_write)
    stopping = store.stop(initial["id"])
    assert stopping["status"] == "stopping"
    assert "磁盘空间不足" in stopping["storage_error"]
    Plant.block.set()
    row = finished(store, initial["id"])
    assert row["status"] == "cancelled"
    assert "磁盘空间不足" in row["error"]
    assert len(row["episodes"]) == 1 and row["episodes"][0]["frames"] == 1
    directory = store._directory(row["id"])
    assert [f.name for f in (directory / "episodes").iterdir()] == ["ep000.npz"]
    assert json.loads((directory / "manifest.json").read_text(encoding="utf-8"))["status"] == "cancelled"


def test_episode_write_exhaustion_reports_failure_and_discards_partial_file(store, monkeypatch):
    attempted = []

    def fail_after_writing(file, **arrays):
        file.write(b"partial file")
        attempted.append(1)
        raise OSError(errno.ENOSPC, "disk full during episode write")

    monkeypatch.setattr(api.np, "savez_compressed", fail_after_writing)
    row = finished(store, store.create(TransportSpec(), "disk-fails-during-episode")["id"])
    assert row["status"] == "failed"
    assert "磁盘空间不足" in row["error"]
    assert not row["episodes"]
    assert attempted == [1]
    assert list((store._directory(row["id"]) / "episodes").iterdir()) == []


def test_restart_marks_running_record_interrupted_without_auto_execution(store):
    row = finished(store, store.create(TransportSpec(), "previous-process")["id"])
    path = store._directory(row["id"]) / "run.json"
    row["status"] = "running"
    path.write_text(json.dumps(row), encoding="utf-8")
    reopened = TransportStore(store.configs, store.lab, env_factory=Plant, realtime=False)
    assert reopened.get(row["id"])["status"] == "interrupted"
    manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "interrupted"
    assert reopened.create(TransportSpec(), "previous-process")["id"] == row["id"]
    assert len(Plant.instances) == 1


def test_api_contract_download_and_invalid_path(store, monkeypatch):
    monkeypatch.setattr(api, "TransportStore", lambda *args: store)
    app = FastAPI()
    register_transport(app, store.configs, store.lab)
    with TestClient(app) as client:
        body = {"spec": TransportSpec().model_dump(), "client_request_id": "http"}
        plan = client.post("/api/v1/transport/plan", json=body["spec"])
        assert plan.status_code == 200 and plan.json()["ready"]
        initial = client.post("/api/v1/transport/runs", json=body)
        assert initial.status_code == 200
        row = finished(store, initial.json()["id"])
        row["policy_diagnostics"]["chunks"] = [{"tick": 1}]
        store._publish(row)
        path = "/api/v1/transport/runs/"+row["id"]
        original = client.get(path).json()
        compact = client.get(path, params={"compact": True}).json()
        assert original["status"] == compact["status"] == "succeeded"
        assert original["policy_diagnostics"].pop("chunks") == [{"tick": 1}]
        assert compact == original
        manifest = client.get(row["manifest_url"])
        assert manifest.status_code == 200
        config = client.get(row["configuration"]["url"])
        assert config.status_code == 200
        assert hashlib.sha256(config.content).hexdigest() == manifest.json()["configuration"]["sha256"]
        episode = client.get(row["episodes"][0]["npz_url"])
        assert episode.status_code == 200
        assert episode.content[:2] == b"PK"
        assert client.get(f"/api/v1/transport/datasets/{row['id']}/episodes/-1").status_code == 404
        assert client.post("/api/v1/transport/plan", json={**body["spec"], "retreat_distance_m": "NaN"}).status_code == 422


def test_disk_full_during_create_is_a_retriable_507_without_phantom_run(store, monkeypatch):
    monkeypatch.setattr(api, "TransportStore", lambda *args: store)
    app = FastAPI()
    register_transport(app, store.configs, store.lab)
    original_replace = Path.replace
    body = {"spec": TransportSpec().model_dump(), "client_request_id": "same-id-after-disk-failure"}

    def fail_replace(path, target):
        if target.name == "run.json":
            raise OSError(errno.ENOSPC, "disk full after writing temp")
        return original_replace(path, target)

    with TestClient(app) as client:
        monkeypatch.setattr(Path, "replace", fail_replace)
        response = client.post("/api/v1/transport/runs", json=body)
        assert response.status_code == 507
        assert "磁盘空间不足" in response.json()["detail"]
        assert not store._rows and not store._workers and not store._cancel
        assert not list(store.root.iterdir())
        monkeypatch.setattr(Path, "replace", original_replace)
        response = client.post("/api/v1/transport/runs", json=body)
        assert response.status_code == 200
        assert finished(store, response.json()["id"])["status"] == "succeeded"


def test_requested_vla_actions_identity_and_diagnostics_are_saved_without_expert_fallback(store, monkeypatch):
    identity = {"id": "test-vla", "kind": "vla", "learned_policy": True, "label": "Test learned policy",
                "weights_sha256": "a"*64, "adapter": "test_mobile7", "action_schema": "mobile7_v1", "action_dim": 7}
    calls = []
    class Controller:
        diagnostics = {"inference_count": 0, "executed_model_actions": 0}
        def reset(self, seed, instruction):
            calls.append((seed, instruction))
        def action(self, env, state, preview):
            assert preview["tick"] == len(calls)-1
            calls.append(preview["tick"])
            self.diagnostics["inference_count"] += 1
            return {"arm": [.2, -.1, .3, .4], "base_linear_m_s": .03,
                    "base_angular_rad_s": .01, "hold_arm": False}
        def mark_applied(self):
            self.diagnostics["executed_model_actions"] += 1
    controller = Controller()
    controller.identity = identity
    class Registry:
        def validate(self, spec):
            assert spec["controller"] == "vla" and spec["policy_id"] == "test-vla"
            return identity
        def create(self, spec):
            self.validate(spec)
            return controller
    monkeypatch.setattr(store, "_policies", Registry)
    def forbidden_expert(_):
        raise AssertionError("VLA must never call privileged expert")
    monkeypatch.setattr(Plant, "expert_action", forbidden_expert)
    row = finished(store, store.create(TransportSpec(controller="vla", policy_id="test-vla"), "vla-run")["id"])
    assert row["status"] == "succeeded", row.get("error")
    assert row["policy_identity"] == identity
    assert row["policy_diagnostics"]["executed_model_actions"] == 4
    directory = store._directory(row["id"])
    with np.load(directory / "episodes/ep000.npz") as data:
        np.testing.assert_allclose(data["requested_action"], np.tile([.2, -.1, .3, .4, .03, .01, 0], (4, 1)))
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf8"))
    config = json.loads((directory / "config.json").read_text(encoding="utf8"))
    assert manifest["policy_identity"] == identity
    assert manifest["physics"]["controller"] == "vla" and "expert_version" not in manifest
    assert manifest["policy_diagnostics"] == row["policy_diagnostics"]
    assert json.loads(config["provenance"]["transport.policy_identity"]) == identity
    assert json.loads(config["provenance"]["transport.task_sequence"]) == api.TASK_SEQUENCES["deliver"]


def test_unavailable_vla_is_rejected_before_generation_or_environment_creation(store, monkeypatch):
    class Registry:
        def validate(self, spec):
            raise ValueError("VLA offline; no fallback")
    monkeypatch.setattr(store, "_policies", Registry)
    with pytest.raises(ValueError, match="VLA offline"):
        store.create(TransportSpec(controller="vla", object_mode="generate"), "offline-vla")
    assert not Plant.instances
    assert not list(store.root.glob("*/run.json"))


@pytest.mark.parametrize("shutdown", [False, True])
def test_stop_or_shutdown_during_blocked_vla_preserves_prior_frames_without_applying_late_action(store, monkeypatch, shutdown):
    from vlab import transport_policy as policy
    identity = {"id": "test-vla", "kind": "vla", "learned_policy": True, "weights_sha256": "a"*64,
                "action_schema": "mobile7_v1", "action_dim": 7}
    agent = policy.VLAController("http://test.local", identity, {"vla_replan_steps": 2})
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()
    class Registry:
        def validate(self, _spec):
            return identity
        def create(self, _spec):
            return agent
    monkeypatch.setattr(store, "_policies", Registry)
    monkeypatch.setattr(agent, "_packet", lambda _env, _state, preview: {
        "request_id": str(preview["tick"]), "session_id": agent.session,
        "tick": preview["tick"], "max_steps": 2})
    def infer(_url, packet, **_kwargs):
        if packet["tick"] == 2:
            entered.set()
            release.wait(timeout=3)
            returned.set()
        return {"request_id": packet["request_id"], "session_id": packet["session_id"],
            "tick": packet["tick"], "weights_sha256": "a"*64, "action_schema": "mobile7_v1",
            "actions": [[0, 0, 0, -.5, .03, 0, 1]]*2}
    monkeypatch.setattr(policy, "_json_call", infer)
    try:
        initial = store.create(TransportSpec(controller="vla", policy_id="test-vla"), f"blocked-inference-{shutdown}")
        rid = initial["id"]
        assert entered.wait(timeout=2)
        assert store.get(rid)["policy_diagnostics"]["inference_pending"] is True
        started = time.monotonic()
        if shutdown:
            store.close()
        else:
            store.stop(rid)
        row = finished(store, rid)
        assert time.monotonic()-started < 1.5
        assert row["status"] == "cancelled" and row["episodes"][0]["frames"] == 2
        assert row["episodes"][0]["interrupted"] and not row["episodes"][0]["success"]
        assert row["policy_diagnostics"]["executed_model_actions"] == 2
        assert row["policy_diagnostics"]["cancelled_inferences"] == 1
        assert not row["policy_diagnostics"]["inference_pending"]
        assert Plant.instances[0].closed and Plant.instances[0].tick == 2
        with np.load(store._directory(rid)/"episodes/ep000.npz") as arrays:
            assert arrays["action"].shape == arrays["requested_action"].shape == (2, 7)
            assert arrays["image_front"].shape[0] == arrays["image_wrist"].shape[0] == 2
            np.testing.assert_allclose(arrays["time"], [0, .05])
        release.set()
        assert returned.wait(timeout=1)
        assert Plant.instances[0].tick == 2 and not agent.queue
    finally:
        release.set()


def test_stop_after_action_returns_does_not_apply_or_count_that_action(store, monkeypatch):
    from vlab import transport_policy as policy
    agent = policy.VLAController("http://test.local", {"kind": "vla", "learned_policy": True}, {})
    def stop_after_action(_env, _state, _preview):
        store.stop(next(iter(store._rows)))
        return {"arm": [0, 0, 0, 0], "base_linear_m_s": 0, "base_angular_rad_s": 0, "hold_arm": False}
    monkeypatch.setattr(agent, "action", stop_after_action)
    class Registry:
        def validate(self, _spec):
            pass
        def create(self, _spec):
            return agent
    monkeypatch.setattr(store, "_policies", Registry)
    row = finished(store, store.create(TransportSpec(controller="vla"), "stop-after-action")["id"])
    assert row["status"] == "cancelled" and not row["episodes"]
    assert row["policy_diagnostics"]["executed_model_actions"] == 0
    assert Plant.instances[0].closed and Plant.instances[0].tick == 0


def test_failure_rendering_next_observation_keeps_the_step_already_applied(store, monkeypatch):
    snapshot = Plant.snapshot
    def broken_after_step(self, render=True):
        if self.tick == 1:
            raise RuntimeError("next camera render failed")
        return snapshot(self, render)
    monkeypatch.setattr(Plant, "snapshot", broken_after_step)
    row = finished(store, store.create(TransportSpec(), "post-step-render-failure")["id"])
    assert row["status"] == "failed" and "next camera render failed" in row["error"]
    assert row["episodes"][0]["frames"] == 1 and row["episodes"][0]["interrupted"]
    with np.load(store._directory(row["id"])/"episodes/ep000.npz") as arrays:
        assert arrays["time"].tolist() == [0.]
        assert arrays["action"].shape == arrays["requested_action"].shape == (1, 7)
        assert arrays["image_front"].shape[0] == arrays["image_wrist"].shape[0] == 1
    assert Plant.instances[0].tick == 1 and Plant.instances[0].closed


@pytest.mark.parametrize("fault", ["missing", "stale", "render_error"])
def test_missing_or_stale_camera_metadata_stops_before_any_next_action(store, monkeypatch, fault):
    snapshot = Plant.snapshot
    def bad_camera(self, render=True):
        result = snapshot(self, render)
        if self.tick == 1:
            if fault == "missing":
                result.pop("camera_status")
            elif fault == "stale":
                result["camera_status"]["front"].update(capture_time=0, age_s=.05)
            else:
                result["camera_status"]["wrist"].update(render_error="renderer unavailable", state="render_unavailable")
        return result
    monkeypatch.setattr(Plant, "snapshot", bad_camera)
    row = finished(store, store.create(TransportSpec(), "invalid-camera-"+fault)["id"])
    assert row["status"] == "failed" and "相机" in row["error"]
    assert row["episodes"][0]["frames"] == 1
    assert Plant.instances[0].tick == 1, "Expired RGB must never advance another expert or VLA action"


def test_missing_camera_status_is_rejected_on_the_initial_observation(store, monkeypatch):
    snapshot = Plant.snapshot
    def no_metadata(self, render=True):
        result = snapshot(self, render)
        result.pop("camera_status")
        return result
    monkeypatch.setattr(Plant, "snapshot", no_metadata)
    row = finished(store, store.create(TransportSpec(), "no-initial-camera-proof")["id"])
    assert row["status"] == "failed" and not row["episodes"]
    assert Plant.instances[0].tick == 0 and Plant.instances[0].closed


def test_missing_actual_action_and_failed_fallback_keeps_a_clear_recording_boundary(store, monkeypatch):
    snapshot = Plant.snapshot
    step = Plant.step
    def missing_actual(self, action):
        step(self, action)
        return {}
    def broken_fallback(self, render=True):
        if self.tick:
            raise RuntimeError("fallback unavailable")
        return snapshot(self, render)
    monkeypatch.setattr(Plant, "step", missing_actual)
    monkeypatch.setattr(Plant, "snapshot", broken_fallback)
    row = finished(store, store.create(TransportSpec(), "unrecordable-fallback")["id"])
    assert row["status"] == "failed" and not row["episodes"]
    assert "只保存此前完整配对" in row["error"]
    assert Plant.instances[0].tick == 1 and Plant.instances[0].closed
