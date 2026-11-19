"""Scenario orchestration contracts with controlled generation and real geometry.

MockLux3DProvider is a test double, never an online generation success. Reused
assets are actual Aholo meshes; all mutable projects and outputs stay in tmp_path.
"""
from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))

from vlab.duck_demo import CHASSIS, DUCK
from vlab.engine import PlatformEnv
from vlab.generation.object_provider import MockLux3DProvider, ObjectJobStore
from vlab.generation.orchestrator import Orchestrator
from vlab.generation.sessions import SessionStore
from vlab.schema import ProjectConfig
from vlab.store import ConfigStore
from vlab.scenarios import ScenarioSpec, ScenarioStore, plan_spec


class SynchronousObjects(ObjectJobStore):
    """Make the controlled provider deterministic without changing production."""

    def submit(self, params, *, background=False):
        return super().submit(params, background=False)

    def poll(self, jid, *, background=False):
        return super().poll(jid, background=False)

    def retry_node(self, jid, node, *, background=False):
        return super().retry_node(jid, node, background=False)


class CaptureJobs:
    def __init__(self):
        self.created = []
        self.rows = {}

    def create(self, kind, params):
        row = {"id": f"collect-{len(self.created)}", "kind": kind,
               "params": copy.deepcopy(params), "status": "running"}
        self.created.append(row)
        self.rows[row["id"]] = row
        return copy.deepcopy(row)

    def get(self, jid):
        return copy.deepcopy(self.rows.get(jid))

    def list(self):
        return copy.deepcopy(list(self.rows.values()))

    def stop(self, jid):
        self.rows[jid]["status"] = "cancelled"
        return self.get(jid)


@pytest.fixture
def lab(tmp_path):
    configs = ConfigStore(tmp_path / "projects")
    cfg = ProjectConfig.model_validate_json((LAB / "configs/default.json").read_text(encoding="utf-8"))
    configs.save(cfg)
    for path in (LAB / "assets/robot").glob("*.obj"):
        dest = tmp_path / "assets/robot" / path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    for ref in (DUCK, CHASSIS):
        dest = tmp_path / "assets" / ref
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("manifest.json", "visual.obj"):
            shutil.copyfile(LAB / "assets" / ref / name, dest / name)
    provider = MockLux3DProvider("success")
    objects = SynchronousObjects(tmp_path / "datasets/generation_jobs", provider, lab_root=tmp_path)
    sessions = SessionStore(tmp_path / "datasets/generation_sessions")
    generation = {"objects": objects, "sessions": sessions,
                  "orchestrator": Orchestrator(sessions, objects, None, config_store=configs)}
    jobs = CaptureJobs()
    return SimpleNamespace(root=tmp_path, configs=configs, jobs=jobs, generation=generation,
                           objects=objects, provider=provider,
                           store=ScenarioStore(tmp_path, configs, jobs, generation))


def reused(**changes):
    data = dict(environment_mode="none", robot_mode="reuse", object_mode="reuse",
                operation="传送带不停，随机姿态抓取物体入箱，采集3次。")
    data.update(changes)
    return ScenarioSpec(**data)


def advance(store, rid, count=10):
    row = store.get(rid)
    for _ in range(count):
        if row["status"] in {"simulating", "failed", "blocked", "cancelled", "succeeded"}:
            break
        row = store.poll(rid)
    return row


def test_plan_parses_work_parameters_and_reports_generation_count():
    result = plan_spec(ScenarioSpec(operation="不停带，带速2 cm/s，固定姿态，抓取入箱，采集4次，种子7。"))
    assert result["ready"], result["issues"]
    assert result["parser"] == "structured_rules_v1"
    assert result["spec"]["episodes"] == 4
    assert result["spec"]["seed"] == 7
    assert result["spec"]["belt_speed_m_s"] == pytest.approx(.02)
    assert result["spec"]["continuous"] is True
    assert result["spec"]["random_pose"] is False
    assert result["generation_count"] >= 1
    assert result["warnings"], "The finite execution scope must remain visible"


@pytest.mark.parametrize("operation", [
    "机器人导航到仓库另一端再抓取入箱", "双臂同时抓取入箱", "抓取后倒水入杯", "抓取并装配零件",
])
def test_unimplemented_actions_are_blocked(operation):
    result = plan_spec(reused(operation=operation))
    assert not result["ready"]
    assert result["issues"]


def test_reused_assets_build_separate_real_project_and_reach_collect(lab):
    before = lab.configs.load().model_dump_json()
    row = lab.store.create(reused(operation="固定姿态，不停带抓取入箱，带速2 cm/s，采集2次，种子4。"), "reuse-1")
    row = advance(lab.store, row["id"])
    assert row["status"] == "simulating", row
    assert lab.provider.submits == 0
    assert len(lab.jobs.created) == 1
    cfg = lab.configs.load(row["project_id"])
    assert cfg.project_id != "default"
    assert cfg.task.asset == DUCK
    assert cfg.task.duck_conveyor.chassis_asset == CHASSIS
    assert cfg.task.duck_conveyor.continuous
    assert cfg.task.duck_conveyor.random_pose is False
    assert cfg.task.duck_conveyor.belt_speed_m_s == pytest.approx(.02)
    assert lab.configs.load().model_dump_json() == before
    params = lab.jobs.created[0]["params"]
    assert params["realtime"] is True
    assert [p["seed"] for p in params["plan"]] == [4, 5]
    assert all(p["asset"] == DUCK for p in params["plan"])


def test_request_id_is_bound_to_spec_and_survives_store_restart(lab):
    spec = reused(environment_mode="generate", environment="仓库货架与标识，前方留空。")
    first = lab.store.create(spec, "stable-request")
    second = lab.store.create(spec, "stable-request")
    assert first["id"] == second["id"]
    assert lab.provider.submits == 1
    reopened = ScenarioStore(lab.root, lab.configs, lab.jobs, lab.generation)
    assert reopened.create(spec, "stable-request")["id"] == first["id"]
    assert lab.provider.submits == 1
    with pytest.raises(ValueError):
        reopened.create(spec.model_copy(update={"object": "红色小鸭"}), "stable-request")


def test_concurrent_duplicate_requests_do_not_create_paid_duplicates(lab):
    spec = reused(environment_mode="generate")
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(lambda _: lab.store.create(spec, "parallel-request"), range(4)))
    assert len({row["id"] for row in rows}) == 1
    assert lab.provider.submits == 1
    assert not lab.jobs.created


def test_lost_role_receipt_recovers_existing_generation_after_restart(lab):
    row = lab.store.create(reused(environment_mode="generate"), "lost-receipt")
    step = next(step for step in row["steps"] if step["role"] == "environment")
    assert step["job_id"]
    # The provider job was persisted, then the process died before its receipt
    # reached the scenario. Recovery must confirm the durable per-role ID.
    step.pop("job_id")
    step["status"] = "pending"
    path = lab.store.dir / f"{row['id']}.json"
    path.write_text(json.dumps(row), encoding="utf-8")
    reopened = ScenarioStore(lab.root, lab.configs, lab.jobs, lab.generation)
    recovered = advance(reopened, row["id"])
    assert recovered["status"] == "simulating", recovered
    assert lab.provider.submits == 1
    assert len(lab.jobs.created) == 1


def test_lost_collect_receipt_recovers_existing_job_after_restart(lab):
    row = advance(lab.store, lab.store.create(reused(), "lost-collect-receipt")["id"])
    assert row["status"] == "simulating", row
    original_job = row.pop("job_id")
    row["status"] = "assembling"
    path = lab.store.dir / f"{row['id']}.json"
    path.write_text(json.dumps(row), encoding="utf-8")
    reopened = ScenarioStore(lab.root, lab.configs, lab.jobs, lab.generation)
    recovered = advance(reopened, row["id"])
    assert recovered["status"] == "simulating", recovered
    assert recovered["job_id"] == original_job
    assert len(lab.jobs.created) == 1


def test_generated_environment_is_registered_and_attached_to_real_scene(lab):
    spec = reused(environment_mode="generate", environment="一组仓库货架，前方留空。")
    row = advance(lab.store, lab.store.create(spec, "generated-environment")["id"])
    assert row["status"] == "simulating", row
    assert lab.provider.submits == 1
    cfg = lab.configs.load(row["project_id"])
    assert len(cfg.scene.props) >= 1
    prop = cfg.scene.props[0]
    assert prop.asset.startswith("imported/")
    manifest = json.loads((lab.root / "assets" / prop.asset / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["kind"] == "static_prop" and manifest["role"] == "static"
    assert manifest["source"] == "lux3d"
    assert manifest["generation_id"].startswith("mock-")
    assert prop.collision
    assert prop.position[1] > .25, "Background prop must not cover the picking volume"


def test_all_three_generated_assets_reach_assembly_once(lab):
    spec = reused(environment_mode="generate", robot_mode="generate", object_mode="generate")
    row = advance(lab.store, lab.store.create(spec, "generate-all")["id"])
    assert row["status"] == "simulating", row
    assert lab.provider.submits == 3
    cfg = lab.configs.load(row["project_id"])
    generated = {step["role"]: step for step in row["steps"]}
    assert all(generated[role].get("session_id") and generated[role].get("job_id")
               and generated[role].get("asset_id") for role in ("environment", "robot", "object"))
    assert cfg.task.asset == generated["object"]["asset_id"]
    assert cfg.task.duck_conveyor.chassis_asset == generated["robot"]["asset_id"]
    assert cfg.scene.props[0].asset == generated["environment"]["asset_id"]
    for _ in range(3):
        lab.store.poll(row["id"])
    assert lab.provider.submits == 3
    assert len(lab.jobs.created) == 1


@pytest.mark.parametrize("behavior", ["failed", "submit_timeout"])
def test_generation_failure_or_unknown_submit_never_starts_simulation(lab, behavior):
    lab.provider.behavior = behavior
    row = lab.store.create(reused(environment_mode="generate"), f"error-{behavior}")
    row = advance(lab.store, row["id"])
    assert row["status"] in {"failed", "blocked"}, row
    assert not lab.jobs.created
    assert lab.provider.submits == 1
    for _ in range(3):
        lab.store.poll(row["id"])
    assert lab.provider.submits == 1
    if behavior == "submit_timeout":
        try:
            lab.store.retry(row["id"])
        except ValueError:
            pass
        assert lab.provider.submits == 1
        assert not lab.jobs.created


def test_stop_before_assets_finish_never_starts_collect(lab):
    row = lab.store.create(reused(environment_mode="generate"), "stop-before-collect")
    assert not lab.jobs.created
    stopped = lab.store.stop(row["id"])
    assert stopped["status"] == "cancelled"
    for _ in range(3):
        assert lab.store.poll(row["id"])["status"] == "cancelled"
    assert not lab.jobs.created
    assert lab.provider.submits == 1


def test_stop_running_scenario_stops_collection(lab):
    row = advance(lab.store, lab.store.create(reused(), "stop-running")["id"])
    assert row["status"] == "simulating", row
    stopped = lab.store.stop(row["id"])
    assert stopped["status"] == "cancelled"
    assert lab.jobs.get(row["job_id"])["status"] == "cancelled"


def test_invalid_reused_asset_never_collects(lab):
    # A real but too-tall object is incompatible with the configured gripper.
    path = lab.root / "assets" / DUCK / "manifest.json"
    mf = json.loads(path.read_text(encoding="utf-8"))
    mf["height_m"] = .15
    path.write_text(json.dumps(mf), encoding="utf-8")
    try:
        row = lab.store.create(reused(environment_mode="generate"), "too-tall")
        row = advance(lab.store, row["id"])
        assert row["status"] in {"blocked", "failed"}, row
    except ValueError:
        pass
    assert not lab.jobs.created
    assert lab.provider.submits == 0


def test_robot_shell_cannot_be_used_as_dynamic_object(lab):
    try:
        row = lab.store.create(reused(environment_mode="generate", object_asset=CHASSIS), "wrong-role")
        row = advance(lab.store, row["id"])
        assert row["status"] in {"blocked", "failed"}, row
    except ValueError:
        pass
    assert not lab.jobs.created
    assert lab.provider.submits == 0


def test_finished_collection_updates_scenario_status(lab):
    row = advance(lab.store, lab.store.create(reused(), "completed-collect")["id"])
    assert row["status"] == "simulating", row
    lab.jobs.rows[row["job_id"]]["status"] = "succeeded"
    assert lab.store.poll(row["id"])["status"] == "succeeded"


def test_stop_during_scene_probe_prevents_new_collection(lab, monkeypatch):
    row = lab.store.create(reused(), "stop-during-probe")
    original = PlatformEnv.step
    stopped = False

    def step_and_stop(env, action):
        nonlocal stopped
        if not stopped:
            stopped = True
            lab.store.stop(row["id"])
        return original(env, action)

    monkeypatch.setattr(PlatformEnv, "step", step_and_stop)
    result = lab.store.poll(row["id"])
    assert stopped, "Cancellation must occur during the real validation probe"
    assert result["status"] == "cancelled"
    assert not lab.jobs.created
    assert lab.provider.submits == 0


def test_stop_during_first_confirm_prevents_later_paid_roles(lab, monkeypatch):
    original = lab.generation["orchestrator"].confirm
    roles = []

    def confirm_and_stop(sid, **kwargs):
        rid, role = kwargs["client_request_id"].split(":")
        roles.append(role)
        result = original(sid, **kwargs)
        lab.store.stop(rid)
        return result

    monkeypatch.setattr(lab.generation["orchestrator"], "confirm", confirm_and_stop)
    row = lab.store.create(reused(environment_mode="generate", robot_mode="generate",
                                  object_mode="generate"), "stop-during-confirm")
    assert row["status"] == "cancelled"
    assert roles == ["environment"]
    assert lab.provider.submits == 1
    assert not lab.jobs.created
    assert lab.store.poll(row["id"])["status"] == "cancelled"
    assert lab.provider.submits == 1


def test_stop_recovers_lost_collection_receipt(lab):
    row = advance(lab.store, lab.store.create(reused(), "stop-lost-receipt")["id"])
    assert row["status"] == "simulating", row
    job_id = row.pop("job_id")
    row["status"] = "assembling"
    (lab.store.dir / f"{row['id']}.json").write_text(json.dumps(row), encoding="utf-8")
    stopped = lab.store.stop(row["id"])
    assert stopped["status"] == "cancelled"
    assert stopped["job_id"] == job_id
    assert lab.jobs.get(job_id)["status"] == "cancelled"
    assert len(lab.jobs.created) == 1


def test_stop_after_cancel_poll_still_reconciles_late_collection(lab):
    row = lab.store.create(reused(), "cancel-poll-late-job")
    # A cancellation can be consumed by poll before stop acquires its mutation
    # lock. A collect job whose durable receipt arrives afterwards still needs
    # reconciliation, even though the scenario already says cancelled.
    (lab.store.dir / f"{row['id']}.cancel").write_text("cancel requested", encoding="ascii")
    assert lab.store.poll(row["id"])["status"] == "cancelled"
    late_job = lab.jobs.create("collect", {"provenance": {"scenario_id": row["id"]}})
    stopped = lab.store.stop(row["id"])
    assert stopped["status"] == "cancelled"
    assert stopped["job_id"] == late_job["id"]
    assert lab.jobs.get(late_job["id"])["status"] == "cancelled"


def test_uncalibrated_environment_reuse_blocks_other_paid_roles(lab):
    source = lab.root / "assets" / DUCK
    destination = lab.root / "assets/imported/uncalibrated-environment"
    shutil.copytree(source, destination)
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(kind="static_prop", role="visual_only", calibrated=False,
                    physics_status="uncalibrated")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    spec = reused(environment_mode="reuse", environment_asset="imported/uncalibrated-environment",
                  robot_mode="generate", object_mode="generate")
    with pytest.raises(ValueError, match="calibrated"):
        lab.store.create(spec, "uncalibrated-preflight")
    assert lab.provider.submits == 0
    assert not lab.jobs.created
    assert not lab.store.list(), "Invalid reuse must fail before persisting a payable run"


@pytest.mark.parametrize("speed,continuous,mount", [
    (.01, True, "four_wheel"), (.04, True, "four_wheel"),
    (.025, False, "four_wheel"), (.025, True, "fixed_arm"),
])
@pytest.mark.parametrize("seed", [0, 3])
def test_work_settings_drive_real_finite_physics(lab, speed, continuous, mount, seed):
    spec = reused(operation="抓取物体放入收纳箱，采集1次。", belt_speed_m_s=speed,
                  continuous=continuous, robot_mount=mount, seed=seed)
    row = advance(lab.store, lab.store.create(spec, "physics-settings")["id"])
    assert row["status"] == "simulating", row
    cfg = lab.configs.load(row["project_id"])
    cfg.sensors = []
    assert cfg.task.duck_conveyor.belt_speed_m_s == speed
    assert cfg.task.duck_conveyor.continuous is continuous
    if mount == "fixed_arm":
        assert all(link.name != "mobile_chassis" for link in cfg.robots[0].links)
        assert all(joint.name != "chassis_mount" for joint in cfg.robots[0].joints)
    env = PlatformEnv(cfg, lab.root)
    try:
        assert env.model.neq == 0, "No artificial grasp attachment"
        for _ in range(500):
            action = env.servo_expert_action()
            assert np.isfinite(action).all()
            env.step(action)
            assert np.isfinite(env.data.qpos).all()
            assert np.isfinite(env.data.qvel).all()
            if continuous and env.scheduler.tick > 3:
                assert env._duck_task.speed == pytest.approx(speed)
            if env.metrics()["success"]:
                break
        metrics = env.metrics()
        if metrics["success"]:
            assert metrics["grasped"] and metrics["lifted"]
            assert metrics["controlled_release"] and not metrics["delivery_failed"]
            assert metrics["tray_contact"] and metrics["contacts"] == 0
            assert env._judge.inside() and metrics["stable_ticks"] >= 6
        if not continuous:
            assert env._duck_task.speed == 0
        print(json.dumps({"physics_settings": {"speed": speed, "continuous": continuous,
                           "mount": mount, "seed": seed}, "success": metrics["success"],
                          "delivery_failed": metrics["delivery_failed"],
                          "stage": env.expert_stage(), "ticks": env.scheduler.tick}))
    finally:
        env.close()
