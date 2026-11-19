"""Asset orchestration uses a controlled provider, never paid online generation."""
import json
import shutil
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_transport_api import Plant, finished, eventually
from vlab.duck_demo import CHASSIS, DUCK
from vlab.generation.object_provider import MockLux3DProvider, ObjectJobStore
from vlab.generation.orchestrator import Orchestrator
from vlab.generation.schema import PlannerReply
from vlab.generation.sessions import SessionStore
from vlab.transport_api import TransportSpec, TransportStore, plan_transport
from vlab.transport_generation import suggest_parameters

LAB = Path(__file__).resolve().parents[1]


def awaiting(store, rid):
    row = eventually(lambda: store.get(rid) if store.get(rid)["status"] == "awaiting_asset_review" else None)
    store._workers[rid].join(timeout=3)
    assert row["asset_reviews"] and not row["episodes"] and "configuration" not in row
    return row


def approve(store, rid, decision="use_once"):
    row = awaiting(store, rid)
    for item in row["asset_reviews"]:
        store.review_asset(rid, item["role"], decision, f"approve-{rid}-{item['role']}")
    return finished(store, rid)


class ImmediateObjects(ObjectJobStore):
    def submit(self, params, *, background=False):
        return super().submit(params, background=False)

    def poll(self, jid, *, background=False):
        return super().poll(jid, background=False)

    def retry_node(self, jid, node, *, background=False):
        return super().retry_node(jid, node, background=False)


@pytest.fixture
def generated(tmp_path):
    for ref in (DUCK, CHASSIS):
        dest = tmp_path / "assets" / ref
        dest.mkdir(parents=True)
        for name in ("manifest.json", "visual.obj"):
            shutil.copyfile(LAB / "assets" / ref / name, dest / name)
    provider = MockLux3DProvider("success")
    objects = ImmediateObjects(tmp_path / "datasets/generation_jobs", provider, lab_root=tmp_path)
    sessions = SessionStore(tmp_path / "datasets/generation_sessions")
    generation = {"objects": objects, "sessions": sessions,
                  "orchestrator": Orchestrator(sessions, objects, None)}
    Plant.instances, Plant.fail, Plant.block, Plant.entered = [], False, None, None
    store = TransportStore(SimpleNamespace(load=lambda: None), tmp_path,
                           env_factory=Plant, realtime=False, generation=generation)
    yield SimpleNamespace(store=store, provider=provider, generation=generation)
    store.close()


def test_rule_suggestion_has_explicit_source_and_preserves_user_numbers():
    result = suggest_parameters("一个金属零件，高5厘米，质量25克，摩擦系数0.8", {"mass_kg": .04})
    assert result["source"] == "rules_estimate" and not result["ai_available"]
    assert result["parameters"]["height_m"] == .05
    assert result["parameters"]["mass_kg"] == .04
    assert result["parameters"]["friction"] == .8
    assert result["provenance"]["mass_kg"]["source"] == "user_explicit"
    assert result["provenance"]["width_m"]["verification"] == "unverified"


def test_configured_model_suggestions_are_validated_and_never_override_user():
    planner = SimpleNamespace(configured=lambda: True, plan=lambda session, registry: PlannerReply(
        spec_patch={"height_m": .055, "width_m": .045, "depth_m": 99},
        physics_patch={"mass_kg": .07, "friction": [.7, .003, .0001]}, complete=True))
    result = suggest_parameters("小型金属件", {"mass_kg": .03}, {"planner": planner})
    assert result["source"] == "model_inferred"
    assert result["parameters"]["height_m"] == .055
    assert result["parameters"]["mass_kg"] == .03
    assert result["parameters"]["depth_m"] == .055, "Unsafe model dimensions must be rejected"
    assert result["provenance"]["depth_m"]["source"] == "rules_estimate"
    assert result["provenance"]["friction"]["source"] == "model_inferred"


def test_model_failure_and_empty_reply_report_rule_fallback_without_leaking_provider_error():
    def fail(*_):
        raise RuntimeError("secret provider credential")
    planner = SimpleNamespace(configured=lambda: True, plan=fail)
    result = suggest_parameters("金属零件", {"mass_kg": .03}, {"planner": planner})
    assert result["ai_available"] and result["source"] == "rules_estimate"
    assert result["parameters"]["mass_kg"] == .03 and result["parameters"]["friction"] == .6
    assert "secret" not in str(result)
    assert any("暂时不可用" in note for note in result["notes"])
    planner.plan = lambda *_: PlannerReply(complete=True)
    result = suggest_parameters("橡胶零件", {}, {"planner": planner})
    assert result["source"] == "rules_estimate"
    assert result["parameters"]["friction"] == 1.8


def test_generated_object_automatically_infers_missing_parameters_and_keeps_sources(generated):
    calls = []
    def plan(*_):
        calls.append(True)
        return PlannerReply(spec_patch={"height_m": .045, "width_m": .04, "depth_m": .04},
                            physics_patch={"mass_kg": .06, "friction": [.9, .003, .0001]}, complete=True)
    generated.generation["planner"] = SimpleNamespace(configured=lambda: True, plan=plan)
    spec = TransportSpec(object_mode="generate", object_description="红色小块",
                         object_parameters={"mass_kg": .025, "width_m": .035},
                         object_parameter_sources={"mass_kg": "user_explicit", "width_m": "rules_estimate"})
    row = approve(generated.store, generated.store.create(spec, "automatic-parameters")["id"])
    assert row["status"] == "succeeded", row
    assert calls == [True] and generated.provider.submits == 1
    assert row["spec"]["object_parameters"]["mass_kg"] == .025
    assert row["spec"]["object_parameters"]["width_m"] == .035
    assert row["spec"]["object_parameters"]["height_m"] == .045
    assert row["spec"]["object_parameter_sources"]["height_m"] == "model_inferred"
    assert row["spec"]["object_parameter_sources"]["width_m"] == "rules_estimate"
    assert row["parameter_provenance"]["mass_kg"]["source"] == "user_explicit"
    session = generated.generation["sessions"].get(row["asset_steps"][0]["session_id"])
    assert session.asset_spec.height.provenance.source.value == "model_inferred"


def test_reused_invalid_mass_and_bin_mesh_are_rejected_before_paid_submit(generated):
    path = generated.store.lab / "assets" / DUCK / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["mass_kg"] = 100
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="mass|质量"):
        generated.store.create(TransportSpec(bin_mode="generate"), "invalid-mass")
    assert generated.provider.submits == 0 and not Plant.instances
    manifest.update(kind="static_prop", role="visual_only", mass_kg=.03)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    path.with_name("visual.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    with pytest.raises(ValueError):
        generated.store.create(TransportSpec(object_mode="generate", bin_mode="reuse", bin_asset=DUCK), "invalid-bin")
    assert generated.provider.submits == 0 and not Plant.instances


@pytest.mark.parametrize("height,bounds", [
    (.12, [[-.02, -.02, 0], [.02, .02, .12]]),
    (.06, [[-.10, -.02, 0], [.10, .02, .06]]),
])
def test_large_original_asset_is_rejected_before_charge_even_with_smaller_override(generated, height, bounds):
    path = generated.store.lab / "assets" / DUCK / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(height_m=height, bounds_m=bounds)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    spec = TransportSpec(bin_mode="generate", object_parameters={
        "height_m": .055, "width_m": .04, "depth_m": .04})
    with pytest.raises(ValueError, match="原始Aholo物品"):
        generated.store.create(spec, "large-source")
    assert generated.provider.submits == 0 and not Plant.instances


def test_two_generated_roles_reach_simulation_once_with_persisted_parameters(generated):
    spec = TransportSpec(object_mode="generate", object_description="红色橡胶球，高4厘米，质量25克",
                         object_parameters={"width_m": .04, "depth_m": .04},
                         bin_mode="generate", bin_description="深蓝色无盖敞口箱",
                         target_xy=[.6, .8])
    assert plan_transport(spec)["generation_count"] == 2
    first = generated.store.create(spec, "two-generated")
    row = approve(generated.store, first["id"])
    assert row["status"] == "succeeded", row
    assert generated.provider.submits == 2
    assert len(Plant.instances) == 1
    assert all(step["status"] == "succeeded" for step in row["asset_steps"])
    assert row["assets"]["object"]["source"] == row["assets"]["bin"]["source"] == "lux3d"
    assert row["spec"]["object_parameters"]["height_m"] == .04
    assert row["spec"]["object_parameters"]["mass_kg"] == .025
    assert generated.store.create(spec, "two-generated")["id"] == first["id"]
    assert generated.provider.submits == 2
    object_step = row["asset_steps"][0]
    session = generated.generation["sessions"].get(object_step["session_id"])
    assert session.physics_spec.mass.value_kg == .025
    assert session.asset_spec.height.value_m == .04


def test_unknown_submit_blocks_simulation_and_never_charges_again(generated):
    generated.provider.behavior = "submit_timeout"
    spec = TransportSpec(object_mode="generate", bin_mode="generate")
    row = finished(generated.store, generated.store.create(spec, "unknown-submit")["id"])
    assert row["status"] == "failed"
    assert row["asset_steps"][0]["status"] == "submit_unknown"
    assert row["asset_steps"][1]["status"] == "pending"
    assert not Plant.instances
    assert generated.provider.submits == 1
    assert generated.store.create(spec, "unknown-submit")["id"] == row["id"]
    generated.store._workers[row["id"]].join(timeout=3)
    with pytest.raises(ValueError, match="未知"):
        generated.store.retry(row["id"])
    assert generated.provider.submits == 1


def test_generation_stop_prevents_second_role_and_simulation(generated):
    entered, release = threading.Event(), threading.Event()
    original = generated.provider.submit
    def blocked(prompt, payload):
        entered.set()
        release.wait(timeout=3)
        return original(prompt, payload)
    generated.provider.submit = blocked
    row = generated.store.create(TransportSpec(object_mode="generate", bin_mode="generate"), "cancel-generation")
    assert entered.wait(timeout=3)
    start = time.monotonic()
    assert generated.store.stop(row["id"])["status"] == "stopping"
    assert time.monotonic()-start < .2
    release.set()
    final = finished(generated.store, row["id"])
    assert final["status"] == "cancelled"
    assert generated.provider.submits == 1
    assert not Plant.instances


def test_known_remote_download_recovers_same_paid_jobs_and_parameters(generated):
    original = generated.provider.download
    attempts = []
    def once(outputs, directory):
        attempts.append(True)
        if len(attempts) == 1:
            raise OSError("controlled download failure")
        return original(outputs, directory)
    generated.provider.download = once
    spec = TransportSpec(object_mode="generate", object_description="红色小球，高4厘米",
                         bin_mode="generate")
    first = generated.store.create(spec, "recover-download")
    row = finished(generated.store, first["id"])
    generated.store._workers[row["id"]].join(timeout=3)
    assert row["status"] == "failed"
    assert generated.provider.submits == 1
    job_id, session_id = row["asset_steps"][0]["job_id"], row["asset_steps"][0]["session_id"]
    resumed = generated.store.retry(row["id"])
    final = approve(generated.store, resumed["id"])
    assert final["status"] == "succeeded", final
    assert generated.provider.submits == 2, "One object and one bin, without repeating the object submit"
    assert final["asset_steps"][0]["job_id"] == job_id
    assert final["asset_steps"][0]["session_id"] == session_id
    assert final["spec"]["object_parameters"]["height_m"] == .04
