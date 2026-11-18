"""Read-only business-code review reproductions. Providers are local test doubles.

Run with the simulation venv Python from outputs/vla-lab. Temporary job/session
stores isolate the probes from real projects. No paid or LLM requests are made.
"""
from __future__ import annotations

import json
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

def _find_lab(start: Path) -> Path:
    """Locate the vla-lab root whether this script sits in outputs/ or in docs/."""
    for p in [start, *start.parents]:
        for cand in (p, p / "vla-lab"):
            if (cand / "vlab" / "generation").is_dir() and (cand / "configs").is_dir():
                return cand
    return start.parents[1] / "vla-lab"


LAB = _find_lab(Path(__file__).resolve().parent)
sys.path.insert(0, str(LAB))

from fastapi import FastAPI
from fastapi.testclient import TestClient
from asset_pipeline.import_object import register_object
from vlab.generation.api import register_generation_api
from vlab.generation.compiler import compile_session
from vlab.generation.object_provider import ObjectJobStore, MockLux3DProvider
from vlab.generation.orchestrator import Orchestrator
from vlab.generation.planner import Planner, apply_reply
from vlab.generation.scene_recipe import recipe_to_config_data, validate_recipe
from vlab.generation.schema import (
    AssetNode, FieldProvenance, FieldSource, Confirmation, PlannerReply,
    PlannerQuestion, SceneRecipe,
)
from vlab.generation.sessions import SessionStore
from vlab.schema import ProjectConfig
from vlab.store import ConfigStore

RESULTS = []


def record(name, fn):
    try:
        detail = fn()
        RESULTS.append({"name": name, "reproduced": True, "detail": detail})
    except Exception as exc:
        RESULTS.append({"name": name, "reproduced": False,
                        "error": f"{type(exc).__name__}: {exc}"})


def stores(root, behavior="success"):
    ss = SessionStore(root / "sessions")
    s = ss.create()
    s.asset_spec.name = "cup"
    s.asset_spec.height.value_m = .09
    s.asset_spec.radius.value_m = .03
    s.physics_spec.mass.value_kg = .12
    s = ss.save(s, s.revision)
    jobs = ObjectJobStore(root / "jobs", MockLux3DProvider(behavior), lab_root=root)
    return ss, s, jobs, Orchestrator(ss, jobs, None)


def probe_preview(root):
    ss, s, jobs, orch = stores(root)
    s.compiled = compile_session(s, prompt_override="A single blue cup. CUSTOM_REVIEW_DETAIL.")
    s = ss.save(s, s.revision)
    preview = s.compiled.prompt_en
    result = orch.confirm(s.session_id, client_request_id="preview", background=False)
    actual = result["job"]["params"]["prompt"]
    assert preview != actual and "CUSTOM_REVIEW_DETAIL" not in actual
    return {"preview": preview, "submitted": actual}


def probe_regeneration(root):
    ss, s, jobs, orch = stores(root)
    first = orch.confirm(s.session_id, client_request_id="first", background=False)["job"]
    jobs.poll(first["id"])
    first = jobs.get(first["id"])
    s = ss.get(s.session_id)
    s.asset_spec.appearance = "blue"
    s = ss.save(s, s.revision)
    second = orch.confirm(s.session_id, client_request_id="second", background=False)["job"]
    jobs.poll(second["id"])
    second = jobs.get(second["id"])
    mf = json.loads((root / "assets" / "imported" / second["slug"] / "manifest.json").read_text())
    assert second["state"] == "succeeded"
    assert first["asset_id"] == second["asset_id"]
    assert mf["generation_id"] == first["remote_id"] != second["remote_id"]
    return {"first_remote": first["remote_id"], "second_remote": second["remote_id"],
            "same_asset_id": second["asset_id"], "registered_generation_id": mf["generation_id"],
            "second_status": second["state"], "hash_matches_second": mf["spec_hash"] == second["spec_hash"]}


def probe_duplicate(root):
    _, _, jobs, _ = stores(root)
    find = jobs.find_by_request
    gate = threading.Barrier(2)

    def concurrent_find(key):
        existing = find(key)
        gate.wait(timeout=10)
        return existing

    params = {"prompt": "cup", "payload": {}, "slug": "same-cup", "client_request_id": "same-key"}
    with patch.object(jobs, "find_by_request", concurrent_find):
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(jobs.submit, params, background=False) for _ in range(2)]
            results = [f.result(timeout=15) for f in futures]
    assert jobs.provider.submits == 2 and results[0]["id"] != results[1]["id"]
    return {"client_request_id": "same-key", "local_jobs": len(jobs.list()),
            "provider_submits": jobs.provider.submits}


def probe_dimensions(root):
    cup = LAB / "assets/imported/lux3d-blue-live-5887561/original.glb"
    p = register_object(cup, root / "tray", kind="static_prop", role="static",
                        height=.02, width=.20, depth=.15)
    mf = json.loads(p.read_text())
    actual = mf["collision_proxy"]["size_m"]
    assert abs(actual[0] - .20) > .01 and abs(actual[1] - .15) > .01
    return {"requested_m": [.20, .15, .02], "actual_proxy_m": actual,
            "calibrated": mf["calibrated"], "proxy_basis": mf["collision_proxy"]["basis"]}


def api_fixture(root):
    app = FastAPI()
    bundle = register_generation_api(app, root)
    # Never use ambient credentials during a review probe.
    bundle["planner"].base_url = ""
    bundle["planner"].api_key = None
    return TestClient(app, raise_server_exceptions=False), bundle


def probe_scene_create(root):
    c, bundle = api_fixture(root)
    with c:
        r = c.post("/api/v1/generation/sessions", json={
            "kind": "environment_splat", "initial_text": "a laboratory background"})
        assert r.status_code == 200
        s = r.json()
        got = c.get(f"/api/v1/generation/sessions/{s['session_id']}")
        assert got.status_code == 500
        return {"create_status": r.status_code, "created_kind": s["asset_spec"]["kind"],
                "created_role": s["asset_spec"]["role"], "reload_status": got.status_code}


def probe_answer(root):
    c, bundle = api_fixture(root)
    s = bundle["sessions"].create()
    s.questions = [PlannerQuestion(id="width", text="How wide?", options=["20 cm", "30 cm"])]
    s = bundle["sessions"].save(s, s.revision)
    with c:
        r = c.post(f"/api/v1/generation/sessions/{s.session_id}/answer", json={
            "question_id": "width", "answer": "20 cm", "expected_revision": s.revision})
        assert r.status_code == 200
    saved = bundle["sessions"].get(s.session_id)
    planner = Planner(base_url="http://unused.invalid", api_key="review-placeholder")
    sent = []
    def capture(messages):
        sent.extend(messages)
        return '{"reply_zh":"ok","complete":true}'
    with patch.object(planner, "_chat", capture):
        planner.plan(saved, {})
    serialized = json.dumps(sent, ensure_ascii=False)
    assert saved.asset_spec.width.value_m is None and "20 cm" not in serialized
    return {"answer_stored": saved.questions[0].answer,
            "structured_width_m": saved.asset_spec.width.value_m,
            "answer_visible_to_next_llm_request": "20 cm" in serialized,
            "state": saved.state.value}


def probe_nodes(root):
    config = ProjectConfig.model_validate_json((LAB / "configs/default.json").read_text())
    store = ConfigStore(root / "projects")
    store.save(config.model_copy(update={"project_id": "default"}))
    recipe = SceneRecipe(recipe_id="missing-node", task_asset_ref="builtin/standard",
                         asset_nodes=[AssetNode(node_id="generated-prop", asset_ref="imported/DOES_NOT_EXIST",
                                                position=[.25, -.15, .05])])
    report = validate_recipe(recipe, LAB, store, "default")
    data = recipe_to_config_data(recipe, config, "default")
    assert report["valid"] and "DOES_NOT_EXIST" not in json.dumps(data)
    return {"nonexistent_asset_ref": recipe.asset_nodes[0].asset_ref,
            "validation_passed": report["valid"], "node_present_in_project": False,
            "warnings": report["warnings"]}


def probe_unknown_recover(root):
    ss, s, jobs, orch = stores(root, "submit_timeout")
    first = orch.confirm(s.session_id, client_request_id="recover", background=False)["job"]
    second = orch.confirm(s.session_id, client_request_id="recover", remote_id="known-remote",
                          background=False)["job"]
    assert first["id"] == second["id"] and second["remote_id"] is None
    return {"same_job": True, "requested_remote_id": "known-remote", "saved_remote_id": None,
            "state": second["state"]}


def probe_partial_download(root):
    _, _, jobs, _ = stores(root)
    job = jobs.submit({"prompt": "cup", "payload": {}, "slug": "partial", "client_request_id": "partial",
                       "kind": "task_object", "height_m": .09, "radius_m": .03, "mass_kg": .12})
    out = jobs.dir / job["id"] / "outputs"
    out.mkdir(parents=True)
    (out / "result.glb").write_bytes(b"glTF-truncated-download")
    with patch.object(jobs.provider, "download", wraps=jobs.provider.download) as download:
        jobs.poll(job["id"])
        jobs.retry_node(job["id"], "download")
        calls = download.call_count
    final = jobs.get(job["id"])
    assert calls == 0 and final["state"] == "retryable_error"
    return {"download_calls_after_explicit_download_retry": calls, "final_state": final["state"],
            "error": final["error"][:180]}


def probe_friction(root):
    ss = SessionStore(root / "sessions")
    s = ss.create()
    s.physics_spec.friction = [.95, .005, .0001]
    s.physics_spec.friction_provenance = FieldProvenance(
        source=FieldSource.user_explicit, confirmation=Confirmation.accepted)
    apply_reply(s, PlannerReply(physics_patch={"friction": [0., 0., 0.]}))
    assert s.physics_spec.friction == [0., 0., 0.]
    return {"user_accepted": [.95, .005, .0001], "after_llm": s.physics_spec.friction,
            "source": s.physics_spec.friction_provenance.source.value}


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="roboforge-review-") as td:
        root = Path(td)
        for name, fn in [
            ("preview_discarded", probe_preview),
            ("regeneration_reuses_old_asset", probe_regeneration),
            ("concurrent_duplicate_remote_submits", probe_duplicate),
            ("width_depth_ignored", probe_dimensions),
            ("world_session_unreadable", probe_scene_create),
            ("clarification_answer_lost", probe_answer),
            ("asset_nodes_not_applied", probe_nodes),
            ("unknown_submit_cannot_attach_remote", probe_unknown_recover),
            ("download_retry_reuses_partial", probe_partial_download),
            ("confirmed_friction_overwritten", probe_friction),
        ]:
            record(name, lambda fn=fn, name=name: fn(root / name))
    result = {"scope": "temporary stores; local mocks/real local assets; no network generation",
              "results": RESULTS}
    out = Path(__file__).with_name("results.json")
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
