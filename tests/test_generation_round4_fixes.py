"""Isolated regression tests for Round 4 code review defects F01-F10.

All tests run against tmp_path stores and controlled local doubles (plus real
MuJoCo builds and the real local cup GLB). No online paid API or LLM requests
are made, and no repository project configs are modified.
"""
from __future__ import annotations

import io
import json
import shutil
import sqlite3
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np
import pytest
import trimesh
from fastapi.testclient import TestClient

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))

from asset_pipeline.import_asset import download_mesh  # noqa: E402
from asset_pipeline.import_object import register_object  # noqa: E402
from vlab.api import create_app  # noqa: E402
from vlab.engine import PlatformEnv  # noqa: E402
from vlab.generation.compiler import compile_session  # noqa: E402
from vlab.generation.object_provider import MockLux3DProvider, ObjectJobStore  # noqa: E402
from vlab.generation.orchestrator import Orchestrator  # noqa: E402
from vlab.generation.planner import Planner, apply_reply  # noqa: E402
from vlab.generation.scene_recipe import recipe_to_config_data, validate_recipe  # noqa: E402
from vlab.generation.schema import (  # noqa: E402
    AssetKind, AssetNode, AssetRole, Confirmation, FieldProvenance, FieldSource,
    PlannerQuestion, PlannerReply, SceneMode, SceneRecipe,
)
from vlab.generation.sessions import SessionStore  # noqa: E402
from vlab.schema import ProjectConfig  # noqa: E402
from vlab.store import ConfigStore  # noqa: E402
from vlab.world_provider import GenerationJobStore, MockWorldProvider  # noqa: E402

CUP_GLB = LAB / "assets" / "imported" / "lux3d-blue-live-5887561" / "original.glb"


def _make_ready_session(ss: SessionStore):
    s = ss.create(project_id="default")
    s.asset_spec.name = "cup"
    s.asset_spec.height.value_m = 0.09
    s.asset_spec.radius.value_m = 0.03
    s.physics_spec.mass.value_kg = 0.12
    return ss.save(s, s.revision)


# --- F01: Atomic idempotency, payload mismatch rejection, response-loss replay ---

def test_f01_concurrent_submit_is_atomic_and_rejects_mismatch(tmp_path):
    class BarrierProvider(MockLux3DProvider):
        def submit(self, prompt, payload):
            return super().submit(prompt, payload)

    jobs = ObjectJobStore(tmp_path / "jobs", BarrierProvider(), lab_root=tmp_path)
    params = {
        "prompt": "cup", "payload": {"version": "G1-Turbo"}, "slug": "cup-conc",
        "client_request_id": "same-key", "kind": "task_object", "role": "dynamic",
        "height_m": 0.09, "radius_m": 0.03, "mass_kg": 0.12,
    }
    start_gate = threading.Barrier(4)

    def worker():
        start_gate.wait(timeout=10)
        return jobs.submit(params, background=False)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = [f.result(timeout=15) for f in [pool.submit(worker) for _ in range(4)]]

    assert len({r["id"] for r in results}) == 1
    assert len(jobs.list()) == 1
    assert jobs.provider.submits == 1

    # Same client_request_id with different payload must be rejected, not silently reused.
    diff = dict(params, prompt="completely different bowl")
    with pytest.raises(ValueError, match="different payload"):
        jobs.submit(diff, background=False)


def test_f01_world_store_concurrent_submit_and_mismatch(tmp_path):
    world = GenerationJobStore(tmp_path / "wjobs", MockWorldProvider(), lab_root=tmp_path)
    params = {"prompt": "empty lab", "client_request_id": "w-same"}
    gate = threading.Barrier(3)

    def worker():
        gate.wait(timeout=10)
        return world.submit("generation", params, background=False)

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = [f.result(timeout=15) for f in [pool.submit(worker) for _ in range(3)]]

    assert len({r["id"] for r in results}) == 1
    assert world.provider.submits == 1
    with pytest.raises(ValueError, match="different payload"):
        world.submit("generation", {"prompt": "different room", "client_request_id": "w-same"}, background=False)


def test_f01_orchestrator_response_loss_replay_and_stale_edit_conflict(tmp_path):
    ss = SessionStore(tmp_path / "sessions")
    s = _make_ready_session(ss)
    s.compiled = compile_session(s)
    s = ss.save(s, s.revision)
    rev_at_confirm = s.revision
    jobs = ObjectJobStore(tmp_path / "jobs", MockLux3DProvider(), lab_root=tmp_path)
    orch = Orchestrator(ss, jobs, None)

    first = orch.confirm(
        s.session_id,
        client_request_id="req-replay",
        expected_revision=rev_at_confirm,
        expected_compiled_revision=s.compiled.spec_revision,
        expected_request_hash=s.compiled.request_hash,
        background=False,
    )
    # Simulate lost HTTP response: client retries with the same pre-confirm revision & crid.
    replay = orch.confirm(
        s.session_id,
        client_request_id="req-replay",
        expected_revision=rev_at_confirm,
        expected_compiled_revision=s.compiled.spec_revision,
        expected_request_hash=s.compiled.request_hash,
        background=False,
    )
    assert replay["job"]["id"] == first["job"]["id"]
    assert jobs.provider.submits == 1

    # A NEW request ID carrying the old stale revision must still trigger RevisionConflict.
    from vlab.generation.sessions import RevisionConflict
    with pytest.raises(RevisionConflict):
        orch.confirm(s.session_id, client_request_id="new-crid", expected_revision=rev_at_confirm, background=False)


# --- F02: Regeneration creates immutable new asset revision without reusing/overwriting old dir ---

def test_f02_regeneration_creates_distinct_immutable_assets(tmp_path):
    ss = SessionStore(tmp_path / "sessions")
    s = _make_ready_session(ss)
    jobs = ObjectJobStore(tmp_path / "jobs", MockLux3DProvider(), lab_root=tmp_path)
    orch = Orchestrator(ss, jobs, None)

    s.compiled = compile_session(s)
    s = ss.save(s, s.revision)
    first = orch.confirm(s.session_id, client_request_id="gen-1", background=False)["job"]
    jobs.poll(first["id"])
    first = jobs.get(first["id"])

    # Modify appearance, recompile, and regenerate in the same session.
    s = ss.get(s.session_id)
    s.asset_spec.appearance = "matte cobalt blue"
    s.compiled = compile_session(s)
    s = ss.save(s, s.revision)
    second = orch.confirm(s.session_id, client_request_id="gen-2", background=False)["job"]
    jobs.poll(second["id"])
    second = jobs.get(second["id"])

    assert first["state"] == "succeeded" and second["state"] == "succeeded"
    assert first["asset_id"] != second["asset_id"]
    assert first["slug"] != second["slug"]
    mf1 = json.loads((tmp_path / "assets" / "imported" / first["slug"] / "manifest.json").read_text())
    mf2 = json.loads((tmp_path / "assets" / "imported" / second["slug"] / "manifest.json").read_text())
    assert mf1["generation_id"] == first["remote_id"] == "mock-1"
    assert mf2["generation_id"] == second["remote_id"] == "mock-2"
    assert mf1["spec_hash"] == first["spec_hash"]
    assert mf2["spec_hash"] == second["spec_hash"]
    assert mf1["spec_hash"] != mf2["spec_hash"]


def test_f02_direct_store_refuses_foreign_manifest_reuse(tmp_path):
    jobs = ObjectJobStore(tmp_path / "jobs", MockLux3DProvider(), lab_root=tmp_path)
    foreign = tmp_path / "assets" / "imported" / "foreign-slug"
    foreign.mkdir(parents=True)
    (foreign / "manifest.json").write_text(json.dumps({
        "generation_id": "other-remote", "spec_hash": "0" * 64, "sha256_glb": "0" * 64, "job_id": "other-job"
    }))
    j = jobs.submit({"prompt": "cup", "payload": {}, "slug": "foreign-slug", "client_request_id": "f1"})
    # Allocated a distinct slug automatically rather than overwriting/reusing foreign-slug.
    assert j["slug"] != "foreign-slug"
    jobs.poll(j["id"])
    assert jobs.get(j["id"])["asset_id"] == f"imported/{j['slug']}"


# --- F06: Confirm uses reviewed compiled prompt/payload, rejects stale spec & conflicting overrides ---

def test_f06_confirm_preserves_reviewed_prompt_and_rejects_stale_or_conflicting(tmp_path):
    ss = SessionStore(tmp_path / "sessions")
    s = _make_ready_session(ss)
    s.asset_spec.has_handle = False
    s = ss.save(s, s.revision)

    # Conflicting prompt override must be rejected on compile.
    from vlab.generation.compiler import CompileError
    with pytest.raises(CompileError, match="把手|handle"):
        compile_session(s, prompt_override="A cup with a handle.")
    with pytest.raises(CompileError, match="height"):
        compile_session(s, prompt_override="A giant cup 30 cm tall.")

    default_comp = compile_session(s)
    edited_comp = compile_session(s, prompt_override="A single blue ceramic cup, 9 cm tall. CUSTOM_REVIEW_DETAIL.")
    assert edited_comp.spec_hash != default_comp.spec_hash
    assert edited_comp.request_hash == edited_comp.spec_hash

    s.compiled = edited_comp
    s = ss.save(s, s.revision)

    jobs = ObjectJobStore(tmp_path / "jobs", MockLux3DProvider(), lab_root=tmp_path)
    orch = Orchestrator(ss, jobs, None)
    res = orch.confirm(
        s.session_id,
        client_request_id="preview-ok",
        expected_compiled_revision=edited_comp.spec_revision,
        expected_request_hash=edited_comp.request_hash,
        background=False,
    )
    assert "CUSTOM_REVIEW_DETAIL" in res["job"]["params"]["prompt"]
    assert res["job"]["params"]["payload"]["prompt"] == edited_comp.prompt_en

    # Mutate spec after compile -> confirm must reject stale preview and require recompile.
    s = ss.get(s.session_id)
    s.asset_spec.appearance = "red glaze"
    s = ss.save(s, s.revision)
    with pytest.raises(CompileError, match="重新编译"):
        orch.confirm(s.session_id, client_request_id="stale-after-edit", background=False)


# --- F03: World/robot session creation validates and persists cleanly without 500 ---

def test_f03_create_world_and_robot_sessions_persist_and_validate(tmp_path):
    with TestClient(create_app(tmp_path)) as c:
        for kind, text, expected_role in (
            ("environment_splat", "a laboratory background", "visual_only"),
            ("environment_splat", None, "visual_only"),
            ("robot_visual", None, "visual_only"),
            ("static_prop", "a steel tray", "static"),
        ):
            payload = {"kind": kind}
            if text is not None:
                payload["initial_text"] = text
            r = c.post("/api/v1/generation/sessions", json=payload)
            assert r.status_code == 200, r.text
            created = r.json()
            assert created["asset_spec"]["kind"] == kind
            assert created["asset_spec"]["role"] == expected_role
            got = c.get(f"/api/v1/generation/sessions/{created['session_id']}")
            assert got.status_code == 200, got.text
            assert got.json()["asset_spec"]["kind"] == kind
            assert got.json()["asset_spec"]["role"] == expected_role

        bad_kind = c.post("/api/v1/generation/sessions", json={"kind": "nonexistent_kind"})
        assert bad_kind.status_code == 400
        bad_combo = c.post("/api/v1/generation/sessions", json={"kind": "environment_splat", "role": "dynamic"})
        assert bad_combo.status_code == 400

    # Legacy DB row with invalid environment_splat/dynamic must not 500 on GET.
    ss = SessionStore(tmp_path / "datasets" / "generation_sessions")
    broken = ss.create()
    raw = json.loads(broken.model_dump_json())
    raw["asset_spec"]["kind"] = "environment_splat"
    raw["asset_spec"]["role"] = "dynamic"
    with sqlite3.connect(ss.path) as conn:
        conn.execute("UPDATE sessions SET data=? WHERE session_id=?", (json.dumps(raw), broken.session_id))
    with TestClient(create_app(tmp_path)) as c:
        repaired = c.get(f"/api/v1/generation/sessions/{broken.session_id}")
        assert repaired.status_code == 200
        assert repaired.json()["asset_spec"]["role"] == "visual_only"


# --- F07: Clarification answer updates structured spec, provenance, messages & next LLM prompt ---

def test_f07_clarification_answer_updates_spec_and_reaches_planner(tmp_path):
    with TestClient(create_app(tmp_path)) as c:
        s = c.post("/api/v1/generation/sessions", json={"kind": "static_prop", "initial_text": "一个托盘"}).json()
        sid = s["session_id"]
        # Seed open questions (one with options, one open free-text question without options).
        ss = SessionStore(tmp_path / "datasets" / "generation_sessions")
        loaded = ss.get(sid)
        loaded.questions = [
            PlannerQuestion(id="width", text="托盘多宽？", options=["20 cm", "30 cm"]),
            PlannerQuestion(id="height", text="托盘多高？", options=[]),
        ]
        loaded = ss.save(loaded, loaded.revision)

        r1 = c.post(f"/api/v1/generation/sessions/{sid}/answer", json={
            "question_id": "width", "answer": "20 cm", "expected_revision": loaded.revision,
        })
        assert r1.status_code == 200, r1.text
        d1 = r1.json()
        assert d1["asset_spec"]["width"]["value_m"] == pytest.approx(0.20)
        assert d1["asset_spec"]["width"]["provenance"]["source"] == "user_explicit"
        assert d1["asset_spec"]["width"]["provenance"]["confirmation"] == "accepted"
        assert d1["state"] == "clarifying"  # height question is still open

        # Invalid answer for typed dimension must be rejected with 400.
        bad = c.post(f"/api/v1/generation/sessions/{sid}/answer", json={
            "question_id": "height", "answer": "not-a-length", "expected_revision": d1["revision"],
        })
        assert bad.status_code == 400

        r2 = c.post(f"/api/v1/generation/sessions/{sid}/answer", json={
            "question_id": "height", "answer": "2 cm", "expected_revision": d1["revision"],
        })
        assert r2.status_code == 200, r2.text
        d2 = r2.json()
        assert d2["asset_spec"]["height"]["value_m"] == pytest.approx(0.02)
        assert d2["state"] == "ready"
        assert any("20 cm" in m["content"] for m in d2["messages"] if m["role"] == "user")

    # Next Planner.plan call sees both the persisted user answer message and questions state.
    saved = SessionStore(tmp_path / "datasets" / "generation_sessions").get(sid)
    planner = Planner(base_url="http://unused.invalid", api_key="placeholder")
    sent = []

    def capture(messages):
        sent.extend(messages)
        return '{"reply_zh":"收到","complete":true}'

    with patch.object(planner, "_chat", capture):
        planner.plan(saved, {})
    serialized = json.dumps(sent, ensure_ascii=False)
    assert "20 cm" in serialized and "2 cm" in serialized


# --- F05: Generic object import respects width/depth or honestly reports unsatisfied ---

def test_f05_import_object_dimension_strategies_and_numerical_assertions(tmp_path):
    assert CUP_GLB.is_file()
    # Default auto strategy with width/depth/height uses anisotropic bbox scaling -> exact extents.
    tray_mf = register_object(
        CUP_GLB, tmp_path / "tray-aniso", kind="static_prop", role="static",
        height=0.02, width=0.20, depth=0.15,
    )
    mf = json.loads(tray_mf.read_text(encoding="utf-8"))
    assert mf["calibrated"] is True
    assert mf["role"] == "static"
    assert mf["scale_mode"] == "bbox_anisotropic"
    assert mf["unsatisfied_dimensions"] == {}
    assert mf["collision_proxy"]["basis"] == "user_explicit"
    np.testing.assert_allclose(mf["collision_proxy"]["size_m"], [0.20, 0.15, 0.02], atol=1e-4)
    np.testing.assert_allclose(mf["actual_extents_m"], [0.20, 0.15, 0.02], atol=1e-4)
    visual_mesh = trimesh.load(tray_mf.parent / "visual.obj", process=False)
    np.testing.assert_allclose(visual_mesh.extents, [0.20, 0.15, 0.02], atol=1e-4)

    # Explicit uniform scaling with conflicting width/depth must NOT claim calibrated or user_explicit!
    uni_mf = register_object(
        CUP_GLB, tmp_path / "tray-uniform", kind="static_prop", role="static",
        height=0.02, width=0.20, depth=0.15, scale_mode="uniform",
    )
    um = json.loads(uni_mf.read_text(encoding="utf-8"))
    assert um["calibrated"] is False
    assert um["role"] == "visual_only"
    assert um["physics_status"] == "uncalibrated"
    assert um["collision_proxy"]["basis"] == "proxy_estimate"
    assert "width_m" in um["unsatisfied_dimensions"] and "depth_m" in um["unsatisfied_dimensions"]

    # Height-only box uses proxy_estimate for mesh-derived X/Y.
    h_mf = register_object(CUP_GLB, tmp_path / "height-only", kind="static_prop", role="static", height=0.10)
    hm = json.loads(h_mf.read_text(encoding="utf-8"))
    assert hm["calibrated"] is True
    assert hm["collision_proxy"]["basis"] == "proxy_estimate"

    # Wide disk with conflicting tiny radius_m must be flagged unsatisfied and uncalibrated.
    disk_path = tmp_path / "disk.glb"
    trimesh.creation.cylinder(radius=0.20, height=0.05).apply_transform(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
    ).export(disk_path)
    disk_mf = register_object(disk_path, tmp_path / "bad-radius", kind="task_object", role="dynamic",
                              height=0.09, radius=0.03)
    dm = json.loads(disk_mf.read_text(encoding="utf-8"))
    assert dm["calibrated"] is False
    assert dm["role"] == "visual_only"
    assert "radius_m" in dm["unsatisfied_dimensions"]
    assert dm["collision_proxy"]["basis"] == "proxy_estimate"


# --- F04: SceneRecipe asset_nodes enter config, API read-back, and real MuJoCo ---

def test_f04_missing_and_unsupported_asset_nodes_rejected(tmp_path):
    shutil.copytree(LAB / "assets" / "robot", tmp_path / "assets" / "robot")
    store = ConfigStore(tmp_path / "projects")
    cfg = ProjectConfig.model_validate_json((LAB / "configs" / "default.json").read_text(encoding="utf-8"))
    store.save(cfg.model_copy(update={"project_id": "default"}))

    missing = SceneRecipe(
        recipe_id="r-missing",
        task_asset_ref="builtin/standard",
        asset_nodes=[AssetNode(node_id="ghost", asset_ref="imported/DOES_NOT_EXIST", position=[0.25, -0.15, 0.0])],
    )
    rep_missing = validate_recipe(missing, tmp_path, store, "default")
    assert rep_missing["valid"] is False
    assert any("DOES_NOT_EXIST" in i for i in rep_missing["issues"])

    # Calibrated dynamic asset_nodes are now supported with real freejoints; uncalibrated dynamic stays blocked.
    dyn_node = SceneRecipe(
        recipe_id="r-dyn",
        task_asset_ref="builtin/standard",
        asset_nodes=[AssetNode(node_id="dyn_prop", asset_ref="builtin/tall", role=AssetRole.dynamic, position=[0.02, -0.09, 0.001])],
    )
    rep_dyn = validate_recipe(dyn_node, tmp_path, store, "default")
    assert rep_dyn["valid"] is True, rep_dyn

    register_object(CUP_GLB, tmp_path / "assets" / "imported" / "uncal-box", kind="task_object", role="dynamic", height=None)
    uncal_dyn = SceneRecipe(
        recipe_id="r-dyn-uncal",
        task_asset_ref="builtin/standard",
        asset_nodes=[AssetNode(node_id="bad_dyn", asset_ref="imported/uncal-box", role=AssetRole.dynamic)],
    )
    rep_uncal = validate_recipe(uncal_dyn, tmp_path, store, "default")
    assert rep_uncal["valid"] is False
    assert any("uncalibrated" in i for i in rep_uncal["issues"])


def test_f04_generated_static_prop_flows_into_recipe_project_and_mujoco(tmp_path):
    shutil.copytree(LAB / "assets" / "robot", tmp_path / "assets" / "robot")
    with TestClient(create_app(tmp_path)) as c:
        base_cfg = c.get("/api/v1/config?project=default").json()
        v0 = base_cfg["version"]

        # 1) Create static_prop session, patch dimensions, compile, confirm with MockLux3DProvider, poll.
        ss = SessionStore(tmp_path / "datasets" / "generation_sessions")
        s = ss.create(project_id="default")
        s.asset_spec.kind = AssetKind.static_prop
        s.asset_spec.role = AssetRole.static
        s.asset_spec.name = "toolbox"
        s.asset_spec.height.value_m = 0.05
        s.asset_spec.width.value_m = 0.10
        s.asset_spec.depth.value_m = 0.10
        s.compiled = compile_session(s)
        s = ss.save(s, s.revision)

        jobs = ObjectJobStore(tmp_path / "datasets" / "generation_jobs", MockLux3DProvider(), lab_root=tmp_path)
        orch = Orchestrator(ss, jobs, None)
        res = orch.confirm(s.session_id, client_request_id="prop-1", background=False)
        jid = res["job"]["id"]
        jobs.poll(jid)
        done = jobs.get(jid)
        assert done["state"] == "succeeded"

        # 2) Build SceneRecipe from session via API (omitting asset_nodes auto-populates for static_prop).
        auto_rec = c.post(f"/api/v1/generation/sessions/{s.session_id}/recipe", json={
            "scene_mode": "composed_scene",
            "task_asset_ref": "builtin/standard",
            "use_default_table": False,
            "colliders": [
                {"name": "workbench", "shape": "box", "full_size_m": [0.64, 0.43, 0.056],
                 "position": [0, 0, -0.028], "quaternion_xyzw": [0, 0, 0, 1]}
            ],
        })
        assert auto_rec.status_code == 200 and len(auto_rec.json()["asset_nodes"]) == 1, auto_rec.text

        # Explicit placement of the generated static_prop at [0.25, -0.15, 0.0].
        rec_resp = c.post(f"/api/v1/generation/sessions/{s.session_id}/recipe", json={
            "scene_mode": "composed_scene",
            "task_asset_ref": "builtin/standard",
            "use_default_table": False,
            "colliders": [
                {"name": "workbench", "shape": "box", "full_size_m": [0.64, 0.43, 0.056],
                 "position": [0, 0, -0.028], "quaternion_xyzw": [0, 0, 0, 1]}
            ],
            "asset_nodes": [
                {"node_id": "toolbox_prop", "asset_ref": done["asset_id"], "role": "static",
                 "position": [0.25, -0.15, 0.0], "quaternion_xyzw": [0, 0, 0, 1], "uniform_scale": 1.0}
            ],
        })
        assert rec_resp.status_code == 200, rec_resp.text
        rid = rec_resp.json()["recipe_id"]

        val = c.post(f"/api/v1/scene-recipes/{rid}/validate?project_id=default")
        assert val.status_code == 200 and val.json()["valid"] is True, val.json()

        ap = c.post(f"/api/v1/scene-recipes/{rid}/apply", json={
            "project_id": "default", "expected_project_version": v0,
        })
        assert ap.status_code == 200, ap.text
        ap_data = ap.json()
        assert len(ap_data["read_back"]["asset_nodes"]) == 1
        assert ap_data["read_back"]["asset_nodes"][0]["node_id"] == "toolbox_prop"
        assert ap_data["read_back"]["asset_nodes"][0]["collider_ref"] == "toolbox_prop"

        # 3) Independent read-back via /scene and /asset-mesh endpoints.
        sc = c.get("/api/v1/projects/default/scene").json()
        assert len(sc["asset_nodes"]) == 1 and sc["asset_nodes"][0]["asset_ref"] == done["asset_id"]
        assert any(col["name"] == "toolbox_prop" for col in sc["colliders"])
        mesh_res = c.get(f"/api/v1/asset-mesh?ref={done['asset_id']}")
        assert mesh_res.status_code == 200 and "v " in mesh_res.text

        # 4) Real MuJoCo build verifies mesh body + geom + auto-landed collider exist at pose!
        saved_cfg = ConfigStore(tmp_path / "projects").load("default")
        env = PlatformEnv(saved_cfg, tmp_path)
        try:
            bid = env.model.body("node_toolbox_prop").id
            np.testing.assert_allclose(env.data.xpos[bid], [0.25, -0.15, 0.0], atol=1e-5)
            vgid = env.model.geom("node_toolbox_prop_visual").id
            assert env.model.geom_type[vgid] == int(mujoco.mjtGeom.mjGEOM_MESH)
            cgid = env.model.geom("toolbox_prop").id
            np.testing.assert_allclose(env.data.geom_xpos[cgid], [0.25, -0.15, 0.025], atol=1e-5)
        finally:
            env.close()


# --- F08: Controlled recovery / attach_remote on submit_unknown ---

def test_f08_attach_remote_and_recover_submit_unknown(tmp_path):
    ss = SessionStore(tmp_path / "sessions")
    s = _make_ready_session(ss)
    s.compiled = compile_session(s)
    s = ss.save(s, s.revision)
    jobs = ObjectJobStore(tmp_path / "jobs", MockLux3DProvider("submit_timeout"), lab_root=tmp_path)
    orch = Orchestrator(ss, jobs, None)

    r1 = orch.confirm(s.session_id, client_request_id="rec-1", background=False)
    jid = r1["job"]["id"]
    assert r1["job"]["state"] == "submit_unknown"
    assert r1["job"]["remote_id"] is None

    # Wrong provider must be rejected.
    with pytest.raises(ValueError, match="provider mismatch"):
        orch.recover(s.session_id, jid, "known-remote", provider="aholo_world")

    # Provider timeout during verification must raise controlled ValueError (not unhandled TimeoutError).
    with patch.object(jobs.provider, "get_status", side_effect=TimeoutError("verify timeout")):
        with pytest.raises(ValueError, match="verification failed"):
            orch.recover(s.session_id, jid, "known-remote", provider="lux3d")

    # Recover onto the original job without creating a new job or resubmitting.
    jobs.provider.behavior = "success"
    rec = orch.recover(s.session_id, jid, "known-remote", provider="lux3d")
    assert rec["job"]["id"] == jid
    assert rec["job"]["remote_id"] == "known-remote"
    assert rec["job"]["state"] == "remote_processing"
    assert rec["job"]["client_request_id"] == "rec-1"
    assert rec["job"]["audit"][-1]["event"] == "attach_remote"
    assert jobs.provider.submits == 1

    # Poll completes the recovered job.
    jobs.poll(jid)
    assert jobs.get(jid)["state"] == "succeeded"

    # Re-attaching a different remote_id after completion must be rejected.
    with pytest.raises(ValueError):
        orch.recover(s.session_id, jid, "other-remote")


# --- F09: Atomic .part download, corrupt residual ignored, explicit retry clears stale cache ---

def test_f09_residual_corrupt_file_redownloaded_and_atomic_part(tmp_path):
    jobs = ObjectJobStore(tmp_path / "jobs", MockLux3DProvider(), lab_root=tmp_path)
    j = jobs.submit({
        "prompt": "cup", "payload": {}, "slug": "partial-test", "client_request_id": "p-1",
        "kind": "task_object", "role": "dynamic", "height_m": 0.09, "radius_m": 0.03, "mass_kg": 0.12,
    })
    outdir = jobs.dir / j["id"] / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)
    # Simulate an unverified residual file left behind by an interrupted download.
    (outdir / "result.glb").write_bytes(b"glTF-truncated-download")

    with patch.object(jobs.provider, "download", wraps=jobs.provider.download) as dl:
        jobs.poll(j["id"])
        assert dl.call_count == 1
        assert jobs.get(j["id"])["state"] == "succeeded"

        # Explicit retry_node("download") must discard cache & stale downstream state and redownload.
        jobs.retry_node(j["id"], "download")
        assert dl.call_count == 2
        assert jobs.get(j["id"])["state"] == "succeeded"

    # download_mesh truncating mid-stream must leave neither result.glb nor .part files.
    class TruncatedResp(io.BytesIO):
        headers = {"Content-Length": "1024"}
        def __enter__(self): return self
        def __exit__(self, *a): return False

    dl_dir = tmp_path / "dl_fail"
    with patch("asset_pipeline.import_asset.urlopen", return_value=TruncatedResp(b"glTF\x02\x00\x00\x00")):
        with pytest.raises(ValueError, match="Truncated download"):
            download_mesh([{"content": "https://example.invalid/cup.glb"}], dl_dir)
    assert not list(dl_dir.glob("*"))


# --- F10: Hard-constraint protection on friction/mass/graspable, whole PhysicsSpec validation, no lost user msgs ---

def test_f10_physics_hard_constraints_invalid_rejected_and_message_preserved(tmp_path):
    ss = SessionStore(tmp_path / "sessions")
    s = ss.create()
    s.physics_spec.friction = [0.95, 0.005, 0.0001]
    s.physics_spec.friction_provenance = FieldProvenance(
        source=FieldSource.user_explicit, confirmation=Confirmation.accepted,
    )
    s.physics_spec.graspable = True
    s.physics_spec.graspable_provenance = FieldProvenance(
        source=FieldSource.user_explicit, confirmation=Confirmation.accepted,
    )

    warnings = apply_reply(s, PlannerReply(physics_patch={"friction": [0.0, 0.0, 0.0], "graspable": False}))
    assert s.physics_spec.friction == [0.95, 0.005, 0.0001]
    assert s.physics_spec.friction_provenance.source == FieldSource.user_explicit
    assert s.physics_spec.graspable is True
    assert any("friction" in w for w in warnings) and any("graspable" in w for w in warnings)

    # Invalid unconfirmed friction must be safely ignored without corrupting PhysicsSpec.
    s2 = ss.create()
    for bad_fric in ([-1.0, 0.0, 0.0], [0.5, 0.5], ["bad", 0, 0]):
        w2 = apply_reply(s2, PlannerReply(physics_patch={"friction": bad_fric}))
        assert s2.physics_spec.friction is None
        assert any("friction" in w for w in w2)

    # Model network/HTTP failure during POST /messages must NOT lose the user's message.
    with patch.dict("os.environ", {"AUX_MODEL_BASE_URL": "http://127.0.0.1:9", "AUX_MODEL_API_KEY": "test-key"}):
        with TestClient(create_app(tmp_path)) as c:
            created = c.post("/api/v1/generation/sessions", json={}).json()
            sid, rev = created["session_id"], created["revision"]
            with patch("vlab.generation.planner.Planner._chat", side_effect=TimeoutError("LLM down")):
                resp = c.post(f"/api/v1/generation/sessions/{sid}/messages", json={
                    "content": "不要丢失这条用户需求", "expected_revision": rev,
                })
            assert resp.status_code == 200, resp.text
            persisted = c.get(f"/api/v1/generation/sessions/{sid}").json()
            assert any(m["role"] == "user" and "不要丢失这条用户需求" in m["content"] for m in persisted["messages"])
            assert any(m["kind"] == "error" for m in persisted["messages"])
