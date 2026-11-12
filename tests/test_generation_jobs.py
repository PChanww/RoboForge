"""Round 4 stage 2: generic object import, node-granular jobs, idempotency, recovery.

MockLux3DProvider is a controlled test double, NOT online success. Real-asset
import uses the existing cup GLB. Covers distinct cup/tray/toolbox defaults,
normalization records, uncalibrated visual-only blocking, bad-mesh errors,
double-confirm idempotency, frozen binding, submit_unknown and node retry.
"""
import json
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
from asset_pipeline.import_object import register_object  # noqa: E402
from vlab.generation.object_provider import ObjectJobStore, MockLux3DProvider  # noqa: E402
from vlab.generation.orchestrator import Orchestrator  # noqa: E402
from vlab.generation.sessions import SessionStore  # noqa: E402
from vlab.generation.schema import SessionState  # noqa: E402

CUP_GLB = LAB / "assets" / "imported" / "lux3d-blue-live-5887561" / "original.glb"


def test_distinct_cup_tray_toolbox_defaults(tmp_path):
    assert CUP_GLB.is_file(), "real cup GLB resource missing"
    cup = register_object(CUP_GLB, tmp_path / "cup", kind="task_object", role="dynamic",
                          height=0.09, radius=0.03, mass=0.12, source="lux3d", task_id="t1")
    tray = register_object(CUP_GLB, tmp_path / "tray", kind="static_prop", role="static",
                           height=0.02, width=0.20, depth=0.15, source="lux3d", task_id="t2")
    box = register_object(CUP_GLB, tmp_path / "toolbox", kind="static_prop", role="static",
                          height=0.10, width=0.30, depth=0.20, source="other")
    cm, tm, bm = (json.loads(p.read_text(encoding="utf-8")) for p in (cup, tray, box))
    assert cm["collision_proxy"]["shape"] == "cylinder" and cm["role"] == "dynamic"
    assert tm["collision_proxy"]["shape"] == "box" and tm["role"] == "static"
    assert bm["collision_proxy"]["shape"] == "box"
    assert tm["collision_proxy"]["size_m"] != bm["collision_proxy"]["size_m"]
    assert cm["calibrated"] and tm["calibrated"]
    for m in (cm, tm, bm):
        assert m["normalization"]["centered_footprint"] is True
        assert m["normalization"]["original_extents_m"]
        assert len(m["sha256_glb"]) == 64
    assert cm["spec_id"] is None  # not from a session here


def test_uncalibrated_blocks_physics(tmp_path):
    mf = register_object(CUP_GLB, tmp_path / "unknown", kind="task_object", role="dynamic",
                         height=None, source="other")
    m = json.loads(mf.read_text(encoding="utf-8"))
    assert m["calibrated"] is False
    assert m["role"] == "visual_only"  # dynamic downgraded until calibrated
    assert m["physics_status"] == "uncalibrated"


def test_bad_mesh_rejected(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    bad = tmp_path / "bad.glb"
    bad.write_bytes(b"not a glb")
    try:
        register_object(bad, tmp_path / "out", height=0.09, source="other")
        raise AssertionError("bad mesh accepted")
    except ValueError:
        pass


def _params(slug, crid, **kw):
    p = {"prompt": "a cup", "payload": {"version": "G1-Turbo"}, "slug": slug,
         "kind": "task_object", "role": "dynamic", "height_m": 0.09, "radius_m": 0.03,
         "mass_kg": 0.12, "client_request_id": crid, "spec_id": "gs_x", "spec_revision": 3,
         "spec_hash": "h" * 64, "project_id": "default", "config_version": 1}
    p.update(kw)
    return p


def test_idempotent_confirm_and_nodes(tmp_path):
    store = ObjectJobStore(tmp_path / "jobs", MockLux3DProvider("success"), lab_root=tmp_path)
    j1 = store.submit(_params("cup-a", "req-1"))
    j2 = store.submit(_params("cup-a", "req-1"))  # duplicate click
    assert j1["id"] == j2["id"]
    assert store.provider.submits == 1  # no duplicate paid remote task
    store.poll(j1["id"])
    done = store.get(j1["id"])
    assert done["state"] == "succeeded", done
    assert all(done["nodes"][n] == "done" for n in ("submit", "remote", "download", "import", "validate"))
    assert done["asset_id"] == "imported/cup-a"
    assert done["download_sha256"]
    mf = json.loads((tmp_path / "assets" / "imported" / "cup-a" / "manifest.json").read_text())
    assert mf["spec_hash"] == "h" * 64 and mf["generation_id"] == done["remote_id"]


def test_submit_unknown_no_blind_resubmit(tmp_path):
    store = ObjectJobStore(tmp_path / "jobs", MockLux3DProvider("submit_timeout"), lab_root=tmp_path)
    j = store.submit(_params("cup-b", "req-2"))
    assert j["state"] == "submit_unknown"
    assert j["remote_id"] is None
    # poll must not resubmit; retry without remote_id is refused
    store.poll(j["id"])
    assert store.get(j["id"])["state"] == "submit_unknown"
    assert store.provider.submits == 1
    try:
        store.retry_node(j["id"], "download")
        raise AssertionError("retry without remote_id accepted")
    except ValueError:
        pass


def test_node_retry_reuses_remote_and_file(tmp_path):
    prov = MockLux3DProvider("success")
    store = ObjectJobStore(tmp_path / "jobs", prov, lab_root=tmp_path)
    j = store.submit(_params("cup-c", "req-3"))
    store.poll(j["id"])
    assert store.get(j["id"])["state"] == "succeeded"
    # force a re-import retry; must reuse remote_id and downloaded file (no new submit)
    before = prov.submits
    store.retry_node(j["id"], "import")
    after = store.get(j["id"])
    assert prov.submits == before
    assert after["state"] == "succeeded"
    assert after["retries"] == 1


def test_restart_marks_submitting_unknown(tmp_path):
    store = ObjectJobStore(tmp_path / "jobs", MockLux3DProvider("success"), lab_root=tmp_path)
    j = store.submit(_params("cup-d", "req-4"))
    # simulate a crash mid-submit
    raw = json.loads((tmp_path / "jobs" / "object_jobs.json").read_text())
    raw[0]["state"] = "submitting"
    (tmp_path / "jobs" / "object_jobs.json").write_text(json.dumps(raw))
    reopened = ObjectJobStore(tmp_path / "jobs", MockLux3DProvider("success"), lab_root=tmp_path)
    assert reopened.get(j["id"])["state"] == "submit_unknown"


def test_orchestrator_freezes_and_binds(tmp_path):
    sessions = SessionStore(tmp_path / "sessions")
    s = sessions.create(project_id="default")
    s.asset_spec.name = "cup"
    s.asset_spec.height.value_m = 0.09
    s.asset_spec.radius.value_m = 0.03
    s.physics_spec.mass.value_kg = 0.12
    s = sessions.save(s, s.revision)
    objects = ObjectJobStore(tmp_path / "jobs", MockLux3DProvider("success"), lab_root=tmp_path)
    orch = Orchestrator(sessions, objects, world_store=None)
    r1 = orch.confirm(s.session_id, client_request_id="c-1", background=False)
    r2 = orch.confirm(s.session_id, client_request_id="c-1", background=False)
    assert r1["job"]["id"] == r2["job"]["id"]  # idempotent
    assert r1["binding"]["spec_hash"] and r1["binding"]["provider"] == "lux3d"
    assert r1["binding"]["spec_revision"] == s.revision
    sess = sessions.get(s.session_id)
    assert sess.state == SessionState.generating
    assert r1["job"]["id"] in sess.job_ids
    tasks = orch.tasks(s.session_id)
    assert len(tasks["tasks"]) == 1


def test_world_recovery_reuses_remote_id(tmp_path):
    from vlab.world_provider import GenerationJobStore, MockWorldProvider
    sessions = SessionStore(tmp_path / "sessions")
    s = sessions.create(project_id="default")
    from vlab.generation.schema import AssetKind, AssetRole
    s.asset_spec.kind = AssetKind.environment_splat
    s.asset_spec.role = AssetRole.visual_only
    s.asset_spec.description_zh = "empty robotics laboratory"
    s = sessions.save(s, s.revision)
    prov = MockWorldProvider("success")
    world = GenerationJobStore(tmp_path / "wjobs", prov, lab_root=tmp_path)
    objects = ObjectJobStore(tmp_path / "jobs", MockLux3DProvider("success"), lab_root=tmp_path)
    orch = Orchestrator(sessions, objects, world_store=world)
    r = orch.confirm(s.session_id, client_request_id="w-1", provider="aholo_world",
                     remote_id="3FO4K4RMF54N", background=False)
    assert r["binding"]["provider"] == "aholo_world"
    assert r["job"]["remote_id"] == "3FO4K4RMF54N"
    assert prov.submits == 0  # recovery: no new paid remote task
    r2 = orch.confirm(s.session_id, client_request_id="w-1", provider="aholo_world",
                      remote_id="3FO4K4RMF54N", background=False)
    assert r2["job"]["id"] == r["job"]["id"]


def test_api_confirm_unconfigured_is_honest(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from vlab.api import create_app
    for k in ("AHOLO_API_KEY", "LUX3D_CN_API_KEY", "LUX3D_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    with TestClient(create_app(tmp_path)) as c:
        s = c.post("/api/v1/generation/sessions", json={}).json()
        sid = s["session_id"]
        c.patch(f"/api/v1/generation/sessions/{sid}/spec",
                json={"expected_revision": s["revision"], "name": "cup", "height_m": 0.09,
                      "radius_m": 0.03, "mass_kg": 0.12})
        r = c.post(f"/api/v1/generation/sessions/{sid}/confirm", json={"client_request_id": "api-1"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["binding"]["provider"] == "lux3d"
        assert body["job"]["state"] == "unconfigured"  # no key -> not a fake success
        tasks = c.get(f"/api/v1/generation/sessions/{sid}/tasks").json()
        assert len(tasks["tasks"]) == 1
        r2 = c.post(f"/api/v1/generation/sessions/{sid}/confirm", json={"client_request_id": "api-1"})
        assert r2.json()["job"]["id"] == body["job"]["id"]


def main():
    import tempfile
    import pytest
    with tempfile.TemporaryDirectory() as td:
        tp = Path(td)
        test_distinct_cup_tray_toolbox_defaults(tp / "a")
        test_uncalibrated_blocks_physics(tp / "b")
        test_bad_mesh_rejected(tp / "c")
        test_idempotent_confirm_and_nodes(tp / "d")
        test_submit_unknown_no_blind_resubmit(tp / "e")
        test_node_retry_reuses_remote_and_file(tp / "f")
        test_restart_marks_submitting_unknown(tp / "g")
        test_orchestrator_freezes_and_binds(tp / "h")
        test_world_recovery_reuses_remote_id(tp / "i")
        mp = pytest.MonkeyPatch()
        test_api_confirm_unconfigured_is_honest(tp / "j", mp)
        mp.undo()
    print("generation stage2 tests OK")


if __name__ == "__main__":
    main()