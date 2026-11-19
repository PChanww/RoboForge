"""Round 4 stage 1: generation sessions, capabilities, planner merge, compiler.

Covers waiting_for_configuration (no fake model reply), form/JSON spec entry,
revision conflicts (409), persistence across store reload, hard-constraint
preservation, contradiction blocking and deterministic compilation.
"""
import sys
from pathlib import Path

from fastapi.testclient import TestClient

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
from vlab.api import create_app  # noqa: E402
from vlab.generation.schema import (PlannerReply, PlannerQuestion, FieldProvenance,  # noqa: E402
                                    FieldSource, Confirmation)
from vlab.generation.planner import apply_reply  # noqa: E402
from vlab.generation.sessions import SessionStore  # noqa: E402


def _client(tmp_path):
    return TestClient(create_app(tmp_path))


def test_capabilities_and_waiting_for_configuration(tmp_path, monkeypatch):
    monkeypatch.delenv("AUX_MODEL_BASE_URL", raising=False)
    monkeypatch.delenv("AUX_MODEL_API_KEY", raising=False)
    with _client(tmp_path) as c:
        cap = c.get("/api/v1/generation/capabilities").json()
        assert cap["aux_model"]["state"] == "waiting_for_configuration"
        assert cap["providers"]["local_import"]["configured"] is True
        assert cap["asset_kinds"]["environment_mesh"]["online"] is None
        assert cap["physics_limits"]["convex_decomposition"] is False
        s = c.post("/api/v1/generation/sessions", json={"initial_text": "一个杯子"}).json()
        sid, rev = s["session_id"], s["revision"]
        r = c.post(f"/api/v1/generation/sessions/{sid}/messages",
                   json={"content": "蓝色无把手杯", "expected_revision": rev}).json()
        # No fabricated assistant reply; a system status instead.
        assert r["aux_state"] == "waiting_for_configuration"
        assert not any(m["role"] == "assistant" for m in r["messages"])
        assert any(m["kind"] == "status" for m in r["messages"])


def test_spec_entry_compile_and_conflict(tmp_path):
    with _client(tmp_path) as c:
        s = c.post("/api/v1/generation/sessions", json={}).json()
        sid, rev = s["session_id"], s["revision"]
        r = c.patch(f"/api/v1/generation/sessions/{sid}/spec",
                    json={"expected_revision": rev, "kind": "task_object", "role": "dynamic",
                          "name": "cup", "height_m": 0.09, "radius_m": 0.03,
                          "has_handle": False, "open_top": True, "appearance": "matte ivory",
                          "mass_kg": 0.12, "user_words": "蓝色无把手杯"})
        assert r.status_code == 200, r.text
        s = r.json()
        assert s["asset_spec"]["height"]["provenance"]["source"] == "user_explicit"
        assert s["asset_spec"]["height"]["provenance"]["confirmation"] == "accepted"
        rev = s["revision"]
        # stale revision -> 409, no silent overwrite
        stale = c.patch(f"/api/v1/generation/sessions/{sid}/spec",
                        json={"expected_revision": rev - 1, "name": "other"})
        assert stale.status_code == 409
        comp = c.post(f"/api/v1/generation/sessions/{sid}/compile", json={}).json()
        assert comp["compiled"]["provider"] == "lux3d"
        assert "handleless" in comp["compiled"]["prompt_en"]
        assert "9 cm" in comp["compiled"]["prompt_en"]
        assert "mass_kg" in comp["compiled"]["excluded_or_postprocess_fields"]
        assert comp["compiled"]["payload"]["face_count"] == 20000
        assert "mass" not in comp["compiled"]["payload"]


def test_contradiction_blocks_compile(tmp_path):
    with _client(tmp_path) as c:
        s = c.post("/api/v1/generation/sessions", json={}).json()
        sid = s["session_id"]
        c.patch(f"/api/v1/generation/sessions/{sid}/spec",
                json={"expected_revision": s["revision"], "user_words": "handleless cup",
                      "has_handle": True})
        r = c.post(f"/api/v1/generation/sessions/{sid}/compile", json={})
        assert r.status_code == 400


def test_persistence_across_reload(tmp_path):
    with _client(tmp_path) as c:
        s = c.post("/api/v1/generation/sessions", json={"initial_text": "杯子"}).json()
        sid = s["session_id"]
        c.patch(f"/api/v1/generation/sessions/{sid}/spec",
                json={"expected_revision": s["revision"], "height_m": 0.09})
    # A fresh app/store on the same root must read the persisted session back.
    with _client(tmp_path) as c:
        again = c.get(f"/api/v1/generation/sessions/{sid}").json()
        assert again["asset_spec"]["height"]["value_m"] == 0.09
        assert again["messages"][0]["content"] == "杯子"


def test_planner_merge_preserves_hard_constraints(tmp_path):
    store = SessionStore(tmp_path / "merge")
    s = store.create()
    reply = PlannerReply(spec_patch={"height_m": 0.09, "has_handle": False},
                         questions=[PlannerQuestion(id="q1", text="确认?")])
    apply_reply(s, reply)
    assert s.asset_spec.height.value_m == 0.09
    assert s.asset_spec.height.provenance.source == FieldSource.model_inferred
    assert s.asset_spec.height.provenance.verification.value == "unverified"
    # user confirms height -> hard constraint; a later model suggestion must not override
    s.asset_spec.height.provenance = FieldProvenance(source=FieldSource.user_explicit,
                                                     confirmation=Confirmation.accepted)
    warnings = apply_reply(s, PlannerReply(spec_patch={"height_m": 0.3}))
    assert s.asset_spec.height.value_m == 0.09
    assert warnings


def main():
    import tempfile
    import pytest
    with tempfile.TemporaryDirectory() as td:
        tp = Path(td)
        mp = pytest.MonkeyPatch()
        test_capabilities_and_waiting_for_configuration(tp / "a", mp)
        mp.undo()
        test_spec_entry_compile_and_conflict(tp / "b")
        test_contradiction_blocks_compile(tp / "c")
        test_persistence_across_reload(tp / "d")
        test_planner_merge_preserves_hard_constraints(tp / "e")
    print("generation stage1 tests OK")


if __name__ == "__main__":
    main()