"""Round 4 stage 3: scene recipes, layout/dependency validation, real MuJoCo physics,
atomic apply with version conflict and read-back.

composed_scene works with NO World background and NO calibration (explicit proxies +
parameterized workbench). background_scene covers mesh (mujoco) and gaussian (hybrid,
fixed-front static_background_overlay). A real generated cup asset flows into a new
trainable project version. Validation builds and short-steps a real MuJoCo model;
mock/remote success is never physics truth.
"""
import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
from vlab.api import create_app  # noqa: E402
from vlab.store import ConfigStore  # noqa: E402
from vlab.generation.scene_recipe import (  # noqa: E402
    build_recipe_from_session, recipe_to_config_data, validate_recipe)
from vlab.generation.schema import AssetKind, AssetNode, SceneMode, SceneRecipe  # noqa: E402
from vlab.generation.sessions import SessionStore  # noqa: E402

PID = "qa_scene_recipe"
WORKBENCH = {"name": "workbench", "shape": "box", "full_size_m": [0.64, 0.43, 0.056],
             "position": [0, 0, -0.028], "quaternion_xyzw": [0, 0, 0, 1]}
TOOLBOX = {"name": "toolbox", "shape": "box", "full_size_m": [0.10, 0.10, 0.05],
           "position": [0.25, -0.15, 0.025], "quaternion_xyzw": [0, 0, 0, 1]}
GEN_CUP = "imported/lux3d-blue-live-5887561"


def _composed(**kw):
    base = dict(recipe_id="r_composed", scene_mode=SceneMode.composed_scene,
                task_asset_ref="builtin/standard", use_default_table=False,
                colliders=[WORKBENCH, TOOLBOX], sensor_renderer="mujoco")
    base.update(kw)
    return SceneRecipe(**base)


def _store():
    return ConfigStore(LAB / "projects")


def _setup(c):
    base = json.loads((LAB / "configs" / "default.json").read_text(encoding="utf-8"))
    r = c.put(f"/api/v1/config?project={PID}", json=base)
    assert r.status_code == 200, r.text
    return r.json()["version"]


def test_composed_scene_validates_without_world():
    rep = validate_recipe(_composed(), LAB, _store(), "default")
    assert rep["valid"], rep["issues"]
    assert rep["checks"]["physics"]["built"] and rep["checks"]["physics"]["stable"], rep["checks"]["physics"]
    assert rep["checks"]["background"] == {}  # no World background needed
    assert rep["checks"]["task_object"]["asset"] == "standard"


def test_generated_cup_flows_into_recipe():
    rep = validate_recipe(_composed(task_asset_ref=GEN_CUP), LAB, _store(), "default")
    assert rep["valid"], rep["issues"]
    assert rep["checks"]["task_object"]["asset"] == GEN_CUP
    assert rep["checks"]["physics"]["stable"]


def test_background_mesh_scene_validates():
    r = _composed(recipe_id="r_mesh", scene_mode=SceneMode.background_scene,
                  background_ref="scenes/demo_workspace/1", sensor_renderer="mujoco",
                  use_default_table=True, colliders=[])
    rep = validate_recipe(r, LAB, _store(), "default")
    assert rep["valid"], rep["issues"]
    assert rep["checks"]["background"]["gaussian"] is False


def test_gaussian_requires_hybrid():
    r = _composed(recipe_id="r_bad", scene_mode=SceneMode.background_scene,
                  background_ref="scenes/world_586c79650960/1", sensor_renderer="mujoco",
                  use_default_table=True, colliders=[])
    rep = validate_recipe(r, LAB, _store(), "default")
    assert not rep["valid"]
    assert any("hybrid" in i for i in rep["issues"]), rep["issues"]


def test_background_hybrid_builds_and_warns_fixed_front():
    r = _composed(recipe_id="r_hybrid", scene_mode=SceneMode.background_scene,
                  background_ref="scenes/world_586c79650960/1", sensor_renderer="hybrid",
                  use_default_table=True, colliders=[])
    rep = validate_recipe(r, LAB, _store(), "default")
    assert rep["valid"], rep["issues"]
    assert rep["checks"]["physics"]["stable"], rep["checks"]["physics"]
    assert any("static_background_overlay" in w for w in rep["warnings"]), rep["warnings"]


def test_dependency_cycle_detected():
    r = _composed(recipe_id="r_cycle", asset_nodes=[
        AssetNode(node_id="a", asset_ref="builtin/standard", depends_on=["b"]),
        AssetNode(node_id="b", asset_ref="builtin/standard", depends_on=["a"])])
    rep = validate_recipe(r, LAB, _store(), "default")
    assert not rep["valid"]
    assert any("cycle" in i for i in rep["issues"]), rep["issues"]


def test_missing_dependency_detected():
    r = _composed(recipe_id="r_missing", asset_nodes=[
        AssetNode(node_id="a", asset_ref="builtin/standard", depends_on=["ghost"])])
    rep = validate_recipe(r, LAB, _store(), "default")
    assert not rep["valid"]
    assert any("unknown node ghost" in i for i in rep["issues"]), rep["issues"]


def test_layout_penetration_warns():
    overlap = dict(TOOLBOX, name="toolbox2", position=[0.25, -0.15, 0.025])
    r = _composed(recipe_id="r_overlap", colliders=[WORKBENCH, TOOLBOX, overlap])
    rep = validate_recipe(r, LAB, _store(), "default")
    assert any("interpenetrate" in w for w in rep["warnings"]), rep["warnings"]


def test_recipe_from_session_mapping(tmp_path):
    sessions = SessionStore(tmp_path / "sessions")
    s = sessions.create(project_id="default")
    s.asset_spec.name = "cup"
    s = sessions.save(s, s.revision)
    r = build_recipe_from_session(s, LAB, overrides={"task_asset_ref": "builtin/standard"})
    assert r.scene_mode == SceneMode.composed_scene and r.task_asset_ref == "builtin/standard"
    base = _store().load("default")
    data = recipe_to_config_data(r, base, "default")
    assert data["task"]["asset"] == "standard"
    # a static_prop session must NOT auto-become a manipulable task object
    s2 = sessions.create(project_id="default")
    s2.asset_spec.kind = AssetKind.static_prop
    s2 = sessions.save(s2, s2.revision)
    r2 = build_recipe_from_session(s2, LAB)
    assert r2.task_asset_ref is None


def _make_recipe(c, **overrides):
    s = c.post("/api/v1/generation/sessions", json={"project_id": PID}).json()
    body = dict(scene_mode="composed_scene", task_asset_ref="builtin/standard",
                use_default_table=False, colliders=[WORKBENCH, TOOLBOX], sensor_renderer="mujoco")
    body.update(overrides)
    r = c.post(f"/api/v1/generation/sessions/{s['session_id']}/recipe", json=body)
    assert r.status_code == 200, r.text
    return r.json()["recipe_id"]


def test_apply_read_back_and_version_conflict():
    app = create_app(LAB)
    with TestClient(app) as c:
        v0 = _setup(c)
        rid = _make_recipe(c, recipe_id="qa_apply_composed")
        rep = c.post(f"/api/v1/scene-recipes/{rid}/validate?project_id={PID}")
        assert rep.status_code == 200 and rep.json()["valid"], rep.text
        ap = c.post(f"/api/v1/scene-recipes/{rid}/apply",
                    json={"project_id": PID, "expected_project_version": v0})
        assert ap.status_code == 200, ap.text
        out = ap.json()
        assert out["version"] == v0 + 1 and out["validation_status"] == "validated"
        assert out["task_asset"] == "standard"
        names = [x["name"] for x in out["read_back"]["colliders"]]
        assert names == ["workbench", "toolbox"] and out["read_back"]["use_default_table"] is False
        # read back through the project scene endpoint (independent path)
        sc = c.get(f"/api/v1/projects/{PID}/scene").json()
        assert sc["version"] == v0 + 1 and [x["name"] for x in sc["colliders"]] == ["workbench", "toolbox"]
        # stale writer -> 409, project unchanged
        conflict = c.post(f"/api/v1/scene-recipes/{rid}/apply",
                          json={"project_id": PID, "expected_project_version": v0})
        assert conflict.status_code == 409, conflict.text
        assert c.get(f"/api/v1/projects/{PID}/scene").json()["version"] == v0 + 1


def test_apply_invalid_recipe_leaves_project_unchanged():
    app = create_app(LAB)
    with TestClient(app) as c:
        v0 = _setup(c)
        rid = _make_recipe(c, recipe_id="qa_apply_bad", task_asset_ref="imported/does_not_exist")
        ap = c.post(f"/api/v1/scene-recipes/{rid}/apply",
                    json={"project_id": PID, "expected_project_version": v0})
        assert ap.status_code == 400, ap.text
        assert c.get(f"/api/v1/projects/{PID}/scene").json()["version"] == v0


def main():
    import tempfile
    test_composed_scene_validates_without_world()
    test_generated_cup_flows_into_recipe()
    test_background_mesh_scene_validates()
    test_gaussian_requires_hybrid()
    test_background_hybrid_builds_and_warns_fixed_front()
    test_dependency_cycle_detected()
    test_missing_dependency_detected()
    test_layout_penetration_warns()
    with tempfile.TemporaryDirectory() as td:
        test_recipe_from_session_mapping(Path(td))
    test_apply_read_back_and_version_conflict()
    test_apply_invalid_recipe_leaves_project_unchanged()
    print("scene recipe stage3 tests OK")


if __name__ == "__main__":
    main()