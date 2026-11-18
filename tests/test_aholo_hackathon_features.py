"""Isolated tests for Aholo hackathon upgrades:
- Account remote task discovery & one-click import without re-submitting paid tasks
- Reference image validation & Lux3D img_to_3d / text_to_3d(img=...) routing
- Parameterized goal tray + custom calibrated receptacle in real MuJoCo physics
"""
from __future__ import annotations

import base64
import json
import shutil
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import trimesh
from fastapi.testclient import TestClient

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))

from vlab.api import create_app  # noqa: E402
from vlab.engine import PlatformEnv  # noqa: E402
from vlab.generation.compiler import (  # noqa: E402
    LUX3D_PAYLOAD_DEFAULTS,
    CompileError,
    compiled_request_hash,
    compile_session,
    spec_hash,
)
from vlab.generation.object_provider import Lux3DProvider, MockLux3DProvider, ObjectJobStore  # noqa: E402
from vlab.generation.orchestrator import Orchestrator  # noqa: E402
from vlab.generation.schema import AssetKind, AssetRole  # noqa: E402
from vlab.generation.sessions import SessionStore  # noqa: E402
from vlab.schema import ProjectConfig  # noqa: E402
from vlab.store import ConfigStore  # noqa: E402
from vlab.world_provider import GenerationJobStore, MockWorldProvider  # noqa: E402


def _make_lab(tmp_path: Path) -> Path:
    (tmp_path / "configs").mkdir(parents=True, exist_ok=True)
    shutil.copy(LAB / "configs" / "default.json", tmp_path / "configs" / "default.json")
    (tmp_path / "projects" / "default").mkdir(parents=True, exist_ok=True)
    shutil.copy(LAB / "configs" / "default.json", tmp_path / "projects" / "default" / "current.json")
    shutil.copytree(LAB / "assets" / "robot", tmp_path / "assets" / "robot", dirs_exist_ok=True)
    return tmp_path


def _tray_glb(path: Path) -> Path:
    mesh = trimesh.creation.box(extents=[0.20, 0.15, 0.02])
    path.write_bytes(mesh.export(file_type="glb"))
    return path


class _TrayGlbProvider(MockLux3DProvider):
    def download(self, outputs, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        out = directory / "result.glb"
        _tray_glb(out)
        return out


def test_spec_hash_backward_compatible_without_reference_image(tmp_path):
    ss = SessionStore(tmp_path / "sessions")
    s = ss.create(project_id="default")
    s.asset_spec.name = "cup"
    s.asset_spec.height.value_m = 0.09
    s.asset_spec.radius.value_m = 0.03
    s.physics_spec.mass.value_kg = 0.12
    s = ss.save(s, s.revision)

    # Without reference_image, dump omits None reference_image for hash parity
    asset_legacy = s.asset_spec.model_dump()
    asset_legacy.pop("reference_image", None)
    import hashlib
    expected_base = hashlib.sha256(
        json.dumps({"asset": asset_legacy, "physics": s.physics_spec.model_dump()},
                   sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    assert spec_hash(s.asset_spec, s.physics_spec) == expected_base

    compiled = compile_session(s)
    assert "img" not in compiled.payload
    assert "generation_mode" not in compiled.payload
    assert compiled.spec_hash == compiled_request_hash(
        compiler_version=compiled.compiler_version,
        base_spec_hash=expected_base,
        provider="lux3d",
        template="single_object",
        prompt_en=compiled.prompt_en,
        payload={"prompt": compiled.prompt_en, **LUX3D_PAYLOAD_DEFAULTS},
    )


def test_reference_image_validation_and_lux3d_img_to_3d_routing(tmp_path, monkeypatch):
    lab = _make_lab(tmp_path)
    monkeypatch.delenv("AHOLO_API_KEY", raising=False)
    app = create_app(lab)
    c = TestClient(app)

    s = c.post("/api/v1/generation/sessions", json={"kind": "task_object", "initial_text": "ceramic mug"}).json()
    sid, rev = s["session_id"], s["revision"]

    # Local paths / file URLs must be rejected
    for bad in ("C:\\secret\\img.png", "file:///etc/passwd", "ftp://example.com/a.png", "not-a-url"):
        r = c.patch(f"/api/v1/generation/sessions/{sid}/spec", json={"expected_revision": rev, "reference_image": bad})
        assert r.status_code == 400, bad

    png_b64 = base64.b64encode(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR").decode("ascii")
    data_uri = f"data:image/png;base64,{png_b64}"
    r = c.patch(f"/api/v1/generation/sessions/{sid}/spec", json={
        "expected_revision": rev, "name": "mug", "height_m": 0.09, "radius_m": 0.03,
        "reference_image": data_uri,
    })
    assert r.status_code == 200, r.text
    rev = r.json()["revision"]

    # Auto mode selects img_to_3d when reference_image is set
    comp_res = c.post(f"/api/v1/generation/sessions/{sid}/compile", json={"expected_revision": rev})
    assert comp_res.status_code == 200, comp_res.text
    comp = comp_res.json()
    assert comp["compiled"]["payload"]["generation_mode"] == "img_to_3d"
    assert comp["compiled"]["payload"]["img"] == data_uri
    assert "style" not in comp["compiled"]["payload"]

    # Lux3DProvider routes img_to_3d -> img_to_3d.create and text_to_3d -> text_to_3d.create(img=...)
    prov = Lux3DProvider()
    fake_sdk = MagicMock()
    fake_sdk.img_to_3d.create.return_value = 99001
    fake_sdk.text_to_3d.create.return_value = 99002
    prov._client = fake_sdk

    rid1 = prov.submit(comp["compiled"]["prompt_en"], comp["compiled"]["payload"])
    assert rid1 == "99001"
    fake_sdk.img_to_3d.create.assert_called_once()
    fake_sdk.text_to_3d.create.assert_not_called()

    comp2_res = c.post(f"/api/v1/generation/sessions/{sid}/compile", json={
        "expected_revision": comp["revision"], "generation_mode": "text_to_3d",
    })
    assert comp2_res.status_code == 200, comp2_res.text
    comp2 = comp2_res.json()
    assert comp2["compiled"]["payload"]["generation_mode"] == "text_to_3d"
    assert comp2["compiled"]["payload"]["use_text_with_img"] is True
    rid2 = prov.submit(comp2["compiled"]["prompt_en"], comp2["compiled"]["payload"])
    assert rid2 == "99002"
    fake_sdk.text_to_3d.create.assert_called_once()

    # World rejects data:image URI cleanly, allows https:// ref_image
    w_sess = SessionStore(tmp_path / "wsess").create(project_id="default")
    w_sess.asset_spec.kind = AssetKind.environment_splat
    w_sess.asset_spec.role = AssetRole.visual_only
    w_sess.asset_spec.reference_image = data_uri
    with pytest.raises(CompileError):
        compile_session(w_sess)

    w_sess.asset_spec.reference_image = "https://example.com/lab.jpg"
    w_comp = compile_session(w_sess)
    assert w_comp.payload["ref_image"] == "https://example.com/lab.jpg"


def test_remote_tasks_listing_and_one_click_import_no_resubmit(tmp_path, monkeypatch):
    lab = _make_lab(tmp_path)
    monkeypatch.delenv("AHOLO_API_KEY", raising=False)
    monkeypatch.delenv("LUX3D_CN_API_KEY", raising=False)
    monkeypatch.delenv("LUX3D_API_KEY", raising=False)
    app = create_app(lab)
    c = TestClient(app)

    # Unconfigured reports cleanly without exposing keys
    rt_unconf = c.get("/api/v1/generation/remote-tasks?provider=lux3d")
    assert rt_unconf.status_code == 200
    assert rt_unconf.json()["configured"] is False
    assert rt_unconf.json()["items"] == []

    # Wire mock provider onto stores for isolated import verification
    ss = SessionStore(tmp_path / "sessions2")
    s = ss.create(project_id="default")
    s.asset_spec.name = "ivory-cup"
    s.asset_spec.height.value_m = 0.08
    s.asset_spec.radius.value_m = 0.03
    s.physics_spec.mass.value_kg = 0.10
    s = ss.save(s, s.revision)

    obj_prov = MockLux3DProvider("success")
    obj_store = ObjectJobStore(tmp_path / "jobs2", obj_prov, lab_root=tmp_path)
    world_prov = MockWorldProvider("success")
    world_store = GenerationJobStore(tmp_path / "wjobs2", world_prov, lab_root=tmp_path)
    orch = Orchestrator(ss, obj_store, world_store)

    listed_lux = orch.list_remote_tasks("lux3d")
    assert listed_lux["configured"] is True
    assert listed_lux["items"][0]["remote_id"] == "mock-1"

    listed_world = orch.list_remote_tasks("aholo_world")
    assert listed_world["configured"] is True
    assert listed_world["items"][0]["remote_id"] == "mock-1"

    # Import remote task without calling provider.submit
    res = orch.import_remote(s.session_id, "5887561", provider="lux3d", background=False)
    assert obj_prov.submits == 0
    assert res["imported_remote"] is True
    assert res["job"]["remote_id"] == "5887561"
    assert res["job"]["state"] == "succeeded"
    assert res["job"]["asset_id"]
    mf = json.loads((tmp_path / "assets" / res["job"]["asset_id"] / "manifest.json").read_text())
    assert mf["generation_id"] == "5887561" and mf["calibrated"] is True

    # Kind/provider mismatch is blocked cleanly
    with pytest.raises(ValueError, match="expects lux3d"):
        orch.import_remote(s.session_id, "3FO4K4RMF54N", provider="aholo_world", background=False)

    # Idempotent replay returns same job
    replay = orch.import_remote(s.session_id, "5887561", provider="lux3d", background=False)
    assert replay["job"]["id"] == res["job"]["id"]
    assert obj_prov.submits == 0


def test_parameterized_goal_tray_and_custom_receptacle_pickup_in_mujoco(tmp_path, monkeypatch):
    lab = _make_lab(tmp_path)
    monkeypatch.delenv("AHOLO_API_KEY", raising=False)
    app = create_app(lab)
    c = TestClient(app)

    # 1) Register a calibrated 20cm x 15cm x 2cm ceramic tray via static_prop job
    s = c.post("/api/v1/generation/sessions", json={
        "kind": "static_prop", "role": "static", "initial_text": "20x15x2cm 白色陶瓷托盘",
    }).json()
    sid, rev = s["session_id"], s["revision"]
    r = c.patch(f"/api/v1/generation/sessions/{sid}/spec", json={
        "expected_revision": rev, "name": "ceramic-tray",
        "width_m": 0.20, "depth_m": 0.15, "height_m": 0.02, "mass_kg": 0.4,
    })
    assert r.status_code == 200, r.text
    ss = SessionStore(lab / "datasets" / "generation_sessions")
    obj_store = ObjectJobStore(lab / "datasets" / "generation_jobs", _TrayGlbProvider(), lab_root=lab)
    orch = Orchestrator(ss, obj_store, None)
    imp = orch.import_remote(sid, "mock-tray-1", provider="lux3d", background=False)
    assert imp["job"]["state"] == "succeeded"

    # 2) Assemble SceneRecipe with use_as_goal_tray=True
    rec = c.post(f"/api/v1/generation/sessions/{sid}/recipe", json={
        "scene_mode": "composed_scene",
        "task_asset_ref": "builtin/standard",
        "use_default_table": True,
        "use_as_goal_tray": True,
    })
    assert rec.status_code == 200, rec.text
    rj = rec.json()
    assert rj["goal_tray_node_id"] == "ceramic-tray"
    assert rj["goal_half_extents"] == pytest.approx([0.10, 0.075], abs=1e-4)
    assert rj["goal_surface_m"] == pytest.approx(0.02, abs=1e-4)

    # 3) Validate in real MuJoCo + apply atomically
    val = c.post(f"/api/v1/scene-recipes/{rj['recipe_id']}/validate?project_id=default")
    assert val.status_code == 200 and val.json()["valid"] is True, val.json()
    assert val.json()["checks"]["goal_tray"]["show_builtin_tray"] is False

    sc0 = c.get("/api/v1/projects/default/scene").json()
    ap = c.post(f"/api/v1/scene-recipes/{rj['recipe_id']}/apply", json={
        "project_id": "default", "expected_project_version": sc0["version"],
    })
    assert ap.status_code == 200, ap.text
    apj = ap.json()
    assert apj["task"]["show_builtin_tray"] is False
    assert apj["task"]["goal_tray_collider_ref"] == "ceramic-tray"
    assert apj["task"]["goal_half_extents"] == pytest.approx([0.10, 0.075], abs=1e-4)
    assert apj["task"]["goal_surface_m"] == pytest.approx(0.02, abs=1e-4)

    # 4) Run real MuJoCo closed-loop servo expert pick-and-place into the custom tray!
    cfg = ConfigStore(lab / "projects").load("default")
    env = PlatformEnv(cfg, lab)
    try:
        with pytest.raises(KeyError):
            env.model.geom("tray")
        assert env._judge.tray_geom == env.model.geom("ceramic-tray").id
        for _ in range(180):
            obs = env.step(env.servo_expert_action())
            if env.metrics()["success"]:
                break
        m = env.metrics()
        assert m["lifted"] and m["tray_contact"] and m["success"], m
    finally:
        env.close()

    # 5) Switching back to a normal recipe restores builtin tray cleanly
    rec2 = c.post(f"/api/v1/generation/sessions/{sid}/recipe", json={
        "scene_mode": "composed_scene",
        "task_asset_ref": "builtin/standard",
        "use_default_table": True,
        "use_as_goal_tray": False,
        "asset_nodes": [],
    }).json()
    ap2 = c.post(f"/api/v1/scene-recipes/{rec2['recipe_id']}/apply", json={
        "project_id": "default", "expected_project_version": apj["version"],
    })
    assert ap2.status_code == 200, ap2.text
    assert ap2.json()["task"]["show_builtin_tray"] is True
    assert ap2.json()["task"]["goal_tray_collider_ref"] is None


def test_concave_compound_tray_and_multi_dynamic_objects_and_nonzero_support_z(tmp_path, monkeypatch):
    lab = _make_lab(tmp_path)
    monkeypatch.delenv("AHOLO_API_KEY", raising=False)
    app = create_app(lab)
    c = TestClient(app)

    # Launch flagship hackathon demo on an elevated support plane z = 0.025 m!
    resp = c.post("/api/v1/hackathon-demo/setup", json={
        "project_id": "hackathon_demo",
        "with_world_background": False,
        "support_plane_z": 0.025,
    })
    assert resp.status_code == 200, resp.text
    rj = resp.json()
    assert rj["tray_compound_boxes"] >= 4
    assert rj["goal_tray_collider_ref"].endswith("_part_0")
    assert "companion_cube" in rj["dynamic_nodes"]
    # Interior floor top is above support_plane_z (0.025), strictly below full rim height (0.025 + 0.022 = 0.047)
    assert 0.025 < rj["goal_surface_m"] < 0.042

    cfg = ConfigStore(lab / "projects").load("hackathon_demo")
    assert cfg.scene.work_surface_z == pytest.approx(0.025, abs=1e-6)
    env = PlatformEnv(cfg, lab)
    try:
        obs0 = env.observation()
        assert "companion_cube" in obs0["state"]["dynamic_objects"]
        cube_z = obs0["state"]["dynamic_objects"]["companion_cube"]["position"][2]
        assert 0.024 <= cube_z <= 0.060

        # Closed-loop servo expert picks cup off elevated z=0.025 table and drops it inside concave tray walls
        for _ in range(180):
            obs = env.step(env.servo_expert_action())
            if env.metrics()["success"]:
                break
        m = env.metrics()
        assert m["lifted"] and m["tray_contact"] and m["success"], m
    finally:
        env.close()


def test_part_split_assemble_robot_and_urdf_export(tmp_path, monkeypatch):
    import xml.etree.ElementTree as ET
    from asset_pipeline.import_object import register_object

    lab = _make_lab(tmp_path)
    monkeypatch.delenv("AHOLO_API_KEY", raising=False)
    app = create_app(lab)
    c = TestClient(app)

    # Create an elongated Y-up arm-like mesh GLB and register as robot_visual
    arm_mesh = trimesh.creation.box([0.08, 0.48, 0.08])
    arm_glb = tmp_path / "arm_visual.glb"
    arm_glb.write_bytes(arm_mesh.export(file_type="glb"))
    register_object(
        arm_glb, lab / "assets" / "imported" / "test-arm-shell",
        kind="robot_visual", role="visual_only", height=0.48, source="lux3d", task_id="5025807",
    )

    # 1) Split locally into 4 link segments
    sp = c.post("/api/v1/assets/test-arm-shell/part-split", json={"segments": 4, "axis": "z", "remote": False})
    assert sp.status_code == 200, sp.text
    spj = sp.json()
    assert spj["method"] == "axis_segments"
    assert spj["part_count"] == 4
    files = {p["name"]: p["obj_file"] for p in spj["parts"]}
    assert set(files) >= {"base", "upper", "fore", "wrist"}

    # 2) Assemble split parts onto a simulated 5R robot & verify MuJoCo validation passes
    asm = c.post("/api/v1/robots/assemble-parts", json={
        "project_id": "default",
        "robot_id": "assembled_custom_5r",
        "parts_slug": "test-arm-shell",
        "mapping": files,
        "replace_primary": True,
    })
    assert asm.status_code == 200, asm.text
    assert asm.json()["robot"]["mesh_dir"] == "assets/robots/assembled_custom_5r"

    # 3) Export standard ROS/MuJoCo URDF
    urdf_res = c.get("/api/v1/projects/default/robot.urdf")
    assert urdf_res.status_code == 200, urdf_res.text
    root = ET.fromstring(urdf_res.text)
    assert root.tag == "robot" and root.attrib["name"] == "assembled_custom_5r"
    joint_names = {el.attrib["name"]: el.attrib["type"] for el in root.findall("joint")}
    assert joint_names.get("waist") == "revolute" and joint_names.get("shoulder") == "revolute"
    mesh_refs = [m.attrib["filename"] for m in root.findall(".//visual/geometry/mesh")]
    assert any("assets/robots/assembled_custom_5r/upper.obj" in ref for ref in mesh_refs)

    # 4) Verify Lux3DProvider cloud part_split SDK call routing
    prov = Lux3DProvider()
    fake_sdk = MagicMock()
    fake_sdk.part_split.create.return_value = 777001
    prov._client = fake_sdk
    assert prov.submit_part_split("https://example.com/model.glb") == "777001"
    fake_sdk.part_split.create.assert_called_once_with(glb_url="https://example.com/model.glb")


def test_environment_mesh_online_compilation_import_and_mujoco(tmp_path, monkeypatch):
    lab = _make_lab(tmp_path)
    monkeypatch.delenv("AHOLO_API_KEY", raising=False)
    app = create_app(lab)
    c = TestClient(app)

    s = c.post("/api/v1/generation/sessions", json={
        "kind": "environment_mesh", "role": "static", "initial_text": "modern wooden workbench with backstop",
    }).json()
    sid, rev = s["session_id"], s["revision"]
    r = c.patch(f"/api/v1/generation/sessions/{sid}/spec", json={
        "expected_revision": rev, "name": "wood-bench", "width_m": 0.64, "depth_m": 0.43, "height_m": 0.12,
    }).json()
    comp = c.post(f"/api/v1/generation/sessions/{sid}/compile", json={"expected_revision": r["revision"]})
    assert comp.status_code == 200, comp.text
    assert comp.json()["compiled"]["provider"] == "lux3d"

    # Simulate downloading L-shaped workbench environment GLB & importing into scenes/
    slab = trimesh.creation.box([0.64, 0.43, 0.02], transform=trimesh.transformations.translation_matrix([0, 0, -0.01]))
    back = trimesh.creation.box([0.64, 0.04, 0.10], transform=trimesh.transformations.translation_matrix([0, 0.195, 0.05]))
    bench = trimesh.util.concatenate([slab, back])

    class _BenchProv(MockLux3DProvider):
        def download(self, outputs, directory):
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            out = directory / "result.glb"
            out.write_bytes(bench.export(file_type="glb"))
            return out

    ss = SessionStore(lab / "datasets" / "generation_sessions")
    obj_store = ObjectJobStore(lab / "datasets" / "generation_jobs", _BenchProv(), lab_root=lab)
    orch = Orchestrator(ss, obj_store, None)
    imp = orch.import_remote(sid, "mock-env-1", provider="lux3d", background=False)
    assert imp["job"]["state"] == "succeeded"
    assert imp["job"]["asset_id"].startswith("scenes/")

    rec = c.post(f"/api/v1/generation/sessions/{sid}/recipe", json={
        "scene_mode": "background_scene",
        "task_asset_ref": "builtin/standard",
        "background_ref": imp["job"]["asset_id"],
        "sensor_renderer": "mujoco",
        "use_default_table": True,
        "colliders": [],
    })
    assert rec.status_code == 200, rec.text
    val = c.post(f"/api/v1/scene-recipes/{rec.json()['recipe_id']}/validate?project_id=default")
    assert val.status_code == 200 and val.json()["valid"] is True, val.json()
    assert val.json()["checks"]["layout"]["colliders"] >= 2


def test_expert_step_batching_dynamic_colliders_and_tray_orientation(tmp_path, monkeypatch):
    from asset_pipeline.import_object import register_object

    lab = _make_lab(tmp_path)
    monkeypatch.delenv("AHOLO_API_KEY", raising=False)

    # Rotated tray GLB (0.15 along X, 0.20 along Z in Y-up -> X/Y swapped after Z-up conversion)
    base = trimesh.creation.box([0.15, 0.20, 0.010], transform=trimesh.transformations.translation_matrix([0, 0, 0.005]))
    tray_glb = tmp_path / "rotated_tray.glb"
    tray_glb.write_bytes(base.export(file_type="glb"))
    mf_path = register_object(
        tray_glb, lab / "assets" / "imported" / "rot-tray",
        kind="static_prop", role="static", width=0.20, depth=0.15, height=0.022,
        source="lux3d", task_id="test-rot", collision_mode="compound",
    )
    mf = json.loads(mf_path.read_text(encoding="utf-8"))
    assert mf["calibrated"] is True
    assert mf["compound_proxy"]["box_count"] >= 4
    assert mf["compound_proxy"]["floor_box_index"] == 0

    app = create_app(lab)
    c = TestClient(app)
    resp = c.post("/api/v1/hackathon-demo/setup", json={"project_id": "hackathon_demo"})
    assert resp.status_code == 200, resp.text

    sess = c.post("/api/v1/sessions", json={"project": "hackathon_demo"}).json()
    sid, epoch = sess["id"], sess["epoch"]
    obs0 = sess["observation"]
    assert obs0["state"]["object_pose"]["radius_m"] > 0
    assert obs0["state"]["object_pose"]["height_m"] > 0
    dyn = obs0["state"]["dynamic_objects"]["companion_cube"]
    assert len(dyn["colliders"]) >= 1
    assert dyn["colliders"][0]["size_m"] == pytest.approx([0.036, 0.036, 0.036], abs=2e-3)

    tok = c.post(f"/api/v1/sessions/{sid}/control/acquire", json={"owner": "test"}).json()["token"]
    step_res = c.post(f"/api/v1/sessions/{sid}/expert_step", json={"token": tok, "epoch": epoch, "steps": 10})
    assert step_res.status_code == 200, step_res.text
    sj = step_res.json()
    assert sj["tick"] == obs0["tick"] + 10
    assert "metrics" in sj and isinstance(sj["metrics"]["grasped"], bool)
    c.delete(f"/api/v1/sessions/{sid}")
