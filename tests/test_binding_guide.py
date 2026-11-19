import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vlab.api import create_app
from vlab.generation.api import register_generation_api
from vlab.generation.object_provider import MockLux3DProvider, ObjectJobStore
from vlab.generation.schema import AssetKind, AssetRole, AssetSpec
from vlab.make_default_config import build
from vlab.make_second_config import build as build_gantry
from vlab.store import ConfigStore
from vlab.world_provider import GenerationJobStore, MockWorldProvider


def json_sections(markdown):
    return [json.loads(block.split("\n```", 1)[0]) for block in markdown.split("```json\n")[1:]]


@pytest.mark.parametrize(("kind", "role"), [
    (AssetKind.task_object, AssetRole.dynamic),
    (AssetKind.static_prop, AssetRole.static),
    (AssetKind.robot_visual, AssetRole.visual_only),
    (AssetKind.environment_mesh, AssetRole.static),
    (AssetKind.environment_splat, AssetRole.visual_only),
])
def test_generated_asset_guide_uses_registered_manifest_and_frozen_project(tmp_path, monkeypatch, kind, role):
    config_store = ConfigStore(tmp_path / "projects")
    original = config_store.save(build())
    jobs_root = tmp_path / "datasets" / "generation_jobs"
    objects = ObjectJobStore(jobs_root, MockLux3DProvider(), lab_root=tmp_path)
    world = GenerationJobStore(jobs_root, MockWorldProvider(), lab_root=tmp_path)
    monkeypatch.setattr("vlab.generation.api.ObjectJobStore", lambda *args, **kwargs: objects)
    app = FastAPI()
    ctx = register_generation_api(app, tmp_path, config_store, world)
    spec = AssetSpec(kind=kind, role=role, name="guide-test")
    if kind not in {AssetKind.environment_splat, AssetKind.robot_visual}:
        spec.height.value_m = 0.09
    sessions = ctx["sessions"]
    session = sessions.create(project_id=original.project_id, asset_spec=spec)
    other = sessions.create(project_id=original.project_id)
    orchestrator = ctx["orchestrator"]
    job_id = orchestrator.confirm(session.session_id, background=False)["job"]["id"]
    path = f"/api/v1/generation/sessions/{session.session_id}/tasks/{job_id}/binding-guide"

    with TestClient(app) as client:
        assert client.get(path).status_code == 409
        assert client.get(path.replace(session.session_id, other.session_id)).status_code == 404

        done = orchestrator.poll(session.session_id, job_id, background=False)
        assert done["state"] == "succeeded"
        updated = original.model_copy(deep=True)
        updated.sensors[0].id = "renamed_front"
        config_store.save(updated)

        response = client.get(path)
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/markdown")
        assert response.headers["content-disposition"] == f'attachment; filename="{job_id}-binding.md"'
        asset, project = json_sections(response.text)
        assert asset["kind"] == kind.value
        if kind == AssetKind.environment_splat:
            assert asset["asset_ref"] == f"scenes/{done['asset_id']}/{done['revision']}"
            assert "固定前视" in response.text
        else:
            assert asset["asset_ref"] == done["asset_id"]
        if kind in {AssetKind.task_object, AssetKind.static_prop}:
            assert asset["height_m"] == 0.09
        elif kind == AssetKind.robot_visual:
            assert "绑定到连杆" in response.text and asset["role"] == "visual_only"
        elif kind == AssetKind.environment_mesh:
            assert "工作面和碰撞代理" in response.text and asset["calibrated"]
        assert project["config_version"] == original.version
        assert project["sensors"][0]["id"] == original.sensors[0].id


def test_project_guide_reports_saved_joint_sensor_and_action_bindings(tmp_path):
    store = ConfigStore(tmp_path / "projects")
    config = store.save(build())
    with TestClient(create_app(tmp_path)) as client:
        path = f"/api/v1/projects/{config.project_id}/binding-guide"
        response = client.get(path, params={"version": config.version})
        assert response.status_code == 200
        project, scene = json_sections(response.text)
        assert project["config_version"] == config.version
        assert [sensor["id"] for sensor in project["sensors"]] == [sensor.id for sensor in config.sensors]
        joint_units = {joint["name"]: joint["position_unit"] for joint in project["robots"][0]["joints"]}
        assert joint_units["waist"] == "rad"
        assert joint_units["left"] == "m"
        assert project["controllers"][0]["action_names"] == config.controllers[0].action_names
        assert scene["task_asset"] == config.task.asset
        assert client.get(path, params={"version": config.version + 99}).status_code == 404


def test_gantry_binding_uses_its_own_controller_and_skips_unmounted_camera(tmp_path):
    config = build()
    config.sensors[1].parent = "wrist"  # Present on the arm, missing from the gantry.
    gantry = config.robots[1]
    gantry_id = gantry.id
    pads = {name: f"gantry_{name}" for name in gantry.gripper.pad_geoms}
    for link in gantry.links:
        for geom in link.collisions:
            if geom.name in pads:
                geom.name = pads[geom.name]
    gantry.gripper.pad_geoms = list(pads.values())
    store = ConfigStore(tmp_path / "projects")
    store.save(config)
    with TestClient(create_app(tmp_path)) as client:
        sessions = "/api/v1/sessions"
        guide_path = "/api/v1/projects/default/binding-guide"
        result = client.post(sessions, json={"robot_id": gantry_id, "controller_id": "joints"})
        assert result.status_code == 400
        assert "waist" in result.json()["error"]
        unavailable, _ = json_sections(client.get(guide_path).text)
        assert unavailable["robots"][1]["controller_ids"] == []
        assert "wrist" in unavailable["robots"][1]["unavailable_sensors"]

        controller = build_gantry().controllers[0].model_copy(update={"id": "gantry_joints"})
        config.controllers.append(controller)
        store.save(config)
        capability = client.get("/api/v1/capabilities").json()["robots"][1]
        guide, _ = json_sections(client.get(guide_path).text)
        assert guide["robots"][1]["controller_ids"] == capability["controller_ids"] == ["gantry_joints"]
        assert guide["robots"][1]["sensor_ids"] == capability["sensor_ids"] == ["front"]
        report = client.post("/api/v1/compat", json={
            "robot_id": gantry_id,
            "controller_id": controller.id,
            "contract": {"action_names": controller.action_names, "camera_keys": ["front"]},
        }).json()
        assert report["compatible"] and report["controller"] == controller.id
        bad_contract = client.post("/api/v1/compat", json={
            "robot_id": gantry_id, "controller_id": controller.id,
            "contract": {"action_units": ["rad"] * 4, "camera_keys": ["wrist"]},
        }).json()
        assert not bad_contract["compatible"]
        assert any("action units" in issue for issue in bad_contract["mismatches"])
        assert any("wrist" in issue for issue in bad_contract["mismatches"])

        response = client.post(sessions, json={"robot_id": gantry_id})
        assert response.status_code == 200, response.text
        session = response.json()
        assert session["controller"] == controller.id
        assert session["observation"]["images"]["wrist"] is None
        assert "mount" in session["observation"]["camera_status"]["wrist"]["render_error"]
        path = f"{sessions}/{session['id']}"
        token = client.post(path + "/control/acquire", json={"owner": "gantry-test"}).json()["token"]
        obs = client.post(path + "/action", json={
            "token": token, "epoch": session["epoch"], "tick": 0, "action": [0.0, 0.0, 0.0, 0.058],
        })
        assert obs.status_code == 200, obs.text
        assert obs.json()["tick"] == 1
        assert set(obs.json()["state"]["joint_pos"]) == {"x", "y", "z", "left", "right"}
        assert obs.json()["camera_status"]["wrist"]["state"] == "render_unavailable"
