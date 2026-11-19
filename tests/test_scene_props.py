"""Imported static meshes render in both sensors; optional box proxies make contact."""
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest
from pydantic import ValidationError

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
from asset_pipeline.import_object import register_object
from vlab.engine import PlatformEnv
from vlab.scene_builder import environment_xml
from vlab.schema import ProjectConfig, SceneCollider, SceneConfig, SceneProp

REAL_GLB = LAB / "assets/imported/yellow-plastic-duck-a08e3e/original.glb"


@pytest.fixture
def prop_asset(tmp_path):
    directory = tmp_path / "assets/imported/generated-scenery"
    register_object(REAL_GLB, directory, kind="static_prop", role="static", height=.10,
                    source="lux3d", task_id="existing-real-asset-fixture")
    return tmp_path, directory


def config():
    cfg = ProjectConfig.model_validate_json((LAB / "configs/default.json").read_text(encoding="utf-8"))
    for robot in cfg.robots:
        robot.mesh_dir = str(LAB / robot.mesh_dir)
    cfg.task.asset = "block"
    return cfg


def test_static_prop_builds_renders_both_cameras_and_tracks_provenance(prop_asset):
    root, directory = prop_asset
    cfg = config()
    cfg.scene.props = [SceneProp(name="scenery", asset="imported/generated-scenery",
                                  position=[0, .40, -.061], uniform_scale=2.)]
    env = PlatformEnv(cfg, root)
    try:
        assert env.model.ncam == 2
        for sensor in cfg.sensors:
            frame = env._render_camera(sensor)
            assert frame.shape == (sensor.resolution[1], sensor.resolution[0], 3)
            assert np.std(frame) > 2
        geom = env.model.geom("scene_prop_scenery_visual")
        assert geom.contype == geom.conaffinity == 0
        assert env.model.body("scene_prop_scenery").dofnum == 0
        provenance = env._env_info["props"][0]
        assert provenance["asset"] == "imported/generated-scenery"
        assert provenance["source"] == "lux3d"
        assert provenance["generation_id"] == "existing-real-asset-fixture"
        assert len(provenance["manifest_sha256"]) == len(provenance["mesh_sha256"]) == 64
        assert provenance["collision_proxy"] is None
        assert np.isclose(np.diff(provenance["bounds_m"], axis=0)[0][2], .2)
    finally:
        env.close()


def test_requested_proxy_physically_supports_dynamic_body(prop_asset):
    root, _ = prop_asset
    cfg = config()
    cfg.scene.use_default_table = False
    cfg.scene.props = [SceneProp(name="scenery", asset="imported/generated-scenery", collision=True,
                                  position=[.4, .4, 0], quaternion_xyzw=[0, 0, 1, 1], uniform_scale=1.5)]
    xml, assets, info = environment_xml(cfg, root)
    declarations = ''.join(f'<mesh name="{Path(name).stem}" file="{name}"/>' for name in assets)
    model = mujoco.MjModel.from_xml_string(f'''<mujoco>
      <option timestep=".002"/>
      <default><default class="visual"><geom contype="0" conaffinity="0" mass="0"/></default></default>
      <asset>{declarations}</asset><worldbody>{xml}
      <body name="probe" pos=".4 .4 .20"><freejoint/><geom name="probe_geom" type="sphere" size=".01" mass=".01"/></body>
      </worldbody></mujoco>''', assets)
    data = mujoco.MjData(model)
    proxy = model.geom("scene_prop_scenery_contact").id
    probe = model.geom("probe_geom").id
    touched = False
    for _ in range(400):
        mujoco.mj_step(model, data)
        touched |= any({int(c.geom1), int(c.geom2)} == {proxy, probe} for c in data.contact)
    assert touched, "explicit static proxy must provide real contact"
    assert data.xpos[model.body("probe").id][2] > .15
    assert "estimated" in info["props"][0]["collision_proxy"]["basis"]


def test_dynamic_task_asset_cannot_be_silently_frozen_as_prop(prop_asset):
    root, directory = prop_asset
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    manifest.update(kind="task_object", role="dynamic")
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    cfg = config()
    cfg.scene.props = [SceneProp(name="scenery", asset="imported/generated-scenery")]
    with pytest.raises(ValueError, match="static or visual_only"):
        environment_xml(cfg, root)


@pytest.mark.parametrize("change", [
    {"name": "../escape"}, {"asset": "imported/../escape"}, {"asset": "builtin/standard"},
    {"position": [0, 0]}, {"position": [0, float("inf"), 0]},
    {"quaternion_xyzw": [0, 0, 0, 0]}, {"quaternion_xyzw": [1e300]*4},
    {"uniform_scale": 0}, {"uniform_scale": float("nan")},
])
def test_prop_rejects_invalid_identifiers_pose_and_scale(change):
    with pytest.raises(ValidationError):
        SceneProp(**{"name": "scenery", "asset": "imported/generated-scenery", **change})


def test_scene_rejects_ambiguous_prop_and_collider_names():
    prop = SceneProp(name="scenery", asset="imported/generated-scenery")
    with pytest.raises(ValidationError, match="unique"):
        SceneConfig(props=[prop, prop])
    with pytest.raises(ValidationError, match="conflict"):
        SceneConfig(props=[prop], colliders=[SceneCollider(name="scenery")])
    with pytest.raises(ValidationError, match="conflict"):
        SceneConfig(props=[prop], colliders=[SceneCollider(name="scene_prop_scenery_contact")])


def test_prop_rejects_manifest_escape_and_nonfinite_vertices(prop_asset):
    root, directory = prop_asset
    cfg = config()
    cfg.scene.props = [SceneProp(name="scenery", asset="imported/generated-scenery")]
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    manifest["obj"] = "../outside.obj"
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes"):
        environment_xml(cfg, root)
    manifest["obj"] = "visual.obj"
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (directory / "visual.obj").write_text("v nan 0 0\nv 0 1 0\nv 0 0 1\nv 1 0 0\nf 1 2 3\n", encoding="ascii")
    with pytest.raises(ValueError, match="non-finite"):
        environment_xml(cfg, root)


def test_empty_props_preserve_previous_scene_output():
    cfg = config()
    xml, assets, info = environment_xml(cfg, LAB)
    assert not assets and "props" not in info
    assert "scene_prop_" not in xml
    assert info["default_table"]
