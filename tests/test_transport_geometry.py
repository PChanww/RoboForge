"""Actual mesh calibration and an open, contact-only custom destination bin."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from vlab import tasks
from vlab.duck_demo import DUCK
from vlab.transport_geometry import bin_geometry, prepare_object


LAB = Path(__file__).resolve().parents[1]


def vertices(mesh):
    return np.array([[float(v) for v in line.split()[1:4]]
                     for line in mesh.decode().splitlines() if line.startswith("v ")])


def test_no_object_overrides_preserve_original_geometry_and_dynamics():
    baseline = tasks.asset_params(DUCK, LAB)
    result = prepare_object(LAB, DUCK, {})
    assert result[:6] == baseline[:6]
    assert result[6]["friction"] == 1.2
    assert set(result[6]["parameter_sources"].values()) == {"asset_or_task_estimate"}


def test_object_dimensions_and_mass_change_physics_without_editing_source():
    source = LAB / "assets" / DUCK / "visual.obj"
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    r, h, mass, color, mesh, offset, meta = prepare_object(LAB, DUCK, {
        "width_m": .04, "depth_m": .05, "height_m": .065, "mass_kg": .08, "friction": .7,
    })
    points = vertices(mesh)
    np.testing.assert_allclose(np.ptp(points, axis=0), [.04, .05, .065], atol=1e-10)
    np.testing.assert_allclose((points.min(axis=0)+points.max(axis=0))[:2], 0, atol=1e-10)
    assert points[:, 2].min() == pytest.approx(0, abs=1e-10)
    assert r == .025 and h == .065 and mass == .08 and offset == -.0325
    assert meta["friction"] == .7
    assert set(meta["parameter_sources"].values()) == {"user_explicit"}
    model = mujoco.MjModel.from_xml_string(
        f'<mujoco><asset><mesh name="object" file="object.obj"/></asset>'
        f'<worldbody><body name="object"><freejoint/><geom type="mesh" mesh="object" mass="{mass}"/></body></worldbody></mujoco>',
        {"object.obj": mesh},
    )
    assert model.body("object").mass[0] == pytest.approx(.08)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before


def test_box_import_without_legacy_radius_uses_actual_mesh_and_preserves_package(tmp_path):
    package = tmp_path / "assets/imported/box-object"
    package.mkdir(parents=True)
    mesh = b"v -.02 -.025 0\nv .02 -.025 0\nv -.02 .025 0\nv -.02 -.025 .06\nf 1 2 3\nf 1 2 4\nf 1 3 4\nf 2 3 4\n"
    manifest = json.dumps({
        "source": "lux3d", "generation_id": "test", "kind": "task_object", "role": "dynamic",
        "obj": "visual.obj", "radius_m": None, "height_m": .06, "mass_kg": .04,
        "collision_proxy": {"shape": "box", "size_m": [.04, .05, .06]},
    })
    (package / "visual.obj").write_bytes(mesh)
    (package / "manifest.json").write_text(manifest)
    r, h, mass, color, actual_mesh, offset, metadata = prepare_object(tmp_path, "imported/box-object")
    assert r == .025 and h == .06 and mass == .04 and offset == -.03
    assert actual_mesh == mesh
    assert metadata["source_mesh_sha256"] == hashlib.sha256(mesh).hexdigest()
    assert metadata["parameters"]["depth_m"] == .05
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><asset><mesh name="object" file="object.obj"/></asset>'
        f'<worldbody><body name="object"><freejoint/><geom type="mesh" mesh="object" mass="{mass}"/></body></worldbody></mujoco>',
        {"object.obj": actual_mesh},
    )
    assert model.body("object").mass[0] == pytest.approx(.04)
    assert (package / "visual.obj").read_bytes() == mesh
    assert (package / "manifest.json").read_text() == manifest


@pytest.mark.parametrize("parameters", [
    {"mass_kg": float("nan")}, {"mass_kg": .151}, {"height_m": .076},
    {"width_m": .001}, {"friction": 0}, {"density": 1000}, {"mass_kg": True},
])
def test_invalid_physics_parameters_are_rejected(parameters):
    with pytest.raises(ValueError):
        prepare_object(LAB, DUCK, parameters)


def test_default_bin_matches_existing_contact_envelope_and_has_open_top():
    body, assets, info = bin_geometry([.12, .84], [.21, .22, .056], "#2E8F85")
    assert not assets
    root = ET.fromstring("<root>"+body+"</root>")
    assert len(root) == 5
    np.testing.assert_allclose(info["inner_half_m"], [.099, .104])
    assert info["rim_z_m"] == pytest.approx(.065)
    np.testing.assert_allclose([float(v) for v in root.find("geom[@name='bin_wall_0']").attrib["pos"].split()], [.018, .84, .037])
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><option timestep=".002"/><worldbody>'+body+
        '<body name="probe" pos=".12 .84 .18"><freejoint/>'
        '<geom name="probe" type="sphere" size=".01" mass=".01"/></body></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    # A lid/convex whole-bin collision would stop this object near .075 m.
    mujoco.mj_step(model, data, nstep=1500)
    assert data.qpos[2] == pytest.approx(.019, abs=.001)
    assert any({int(c.geom1), int(c.geom2)} == {model.geom("tray").id, model.geom("probe").id} for c in data.contact)


def test_custom_aholo_bin_is_scaled_visual_with_independent_open_contacts(tmp_path):
    package = tmp_path / "assets/imported/test-bin"
    package.mkdir(parents=True)
    source = b"v 1 2 3\nv 3 2 3\nv 1 6 3\nv 1 2 9\nf 1 2 3\nf 1 2 4\nf 1 3 4\nf 2 3 4\n"
    (package / "visual.obj").write_bytes(source)
    (package / "manifest.json").write_text(json.dumps({
        "source": "lux3d", "generation_id": "test", "kind": "static_prop", "role": "visual_only", "obj": "visual.obj",
    }))
    body, assets, info = bin_geometry([-.4, .6], [.24, .28, .08], "#FF0000", tmp_path, "imported/test-bin")
    points = vertices(assets["bin_visual.obj"])
    np.testing.assert_allclose(np.ptp(points, axis=0), [.24, .28, .089], atol=1e-10)
    np.testing.assert_allclose(points.min(axis=0), [-.12, -.14, 0], atol=1e-10)
    assert (package / "visual.obj").read_bytes() == source
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><asset><mesh name="bin_visual" file="bin_visual.obj"/></asset>'
        '<default><default class="visual"><geom contype="0" conaffinity="0"/></default></default>'
        '<worldbody>'+body+'</worldbody></mujoco>', assets,
    )
    assert model.ngeom == 6
    visual = model.geom("transport_bin_visual").id
    assert model.geom_contype[visual] == model.geom_conaffinity[visual] == 0
    assert sum(model.geom_contype != 0) == 5
    np.testing.assert_allclose(info["inner_half_m"], [.114, .134])
    assert info["visual_asset"]["source"] == "lux3d"


@pytest.mark.parametrize("goal,size,color", [
    ([float("inf"), 0], [.21, .22, .056], "#00AAFF"),
    ([0, 0], [.1, .22, .056], "#00AAFF"),
    ([0, 0], [.21, .22, .1], "#00AAFF"),
    ([0, 0], [.21, .22, .056], '#fff"/><geom type="plane'),
])
def test_invalid_bin_configuration_is_rejected(goal, size, color):
    with pytest.raises(ValueError):
        bin_geometry(goal, size, color)


def test_import_paths_cannot_escape_asset_package():
    with pytest.raises(ValueError):
        prepare_object(LAB, "imported/../../.env", {})
