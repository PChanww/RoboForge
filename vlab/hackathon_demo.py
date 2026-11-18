"""Turnkey Hackathon Flagship Demo builder for RoboForge x Aholo.

Demonstrates all physics & generation breakthroughs together in real MuJoCo:
- Concave 5-box compound tray (recessed floor + 4 physical rim walls; no convex bridging)
- Multiple coexisting freejoint dynamic objects (task_asset cup + dynamic calibration block)
- Optional Aholo Lux3D cup (imported/lux3d-blue-live-5887561) & Aholo World background
- Parameterized support_plane_z, custom tray goal judging, part_split + URDF export ready
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import trimesh

from asset_pipeline.import_object import register_object
from .generation.scene_recipe import apply_mutate, validate_recipe
from .generation.schema import AssetNode, AssetRole, SceneMode, SceneRecipe
from .scene_assets import load_scene
from .store import ConfigStore


def _build_concave_tray_glb(dest_glb: Path, outer_w: float = 0.20, outer_d: float = 0.15,
                            floor_h: float = 0.008, wall_h: float = 0.022, wall_t: float = 0.012) -> None:
    """Create a realistic concave open-top ceramic tray GLB (Y-up for register_object)."""
    base = trimesh.creation.box(
        [outer_w, outer_d, floor_h],
        transform=trimesh.transformations.translation_matrix([0.0, 0.0, floor_h / 2.0]),
    )
    w_left = trimesh.creation.box(
        [wall_t, outer_d, wall_h - floor_h],
        transform=trimesh.transformations.translation_matrix([-(outer_w - wall_t) / 2.0, 0.0, floor_h + (wall_h - floor_h) / 2.0]),
    )
    w_right = trimesh.creation.box(
        [wall_t, outer_d, wall_h - floor_h],
        transform=trimesh.transformations.translation_matrix([(outer_w - wall_t) / 2.0, 0.0, floor_h + (wall_h - floor_h) / 2.0]),
    )
    w_front = trimesh.creation.box(
        [outer_w - 2 * wall_t, wall_t, wall_h - floor_h],
        transform=trimesh.transformations.translation_matrix([0.0, -(outer_d - wall_t) / 2.0, floor_h + (wall_h - floor_h) / 2.0]),
    )
    w_back = trimesh.creation.box(
        [outer_w - 2 * wall_t, wall_t, wall_h - floor_h],
        transform=trimesh.transformations.translation_matrix([0.0, (outer_d - wall_t) / 2.0, floor_h + (wall_h - floor_h) / 2.0]),
    )
    tray_zup = trimesh.util.concatenate([base, w_left, w_right, w_front, w_back])
    # Convert Z-up -> Y-up for GLB export because register_object(up="y") rotates Y-up -> Z-up
    rot_y_up = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)
    tray_zup.apply_transform(rot_y_up)
    tray_zup.visual.vertex_colors = [225, 220, 210, 255]
    dest_glb.write_bytes(tray_zup.export(file_type="glb"))


def _build_cube_glb(dest_glb: Path, size: float = 0.036) -> None:
    cube = trimesh.creation.box([size, size, size])
    cube.visual.vertex_colors = [235, 130, 50, 255]
    dest_glb.write_bytes(cube.export(file_type="glb"))


def _build_workbench_glb(dest_glb: Path, outer_w: float = 0.66, outer_d: float = 0.45, height: float = 0.056) -> None:
    top = trimesh.creation.box([outer_w, outer_d, height * 0.6],
                               transform=trimesh.transformations.translation_matrix([0.0, 0.0, height * 0.7]))
    frame = trimesh.creation.box([outer_w * 0.94, outer_d * 0.92, height * 0.4],
                                 transform=trimesh.transformations.translation_matrix([0.0, 0.0, height * 0.2]))
    bench = trimesh.util.concatenate([frame, top])
    rot_y_up = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)
    bench.apply_transform(rot_y_up)
    bench.visual.vertex_colors = [88, 102, 110, 255]
    dest_glb.write_bytes(bench.export(file_type="glb"))


def ensure_demo_assets(lab_root: Path) -> dict:
    lab_root = Path(lab_root)
    repo_imported = Path(__file__).resolve().parents[1] / "assets" / "imported"
    tray_dir = lab_root / "assets" / "imported" / "demo-concave-tray"
    cube_dir = lab_root / "assets" / "imported" / "demo-dynamic-cube"
    bench_dir = lab_root / "assets" / "imported" / "demo-workbench"
    for slug, dst_dir in (("demo-concave-tray", tray_dir), ("demo-dynamic-cube", cube_dir), ("demo-workbench", bench_dir)):
        src_dir = repo_imported / slug
        if not (dst_dir / "manifest.json").is_file() and (src_dir / "manifest.json").is_file() and src_dir.resolve() != dst_dir.resolve():
            dst_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        if not (tray_dir / "manifest.json").is_file():
            tray_glb = tmp / "tray.glb"
            _build_concave_tray_glb(tray_glb)
            register_object(
                tray_glb, tray_dir,
                kind="static_prop", role="static",
                width=0.20, depth=0.15, height=0.022, mass=0.45,
                source="other", source_detail="RoboForge concave compound demo tray",
                prompt="Open ceramic sorting tray 20x15x2.2 cm with recessed cavity and raised rim",
                collision_mode="compound",
            )
        if not (cube_dir / "manifest.json").is_file():
            cube_glb = tmp / "cube.glb"
            _build_cube_glb(cube_glb, size=0.036)
            register_object(
                cube_glb, cube_dir,
                kind="task_object", role="dynamic",
                width=0.036, depth=0.036, height=0.036, mass=0.06,
                source="other", source_detail="RoboForge dynamic desktop companion block",
                prompt="Orange wooden calibration cube 3.6 cm",
            )
        if not (bench_dir / "manifest.json").is_file():
            bench_glb = tmp / "workbench.glb"
            _build_workbench_glb(bench_glb)
            register_object(
                bench_glb, bench_dir,
                kind="static_prop", role="static",
                width=0.66, depth=0.45, height=0.056, mass=2.5,
                source="other", source_detail="RoboForge laboratory workbench platform",
                prompt="Modern robotics laboratory workbench platform 66x45x5.6 cm",
                collision_mode="primitive",
            )
    tray_mf = json.loads((tray_dir / "manifest.json").read_text(encoding="utf-8"))
    cube_mf = json.loads((cube_dir / "manifest.json").read_text(encoding="utf-8"))
    bench_mf = json.loads((bench_dir / "manifest.json").read_text(encoding="utf-8"))
    return {"tray": tray_mf, "cube": cube_mf, "workbench": bench_mf}


def setup_hackathon_demo(
    lab_root: Path,
    store: ConfigStore,
    *,
    project_id: str = "hackathon_demo",
    with_world_background: bool = False,
    support_plane_z: float = 0.0,
) -> dict:
    """Create/update the flagship showcase project with concave compound tray + multi-dynamic objects."""
    lab_root = Path(lab_root)
    ensure_demo_assets(lab_root)
    cup_slug = "imported/lux3d-blue-live-5887561"
    if not (lab_root / "assets" / "imported" / "lux3d-blue-live-5887561" / "manifest.json").is_file():
        cup_slug = "builtin/standard"

    bg_ref = None
    mode = SceneMode.composed_scene
    renderer = "mujoco"
    if with_world_background:
        world_mf = lab_root / "assets" / "scenes" / "world_586c79650960" / "1" / "manifest.json"
        if world_mf.is_file():
            load_scene(lab_root, "world_586c79650960", 1, physics=False)
            bg_ref = "scenes/world_586c79650960/1"
            mode = SceneMode.background_scene
            renderer = "hybrid"

    tray_mf = json.loads((lab_root / "assets" / "imported" / "demo-concave-tray" / "manifest.json").read_text(encoding="utf-8"))
    comp = tray_mf.get("compound_proxy") or {}
    boxes = comp.get("boxes") or []
    fidx = int(comp.get("floor_box_index") or 0)
    fbox = boxes[fidx] if 0 <= fidx < len(boxes) else None
    goal_surface = support_plane_z + (float(fbox["position"][2]) + float(fbox["size_m"][2]) / 2.0 if fbox else 0.008)

    recipe = SceneRecipe(
        recipe_id=f"recipe_{project_id}",
        task_goal="Aholo x RoboForge: Concave Tray & Multi-Dynamic Tabletop Manipulation Demo",
        scene_mode=mode,
        task_asset_ref=cup_slug,
        support_plane_z=float(support_plane_z),
        background_ref=bg_ref,
        use_default_table=True,
        sensor_renderer=renderer,
        goal_tray_node_id="demo_tray",
        goal_xy=[0.133, 0.050],
        goal_half_extents=[0.072, 0.052],
        goal_surface_m=goal_surface,
        asset_nodes=[
            AssetNode(
                node_id="demo_workbench",
                asset_ref="imported/demo-workbench",
                role=AssetRole.visual_only,
                position=[0.0, 0.0, float(support_plane_z) - 0.0556],
            ),
            AssetNode(
                node_id="demo_tray",
                asset_ref="imported/demo-concave-tray",
                role=AssetRole.static,
                position=[0.133, 0.050, float(support_plane_z)],
                collision_mode="compound",
            ),
            AssetNode(
                node_id="companion_cube",
                asset_ref="imported/demo-dynamic-cube",
                role=AssetRole.dynamic,
                position=[0.015, -0.095, float(support_plane_z) + 0.001],
                mass_kg=0.06,
            ),
        ],
    )

    if not store.exists(project_id):
        base = store.load("default").model_copy(update={"project_id": project_id, "version": 1})
        store.save(base)

    report = validate_recipe(recipe, lab_root, store, project_id)
    if not report["valid"]:
        raise ValueError("Demo recipe failed MuJoCo validation: " + "; ".join(report["issues"]))

    cur = store.load(project_id)
    data = cur.model_dump()
    apply_mutate(recipe, lab_root=lab_root)(data)
    data["project_id"] = project_id
    data["validation_status"] = "validated"
    data["task"]["instruction"] = "Pick up the ceramic cup and place it inside the concave tray cavity alongside the dynamic block."
    from .schema import ProjectConfig
    saved = store.save(ProjectConfig.model_validate(data))

    return {
        "project_id": saved.project_id,
        "version": saved.version,
        "validation_status": saved.validation_status,
        "task_asset": saved.task.asset,
        "goal_surface_m": saved.task.goal_surface_m,
        "goal_tray_collider_ref": saved.task.goal_tray_collider_ref,
        "tray_compound_boxes": len(boxes),
        "dynamic_nodes": [n.node_id for n in saved.scene.asset_nodes if n.role == "dynamic"],
        "background": bg_ref,
        "sensor_renderer": saved.scene.sensor_renderer,
        "report": report,
    }
