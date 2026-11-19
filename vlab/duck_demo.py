"""Duck demo template. Execution uses the normal config/session/job APIs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from fastapi import HTTPException

from .schema import JointConfig, LinkConfig, VisualGeom
from .scene_assets import valid_id

DUCK = "imported/yellow-plastic-duck-a08e3e"
CHASSIS = "imported/four-wheel-robot-chassis-82e8e5"


def asset_info(lab_root: Path, ref: str, name: str):
    if not ref.startswith("imported/"):
        raise ValueError("Please select an imported Aholo asset")
    valid_id(ref.split("/", 1)[1])
    path = lab_root / "assets" / ref / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return {"id": ref, "name": name, "source": manifest["source"],
            "generation_id": manifest["generation_id"], "height_m": manifest["height_m"],
            "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "mesh_sha256": manifest["sha256_glb"], "kind": manifest.get("kind"),
            "role": manifest.get("role"), "bounds_m": manifest.get("bounds_m")}


def make_config(base, lab_root: Path, seed=0, asset=DUCK, chassis=CHASSIS):
    obj = asset_info(lab_root, asset, "抓取物体")
    body = asset_info(lab_root, chassis, "四轮底盘")
    if obj["kind"] != "task_object" or obj["role"] != "dynamic" or obj["source"] != "lux3d":
        raise ValueError("抓取物体需要 Aholo 生成的动态单体模型")
    extent = np.diff(np.array(obj["bounds_m"]), axis=0)[0]
    if obj["height_m"] > .075 or np.max(extent[:2]) > .085:
        raise ValueError("当前夹爪仅支持高度不超过 7.5 cm、水平尺寸不超过 8.5 cm 的小型刚体")
    if body["kind"] != "robot_visual" or body["source"] != "lux3d":
        raise ValueError("底盘需要 Aholo robot_visual 外观模型")
    cfg = base.model_copy(deep=True)
    cfg.project_id, cfg.version = "duck_demo_template", 1
    cfg.robots = [cfg.robots[0]]
    robot = cfg.robots[0]
    # Meshes are assembled once as data; original imported meshes stay intact.
    assembly = hashlib.sha256(chassis.encode()).hexdigest()[:12]
    target = lab_root / "assets/robot" / f"duck_sorter_{assembly}"
    target.mkdir(parents=True, exist_ok=True)
    files = {f"{v.mesh}.obj": lab_root / robot.mesh_dir / f"{v.mesh}.obj"
             for link in robot.links for v in link.visuals}
    files["chassis.obj"] = lab_root / "assets" / chassis / "visual.obj"
    for name, source in files.items():
        dest = target / name
        data = source.read_bytes()
        if not dest.exists() or dest.read_bytes() != data:
            dest.write_bytes(data)
    robot.mesh_dir = str(target.relative_to(lab_root)).replace("\\", "/")
    robot.links.append(LinkConfig(name="mobile_chassis", parent=robot.root_link,
                                  origin=[0, .025, -.061], gravcomp=False,
                                  visuals=[VisualGeom(mesh="chassis", rgba=[.38, .43, .45, 1])]))
    robot.joints.append(JointConfig(name="chassis_mount", child="mobile_chassis", type="fixed"))
    for link in robot.links:
        for geom in link.collisions:
            if geom.name in ("left_pad", "right_pad"):
                geom.friction = [2., .005, .0001]
                geom.size = [.022, .006, .026]
    cfg.scene.seed = seed
    cfg.scene.environment = None
    cfg.scene.sensor_renderer = "mujoco"
    cfg.scene.use_default_table = False
    cfg.scene.colliders = []
    cfg.scene.props = []
    cfg.task.plugin = cfg.task.id = "duck_conveyor"
    cfg.task.asset = asset
    cfg.task.duck_conveyor.chassis_asset = chassis
    cfg.task.duck_conveyor.continuous = True
    cfg.task.duck_conveyor.belt_speed_m_s = .025
    cfg.task.time_limit_s = 25
    cfg.task.spawn_min, cfg.task.spawn_max = [-.22, -.079], [-.18, -.061]
    cfg.task.goal_xy = [.19, .035]
    cfg.task.instruction = "将传送带上的抓取物体放入机器人旁的收纳箱内。"
    for sensor in cfg.sensors:
        sensor.fps = 20
        if sensor.id != "front":
            continue
        sensor.position = [-.48, -.64, .48]
        z = np.array(sensor.position) - np.array([-.025, .065, .075])
        z /= np.linalg.norm(z)
        x = np.cross([0, 0, 1], z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        sensor.xyaxes = [*x.tolist(), *y.tolist()]
        sensor.resolution = [640, 480]
        sensor.intrinsics.fx = sensor.intrinsics.fy = 600
        sensor.intrinsics.cx, sensor.intrinsics.cy = 320, 240
    cfg.provenance.update({
        "duck.collision": "convex hull of generated mesh; estimated mass/friction",
        "conveyor.contact": "driven rollers; visual slats are not collision surfaces",
        "robot.base": "parked fixed mount; Aholo chassis visual, no driving policy",
        "robot.chassis": chassis,
        "expert": "privileged object pose, fixed downward gripper attitude",
    })
    return cfg


def register_duck_demo(app, store, lab_root):
    @app.get("/api/v1/duck-demo")
    def template(seed: int = 0, asset: str = DUCK, chassis: str = CHASSIS):
        if not 0 <= seed <= 1_000_018:
            raise HTTPException(400, "seed out of range")
        try:
            assets = [asset_info(lab_root, asset, "塑料小鸭 · Aholo" if asset == DUCK else asset.split("/")[-1]),
                      asset_info(lab_root, chassis, "四轮底盘 · Aholo")]
            config = make_config(store.load(), lab_root, seed, asset, chassis)
            return {"assets": assets, "config": config.model_dump(mode="json")}
        except (FileNotFoundError, KeyError, ValueError) as exc:
            raise HTTPException(409, f"Duck demo asset unavailable: {exc}")

    @app.get("/api/v1/duck-demo/assets")
    def catalog():
        objects, chassis = [], []
        for path in sorted((lab_root / "assets/imported").glob("*/manifest.json")):
            try:
                ref = f"imported/{path.parent.name}"
                item = asset_info(lab_root, ref, "塑料小鸭 · Aholo" if ref == DUCK else path.parent.name)
                if item["source"] != "lux3d":
                    continue
                extent = np.diff(np.array(item["bounds_m"]), axis=0)[0]
                if (item["kind"] == "task_object" and item["role"] == "dynamic"
                        and item["height_m"] <= .075 and np.max(extent[:2]) <= .085):
                    objects.append(item)
                elif item["kind"] == "robot_visual":
                    chassis.append(item)
            except (ValueError, KeyError):
                continue
        objects.sort(key=lambda a: a["id"] != DUCK)
        chassis.sort(key=lambda a: a["id"] != CHASSIS)
        return {"objects": objects, "chassis": chassis}
