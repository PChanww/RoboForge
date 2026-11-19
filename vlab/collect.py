"""Collect expert demonstrations into a staging dataset (npz per episode).

Runs the privileged scripted expert in the configured scene and records, per
control tick: named joint state, eef pose, gripper opening, applied action, both
camera frames with capture/available times, and the success label. The staging
data is converted to LeRobot v3 in WSL (see scripts/lerobot_writer.py).
"""
from __future__ import annotations

import base64
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .engine import PlatformEnv
from .schema import ProjectConfig


def _jpeg_bytes(frame):
    return base64.b64decode(frame["jpeg_b64"]) if frame else b""


class EpisodeRecorder:
    def __init__(self, env: PlatformEnv):
        self.env = env
        self.joints = [n for n in env._joints]
        self.cameras = [s.id for s in env.config.sensors]
        self.reset()

    def reset(self):
        self.joint_pos, self.joint_vel, self.eef_pos, self.eef_mat = [], [], [], []
        self.gripper, self.action, self.frame_time = [], [], []
        self.object_xyz = []
        self.object_mat = []
        self.task_state = []
        self.capture_time = {c: [] for c in self.cameras}
        self.age = {c: [] for c in self.cameras}
        self.images = {c: [] for c in self.cameras}

    def capture(self):
        obs = self.env.observation()
        st = obs["state"]
        self.joint_pos.append([st["joint_pos"][n] for n in self.joints])
        self.joint_vel.append([st["joint_vel"][n] for n in self.joints])
        self.eef_pos.append(st["eef_position"])
        self.eef_mat.append([v for row in st["eef_matrix"] for v in row])
        self.gripper.append(st["gripper"] if st["gripper"] is not None else 0.0)
        self.object_xyz.append(st["object_pose"]["position"])
        self.object_mat.append(st["object_pose"]["matrix"])
        if self.env._duck_task is not None:
            task = self.env._duck_task
            self.task_state.append([task.speed, task.distance, task.phase, self.env._judge.contacts])
        self.action.append(None)  # paired with the next action applied at this state
        self.frame_time.append(obs["sim_time"])
        for c in self.cameras:
            f = obs["images"][c]
            if f is None:
                raise ValueError(f"Camera {c} has no available frame; wait for latency warm-up or fix rendering before collecting")
            self.capture_time[c].append(f["capture_time"] if f else -1.0)
            self.age[c].append(obs["timing"]["image_age_s"].get(c, -1.0))
            self.images[c].append(_jpeg_bytes(f))

    def set_action(self, action):
        """Attach the action applied at the most recently captured state."""
        self.action[-1] = [float(v) for v in action]

    def arrays(self, success, asset, seed, control_source):
        if any(a is None for a in self.action):
            raise RuntimeError("episode has frames without a paired action")
        return {
            "joint_pos": np.asarray(self.joint_pos, dtype=np.float32),
            "joint_vel": np.asarray(self.joint_vel, dtype=np.float32),
            "eef_pos": np.asarray(self.eef_pos, dtype=np.float32),
            "eef_mat": np.asarray(self.eef_mat, dtype=np.float32),
            "gripper": np.asarray(self.gripper, dtype=np.float32),
            "object_xyz": np.asarray(self.object_xyz, dtype=np.float32),
            "object_mat": np.asarray(self.object_mat, dtype=np.float32),
            **({"belt_speed_m_s": np.asarray(self.task_state, dtype=np.float32)[:, 0],
                "belt_travel_m": np.asarray(self.task_state, dtype=np.float32)[:, 1],
                "task_phase": np.asarray(self.task_state, dtype=np.int32)[:, 2],
                "grasp_contacts": np.asarray(self.task_state, dtype=np.int32)[:, 3]}
               if self.task_state else {}),
            "action": np.asarray(self.action, dtype=np.float32),
            "frame_time": np.asarray(self.frame_time, dtype=np.float32),
            "success": np.asarray([success], dtype=bool),
            "asset": np.asarray([asset]),
            "seed": np.asarray([seed], dtype=np.int64),
            "control_source": np.asarray([control_source]),
            **{f"capture_time_{c}": np.asarray(self.capture_time[c], dtype=np.float32) for c in self.cameras},
            **{f"age_{c}": np.asarray(self.age[c], dtype=np.float32) for c in self.cameras},
            **{f"jpeg_{c}": np.asarray(self.images[c], dtype=object) for c in self.cameras},
        }


def _renderer_version(env):
    if env.config.scene.sensor_renderer == "hybrid":
        from .rendering.hybrid import RENDERER_VERSION
        return f"hybrid:{RENDERER_VERSION}"
    return "mujoco:1"


def _task_asset_provenance(config: ProjectConfig, lab_root: Path) -> dict:
    """Hash + generation source + physics basis of the manipulable task object."""
    ref = config.task.asset
    info = {"ref": ref}
    if ref.startswith("imported/"):
        slug = ref.split("/", 1)[1]
        mp = Path(lab_root) / "assets" / "imported" / slug / "manifest.json"
        if mp.exists():
            m = json.loads(mp.read_text(encoding="utf-8"))
            info.update({
                "manifest_sha256": hashlib.sha256(mp.read_bytes()).hexdigest(),
                "mesh_sha256": m.get("sha256_glb"),
                "source": m.get("source"),
                "generation_id": m.get("generation_id"),
                "source_detail": m.get("source_detail"),
                "radius_m": m.get("radius_m"), "height_m": m.get("height_m"),
                "mass_kg": m.get("mass_kg"),
                "radius_basis": m.get("radius_basis"), "scale_basis": m.get("scale_basis"),
                "collision_proxy": m.get("collision_proxy"),
            })
    else:
        info["builtin"] = True
    if config.task.plugin == "duck_conveyor":
        info["imported_default_collision_proxy"] = info.get("collision_proxy")
        info["collision_proxy"] = {
            "shape": "generated_mesh_convex_hull", "mass_kg": info.get("mass_kg"),
            "basis": "generated mesh outer envelope; mass/friction are uncalibrated",
        }
    return info


def collect(config: ProjectConfig, lab_root: Path, plan: list[dict], out_dir: Path, max_ticks=200,
            expert="servo", provenance=None, preview_dir: Path | None = None, realtime=False):
    """plan: [{'asset': ..., 'seed': ...}]. expert: 'servo' (closed-loop) or 'openloop'."""
    source = "closed_loop_servo_expert" if expert == "servo" else "script_expert"
    act_fn = (lambda e: e.servo_expert_action()) if expert == "servo" else (lambda e: e.expert_action())
    out_dir = Path(out_dir)
    if (out_dir / "manifest.json").exists() or any((out_dir / "episodes").glob("ep*.npz")):
        raise ValueError("Collection output already contains data; choose a new directory")
    (out_dir / "episodes").mkdir(parents=True, exist_ok=True)
    env = PlatformEnv(config, lab_root)
    try:
        rows = []
        def publish_preview(index):
            if preview_dir is None:
                return
            preview_root = Path(preview_dir)
            preview_root.mkdir(parents=True, exist_ok=True)
            obs = env.observation()
            snapshot = {"episode_index": index, "total": len(plan),
                        "tick": obs["tick"], "sim_time": obs["sim_time"],
                        "images": {key: frame["jpeg_b64"] if frame else None
                                   for key, frame in obs["images"].items()},
                        "stage": env.expert_stage(), "state": obs["state"],
                        "completed": list(rows), "metrics": env.metrics()}
            pending = preview_root / "preview.tmp"
            pending.write_text(json.dumps(snapshot), encoding="utf-8")
            try:
                pending.replace(preview_root / "preview.json")
            except PermissionError:
                # Windows may briefly hold the previous snapshot open in the API.
                # A skipped preview must never interrupt recording training data.
                pass

        for i, item in enumerate(plan):
            cfg = config.model_copy(deep=True)
            cfg.task.asset = item["asset"]
            env.config = cfg
            env.reset(item["seed"])
            # Preroll the existing Cartesian expert with an open gripper until
            # configured sensor latency has made the first frames observable.
            warmup = int(np.ceil(max((s.latency_s for s in cfg.sensors), default=0) * env.controller.frequency_hz)) + 2
            for _ in range(warmup):
                if all(f is not None for f in env.observation()["images"].values()):
                    break
                if env._render_error or env.controller.mode != "cartesian_delta_ik":
                    raise ValueError("Cameras unavailable or unsupported expert warm-up mode")
                env.step([0, 0, 0, 1])
            start_tick = env.scheduler.tick
            rec = EpisodeRecorder(env)
            rec.capture()  # initial state S0
            publish_preview(i)
            wall_start, sim_start = time.monotonic(), env.episode_time()
            success = False
            episode_ticks = (max(max_ticks, int(np.ceil(cfg.task.time_limit_s * env.controller.frequency_hz)))
                             if cfg.task.plugin == "duck_conveyor" else max_ticks)
            while env.scheduler.tick < start_tick + episode_ticks:
                action = act_fn(env)
                rec.set_action(action)  # pair S_t with the action taken at S_t
                env.step(action)
                if realtime:
                    delay = env.episode_time() - sim_start - (time.monotonic() - wall_start)
                    if delay > 0:
                        time.sleep(delay)
                publish_preview(i)
                if env.metrics()["success"]:
                    success = True
                    break
                if env.scheduler.tick < start_tick + episode_ticks:
                    rec.capture()  # next state S_{t+1}
            arrays = rec.arrays(success, item["asset"], env.seed, source)
            np.savez_compressed(out_dir / "episodes" / f"ep{i:03d}.npz", **arrays)
            row = {"index": i, "asset": item["asset"], "seed": env.seed, "frames": len(rec.frame_time),
                   "success": bool(success), "duration_s": float(rec.frame_time[-1]),
                   "control_source": source}
            if cfg.task.plugin == "duck_conveyor":
                row["task_metrics"] = env.metrics()
            rows.append(row)
            publish_preview(i)
            print(i, row, flush=True)
        manifest = {
            "schema_version": "1.0",
            "config_snapshot": json.loads(config.model_dump_json()),
            "config_sha256": hashlib.sha256(config.model_dump_json().encode()).hexdigest(),
            "fps": config.controllers[0].frequency_hz,
            "render_mode": env.render_mode,
            "sensor_renderer": config.scene.sensor_renderer,
            "renderer_version": _renderer_version(env),
            "scene_props": env._env_info.get("props", []),
            "robot": config.robots[0].id,
            "joints": [n for n in env._joints],
            "cameras": [s.id for s in config.sensors],
            "task_asset": _task_asset_provenance(config, lab_root),
            "control_source": f"{source} (privileged object pose; not a learned policy)",
            "expert_version": ("duck_conveyor_tracking_v2" if config.task.plugin == "duck_conveyor" else "push_smooth_v1" if config.task.plugin == "tabletop_push" else "servo_smooth_v1") if expert == "servo" else "openloop_v1",
            "episodes": rows,
            "summary": {"episodes": len(rows), "successes": sum(r["success"] for r in rows),
                        "frames": sum(r["frames"] for r in rows)},
        }
        if config.scene.environment:
            from .scene_assets import load_scene
            scene = config.scene.environment
            hybrid = config.scene.sensor_renderer == "hybrid"
            if hybrid and getattr(env, "_hybrid_ply", None):
                from .rendering.hybrid import asset_sha256
                manifest["gaussian_sha256"] = asset_sha256(env._hybrid_ply)
            manifest["scene_asset_manifest"] = load_scene(
                lab_root, scene.asset_id, scene.revision, physics=not hybrid)[1]
        if provenance:
            manifest["generation_provenance"] = provenance
        if config.task.plugin == "duck_conveyor":
            from .duck_demo import asset_info
            has_chassis = any(l.name == "mobile_chassis" for l in env.robot.links)
            manifest["robot_chassis_asset"] = asset_info(Path(lab_root), config.task.duck_conveyor.chassis_asset, "四轮底盘") if has_chassis else None
            manifest["task_physics"] = {
                "object_collision": "convex hull of generated mesh; mass/friction not calibrated",
                "conveyor_contact": "driven rollers, visual slats",
                "base_motion": "parked" if has_chassis else "fixed_arm", "expert_observations": "privileged object pose",
                "object_mat": "world rotation matrix (N,3,3)",
                "task_phase": "0 feed,1 track,2 follow approach,3 follow descend,4 follow grip,5 lift,6 carry,7 lower,8 release,9 retreat,10 failed grasp",
                "continuous_conveyor": config.task.duck_conveyor.continuous,
            }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps(manifest["summary"]))
        return manifest
    finally:
        env.close()
