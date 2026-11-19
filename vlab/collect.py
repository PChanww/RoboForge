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
    return info


def collect(config: ProjectConfig, lab_root: Path, plan: list[dict], out_dir: Path, max_ticks=200,
            expert="servo", provenance=None):
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
        for i, item in enumerate(plan):
            base_cfg = item.get("config") or config
            cfg = base_cfg.model_copy(deep=True)
            cfg.task.asset = item["asset"]
            instruction = (item.get("instruction") or base_cfg.task.instruction or "").strip()
            if instruction:
                cfg.task.instruction = instruction
            env.config = cfg
            env.reset(item["seed"])
            if item.get("object_xy"):
                for tgt_k, xy_val in item["object_xy"].items():
                    env.set_object_xy(tgt_k, xy_val)
            if item.get("active_target"):
                env.set_active_target(item["active_target"])
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
            success = False
            while env.scheduler.tick < start_tick + max_ticks:
                action = act_fn(env)
                rec.set_action(action)  # pair S_t with the action taken at S_t
                env.step(action)
                if env.metrics()["success"]:
                    success = True
                    break
                if env.scheduler.tick < start_tick + max_ticks:
                    rec.capture()  # next state S_{t+1}
            arrays = rec.arrays(success, item["asset"], env.seed, source)
            np.savez_compressed(out_dir / "episodes" / f"ep{i:03d}.npz", **arrays)
            row = {"index": i, "asset": item["asset"], "seed": env.seed, "frames": len(rec.frame_time),
                   "success": bool(success), "duration_s": float(rec.frame_time[-1]),
                   "instruction": instruction,
                   "active_target": item.get("active_target") or "cup",
                   "scene_tag": item.get("scene_tag") or cfg.project_id,
                   "control_source": source}
            rows.append(row)
            print(i, row, flush=True)
        manifest = {
            "schema_version": "1.0",
            "config_snapshot": json.loads(config.model_dump_json()),
            "config_sha256": hashlib.sha256(config.model_dump_json().encode()).hexdigest(),
            "fps": config.controllers[0].frequency_hz,
            "render_mode": env.render_mode,
            "sensor_renderer": config.scene.sensor_renderer,
            "renderer_version": _renderer_version(env),
            "robot": config.robots[0].id,
            "joints": [n for n in env._joints],
            "cameras": [s.id for s in config.sensors],
            "task_asset": _task_asset_provenance(config, lab_root),
            "control_source": f"{source} (privileged object pose; not a learned policy)",
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
        (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps(manifest["summary"]))
        return manifest
    finally:
        env.close()
