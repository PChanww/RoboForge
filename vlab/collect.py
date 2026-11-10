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


def collect(config: ProjectConfig, lab_root: Path, plan: list[dict], out_dir: Path, max_ticks=200):
    """plan: [{'asset': ..., 'seed': ...}]. Writes episodes + manifest."""
    out_dir = Path(out_dir)
    (out_dir / "episodes").mkdir(parents=True, exist_ok=True)
    env = PlatformEnv(config, lab_root)
    try:
        rows = []
        for i, item in enumerate(plan):
            cfg = config.model_copy(deep=True)
            cfg.task.asset = item["asset"]
            env.config = cfg
            env.reset(item["seed"])
            rec = EpisodeRecorder(env)
            rec.capture()  # initial state S0
            success = False
            while env.scheduler.tick < max_ticks:
                action = env.expert_action()
                rec.set_action(action)  # pair S_t with the action taken at S_t
                env.step(action)
                if env.metrics()["success"]:
                    success = True
                    break
                if env.scheduler.tick < max_ticks:
                    rec.capture()  # next state S_{t+1}
            arrays = rec.arrays(success, item["asset"], env.seed, "script_expert")
            np.savez_compressed(out_dir / "episodes" / f"ep{i:03d}.npz", **arrays)
            row = {"index": i, "asset": item["asset"], "seed": env.seed, "frames": len(rec.frame_time),
                   "success": bool(success), "duration_s": float(rec.frame_time[-1]),
                   "control_source": "script_expert"}
            rows.append(row)
            print(i, row, flush=True)
        manifest = {
            "schema_version": "1.0",
            "config_snapshot": json.loads(config.model_dump_json()),
            "config_sha256": hashlib.sha256(config.model_dump_json().encode()).hexdigest(),
            "fps": config.controllers[0].frequency_hz,
            "robot": config.robots[0].id,
            "joints": [n for n in env._joints],
            "cameras": [s.id for s in config.sensors],
            "control_source": "script_expert (privileged initial pose; not a learned policy)",
            "episodes": rows,
            "summary": {"episodes": len(rows), "successes": sum(r["success"] for r in rows),
                        "frames": sum(r["frames"] for r in rows)},
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps(manifest["summary"]))
        return manifest
    finally:
        env.close()