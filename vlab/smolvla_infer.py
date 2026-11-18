"""Closed-loop inference runner for fine-tuned SmolVLAPolicy models via persistent worker."""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

from .scene_assets import valid_id


def _to_wsl_path(p: Path) -> str:
    p = p.resolve()
    if p.drive:
        return f"/mnt/{p.drive[0].lower()}" + p.as_posix()[2:]
    return p.as_posix()


class SmolVLAPolicyRunner:
    """Runs 450M SmolVLAPolicy closed-loop with replanning + max_q=2 temporal ensembling."""

    def __init__(self, lab_root: Path, model_id: str = "so100_smolvla"):
        self.lab_root = Path(lab_root).resolve()
        self.model_id = valid_id(model_id)
        self.model_dir = self.lab_root / "datasets" / "models" / self.model_id
        stats_path = self.model_dir / "so100_stats.json"
        if not stats_path.is_file() or not (self.model_dir / "config.json").is_file():
            raise FileNotFoundError(f"SmolVLA model files missing under {self.model_dir}")
        self.stats = json.loads(stats_path.read_text(encoding="utf-8"))
        self.camera_keys = list(self.stats.get("camera_keys") or ["front", "wrist"])
        self.state_joints = list(
            self.stats.get("state_joints")
            or ["waist", "shoulder", "elbow", "wrist_pitch", "wrist_yaw", "left", "right"]
        )
        self.instruction = str(self.stats.get("task_instruction") or "")
        self.replan_every = 2
        self._lock = threading.Lock()
        self._proc = self._start_worker()
        self._chunks: list[tuple[np.ndarray, int]] = []
        self._step_idx = 0

    def _start_worker(self):
        repo_vla = Path(__file__).resolve().parents[1]
        worker_script = repo_vla / "scripts" / "smolvla_worker.py"
        if sys.platform.startswith("win"):
            distro = os.environ.get("VLAB_WSL_DISTRO", "Ubuntu-22.04")
            deps_wsl = _to_wsl_path(repo_vla.parents[1] / "work" / "policy" / "deps")
            script_wsl = _to_wsl_path(worker_script)
            model_wsl = _to_wsl_path(self.model_dir)
            cmd = [
                "wsl", "-d", distro, "--", "bash", "-lc",
                f"PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false "
                f"PYTHONPATH={deps_wsl}:. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "
                f"~/miniconda3/envs/voxcpm/bin/python -u {script_wsl} --model-dir {model_wsl}",
            ]
        else:
            cmd = [sys.executable, "-u", str(worker_script), "--model-dir", str(self.model_dir)]
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        hdr = None
        while proc.stdout is not None:
            line = proc.stdout.readline()
            if not line:
                break
            s = line.strip()
            if s.startswith("{"):
                try:
                    hdr = json.loads(s)
                    break
                except json.JSONDecodeError:
                    continue
        if hdr is None:
            err = proc.stderr.read() if proc.stderr else ""
            proc.kill()
            raise RuntimeError(f"SmolVLA worker failed to start: {err.strip()}")
        if not hdr.get("ready"):
            proc.kill()
            raise RuntimeError(f"Unexpected SmolVLA worker handshake: {hdr}")
        return proc

    def _rpc(self, payload: dict) -> dict:
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                self._proc = self._start_worker()
            assert self._proc.stdin is not None and self._proc.stdout is not None
            self._proc.stdin.write(json.dumps(payload) + "\n")
            self._proc.stdin.flush()
            resp_line = self._proc.stdout.readline()
            if not resp_line:
                err = self._proc.stderr.read() if self._proc.stderr else ""
                raise RuntimeError(f"SmolVLA worker exited unexpectedly: {err.strip()}")
            resp = json.loads(resp_line)
            if "error" in resp:
                raise RuntimeError(resp["error"])
            return resp

    def reset(self) -> None:
        self._chunks.clear()
        self._step_idx = 0
        self._rpc({"cmd": "reset"})

    def _state_vec(self, st: dict) -> list[float]:
        return (
            [float(st["joint_pos"][n]) for n in self.state_joints]
            + [float(st["joint_vel"][n]) for n in self.state_joints]
            + [float(v) for v in st["eef_position"]]
            + [float(st["gripper"]) if st.get("gripper") is not None else 0.0]
        )

    def act(self, observation: dict, instruction: str | None = None) -> list[float]:
        effective_instruction = (instruction or "").strip() or self.instruction
        need_replan = (self._step_idx % self.replan_every == 0) or not self._chunks
        if need_replan:
            images = observation.get("images") or {}
            imgs_b64 = {}
            for cam in self.camera_keys:
                frame = images.get(cam)
                if not frame or not frame.get("jpeg_b64"):
                    raise ValueError(f"Camera {cam!r} frame unavailable for SmolVLA")
                imgs_b64[cam] = frame["jpeg_b64"]
            resp = self._rpc({
                "cmd": "act",
                "images": imgs_b64,
                "state_vec": self._state_vec(observation["state"]),
                "instruction": effective_instruction,
            })
            chunk_arr = np.asarray(resp["chunk"], dtype=np.float32)
            self._chunks.append((chunk_arr, self._step_idx))
            if len(self._chunks) > 2:
                self._chunks = self._chunks[-2:]

        num_xyz = np.zeros(3, dtype=np.float32)
        denom = 0.0
        for ch, start_tick in self._chunks:
            age = self._step_idx - start_tick
            if 0 <= age < len(ch):
                w = math.exp(-0.85 * age)
                num_xyz += w * ch[age, :3]
                denom += w
        newest_ch, newest_start = self._chunks[-1]
        newest_idx = min(len(newest_ch) - 1, max(0, self._step_idx - newest_start))
        grip = float(newest_ch[newest_idx, 3])
        xyz = num_xyz / max(denom, 1e-8)
        self._step_idx += 1
        out = np.clip(np.array([xyz[0], xyz[1], xyz[2], grip], dtype=np.float32), -1.0, 1.0)
        return [float(v) for v in out]

    def close(self) -> None:
        p = getattr(self, "_proc", None)
        if p is not None and p.poll() is None:
            try:
                if p.stdin:
                    p.stdin.write(json.dumps({"cmd": "close"}) + "\n")
                    p.stdin.flush()
                p.wait(timeout=2)
            except Exception:
                p.kill()
        self._proc = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
