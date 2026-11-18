"""Pure-NumPy + Pillow runtime inference for VisualBC checkpoints (.npz).

Allows closed-loop single- or multi-camera (e.g. front + wrist) visual policies trained
in WSL PyTorch to execute directly inside the Windows MuJoCo simulation service.
"""
from __future__ import annotations

import base64
import io
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from .scene_assets import valid_id


def _convert_pt_to_npz(pt_path: Path, npz_path: Path) -> None:
    """Convert a PyTorch VisualBC checkpoint (.pt) to pure-NumPy (.npz) weights."""
    try:
        import torch
        ck = torch.load(pt_path, map_location="cpu", weights_only=False)
        sd = ck["state_dict"]
        payload = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in sd.items()}
        payload.update({
            "state_mean": ck["state_mean"].detach().cpu().numpy().astype(np.float32),
            "state_std": ck["state_std"].detach().cpu().numpy().astype(np.float32),
            "prev_mean": ck["prev_mean"].detach().cpu().numpy().astype(np.float32),
            "prev_std": ck["prev_std"].detach().cpu().numpy().astype(np.float32),
            "img_size": np.array(int(ck.get("img_size", 64)), dtype=np.int32),
            "chunk_size": np.array(int(ck.get("chunk_size", 1)), dtype=np.int32),
            "camera_keys": np.array(json.dumps(ck.get("camera_keys", ["front"]))),
            "state_joints": np.array(json.dumps(
                (ck.get("contract") or {}).get("state_joint_names")
                or ["waist", "shoulder", "elbow", "wrist_pitch", "wrist_yaw", "left", "right"]
            )),
        })
        np.savez(npz_path, **payload)
        return
    except ImportError:
        pass
    # Fallback on Windows when torch is installed in WSL conda environment
    py_code = (
        "import json, torch, numpy as np; from pathlib import Path; "
        f"pt = Path({str(pt_path)!r}); npz = Path({str(npz_path)!r}); "
        "ck = torch.load(pt, map_location='cpu', weights_only=False); sd = ck['state_dict']; "
        "p = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in sd.items()}; "
        "p.update({'state_mean': ck['state_mean'].detach().cpu().numpy().astype(np.float32), "
        "'state_std': ck['state_std'].detach().cpu().numpy().astype(np.float32), "
        "'prev_mean': ck['prev_mean'].detach().cpu().numpy().astype(np.float32), "
        "'prev_std': ck['prev_std'].detach().cpu().numpy().astype(np.float32), "
        "'img_size': np.array(int(ck.get('img_size', 64)), dtype=np.int32), "
        "'chunk_size': np.array(int(ck.get('chunk_size', 1)), dtype=np.int32), "
        "'camera_keys': np.array(json.dumps(ck.get('camera_keys', ['front']))), "
        "'state_joints': np.array(json.dumps((ck.get('contract') or {}).get('state_joint_names') or "
        "['waist', 'shoulder', 'elbow', 'wrist_pitch', 'wrist_yaw', 'left', 'right']))}); "
        "np.savez(npz, **p)"
    )
    if pt_path.drive:
        drive_letter = pt_path.drive[0].lower()
        pt_wsl = f"/mnt/{drive_letter}" + pt_path.as_posix()[2:]
        npz_wsl = f"/mnt/{drive_letter}" + npz_path.as_posix()[2:]
        py_code = py_code.replace(str(pt_path), pt_wsl).replace(str(npz_path), npz_wsl)
    subprocess.run(
        ["wsl", "-d", "Ubuntu-22.04", "--", "bash", "-lc", f"~/miniconda3/envs/voxcpm/bin/python -c {json.dumps(py_code)}"],
        check=True,
    )


def _bilinear_resize_chw(chw: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Exact bilinear resize matching PyTorch interpolate(..., mode='bilinear', align_corners=False)."""
    _, in_h, in_w = chw.shape
    if in_h == out_h and in_w == out_w:
        return chw.astype(np.float32, copy=False)
    sy = (np.arange(out_h, dtype=np.float32) + 0.5) * (in_h / out_h) - 0.5
    sx = (np.arange(out_w, dtype=np.float32) + 0.5) * (in_w / out_w) - 0.5
    sy = np.clip(sy, 0.0, in_h - 1.0)
    sx = np.clip(sx, 0.0, in_w - 1.0)
    y0 = np.floor(sy).astype(np.int64)
    x0 = np.floor(sx).astype(np.int64)
    y1 = np.minimum(y0 + 1, in_h - 1)
    x1 = np.minimum(x0 + 1, in_w - 1)
    wy = (sy - y0)[:, None]
    wx = (sx - x0)[None, :]
    top = chw[:, y0][:, :, x0] * (1.0 - wx) + chw[:, y0][:, :, x1] * wx
    bot = chw[:, y1][:, :, x0] * (1.0 - wx) + chw[:, y1][:, :, x1] * wx
    return (top * (1.0 - wy) + bot * wy).astype(np.float32)


def _conv2d_s2_relu(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    k = int(w.shape[2])
    win = np.lib.stride_tricks.sliding_window_view(x, (k, k), axis=(1, 2))[:, ::2, ::2, :, :]
    out = np.einsum("ocpq,cyxpq->oyx", w, win, optimize=True) + b[:, None, None]
    return np.maximum(0.0, out)


def decode_camera_chw(jpeg_b64: str, img_size: int = 64) -> np.ndarray:
    data = base64.b64decode(jpeg_b64, validate=True)
    with Image.open(io.BytesIO(data)) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    chw = np.transpose(rgb, (2, 0, 1))
    return _bilinear_resize_chw(chw, img_size, img_size)


class VisualPolicyRunner:
    def __init__(self, lab_root: Path, model_id: str = "hackathon_vla_bc"):
        mid = valid_id(model_id)
        models_dir = Path(lab_root) / "datasets" / "models"
        npz_path = models_dir / f"{mid}.npz"
        pt_path = models_dir / f"{mid}.pt"
        if not npz_path.is_file():
            if pt_path.is_file():
                _convert_pt_to_npz(pt_path, npz_path)
            else:
                raise FileNotFoundError(f"Visual policy weights not found: {npz_path}")
        d = np.load(npz_path, allow_pickle=False)
        self.model_id = mid
        self.img_size = int(d["img_size"].item()) if "img_size" in d else 64
        self.camera_keys = json.loads(str(d["camera_keys"].item())) if "camera_keys" in d else ["front"]
        self.state_joints = json.loads(str(d["state_joints"].item())) if "state_joints" in d else [
            "waist", "shoulder", "elbow", "wrist_pitch", "wrist_yaw", "left", "right"
        ]
        self.state_mean = d["state_mean"].astype(np.float32)
        self.state_std = d["state_std"].astype(np.float32)
        self.prev_mean = d["prev_mean"].astype(np.float32)
        self.prev_std = d["prev_std"].astype(np.float32)
        self.w = {k: d[k].astype(np.float32) for k in (
            "conv.0.weight", "conv.0.bias",
            "conv.2.weight", "conv.2.bias",
            "conv.4.weight", "conv.4.bias",
            "img_fc.weight", "img_fc.bias",
            "state_fc.weight", "state_fc.bias",
            "head.0.weight", "head.0.bias",
            "head.2.weight", "head.2.bias",
        )}
        self.chunk_size = int(d["chunk_size"].item()) if "chunk_size" in d else max(1, int(self.w["head.2.bias"].shape[0]) // 4)
        self.prev = np.zeros(len(self.prev_mean), dtype=np.float32)
        self._chunk_queue: list[np.ndarray] = []

    def reset(self) -> None:
        self.prev = np.zeros(len(self.prev_mean), dtype=np.float32)
        self._chunk_queue.clear()

    def _state_vec(self, st: dict) -> np.ndarray:
        vals = (
            [float(st["joint_pos"][n]) for n in self.state_joints]
            + [float(st["joint_vel"][n]) for n in self.state_joints]
            + [float(v) for v in st["eef_position"]]
            + [float(st["gripper"]) if st.get("gripper") is not None else 0.0]
        )
        return np.asarray(vals, dtype=np.float32)

    def forward(self, img_chw: np.ndarray, state_vec: np.ndarray, prev_act: np.ndarray | None = None) -> np.ndarray:
        prev_in = self.prev if prev_act is None else np.asarray(prev_act, dtype=np.float32)
        x = _conv2d_s2_relu(img_chw.astype(np.float32), self.w["conv.0.weight"], self.w["conv.0.bias"])
        x = _conv2d_s2_relu(x, self.w["conv.2.weight"], self.w["conv.2.bias"])
        x = _conv2d_s2_relu(x, self.w["conv.4.weight"], self.w["conv.4.bias"])
        flat = np.ascontiguousarray(x).reshape(-1)
        img_feat = np.maximum(0.0, self.w["img_fc.weight"] @ flat + self.w["img_fc.bias"])
        sn = (state_vec.astype(np.float32) - self.state_mean) / self.state_std
        pn = (prev_in - self.prev_mean) / self.prev_std
        st_in = np.concatenate([sn, pn], axis=0)
        st_feat = np.maximum(0.0, self.w["state_fc.weight"] @ st_in + self.w["state_fc.bias"])
        h = np.maximum(0.0, self.w["head.0.weight"] @ np.concatenate([img_feat, st_feat], axis=0) + self.w["head.0.bias"])
        out = np.tanh(self.w["head.2.weight"] @ h + self.w["head.2.bias"])
        out = np.clip(out, -1.0, 1.0).astype(np.float32)
        if self.chunk_size > 1:
            return out.reshape(self.chunk_size, 4)
        return out

    def act(self, observation: dict, instruction: str | None = None) -> list[float]:
        # Fixed-action (image BC) policy: language input is accepted for a uniform
        # contract with VLA runners but intentionally ignored.
        images = observation.get("images") or {}
        chans = []
        for cam in self.camera_keys:
            frame = images.get(cam)
            if not frame or not frame.get("jpeg_b64"):
                raise ValueError(f"Camera {cam!r} frame unavailable for visual policy")
            chans.append(decode_camera_chw(frame["jpeg_b64"], self.img_size))
        img_chw = np.concatenate(chans, axis=0)
        st_vec = self._state_vec(observation["state"])
        pred = self.forward(img_chw, st_vec)
        if self.chunk_size == 1:
            self.prev = pred
            return [float(v) for v in pred]
        self._chunk_queue.append(pred)
        max_q = min(3, self.chunk_size)
        if len(self._chunk_queue) > max_q:
            self._chunk_queue = self._chunk_queue[-max_q:]
        m = len(self._chunk_queue)
        num_xyz = np.zeros(3, dtype=np.float32)
        denom = 0.0
        for i, ch in enumerate(self._chunk_queue):
            age = (m - 1) - i
            w = math.exp(-0.85 * age)
            num_xyz += w * ch[age, :3]
            denom += w
        xyz = num_xyz / max(denom, 1e-8)
        grip = float(self._chunk_queue[-1][0, 3])
        act_4d = np.clip(np.array([xyz[0], xyz[1], xyz[2], grip], dtype=np.float32), -1.0, 1.0)
        self.prev = act_4d
        return [float(v) for v in act_4d]
