"""VisualBC policy adapter: preprocessing, state order and action post-processing.

Loads either a raw checkpoint (`from_checkpoint`) or a self-contained exported package
(`from_package`, verifies manifest SHA256). Runtime imports are torch/numpy/pillow only.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from visual_bc_model import VisualBC, fallback_contract


class VisualBCAdapter:
    def __init__(self, lab_root=None, model_id="visual_hybrid", device="cpu",
                 *, checkpoint=None, contract=None):
        if checkpoint is None:
            if lab_root is None:
                raise ValueError("provide checkpoint=<path or dict> or lab_root+model_id")
            path = Path(lab_root) / "datasets" / "models" / f"{model_id}.pt"
            checkpoint = torch.load(path, weights_only=False)
        self.ck = checkpoint
        self.device = device
        in_ch = int(self.ck["state_dict"]["conv.0.weight"].shape[1])
        self.model = VisualBC(in_channels=in_ch)
        self.model.load_state_dict(self.ck["state_dict"])
        self.model.eval()
        self.size = int(self.ck.get("img_size", 64))
        self.contract_data = contract or self.ck.get("contract") or fallback_contract(self.size)
        self.image_key = self.contract_data.get("image_key", "front")
        self.camera_keys = self.ck.get("camera_keys") or self.contract_data.get("camera_keys") or [self.image_key]
        self.prev = np.zeros(len(self.contract_data.get("action_names", [0, 0, 0, 0])), dtype=np.float32)
        self.manifest = None

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        ck = torch.load(Path(path), weights_only=False)
        return cls(checkpoint=ck, device=device)

    @classmethod
    def from_package(cls, package_dir, device="cpu"):
        d = Path(package_dir).resolve()
        manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        name = manifest.get("weights") or "model.pt"
        wp = (d / name).resolve()
        if d not in wp.parents:
            raise ValueError("package weights path escapes the package directory")
        expected = manifest.get("weights_sha256")
        if not expected:
            raise ValueError("package manifest has no weights_sha256")
        actual = hashlib.sha256(wp.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"package weights sha256 mismatch: {actual} != {expected}")
        ad = cls.from_checkpoint(wp, device)
        ad.manifest = manifest
        ad.contract_data = manifest.get("contract") or ad.contract_data
        ad.image_key = ad.contract_data.get("image_key", "front")
        return ad

    def contract(self):
        return dict(self.contract_data)

    def reset(self):
        self.prev = np.zeros_like(self.prev)

    def _image(self, obs):
        chans = []
        for cam in self.camera_keys:
            f = obs["images"].get(cam)
            if not f:
                raise ValueError(f"observation has no {cam} image (renderer unavailable)")
            img = np.asarray(Image.open(io.BytesIO(base64.b64decode(f["jpeg_b64"]))).convert("RGB"),
                             dtype=np.float32) / 255.0
            t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            t = torch.nn.functional.interpolate(t, size=(self.size, self.size), mode="bilinear", align_corners=False)
            chans.append(t)
        return torch.cat(chans, dim=1)

    def _state(self, st):
        return np.array(st["state_vec"], dtype=np.float32) if "state_vec" in st else self._state_from_names(st)

    def _state_from_names(self, st):
        names = self.contract_data.get("state_joint_names") or []
        joints = [n for n in names if not n.endswith("_vel")] or list(st["joint_pos"])
        vals = [st["joint_pos"][n] for n in joints] + [st["joint_vel"][n] for n in joints]
        vals += list(st["eef_position"]) + [st["gripper"] if st["gripper"] is not None else 0.0]
        return np.asarray(vals, dtype=np.float32)

    def act(self, observation):
        st = (torch.tensor(self._state(observation["state"])) - self.ck["state_mean"]) / self.ck["state_std"]
        pn = (torch.tensor(self.prev) - self.ck["prev_mean"]) / self.ck["prev_std"]
        with torch.no_grad():
            a = self.model(self._image(observation), st.unsqueeze(0), pn.unsqueeze(0))[0].numpy()
        a = np.clip(a, -1, 1).astype(np.float32)
        self.prev = a
        return a.tolist()
