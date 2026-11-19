"""Shared model, preprocessing and checkpoint inference for mobile transport."""
import base64
from contextlib import nullcontext
from dataclasses import dataclass
import io
import time

import numpy as np
from PIL import Image
from safetensors.torch import load_file
import torch

from scripts.train_bc import MLP
from scripts.visual_bc_model import VisualBC
from vlab.transport_training import MODELS, TASKS, checkpoint_dir, file_sha256, read_json


@dataclass
class MobileCheckpoint:
    policy: torch.nn.Module
    mean: torch.Tensor
    std: torch.Tensor
    identity: dict


def features(state, goal, task, sim_time, belt_speed, speed_scale, privileged=None):
    state = np.atleast_2d(np.asarray(state, dtype=np.float32))
    n = len(state)
    relative_goal = np.asarray(goal, dtype=np.float32)[None, :] - state[:, 7:9]
    task_code = np.broadcast_to(np.eye(len(TASKS), dtype=np.float32)[TASKS.index(task)], (n, len(TASKS)))
    context = [np.broadcast_to(np.asarray(value, dtype=np.float32), (n,)) for value in
               (sim_time, belt_speed, speed_scale)]
    parts = [state, relative_goal, task_code, np.stack(context, axis=1)]
    if privileged is not None:
        parts.append(np.asarray(privileged, dtype=np.float32).reshape(n, 13))
    result = np.concatenate(parts, axis=1)
    if state.shape[1] != 28 or not np.isfinite(result).all():
        raise ValueError("Invalid mobile observation")
    return result


def resize_camera(image, model):
    size = (64, 64) if model == "visual_bc" else (160, 120) if image.width != image.height else (160, 160)
    return np.asarray(image.resize(size, Image.Resampling.BILINEAR).convert("RGB"), dtype=np.uint8)


def camera_tensor(frames, device):
    return torch.as_tensor(frames, device=device).permute(0, 3, 1, 2).float() / 255


@torch.no_grad()
def smolvla_features(runtime, cameras, instructions):
    policy = runtime.model
    config_keys = list(policy.config.image_features)
    batch = {key: camera_tensor(frames, runtime.device) for key, frames in zip(config_keys, cameras)}
    images, _ = policy.prepare_images(batch)
    encoder = policy.model.vlm_with_expert
    tokens = runtime.tokenizer(instructions, padding="max_length", max_length=48, truncation=True, return_tensors="pt")
    mask = tokens.attention_mask.to(runtime.device).unsqueeze(-1)
    autocast = torch.autocast("cuda", dtype=runtime.amp_dtype) if runtime.amp_dtype is not None else nullcontext()
    with autocast:
        visual = [encoder.embed_image(image).float().mean(dim=1) for image in images[:2]]
        text = encoder.embed_language_tokens(tokens.input_ids.to(runtime.device)).float()
        pooled_text = (text * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    return torch.cat([*visual, pooled_text], dim=1)


def instruction_for(task, text):
    descriptions = {
        "deliver": "Pick from the conveyor and put in the box.",
        "pick_return": "Pick up and return to the original spot on the conveyor.",
        "bin_retrieve": "Put the object in the box, then retrieve and hold it.",
        "bin_return": "Put the object in the box, retrieve it and return it to the conveyor.",
    }
    return descriptions[task] + " " + text.strip()


def make_model(kind, size):
    if kind == "visual_bc":
        return VisualBC(state_dim=size, prev_dim=0, action_dim=7, image_channels=6)
    return MLP(n_in=size, n_out=7)


def forward_model(kind, model, state, cameras=None):
    return model(cameras, state, state[:, :0]) if kind == "visual_bc" else model(state)


def action_targets(actions, speed_scale):
    result = np.asarray(actions, dtype=np.float32).copy()
    result[:, 4] /= .065 * speed_scale
    result[:, 5] /= .8
    result[:, 6] = 2 * result[:, 6] - 1
    return result


class MobilePolicies:
    def __init__(self, lab, runtime):
        self.lab = lab
        self.runtime = runtime
        self.cached = {}

    def health(self, model_id):
        with self.runtime.lock:
            checkpoint = self.load(model_id)
            return {"ready": True, "state": "ready", "identity": checkpoint.identity}

    def load(self, model_id):
        root = checkpoint_dir(self.lab, model_id)
        report = read_json(root / "report.json")
        if report["id"] != model_id or report["model"] not in MODELS or report["action_schema"] != "mobile7_v1":
            raise ValueError("Unknown mobile model checkpoint")
        weights = root / "weights.safetensors"
        signature = (model_id, weights.stat().st_mtime_ns, report["weights_sha256"])
        cached = self.cached.get(model_id)
        if cached and cached[0] == signature:
            return cached[1]
        runtime = self.runtime
        if not runtime.ready:
            raise ValueError("Local GPU policy server is not ready")
        kind = report["model"]
        if kind == "smolvla" and report["base_weights_sha256"] != runtime.identity["weights_sha256"]:
            raise ValueError("Mobile head and SmolVLA backbone checksums differ")
        if file_sha256(weights) != report["weights_sha256"]:
            raise ValueError("Mobile weights checksum mismatch")
        tensors = load_file(str(weights), device=runtime.device)
        mean, std = tensors.pop("mean"), tensors.pop("std")
        policy = make_model(kind, len(mean)).to(runtime.device).eval()
        policy.load_state_dict(tensors, strict=True)
        identity = {**{k: runtime.identity[k] for k in ("device", "device_name", "cuda_version")},
                    "label": MODELS[kind], "kind": "vla" if kind == "smolvla" else "learned_bc",
                    "learned_policy": True, "training_model": kind, "training_job_id": model_id,
                    "action_schema": "mobile7_v1", "action_dim": 7, "mobile_base_policy": True,
                    "raw_action_schema": "mobile7_v1_normalized", "adapter": "transport_learned_mobile7_v1",
                    "privileged_observations": kind == "privileged_bc", "weights_sha256": report["weights_sha256"],
                    "base_weights_sha256": report.get("base_weights_sha256")}
        checkpoint = MobileCheckpoint(policy, mean, std, identity)
        if len(self.cached) >= 8:
            self.cached.pop(next(iter(self.cached)))
        self.cached[model_id] = (signature, checkpoint)
        return checkpoint

    @torch.inference_mode()
    def predict(self, model_id, packet):
        with self.runtime.lock:
            start = time.perf_counter()
            checkpoint = self.load(model_id)
            kind = checkpoint.identity["training_model"]
            if packet.get("action_schema") != "mobile7_v1" or not 1 <= int(packet["max_steps"]) <= 10:
                raise ValueError("Invalid mobile action request")
            obs = packet["observation"]
            spec = packet["mobile_context"]
            task = packet["task_kind"]
            speed = float(spec["speed_scale"])
            if not 1 <= speed <= 2:
                raise ValueError("Invalid transport speed")
            state = features(obs["state"], spec["goal"], task, spec["time"], spec["belt_speed"], speed,
                             obs["privileged_state"] if kind == "privileged_bc" else None)
            visual = None
            if kind in ("smolvla", "visual_bc"):
                frames = []
                for key in ("front", "wrist"):
                    payload = base64.b64decode(obs["images"][key].split(",", 1)[-1], validate=True)
                    with Image.open(io.BytesIO(payload)) as image:
                        if image.width * image.height > 1920 * 1080:
                            raise ValueError("Camera exceeds protocol resolution")
                        frames.append(resize_camera(image, kind)[None].copy())
                if kind == "visual_bc":
                    visual = torch.cat([camera_tensor(frame, self.runtime.device) for frame in frames], dim=1)
                else:
                    language = instruction_for(task, packet["instruction"])
                    embedding = smolvla_features(self.runtime, frames, [language]).cpu().numpy()
                    state = np.concatenate([state, embedding], axis=1)
            x = torch.as_tensor(state, device=self.runtime.device)
            raw = forward_model(kind, checkpoint.policy, ((x-checkpoint.mean)/checkpoint.std).clamp(-12, 12),
                                visual).cpu().numpy()
            if raw.shape != (1, 7) or not np.isfinite(raw).all():
                raise ValueError("The mobile model did not produce a finite 7D action")
            action = np.clip(raw.copy(), -1., 1.)
            action[:, 4] *= .065 * speed
            action[:, 5] *= .8
            action[:, 6] = raw[:, 6] >= 0
            return {"request_id": packet["request_id"], "session_id": packet["session_id"], "tick": packet["tick"],
                    "action_schema": "mobile7_v1", "weights_sha256": checkpoint.identity["weights_sha256"],
                    "raw_actions": raw.tolist(), "actions": action.tolist(), "device": self.runtime.device,
                    "model_inference_s": time.perf_counter() - start}
