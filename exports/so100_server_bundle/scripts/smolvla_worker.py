"""Persistent stdin/stdout JSONL inference worker for fine-tuned SmolVLAPolicy checkpoints."""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def _select_device(requested: str) -> torch.device:
    req = (requested or "auto").strip().lower()
    if req != "auto":
        return torch.device(req)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu")
    return torch.device("cpu")


def _find_processor(model_dir: Path, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    for base in (model_dir, model_dir.parent, model_dir.parents[2] if len(model_dir.parents) >= 3 else model_dir.parent):
        for rel in ("work/smolvlm-processor", "base_weights/smolvlm-processor"):
            cand = base / rel
            if cand.is_dir():
                return cand
    root_cand = Path(__file__).resolve().parents[3] / "work" / "smolvlm-processor"
    if root_cand.is_dir():
        return root_cand
    raise FileNotFoundError("Could not locate smolvlm-processor directory")


def _decode_rgb_tensor(b64: str) -> torch.Tensor:
    raw = base64.b64decode(b64, validate=True)
    with Image.open(io.BytesIO(raw)) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    if t.shape[2] != 256 or t.shape[3] != 256:
        t = F.interpolate(t, size=(256, 256), mode="bilinear", align_corners=False)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--processor", default=None)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    model_dir = Path(args.model_dir).resolve()
    stats = json.loads((model_dir / "so100_stats.json").read_text(encoding="utf-8"))
    proc_dir = _find_processor(model_dir, args.processor)
    dev = _select_device(args.device)

    cfg = PreTrainedConfig.from_pretrained(model_dir, local_files_only=True)
    cfg.vlm_model_name = str(proc_dir)
    cfg.load_vlm_weights = False
    cfg.device = "cpu"
    policy = SmolVLAPolicy.from_pretrained(model_dir, config=cfg, strict=True)
    policy.to(dev)
    policy.eval()

    tok = policy.model.vlm_with_expert.processor.tokenizer
    tok.padding_side = "right"

    state_mean = torch.tensor(stats["state_mean"], dtype=torch.float32, device=dev)
    state_std = torch.tensor(stats["state_std"], dtype=torch.float32, device=dev)
    act_mean = torch.tensor(stats["action_mean"], dtype=torch.float32, device=dev)
    act_std = torch.tensor(stats["action_std"], dtype=torch.float32, device=dev)
    camera_keys = list(stats.get("camera_keys") or ["front", "wrist"])
    default_instr = str(stats.get("task_instruction") or "Pick up the object and place it inside the tray.")
    chunk_size = int(stats.get("chunk_size") or cfg.chunk_size)
    max_act_dim = int(cfg.max_action_dim)
    use_bf16 = dev.type in ("cuda", "npu")
    zero_noise = torch.zeros((1, chunk_size, max_act_dim), dtype=torch.float32, device=dev)

    token_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    print(json.dumps({"ready": True, "device": str(dev), "model_dir": str(model_dir)}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        cmd = req.get("cmd")
        if cmd == "reset":
            policy.reset()
            print(json.dumps({"ok": True}), flush=True)
            continue
        if cmd == "close":
            print(json.dumps({"closed": True}), flush=True)
            break
        if cmd == "act":
            instr = str(req.get("instruction") or default_instr).rstrip() + "\n"
            if instr not in token_cache:
                enc = tok([instr], padding="max_length", max_length=cfg.tokenizer_max_length,
                          truncation=True, return_tensors="pt")
                token_cache[instr] = (enc.input_ids.to(dev), enc.attention_mask.bool().to(dev))
            ids, mask = token_cache[instr]
            st_raw = torch.tensor(req["state_vec"], dtype=torch.float32, device=dev)
            sn = ((st_raw - state_mean) / state_std).unsqueeze(0)
            imgs_map = req.get("images") or {}
            batch = {
                "observation.state": sn,
                "observation.language.tokens": ids,
                "observation.language.attention_mask": mask,
            }
            for cam in camera_keys:
                batch[f"observation.images.{cam}"] = _decode_rgb_tensor(imgs_map[cam]).to(dev)
            with torch.inference_mode():
                with torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_bf16):
                    pred_norm = policy.predict_action_chunk(batch, noise=zero_noise)
                pred = torch.clamp(pred_norm[0].float() * act_std + act_mean, -1.0, 1.0)
            print(json.dumps({"chunk": pred.cpu().tolist()}), flush=True)
            continue
        print(json.dumps({"error": f"unknown cmd {cmd!r}"}), flush=True)


if __name__ == "__main__":
    main()
