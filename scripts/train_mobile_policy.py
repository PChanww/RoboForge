"""Train a GPU policy for the 20 Hz mobile transport action contract."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from safetensors.torch import save_file
import torch
from torch.nn import functional as F

from integrations.transport_mobile_models import (action_targets, camera_tensor, features, forward_model,
                                                   instruction_for, make_model, resize_camera, smolvla_features)
from integrations.transport_smolvla_server import Runtime
from vlab.transport_training import MODELS, STATE_FIELDS, checkpoint_dir, datasets, file_sha256, split_episodes


def load_episode(lab, episode, kind, runtime):
    path = lab / "datasets/transport" / episode["id"] / "episodes" / f"ep{episode['index']:03d}.npz"
    n = episode["frames"]
    stride = max(1, (n + 2047) // 2048, 6 if kind == "smolvla" else 1)
    indices = np.arange(0, n, stride)
    with np.load(path, allow_pickle=False) as row:
        times, actions = row["time"], row["action"]
        if (times.shape != (n,) or not np.isfinite(times).all()
                or not np.allclose(np.diff(times), .05, rtol=0, atol=1e-5)
                or actions.shape != (n, 7) or not np.isfinite(actions).all()
                or not np.isin(actions[:, 6], [0, 1]).all()):
            raise ValueError(f"Invalid 20 Hz timestamps or 7D actions in {path}")
        targets = action_targets(actions, episode["speed_scale"])
        if np.abs(targets).max() > 1.00001:
            raise ValueError(f"Recorded action exceeds contract in {path}")
        state = np.concatenate([row[key][indices] for key in STATE_FIELDS], axis=1)
        privileged = (np.concatenate([row[key][indices] for key in ("object_pose", "object_velocity")], axis=1)
                      if kind == "privileged_bc" else None)
        x = features(state, episode["goal"], episode["task"], times[indices],
                     row["belt_speed_m_s"][indices], episode["speed_scale"], privileged)
        cameras = []
        if kind in ("visual_bc", "smolvla"):
            for key in ("image_front", "image_wrist"):
                rgb = row[key]
                if rgb.ndim != 4 or rgb.shape[0] != n or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
                    raise ValueError(f"Invalid RGB field {key} in {path}")
                cameras.append(np.stack([resize_camera(Image.fromarray(rgb[i]), kind) for i in indices]))
                del rgb
    if kind == "smolvla":
        language = instruction_for(episode["task"], episode["instruction"])
        encodings = []
        for start in range(0, len(indices), 8):
            batch = [camera[start:start+8] for camera in cameras]
            encodings.append(smolvla_features(runtime, batch, [language] * len(batch[0])).cpu().numpy())
        x = np.concatenate([x, np.concatenate(encodings)], axis=1)
    visual = np.concatenate(cameras, axis=-1) if kind == "visual_bc" else None
    return x, targets[indices].clip(-1., 1.), visual


def load_split(lab, episodes, kind, runtime, split):
    loaded = []
    for episode in episodes:
        item = load_episode(lab, episode, kind, runtime)
        loaded.append(item)
        print(f"{split}: {episode['id']}/{episode['index']} ({episode['task']}, seed={episode['seed']}) "
              f"loaded {len(item[0])}/{episode['frames']} frames", flush=True)
    x, y, views = zip(*loaded)
    return (torch.from_numpy(np.concatenate(x)), torch.from_numpy(np.concatenate(y)),
            torch.from_numpy(np.concatenate(views)) if kind == "visual_bc" else None)


def batches(data, *, shuffle):
    x, y, images = data
    indices = torch.randperm(len(x)) if shuffle else torch.arange(len(x))
    for rows in indices.split(128):
        yield (x[rows].cuda(), y[rows].cuda(),
               camera_tensor(images[rows].numpy(), "cuda") if images is not None else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--datasets", required=True, nargs="+")
    parser.add_argument("--epochs", required=True, type=int, choices=range(1, 101), metavar="1..100")
    args = parser.parse_args()
    lab = Path(__file__).resolve().parents[1]
    if not torch.cuda.is_available():
        sys.exit("CUDA is unavailable in the dedicated GPU environment.")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("high")
    catalog = datasets(lab, args.datasets)
    if {row["id"] for row in catalog} != set(args.datasets):
        sys.exit("One or more recordings are unavailable or violate the mobile expert data contract.")
    train_episodes, val_episodes = split_episodes(catalog)
    print(f"CUDA: {torch.cuda.get_device_name(0)} | {args.model} | "
          f"{len(train_episodes)} training episodes, {len(val_episodes)} validation episodes", flush=True)
    runtime = None
    if args.model == "smolvla":
        runtime = Runtime(lab / "work/smolvla-libero", lab / "work/smolvlm-processor", "cuda")
        runtime.load()
        if not runtime.ready:
            sys.exit(runtime.error)
    train = load_split(lab, train_episodes, args.model, runtime, "train")
    val = load_split(lab, val_episodes, args.model, runtime, "validation")
    base_sha = runtime.identity["weights_sha256"] if runtime else None
    del runtime
    torch.cuda.empty_cache()
    mean, std = train[0].mean(dim=0), train[0].std(dim=0, unbiased=False).clamp(min=.01)
    train = (((train[0]-mean)/std).clamp(-12, 12), *train[1:])
    val = (((val[0]-mean)/std).clamp(-12, 12), *val[1:])
    model = make_model(args.model, len(mean)).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    history, best, best_loss = [], None, float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.
        for x, y, images in batches(train, shuffle=True):
            prediction = forward_model(args.model, model, x, images)
            loss = F.mse_loss(prediction, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item()*len(x)
        model.eval()
        with torch.inference_mode():
            error, holds = [], []
            for x, y, images in batches(val, shuffle=False):
                predicted = forward_model(args.model, model, x, images)
                error.append((predicted-y).cpu())
                holds.append(((predicted[:, 6] >= 0) == (y[:, 6] >= 0)).float().cpu())
        errors = torch.cat(error)
        val_loss = float(errors.square().mean())
        if not np.isfinite(val_loss):
            raise ValueError("Model produced non-finite validation errors.")
        metrics = {"epoch": epoch, "train_mse": total / len(train[0]), "val_mse": val_loss,
                   "val_normalized_mae": errors.abs().mean(dim=0).tolist(), "val_hold_accuracy": float(torch.cat(holds).mean())}
        history.append(metrics)
        print(json.dumps(metrics), flush=True)
        if val_loss < best_loss:
            best_loss = val_loss
            best = (deepcopy(metrics), {key: value.detach().cpu().clone() for key, value in model.state_dict().items()})
    result, weights = best
    output = checkpoint_dir(lab, args.job_id)
    output.mkdir(parents=True, exist_ok=False)
    path = output / "weights.safetensors"
    save_file({**weights, "mean": mean, "std": std}, str(path))
    report = {"id": args.job_id, "model": args.model, "created": datetime.now(timezone.utc).isoformat(),
              "device": torch.cuda.get_device_name(0), "action_schema": "mobile7_v1",
              "weights_sha256": file_sha256(path), "base_weights_sha256": base_sha,
              "train_frames": len(train[0]), "val_frames": len(val[0]), "best": result, "epochs": history,
              "train_episodes": train_episodes, "val_episodes": val_episodes}
    temporary = output / "report.tmp"
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(output / "report.json")
    print(f"Saved {path} with validation MSE {best_loss:.6f}", flush=True)


if __name__ == "__main__":
    main()
