"""Image-based behavior cloning (visual policy, NO privileged object pose).

Inputs: front camera image (64x64) + joint state (18) + previous action (4).
Trains a small CNN+MLP on the LeRobot expert dataset, saves/reloads a checkpoint,
and exports metadata for the HTTP closed-loop evaluator (eval_image_policy.py).
"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vlab.schema import ProjectConfig
from vlab.compat import visual_contract
from vlab.scene_assets import valid_id

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lerobot.datasets.lerobot_dataset import LeRobotDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from visual_bc_model import IMG, PREV_DIM, STATE_DIM, VisualBC  # noqa: E402
from vlab.visual_infer import VisualPolicyRunner  # noqa: E402


def load(ds, camera_keys):
    imgs, states, acts, eps = [], [], [], []
    for i in range(len(ds)):
        s = ds[i]
        chans = []
        for cam in camera_keys:
            key = f"observation.images.{cam}"
            if key not in s:
                raise ValueError(f"Dataset missing camera stream {key!r}")
            img = s[key]
            if img.shape[0] != 3:
                img = img.permute(2, 0, 1)
            img = F.interpolate(img.unsqueeze(0), size=(IMG, IMG), mode="bilinear", align_corners=False)[0]
            chans.append(img)
        imgs.append(torch.cat(chans, dim=0))
        states.append(s["observation.state"])
        acts.append(s["action"])
        eps.append(int(s["episode_index"]))
    return (torch.stack(imgs), torch.stack(states), torch.stack(acts), np.asarray(eps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--dataset", default="expert21")
    ap.add_argument("--model-out", default="visual_bc")
    ap.add_argument("--cameras", default="front", help="Comma-separated camera keys, e.g. front,wrist")
    args = ap.parse_args()
    valid_id(args.dataset)
    valid_id(args.model_out)
    camera_keys = [valid_id(c.strip()) for c in args.cameras.split(",") if c.strip()]
    if not camera_keys:
        raise ValueError("Provide at least one camera key")
    lab = Path(args.lab)
    metadata = lab / "datasets" / "lerobot" / args.dataset / "meta" / "simulation.json"
    if not metadata.is_file():
        raise ValueError("Dataset lacks its simulation/config contract; reconvert to a new dataset ID before training")
    source = json.loads(metadata.read_text(encoding="utf-8"))
    contract = visual_contract(ProjectConfig.model_validate(source["config_snapshot"]))
    contract["camera_keys"] = camera_keys
    ds = LeRobotDataset(repo_id=f"local/{args.dataset}", root=lab / "datasets" / "lerobot" / args.dataset)
    img, state, act, ep = load(ds, camera_keys)
    if state.shape[1] != STATE_DIM or act.shape[1] != PREV_DIM:
        raise ValueError("VisualBC only supports 18 state values and 4 actions")
    prev = torch.zeros_like(act)
    for e in np.unique(ep):
        idx = np.where(ep == e)[0]
        prev[idx[1:]] = act[idx[:-1]]
    unique_eps = sorted(np.unique(ep).tolist())
    if len(unique_eps) >= 20 and max(unique_eps) >= 19:
        tr, va, gen = ep <= 14, (ep >= 15) & (ep <= 18), ep >= 19
    elif len(unique_eps) >= 2:
        n_ep = len(unique_eps)
        n_tr = max(1, min(n_ep - 1, int(round(0.7 * n_ep))))
        tr_set = set(unique_eps[:n_tr])
        rest = unique_eps[n_tr:]
        va_set = set(rest[:max(1, len(rest) // 2)])
        gen_set = set(rest[max(1, len(rest) // 2):]) or va_set
        tr = np.isin(ep, list(tr_set))
        va = np.isin(ep, list(va_set))
        gen = np.isin(ep, list(gen_set))
    else:
        tr = va = gen = np.ones(len(ep), dtype=bool)
    if not all(mask.any() for mask in (tr, va, gen)):
        raise ValueError("Dataset split produced empty train/val/gen partition")
    smean, sstd = state[tr].mean(0), state[tr].std(0) + 1e-6
    pmean, pstd = prev[tr].mean(0), prev[tr].std(0) + 1e-6
    sn = (state - smean) / sstd
    pn = (prev - pmean) / pstd
    torch.manual_seed(0)
    in_channels = 3 * len(camera_keys)
    model = VisualBC(in_channels=in_channels)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    loss_fn = nn.MSELoss()
    n, steps = int(tr.sum()), 0
    bs = 32 if n < 800 else 64
    xt = (img[tr], sn[tr], pn[tr], act[tr])
    for epoch in range(args.epochs):
        perm = torch.randperm(n)
        model.train()
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(model(xt[0][idx], xt[1][idx], xt[2][idx]), xt[3][idx])
            loss.backward()
            opt.step()
            steps += 1
        sched.step()
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                vl = float(loss_fn(model(img[va], sn[va], pn[va]), act[va]))
            print({"epoch": epoch + 1, "val_mse": round(vl, 5)}, flush=True)
    model.eval()
    with torch.no_grad():
        val_mse = float(loss_fn(model(img[va], sn[va], pn[va]), act[va]))
        gen_mse = float(loss_fn(model(img[gen], sn[gen], pn[gen]), act[gen]))
    models = lab / "datasets" / "models"
    models.mkdir(parents=True, exist_ok=True)
    render_meta = {}
    manifest_path = lab / "datasets" / "collections" / args.dataset / "manifest.json"
    if manifest_path.is_file():
        mf = json.loads(manifest_path.read_text(encoding="utf-8"))
        render_meta = {k: mf.get(k) for k in ("render_mode", "sensor_renderer", "renderer_version",
                                             "gaussian_sha256") if mf.get(k) is not None}
    ck = {"state_dict": model.state_dict(), "state_mean": smean, "state_std": sstd,
          "prev_mean": pmean, "prev_std": pstd, "img_size": IMG, "steps": steps,
          "in_channels": in_channels, "camera_keys": camera_keys,
          "privileged": False, "inputs": camera_keys + ["state", "prev_action"],
          "contract": contract, "dataset": args.dataset, "config_sha256": source["config_sha256"],
          "render_meta": render_meta}
    torch.save(ck, models / f"{args.model_out}.pt")
    npz_payload = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in model.state_dict().items()}
    npz_payload.update({
        "state_mean": smean.cpu().numpy().astype(np.float32),
        "state_std": sstd.cpu().numpy().astype(np.float32),
        "prev_mean": pmean.cpu().numpy().astype(np.float32),
        "prev_std": pstd.cpu().numpy().astype(np.float32),
        "img_size": np.array(IMG, dtype=np.int32),
        "camera_keys": np.array(json.dumps(camera_keys)),
        "state_joints": np.array(json.dumps(source.get("joints") or ["waist", "shoulder", "elbow", "wrist_pitch", "wrist_yaw", "left", "right"])),
    })
    np.savez(models / f"{args.model_out}.npz", **npz_payload)
    runner = VisualPolicyRunner(lab, args.model_out)
    with torch.no_grad():
        t_out = model(img[:4], sn[:4], pn[:4]).numpy()
    np_out = np.stack([runner.forward(img[k].numpy(), state[k].numpy(), prev[k].numpy()) for k in range(min(4, len(img)))])
    numpy_parity = float(np.abs(t_out[:len(np_out)] - np_out).max())
    if numpy_parity >= 1e-4:
        raise RuntimeError(f"PyTorch/NumPy inference parity check failed: {numpy_parity}")
    ck2 = torch.load(models / f"{args.model_out}.pt", weights_only=False)
    m2 = VisualBC(in_channels=in_channels)
    m2.load_state_dict(ck2["state_dict"])
    m2.eval()
    with torch.no_grad():
        diff = float((model(img[:4], sn[:4], pn[:4]) - m2(img[:4], sn[:4], pn[:4])).abs().max())
    summary = {"type": f"visual BC ({'+'.join(camera_keys)}+state, no object pose)",
               "camera_keys": camera_keys, "in_channels": in_channels,
               "episodes": len(unique_eps), "frames": int(len(img)),
               "train_frames": int(tr.sum()), "optimizer_steps": steps,
               "val_mse": round(val_mse, 5), "gen_mse": round(gen_mse, 5),
               "reload_max_abs_diff": diff, "numpy_parity_max_diff": round(numpy_parity, 8),
               "dataset": args.dataset, "render_meta": render_meta,
               "checkpoint": str(models / f"{args.model_out}.pt"),
               "numpy_weights": str(models / f"{args.model_out}.npz")}
    (lab / "datasets" / "reports").mkdir(parents=True, exist_ok=True)
    (lab / "datasets" / "reports" / f"train_{args.model_out}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
