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


def load(ds):
    imgs, states, acts, eps = [], [], [], []
    for i in range(len(ds)):
        s = ds[i]
        img = s["observation.images.front"]
        if img.shape[0] != 3:
            img = img.permute(2, 0, 1)
        img = F.interpolate(img.unsqueeze(0), size=(IMG, IMG), mode="bilinear", align_corners=False)[0]
        imgs.append(img)
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
    args = ap.parse_args()
    valid_id(args.dataset)
    valid_id(args.model_out)
    lab = Path(args.lab)
    metadata = lab / "datasets" / "lerobot" / args.dataset / "meta" / "simulation.json"
    if not metadata.is_file():
        raise ValueError("Dataset lacks its simulation/config contract; reconvert to a new dataset ID before training")
    source = json.loads(metadata.read_text(encoding="utf-8"))
    contract = visual_contract(ProjectConfig.model_validate(source["config_snapshot"]))
    ds = LeRobotDataset(repo_id=f"local/{args.dataset}", root=lab / "datasets" / "lerobot" / args.dataset)
    img, state, act, ep = load(ds)
    if state.shape[1] != STATE_DIM or act.shape[1] != PREV_DIM:
        raise ValueError("VisualBC only supports 18 state values and 4 actions")
    prev = torch.zeros_like(act)
    for e in np.unique(ep):
        idx = np.where(ep == e)[0]
        prev[idx[1:]] = act[idx[:-1]]
    tr, va, gen = ep <= 14, (ep >= 15) & (ep <= 18), ep >= 19
    if not all(mask.any() for mask in (tr, va, gen)):
        raise ValueError("This baseline split requires the 21-episode collection plan; training/validation/generalization groups must be nonempty")
    smean, sstd = state[tr].mean(0), state[tr].std(0) + 1e-6
    pmean, pstd = prev[tr].mean(0), prev[tr].std(0) + 1e-6
    sn = (state - smean) / sstd
    pn = (prev - pmean) / pstd
    torch.manual_seed(0)
    model = VisualBC()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    n, bs, steps = int(tr.sum()), 128, 0
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
          "privileged": False, "inputs": ["front_image", "state", "prev_action"],
          "contract": contract, "dataset": args.dataset, "config_sha256": source["config_sha256"],
          "render_meta": render_meta}
    torch.save(ck, models / f"{args.model_out}.pt")
    ck2 = torch.load(models / f"{args.model_out}.pt", weights_only=False)
    m2 = VisualBC()
    m2.load_state_dict(ck2["state_dict"])
    m2.eval()
    with torch.no_grad():
        diff = float((model(img[:4], sn[:4], pn[:4]) - m2(img[:4], sn[:4], pn[:4])).abs().max())
    summary = {"type": "visual BC (image+state, no object pose)", "frames": int(len(img)),
               "train_frames": int(tr.sum()), "optimizer_steps": steps,
               "val_mse": round(val_mse, 5), "gen_mse": round(gen_mse, 5),
               "reload_max_abs_diff": diff, "dataset": args.dataset,
               "render_meta": render_meta, "checkpoint": str(models / f"{args.model_out}.pt")}
    (lab / "datasets" / "reports").mkdir(parents=True, exist_ok=True)
    (lab / "datasets" / "reports" / f"train_{args.model_out}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
