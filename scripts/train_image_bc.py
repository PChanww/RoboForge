"""Image-based behavior cloning (visual policy, NO privileged object pose).

Inputs: front camera image (64x64) + joint state (18) + previous action (4).
Trains a small CNN+MLP on the LeRobot expert dataset, saves/reloads a checkpoint,
and exports metadata for the HTTP closed-loop evaluator (eval_image_policy.py).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lerobot.datasets.lerobot_dataset import LeRobotDataset

STATE_DIM = 18
PREV_DIM = 4
IMG = 64


class VisualBC(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2), nn.ReLU(),   # 30
            nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(),  # 14
            nn.Conv2d(32, 64, 3, stride=2), nn.ReLU(),  # 6
            nn.Flatten())
        self.img_fc = nn.Linear(64 * 6 * 6, 64)
        self.state_fc = nn.Linear(STATE_DIM + PREV_DIM, 32)
        self.head = nn.Sequential(nn.Linear(64 + 32, 128), nn.ReLU(),
                                  nn.Linear(128, 4), nn.Tanh())

    def forward(self, img, state, prev):
        i = F.relu(self.img_fc(self.conv(img)))
        s = F.relu(self.state_fc(torch.cat([state, prev], dim=1)))
        return self.head(torch.cat([i, s], dim=1))


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
    args = ap.parse_args()
    lab = Path(args.lab)
    ds = LeRobotDataset(repo_id="local/vla-lab-expert21", root=lab / "datasets" / "lerobot" / "expert21")
    img, state, act, ep = load(ds)
    prev = torch.zeros_like(act)
    for e in np.unique(ep):
        idx = np.where(ep == e)[0]
        prev[idx[1:]] = act[idx[:-1]]
    tr, va, gen = ep <= 14, (ep >= 15) & (ep <= 18), ep >= 19
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
    ck = {"state_dict": model.state_dict(), "state_mean": smean, "state_std": sstd,
          "prev_mean": pmean, "prev_std": pstd, "img_size": IMG, "steps": steps,
          "privileged": False, "inputs": ["front_image", "state", "prev_action"]}
    torch.save(ck, models / "visual_bc.pt")
    ck2 = torch.load(models / "visual_bc.pt", weights_only=False)
    m2 = VisualBC()
    m2.load_state_dict(ck2["state_dict"])
    m2.eval()
    with torch.no_grad():
        diff = float((model(img[:4], sn[:4], pn[:4]) - m2(img[:4], sn[:4], pn[:4])).abs().max())
    summary = {"type": "visual BC (image+state, no object pose)", "frames": int(len(img)),
               "train_frames": int(tr.sum()), "optimizer_steps": steps,
               "val_mse": round(val_mse, 5), "gen_mse": round(gen_mse, 5),
               "reload_max_abs_diff": diff, "checkpoint": str(models / "visual_bc.pt")}
    (lab / "datasets" / "reports").mkdir(parents=True, exist_ok=True)
    (lab / "datasets" / "reports" / "train_visual.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()