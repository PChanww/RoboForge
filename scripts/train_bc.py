"""Small behavior-cloning MLP (state -> cartesian action) on the LeRobot dataset.

Runs in WSL with torch. Trains on episodes 0-14, validates on held-out spawns
15-18 and held-out asset 19-20, saves a checkpoint, reloads it, and exports
numpy weights for closed-loop evaluation in the MuJoCo sim.
"""
import argparse
import glob
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

STATE_NAMES = ["waist", "shoulder", "elbow", "wrist_pitch", "wrist_yaw", "left", "right",
               "waist_vel", "shoulder_vel", "elbow_vel", "wrist_pitch_vel", "wrist_yaw_vel",
               "left_vel", "right_vel", "eef_x", "eef_y", "eef_z", "gripper"]
ACTION_NAMES = ["dx", "dy", "dz", "grip"]


class MLP(nn.Module):
    def __init__(self, n_in=18, n_out=4, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_out), nn.Tanh())

    def forward(self, x):
        return self.net(x)


def load_arrays(dataset_root: Path):
    files = sorted(glob.glob(str(dataset_root / "data" / "**" / "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no parquet under {dataset_root}/data")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    X = np.stack(df["observation.state"].map(lambda v: np.asarray(v, dtype=np.float32)))
    A = np.stack(df["action"].map(lambda v: np.asarray(v, dtype=np.float32)))
    E = df["episode_index"].to_numpy()
    return X, A, E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dagger", default="", help="glob of DAgger npz files (default: <lab>/datasets/dagger/*.npz)")
    args = ap.parse_args()
    lab = Path(args.lab)
    if not args.dagger:
        args.dagger = str(lab / "datasets" / "dagger" / "*.npz")
    root = lab / "datasets" / "lerobot" / "expert21"
    X, A, E = load_arrays(root)
    tr = E <= 14
    va = (E >= 15) & (E <= 18)
    gen = E >= 19
    # Previous action is a timing cue for the open-loop scripted expert.
    P = np.zeros_like(A)
    for ep in np.unique(E):
        idx = np.where(E == ep)[0]
        P[idx[1:]] = A[idx[:-1]]
    Xin = np.concatenate([X, P], axis=1)
    mean = Xin[tr].mean(0)
    std = Xin[tr].std(0) + 1e-6
    Xn = (Xin - mean) / std

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = MLP(n_in=Xin.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    xt, at = torch.tensor(Xn[tr]), torch.tensor(A[tr])
    xv, av = torch.tensor(Xn[va]), torch.tensor(A[va])
    dagger_files = sorted(glob.glob(args.dagger))
    if dagger_files:
        ds = np.concatenate([np.load(f)["state"] for f in dagger_files])
        da = np.concatenate([np.load(f)["action"] for f in dagger_files])
        xt = torch.cat([xt, torch.tensor((ds - mean) / std, dtype=torch.float32)])
        at = torch.cat([at, torch.tensor(da, dtype=torch.float32)])
        print(f"added {len(ds)} DAgger pairs from {len(dagger_files)} file(s)", flush=True)
    n, bs = len(xt), 256
    steps, history = 0, []
    for epoch in range(args.epochs):
        perm = torch.randperm(n)
        model.train()
        total = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(model(xt[idx]), at[idx])
            loss.backward()
            opt.step()
            steps += 1
            total += float(loss) * len(idx)
        model.eval()
        with torch.no_grad():
            val = float(loss_fn(model(xv), av))
        history.append({"epoch": epoch + 1, "train_mse": round(total / n, 5), "val_mse": round(val, 5)})
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(history[-1], flush=True)

    model.eval()
    with torch.no_grad():
        gen_mse = float(loss_fn(model(torch.tensor(Xn[gen])), torch.tensor(A[gen])))
    ckpt = {"state_dict": model.state_dict(), "state_mean": mean, "state_std": std,
            "state_names": STATE_NAMES, "action_names": ACTION_NAMES, "steps": steps,
            "train_episodes": "0-14", "val_episodes": "15-18", "gen_episodes": "19-20",
            "val_mse": history[-1]["val_mse"], "gen_mse": round(gen_mse, 5)}
    models = lab / "datasets" / "models"
    models.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, models / "bc_state_mlp.pt")

    # Reload from disk and verify identical predictions.
    ck = torch.load(models / "bc_state_mlp.pt", weights_only=False)
    model2 = MLP(n_in=Xin.shape[1])
    model2.load_state_dict(ck["state_dict"])
    model2.eval()
    probe = torch.tensor(Xn[:8], dtype=torch.float32)
    with torch.no_grad():
        d = float((model(probe) - model2(probe)).abs().max())
    sd = ck["state_dict"]
    np.savez(models / "bc_state_mlp.npz",
             w1=sd["net.0.weight"].numpy(), b1=sd["net.0.bias"].numpy(),
             w2=sd["net.2.weight"].numpy(), b2=sd["net.2.bias"].numpy(),
             w3=sd["net.4.weight"].numpy(), b3=sd["net.4.bias"].numpy(),
             state_mean=mean, state_std=std)
    summary = {"frames": int(len(X)), "train_frames": int(tr.sum()), "val_frames": int(va.sum()),
               "gen_frames": int(gen.sum()), "optimizer_steps": steps,
               "final_train_mse": history[-1]["train_mse"], "val_mse": history[-1]["val_mse"],
               "gen_mse": round(gen_mse, 5), "reload_max_abs_diff": d,
               "checkpoint": str(models / "bc_state_mlp.pt")}
    (lab / "datasets" / "reports").mkdir(parents=True, exist_ok=True)
    (lab / "datasets" / "reports" / "train_bc.json").write_text(json.dumps(
        {**summary, "history": history}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()