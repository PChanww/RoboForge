"""Privileged imitation baseline: state + object pose + previous action -> action.

This is a sanity check for the train/save/reload/eval loop, NOT a deployable
policy: it reads the object pose, which a real deployment would need to perceive.
The deployable path is vision (SmolVLA), reported separately.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from train_bc import MLP, STATE_NAMES, ACTION_NAMES


def ep_arrays(files):
    X, A, E = [], [], []
    for f in files:
        z = np.load(f, allow_pickle=True)
        base = np.concatenate([z["joint_pos"], z["joint_vel"], z["eef_pos"],
                               z["gripper"][:, None], z["object_xyz"]], axis=1)
        prev = np.zeros_like(z["action"])
        prev[1:] = z["action"][:-1]
        X.append(np.concatenate([base, prev], axis=1))
        A.append(z["action"])
        E.append(np.full(len(base), int(f.rsplit("ep", 1)[1].split(".")[0])))
    return np.concatenate(X).astype(np.float32), np.concatenate(A).astype(np.float32), np.concatenate(E)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()
    lab = Path(args.lab)
    files = sorted(glob.glob(str(lab / "datasets" / "staging" / "episodes" / "ep*.npz")))
    X, A, E = ep_arrays(files)
    tr, va, gen = E <= 14, (E >= 15) & (E <= 18), E >= 19
    mean, std = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xn = (X - mean) / std
    torch.manual_seed(0)
    model = MLP(n_in=X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    xt, at = torch.tensor(Xn[tr]), torch.tensor(A[tr])
    xv, av = torch.tensor(Xn[va]), torch.tensor(A[va])
    n, bs, steps = len(xt), 256, 0
    for epoch in range(args.epochs):
        perm = torch.randperm(n)
        model.train()
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss_fn(model(xt[idx]), at[idx]).backward()
            opt.step()
            steps += 1
        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                print({"epoch": epoch + 1, "val_mse": round(float(loss_fn(model(xv), av)), 5)}, flush=True)
    model.eval()
    with torch.no_grad():
        val_mse = float(loss_fn(model(xv), av))
        gen_mse = float(loss_fn(model(torch.tensor(Xn[gen])), torch.tensor(A[gen])))
    ckpt = {"state_dict": model.state_dict(), "state_mean": mean, "state_std": std, "steps": steps}
    models = lab / "datasets" / "models"
    models.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, models / "bc_priv_mlp.pt")
    ck = torch.load(models / "bc_priv_mlp.pt", weights_only=False)
    m2 = MLP(n_in=X.shape[1])
    m2.load_state_dict(ck["state_dict"])
    m2.eval()
    probe = torch.tensor(Xn[:8], dtype=torch.float32)
    with torch.no_grad():
        diff = float((model(probe) - m2(probe)).abs().max())
    sd = ck["state_dict"]
    np.savez(models / "bc_priv_mlp.npz",
             w1=sd["net.0.weight"].numpy(), b1=sd["net.0.bias"].numpy(),
             w2=sd["net.2.weight"].numpy(), b2=sd["net.2.bias"].numpy(),
             w3=sd["net.4.weight"].numpy(), b3=sd["net.4.bias"].numpy(),
             state_mean=mean, state_std=std)
    summary = {"input_dim": int(X.shape[1]), "frames": int(len(X)), "optimizer_steps": steps,
               "val_mse": round(val_mse, 5), "gen_mse": round(gen_mse, 5), "reload_max_abs_diff": diff,
               "note": "privileged (object pose included); sanity check of the training loop"}
    (lab / "datasets" / "reports" / "train_priv.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()