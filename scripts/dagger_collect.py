"""DAgger data collection: run the current BC policy, label visited states with the
expert action, and save (state, expert_action) pairs for retraining.

Aggregating these fixes the covariate shift that causes plain BC to brake early.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
from vlab.engine import PlatformEnv
from vlab.schema import ProjectConfig
from eval_policy import forward, state_vec

JOINTS = ["waist", "shoulder", "elbow", "wrist_pitch", "wrist_yaw", "left", "right"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(LAB / "datasets" / "models" / "bc_state_mlp.npz"))
    ap.add_argument("--iter", type=int, default=1)
    ap.add_argument("--out", default=str(LAB / "datasets" / "dagger"))
    args = ap.parse_args()
    cfg = ProjectConfig.model_validate_json((LAB / "configs" / "default.json").read_text(encoding="utf-8"))
    npz = np.load(args.model)
    env = PlatformEnv(cfg, LAB)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plan = [("standard", s) for s in (5, 6, 7)] + [("tall", 5), ("tall", 6),
                                                    ("wide", 5), ("wide", 6)]
    S, A = [], []
    try:
        for asset, seed in plan:
            env.config = env.config.model_copy(deep=True)
            env.config.task.asset = asset
            env.reset(seed)
            obs = env.observation()
            prev = np.zeros(4, dtype=np.float32)
            for _ in range(200):
                inp = np.concatenate([state_vec(obs), prev])
                S.append(inp)
                A.append(np.asarray(env.expert_action(), dtype=np.float32))
                action = np.clip(forward(npz, inp), -1, 1)
                prev = action.astype(np.float32)
                obs = env.step(action.tolist())
                if env.metrics()["success"] or env.metrics()["truncated"]:
                    break
    finally:
        env.close()
    S, A = np.asarray(S, dtype=np.float32), np.asarray(A, dtype=np.float32)
    path = out / f"dagger_iter{args.iter}.npz"
    np.savez(path, state=S, action=A)
    print(json.dumps({"pairs": int(len(S)), "path": str(path),
                      "episodes": len(plan)}))


if __name__ == "__main__":
    main()