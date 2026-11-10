"""Closed-loop evaluation of the trained BC MLP in the MuJoCo sim (Windows).

Loads datasets/models/bc_state_mlp.npz, runs the policy on held-out episodes
(different spawn seeds and a held-out asset) and reports real success rates.
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


def forward(npz, state):
    x = (state - npz["state_mean"]) / npz["state_std"]
    h1 = np.maximum(x @ npz["w1"].T + npz["b1"], 0)
    h2 = np.maximum(h1 @ npz["w2"].T + npz["b2"], 0)
    return np.tanh(h2 @ npz["w3"].T + npz["b3"])


def state_vec(obs):
    st = obs["state"]
    joints = ["waist", "shoulder", "elbow", "wrist_pitch", "wrist_yaw", "left", "right"]
    return np.array([st["joint_pos"][n] for n in joints] + [st["joint_vel"][n] for n in joints]
                    + st["eef_position"] + [st["gripper"] or 0.0], dtype=np.float32)


def state_vec_p(obs):
    st = obs["state"]
    return np.concatenate([state_vec(obs), np.asarray(st["object_pose"]["position"], dtype=np.float32)])


def run_episode(env, npz, asset, seed, max_ticks=200, priv=False):
    env.config = env.config.model_copy(deep=True)
    env.config.task.asset = asset
    env.reset(seed)
    obs = env.observation()
    prev = np.zeros(4, dtype=np.float32)
    for _ in range(max_ticks):
        base = state_vec_p(obs) if priv else state_vec(obs)
        action = np.clip(forward(npz, np.concatenate([base, prev])), -1, 1)
        prev = action.astype(np.float32)
        obs = env.step(action.tolist())
        m = env.metrics()
        if m["success"]:
            return {"asset": asset, "seed": seed, "success": True, "tick": m["tick"],
                    "lifted": m["lifted"], "grasped": m["grasped"]}
    m = env.metrics()
    return {"asset": asset, "seed": seed, "success": False, "tick": m["tick"],
            "lifted": m["lifted"], "grasped": m["grasped"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(LAB / "datasets" / "models" / "bc_state_mlp.npz"))
    ap.add_argument("--out", default=str(LAB / "datasets" / "reports" / "eval_policy.json"))
    ap.add_argument("--group", default="all", choices=["all", "in_distribution", "heldout_spawn", "heldout_asset"])
    ap.add_argument("--priv", action="store_true", help="use the privileged policy (object pose input)")
    args = ap.parse_args()
    if args.priv and args.model == str(LAB / "datasets" / "models" / "bc_state_mlp.npz"):
        args.model = str(LAB / "datasets" / "models" / "bc_priv_mlp.npz")
    cfg = ProjectConfig.model_validate_json((LAB / "configs" / "default.json").read_text(encoding="utf-8"))
    npz = np.load(args.model)
    env = PlatformEnv(cfg, LAB)
    groups = {
        "in_distribution": [("standard", s) for s in (0, 1, 2)] + [("tall", 0), ("wide", 0)],
        "heldout_spawn": [("standard", 30), ("standard", 31), ("tall", 30), ("wide", 30)],
        "heldout_asset": [("imported/lux3d-ivory", 0), ("imported/lux3d-ivory", 1)],
    }
    report = {"model": str(args.model), "policy": "behavior-cloning MLP (state->4D cartesian), not a VLA",
              "groups": {}}
    try:
        for name, plan in groups.items():
            if args.group != "all" and name != args.group:
                continue
            rows = [run_episode(env, npz, a, s, priv=args.priv) for a, s in plan]
            report["groups"][name] = {"episodes": rows,
                                      "successes": sum(r["success"] for r in rows), "n": len(rows)}
            print(name, report["groups"][name]["successes"], "/", len(rows), flush=True)
    finally:
        env.close()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: f"{v['successes']}/{v['n']}" for k, v in report["groups"].items()}))


if __name__ == "__main__":
    main()