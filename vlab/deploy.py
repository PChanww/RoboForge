"""Export a portable deployment package and a standalone loader.

The package bundles the config snapshot, robot/object assets, policy weights and
the observation/action contract. `run.py` inside the package rebuilds the MuJoCo
environment from the package alone (in a fresh process) and runs a controller.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .schema import ProjectConfig

RUN_PY = '''"""Standalone package runner. Rebuilds the env from this package and runs a controller.

  python run.py --controller expert --ticks 200 --report result.json
  python run.py --controller policy --ticks 200 --report result.json
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

PKG = Path(__file__).resolve().parent
LAB = Path(__import__("os").environ.get("VLAB_LAB", PKG))
sys.path.insert(0, str(LAB))

from vlab.engine import PlatformEnv
from vlab.schema import ProjectConfig

def policy_forward(npz, inp):
    x = (inp - npz["state_mean"]) / npz["state_std"]
    h = np.maximum(x @ npz["w1"].T + npz["b1"], 0)
    h = np.maximum(h @ npz["w2"].T + npz["b2"], 0)
    return np.tanh(h @ npz["w3"].T + npz["b3"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", default="expert", choices=["expert", "policy"])
    ap.add_argument("--ticks", type=int, default=200)
    ap.add_argument("--report", default="result.json")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    cfg = ProjectConfig.model_validate_json((PKG / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((PKG / "manifest.json").read_text(encoding="utf-8"))
    env = PlatformEnv(cfg, PKG)
    obs = env.observation() if a.seed is None else env.reset(a.seed)
    npz = np.load(PKG / manifest["policy"]["weights_file"]) if a.controller == "policy" else None
    joints = manifest["observation"]["state_names"][:7]
    prev = np.zeros(4, dtype=np.float32)
    for _ in range(a.ticks):
        if a.controller == "expert":
            action = env.expert_action()
        else:
            st = obs["state"]
            base = np.array([st["joint_pos"][n] for n in joints] + [st["joint_vel"][n] for n in joints]
                            + st["eef_position"] + [st["gripper"] or 0.0], dtype=np.float32)
            if manifest["policy"].get("privileged"):
                base = np.concatenate([base, np.asarray(st["object_pose"]["position"], dtype=np.float32)])
            action = np.clip(policy_forward(npz, np.concatenate([base, prev])), -1, 1)
            prev = action.astype(np.float32)
            action = action.tolist()
        obs = env.step(action)
        if env.metrics()["success"]:
            break
    m = env.metrics()
    env.close()
    result = {"controller": a.controller, "success": m["success"], "lifted": m["lifted"],
              "grasped": m["grasped"], "tick": m["tick"], "config_version": cfg.version,
              "package": manifest["package_id"]}
    Path(a.report).write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result))

if __name__ == "__main__":
    main()
'''


def export_package(config: ProjectConfig, lab_root: Path, out_dir: Path, policy_npz: Path,
                   policy_meta: dict, extra_assets=()) -> dict:
    lab_root, out_dir = Path(lab_root), Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "assets").mkdir(parents=True)

    shutil.copytree(lab_root / "assets" / "robot", out_dir / "assets" / "robot")
    for asset in extra_assets:
        src = lab_root / "assets" / asset
        if src.is_dir():
            shutil.copytree(src, out_dir / "assets" / asset)
    (out_dir / "config.json").write_text(config.model_dump_json(indent=2), encoding="utf-8")
    shutil.copy(policy_npz, out_dir / "model.npz")
    weights_sha = hashlib.sha256((out_dir / "model.npz").read_bytes()).hexdigest()
    (out_dir / "run.py").write_text(RUN_PY, encoding="utf-8")

    contract = {
        "observation": {
            "state_names": policy_meta["state_names"],
            "image_keys": [s.id for s in config.sensors],
            "image_fps": {s.id: s.fps for s in config.sensors},
            "state_semantics": "joint pos (rad/m), joint vel, eef position (m), gripper [0..1]"
                               + (", object position (m)" if policy_meta.get("privileged") else ""),
        },
        "action": {
            "names": config.controllers[0].action_names,
            "dim": config.controllers[0].action_dim,
            "range": [[-1, 1]] * config.controllers[0].action_dim,
            "frame": config.controllers[0].frame,
            "max_delta_m": config.controllers[0].max_delta_m,
            "frequency_hz": config.controllers[0].frequency_hz,
        },
        "timing": {"physics_dt": config.scene.physics_dt,
                   "control_hz": config.controllers[0].frequency_hz},
    }
    manifest = {
        "package_id": f"{config.project_id}-v{config.version}-{datetime.now(timezone.utc):%Y%m%d%H%M%S}",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": config.schema_version,
        "config_version": config.version,
        "robot": config.robots[0].id,
        "controller": config.controllers[0].id,
        "observation": contract["observation"],
        "action": contract["action"],
        "timing": contract["timing"],
        "policy": {**policy_meta, "weights_file": "model.npz", "weights_sha256": weights_sha},
        "assets": sorted(str(p.relative_to(out_dir)) for p in (out_dir / "assets").rglob("*") if p.is_file()),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "README.md").write_text(
        "# 部署包\n\n此目录自包含配置、资产与策略权重。运行：\n\n"
        "```powershell\n"
        "python run.py --controller expert --ticks 200 --report result.json\n"
        "python run.py --controller policy --ticks 200 --report result.json\n"
        "```\n\n"
        "`run.py` 仅依赖本机 vla-lab 的 `vlab` 包（可移植到其他机器时一并复制）。\n",
        encoding="utf-8")
    return manifest