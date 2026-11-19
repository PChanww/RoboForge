"""Closed-loop evaluation of the visual BC policy over HTTP (runs in WSL).

Loads visual_bc.pt, drives the Windows MuJoCo backend via /api/v1 (observation ->
CNN -> action). Uses only camera image + joint state + previous action (no object
pose). Creates per-asset evaluation projects so held-out assets can be tested.
"""
import argparse
import base64
import io
import json
import sys
import uuid
import urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vlab.schema import ProjectConfig
from vlab.compat import check_contract, visual_contract
from vlab.scene_assets import valid_id

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from train_image_bc import VisualBC, IMG

BASE = "http://127.0.0.1:8767"
JOINTS = ["waist", "shoulder", "elbow", "wrist_pitch", "wrist_yaw", "left", "right"]


def req(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(BASE + path, data=data, method=method,
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=180) as resp:
        return json.loads(resp.read())


BASE_PROJECT = "default"
BASE_CONFIG = None
RUN_ID = uuid.uuid4().hex[:8]


def ensure_project(asset):
    pid = "eval_" + RUN_ID + "_" + asset.replace("/", "_")
    cfg = json.loads(json.dumps(BASE_CONFIG))
    cfg["task"]["asset"] = asset
    cfg["project_id"] = pid
    req("PUT", f"/api/v1/config?project={pid}", cfg)
    return pid


def state_vec(st):
    return np.array([st["joint_pos"][n] for n in JOINTS] + [st["joint_vel"][n] for n in JOINTS]
                    + st["eef_position"] + [st["gripper"] or 0.0], dtype=np.float32)


def run_episode(model, ck, asset, seed, ticks=200):
    pid = ensure_project(asset)
    s = req("POST", "/api/v1/sessions", {"project": pid})
    sid = s["id"]
    tok = req("POST", f"/api/v1/sessions/{sid}/control/acquire", {"owner": "visual_eval"})["token"]
    epoch = req("POST", f"/api/v1/sessions/{sid}/reset", {"seed": seed})["epoch"]
    prev = np.zeros(4, dtype=np.float32)
    success = False
    render_mode = None
    try:
        for _ in range(ticks):
            obs = req("GET", f"/api/v1/sessions/{sid}/observation")
            render_mode = obs.get("render_mode", render_mode)
            f = obs["images"].get("front")
            if not f:
                break
            img = np.asarray(Image.open(io.BytesIO(base64.b64decode(f["jpeg_b64"]))).convert("RGB"),
                             dtype=np.float32) / 255.0
            t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            t = F.interpolate(t, size=(IMG, IMG), mode="bilinear", align_corners=False)
            st = (torch.tensor(state_vec(obs["state"])) - ck["state_mean"]) / ck["state_std"]
            pn = (torch.tensor(prev) - ck["prev_mean"]) / ck["prev_std"]
            with torch.no_grad():
                action = model(t, st.unsqueeze(0), pn.unsqueeze(0))[0].numpy()
            action = np.clip(action, -1, 1)
            prev = action.astype(np.float32)
            req("POST", f"/api/v1/sessions/{sid}/action",
                {"token": tok, "action": action.tolist(), "epoch": epoch})
            m = req("GET", f"/api/v1/sessions/{sid}/result")["metrics"]
            if m["success"]:
                success = True
                break
    finally:
        req("DELETE", f"/api/v1/sessions/{sid}")
    return {"asset": asset, "seed": seed, "success": success, "render_mode": render_mode}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--model", default="visual_bc")
    ap.add_argument("--base-project", default="default")
    ap.add_argument("--plan", default="full", choices=["full", "small"])
    ap.add_argument("--config", help="Immutable project config snapshot")
    ap.add_argument("--allow-legacy-contract", action="store_true", help="Explicitly assume the original default robot/cameras for legacy checkpoints")
    ap.add_argument("--api-base", default="http://127.0.0.1:8767")
    args = ap.parse_args()
    global BASE_PROJECT, BASE_CONFIG, BASE
    BASE = args.api_base.rstrip("/")
    BASE_PROJECT = args.base_project
    lab = Path(args.lab)
    valid_id(args.model)
    valid_id(args.base_project)
    ck = torch.load(lab / "datasets" / "models" / f"{args.model}.pt", weights_only=False)
    model = VisualBC()
    model.load_state_dict(ck["state_dict"])
    model.eval()
    base_cfg = (json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config
                else req("GET", f"/api/v1/config?project={BASE_PROJECT}"))
    BASE_CONFIG = base_cfg
    robot_id = base_cfg["robots"][0]["id"]
    contract = ck.get("contract")
    if contract is None:
        if not args.allow_legacy_contract:
            raise SystemExit("Checkpoint has no input/action contract. Retrain or explicitly use --allow-legacy-contract for the historical baseline")
        contract = visual_contract(ProjectConfig.model_validate_json((lab / "configs/default.json").read_text(encoding="utf-8")))
        print("WARNING: assuming historical default contract for legacy checkpoint", flush=True)
    visual_contract(ProjectConfig.model_validate(base_cfg))
    compat = check_contract(contract, ProjectConfig.model_validate(base_cfg), robot_id)
    if not compat["compatible"]:
        raise SystemExit(f"visual policy incompatible with {robot_id}: {compat['mismatches']}")
    if getattr(args, "plan", "full") == "small":
        plan = [("standard", 0), ("standard", 1)]
    else:
        plan = [("standard", 0), ("standard", 1), ("standard", 30), ("standard", 31),
                ("imported/lux3d-ivory", 0), ("imported/lux3d-ivory", 1)]
    rows = [run_episode(model, ck, a, s) for a, s in plan]
    groups = {
        "in_distribution": [r for r in rows if r["asset"] == "standard" and r["seed"] < 10],
        "heldout_spawn": [r for r in rows if r["asset"] == "standard" and r["seed"] >= 10],
        "heldout_asset": [r for r in rows if r["asset"] != "standard"],
    }
    report = {"policy": "visual BC (image+state, no object pose)",
              "render_mode": rows[0].get("render_mode") if rows else None, "episodes": rows,
              "model": args.model, "project": BASE_PROJECT, "config_snapshot": base_cfg,
              "contract": contract, "legacy_contract_assumed": "contract" not in ck,
              "groups": {k: {"successes": sum(r["success"] for r in v), "n": len(v)} for k, v in groups.items()}}
    out = Path(args.out) if args.out else lab / "datasets" / "reports" / f"eval_{args.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report["groups"]))


if __name__ == "__main__":
    main()
