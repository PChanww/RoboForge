"""Export the deployment package and verify it in independent processes.

Runs the packaged env with the expert (expects success) and with the exported
policy (records the outcome) in fresh Python processes.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
from vlab.deploy import export_package
from vlab.schema import ProjectConfig

STATE_NAMES = ["waist", "shoulder", "elbow", "wrist_pitch", "wrist_yaw", "left", "right",
               "waist_vel", "shoulder_vel", "elbow_vel", "wrist_pitch_vel", "wrist_yaw_vel",
               "left_vel", "right_vel", "eef_x", "eef_y", "eef_z", "gripper"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(LAB / "deploy" / "lux3d_5r-v4"))
    args = ap.parse_args()
    cfg = ProjectConfig.model_validate_json((LAB / "configs" / "default.json").read_text(encoding="utf-8"))
    train = json.loads((LAB / "datasets" / "reports" / "train_bc.json").read_text(encoding="utf-8"))
    evalr = json.loads((LAB / "datasets" / "reports" / "eval_policy.json").read_text(encoding="utf-8"))
    meta = {"type": "behavior-cloning MLP (state->4D cartesian)", "state_names": STATE_NAMES,
            "privileged": False, "normalization": "train-split mean/std",
            "data_version": "expert21 (21 episodes, 3297 frames)", "train": train,
            "eval": {"groups": evalr["groups"]}, "deployable_observation": True}
    out = Path(args.out)
    manifest = export_package(cfg, LAB, out, LAB / "datasets" / "models" / "bc_state_mlp.npz", meta)
    print("exported", out, "package", manifest["package_id"], flush=True)

    env = {**os.environ, "VLAB_LAB": str(LAB), "PYTHONIOENCODING": "utf-8"}
    results = {}
    for controller in ("expert", "policy"):
        rep = out / f"result_{controller}.json"
        proc = subprocess.run([sys.executable, "run.py", "--controller", controller, "--ticks", "200",
                               "--report", str(rep)], cwd=out, env=env, capture_output=True, text=True)
        results[controller] = {"returncode": proc.returncode,
                               "stdout": proc.stdout.strip()[-400:],
                               "stderr": proc.stderr.strip()[-400:],
                               "result": json.loads(rep.read_text(encoding="utf-8")) if rep.is_file() else None}
        print(controller, results[controller]["returncode"], results[controller]["result"], flush=True)
    compat = {"package": manifest["package_id"], "weights_sha256": manifest["policy"]["weights_sha256"],
              "independent_process": results,
              "note": "expert expected to succeed; policy is a data-limited imitation baseline"}
    (LAB / "datasets" / "reports" / "deploy_check.json").write_text(
        json.dumps(compat, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: (v["result"] or {}).get("success", v["returncode"]) for k, v in results.items()}))


if __name__ == "__main__":
    main()