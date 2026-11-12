"""Convert staging episodes to a LeRobot v3 dataset (runs in WSL with lerobot 0.4.4).

Usage:
  PYTHONPATH=<deps> python scripts/lerobot_writer.py \
      --staging datasets/staging --out datasets/lerobot/expert21
Reads every episodes/ep*.npz (allow_pickle) and writes one LeRobot episode per npz.
"""
import argparse
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset

STATE_NAMES = ["waist", "shoulder", "elbow", "wrist_pitch", "wrist_yaw", "left", "right",
               "waist_vel", "shoulder_vel", "elbow_vel", "wrist_pitch_vel", "wrist_yaw_vel",
               "left_vel", "right_vel", "eef_x", "eef_y", "eef_z", "gripper"]
ACTION_NAMES = ["dx", "dy", "dz", "grip"]


def decode(b):
    return np.asarray(Image.open(io.BytesIO(bytes(b))).convert("RGB"), dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    staging, out = Path(args.staging), Path(args.out)
    manifest = json.loads((staging / "manifest.json").read_text(encoding="utf-8"))
    if out.exists():
        raise ValueError("Dataset output exists; choose a new dataset ID (existing data is never deleted)")
    fps = int(manifest["fps"])
    if fps != manifest["fps"]:
        raise ValueError("LeRobot baseline requires an integer control FPS")
    joint_names = manifest["joints"]
    state_names = joint_names + [n + "_vel" for n in joint_names] + ["eef_x", "eef_y", "eef_z", "gripper"]
    cfg = manifest["config_snapshot"]
    action_names = cfg["controllers"][0]["action_names"]
    features = {
        "observation.state": {"dtype": "float32", "shape": (len(state_names),), "names": state_names},
        "action": {"dtype": "float32", "shape": (len(action_names),), "names": action_names},
    }
    for camera in cfg["sensors"]:
        w, h = camera["resolution"]
        features[f"observation.images.{camera['id']}"] = {"dtype": "image", "shape": (h, w, 3), "names": ["height", "width", "channel"]}
    repo_id = "local/" + out.name
    ds = LeRobotDataset.create(repo_id=repo_id, fps=fps, features=features,
                               root=out, robot_type=manifest["robot"], use_videos=False)
    for ep in manifest["episodes"]:
        z = np.load(staging / "episodes" / f"ep{ep['index']:03d}.npz", allow_pickle=True)
        n = z["joint_pos"].shape[0]
        for i in range(n):
            state = np.concatenate([z["joint_pos"][i], z["joint_vel"][i], z["eef_pos"][i],
                                    np.array([z["gripper"][i]], dtype=np.float32)]).astype(np.float32)
            frame = {
                "observation.state": state,
                "action": z["action"][i].astype(np.float32),
                "task": cfg["task"]["instruction"],
            }
            for camera in manifest["cameras"]:
                frame[f"observation.images.{camera}"] = decode(z[f"jpeg_{camera}"][i])
            ds.add_frame(frame)
        ds.save_episode()
        print(f"wrote episode {ep['index']} ({ep['asset']}, seed {ep['seed']}) frames={n} success={ep['success']}", flush=True)
    ds.finalize()
    (out / "meta" / "simulation.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    # Companion per-frame timing (real capture/available/age) so time info survives export.
    with (out / "meta" / "frame_timing.jsonl").open("w", encoding="utf-8") as tf:
        for row in manifest["episodes"]:
            z = np.load(staging / "episodes" / f"ep{row['index']:03d}.npz", allow_pickle=True)
            for i in range(len(z["frame_time"])):
                rec = {"episode": row["index"], "frame": i, "sim_time": float(z["frame_time"][i]),
                       "cameras": {c: {"capture_time": float(z[f"capture_time_{c}"][i]),
                                       "age_s": float(z[f"age_{c}"][i])} for c in manifest["cameras"]}}
                tf.write(json.dumps(rec) + "\n")

    # Verify by reading the dataset back through the official loader.
    ds2 = LeRobotDataset(repo_id=repo_id, root=out)
    print(json.dumps({"episodes": ds2.num_episodes, "frames": ds2.num_frames,
                      "fps": ds2.fps, "features": list(ds2.features.keys()),
                      "video_keys": list(ds2.meta.video_keys), "image_keys": list(ds2.meta.image_keys)}))
    sample = ds2[0]
    print("sample keys", sorted(sample.keys()), "state", tuple(sample["observation.state"].shape),
          "action", tuple(sample["action"].shape), "task", sample["task"])


if __name__ == "__main__":
    main()
