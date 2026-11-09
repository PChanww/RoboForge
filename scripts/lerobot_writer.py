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
    fps = int(manifest["fps"])
    features = {
        "observation.state": {"dtype": "float32", "shape": (len(STATE_NAMES),), "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": (len(ACTION_NAMES),), "names": ACTION_NAMES},
        "observation.images.front": {"dtype": "image", "shape": (256, 256, 3),
                                     "names": ["height", "width", "channel"]},
        "observation.images.wrist": {"dtype": "image", "shape": (256, 256, 3),
                                     "names": ["height", "width", "channel"]},
    }
    if out.exists():
        import shutil
        shutil.rmtree(out)
    ds = LeRobotDataset.create(repo_id="local/vla-lab-expert21", fps=fps, features=features,
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
                "observation.images.front": decode(z["jpeg_front"][i]),
                "observation.images.wrist": decode(z["jpeg_wrist"][i]),
                "task": "Place the cup in the tray.",
            }
            ds.add_frame(frame)
        ds.save_episode()
        print(f"wrote episode {ep['index']} ({ep['asset']}, seed {ep['seed']}) frames={n} success={ep['success']}", flush=True)
    ds.finalize()

    # Verify by reading the dataset back through the official loader.
    ds2 = LeRobotDataset(repo_id="local/vla-lab-expert21", root=out)
    print(json.dumps({"episodes": ds2.num_episodes, "frames": ds2.num_frames,
                      "fps": ds2.fps, "features": list(ds2.features.keys()),
                      "video_keys": list(ds2.meta.video_keys), "image_keys": list(ds2.meta.image_keys)}))
    sample = ds2[0]
    print("sample keys", sorted(sample.keys()), "state", tuple(sample["observation.state"].shape),
          "action", tuple(sample["action"].shape), "task", sample["task"])


if __name__ == "__main__":
    main()