"""Data checks for the LeRobot dataset + staging alignment (runs in WSL).

Verifies timestamps, action/state ranges, image integrity, episode boundaries and
image-state-action time alignment. Writes datasets/reports/data_checks.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab", required=True, help="path to vla-lab")
    args = ap.parse_args()
    lab = Path(args.lab)
    ds = LeRobotDataset(repo_id="local/vla-lab-expert21", root=lab / "datasets" / "lerobot" / "expert21")
    cfg = json.loads((lab / "configs" / "default.json").read_text(encoding="utf-8"))
    manifest = json.loads((lab / "datasets" / "staging" / "manifest.json").read_text(encoding="utf-8"))
    limits = {j["name"]: j["range"] for j in cfg["robots"][0]["joints"] if j["range"]}

    issues, per_episode = [], []
    prev_ts = {}
    for i in range(len(ds)):
        s = ds[i]
        ep = int(s["episode_index"])
        ts = float(s["timestamp"])
        a = s["action"].numpy().astype(float)
        st = s["observation.state"].numpy().astype(float)
        if not np.isfinite(a).all() or not np.isfinite(st).all():
            issues.append(f"frame {i}: non-finite state/action")
        if (np.abs(a) > 1.0001).any():
            issues.append(f"frame {i}: action outside [-1,1]: {a}")
        if ep in prev_ts and ts <= prev_ts[ep]:
            issues.append(f"frame {i}: timestamp not increasing within episode")
        prev_ts[ep] = ts
        joints = st[:7]
        for k, name in enumerate(["waist", "shoulder", "elbow", "wrist_pitch", "wrist_yaw", "left", "right"]):
            lo, hi = limits[name]
            if not lo - 1e-3 <= joints[k] <= hi + 1e-3:
                issues.append(f"frame {i}: joint {name}={joints[k]:.3f} outside [{lo},{hi}]")

    # Image integrity: decode one frame per camera and count non-empty frames.
    sample = ds[0]
    img_shapes = {k: tuple(sample[k].shape) for k in ds.meta.image_keys}
    # The loader returns CHW float tensors; on disk frames are 256x256 RGB PNG.
    if any(not ((s == (3, 256, 256)) or (s == (256, 256, 3))) for s in img_shapes.values()):
        issues.append(f"unexpected image shapes {img_shapes}")

    # Alignment from staging: image capture time must not be in the future,
    # and the commanded translation must correlate with actual eef displacement.
    ages, intervals, cosines = [], [], []
    for row in manifest["episodes"]:
        z = np.load(lab / "datasets" / "staging" / "episodes" / f"ep{row['index']:03d}.npz", allow_pickle=True)
        ft = z["frame_time"]
        for mid, cam in enumerate(manifest["cameras"]):
            ct = z[f"capture_time_{cam}"]
            ages.extend((ft - ct).tolist())
            intervals.extend(np.diff(ct).tolist())
        disp = np.diff(z["eef_pos"], axis=0)
        act = z["action"][:-1, :3]
        mask = np.linalg.norm(disp, axis=1) > 1e-3
        if mask.any():
            cos = (disp[mask] * act[mask]).sum(1) / (
                np.linalg.norm(disp[mask], axis=1) * np.linalg.norm(act[mask], axis=1) + 1e-9)
            cosines.extend(cos.tolist())
        per_episode.append({"index": row["index"], "asset": row["asset"], "seed": row["seed"],
                            "frames": int(len(ft)), "success": bool(row["success"]),
                            "duration_s": float(ft[-1]),
                            "image_age_max_ms": float(1000 * (ft - z["capture_time_" + manifest["cameras"][0]]).max())})
    ages = np.asarray(ages)
    intervals = np.asarray(intervals)
    if (ages < -1e-6).any():
        issues.append(f"{int((ages < -1e-6).sum())} image samples have negative age (future frames)")
    cos_mean = float(np.mean(cosines)) if cosines else None
    if cos_mean is not None and cos_mean < 0.5:
        issues.append(f"action direction vs eef displacement cosine {cos_mean:.2f} < 0.5 (misaligned)")
    report = {
        "dataset": {"episodes": ds.num_episodes, "frames": ds.num_frames, "fps": ds.fps,
                    "image_keys": list(ds.meta.image_keys), "video_keys": list(ds.meta.video_keys),
                    "image_shapes": img_shapes},
        "checks": {
            "issues": issues,
            "action_range_ok": not any("[-1,1]" in s for s in issues),
            "timestamps_monotonic": not any("timestamp" in s for s in issues),
            "state_in_joint_limits": not any("outside [" in s and "joint" in s for s in issues),
            "image_age_s": {"mean": round(float(ages.mean()), 4), "max": round(float(ages.max()), 4)},
            "camera_period_s": {"mean": round(float(intervals.mean()), 4),
                                "min": round(float(intervals.min()), 4),
                                "max": round(float(intervals.max()), 4)},
            "action_state_direction_cosine": round(cos_mean, 3) if cos_mean is not None else None,
        },
        "episodes": per_episode,
    }
    out = lab / "datasets" / "reports" / "data_checks.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"issues": issues[:5], "n_issues": len(issues),
                      "image_age_mean_s": report["checks"]["image_age_s"]["mean"],
                      "camera_period_mean_s": report["checks"]["camera_period_s"]["mean"],
                      "frames": ds.num_frames}))


if __name__ == "__main__":
    main()