"""Training data and model manifests for the mobile transport action contract."""
from pathlib import Path
import hashlib
import json

from .scene_assets import contained_file, valid_id

STATE_FIELDS = ("arm_qpos", "base_pose", "base_velocity", "wheel_qpos", "wheel_qvel")
TASKS = ("deliver", "pick_return", "bin_retrieve", "bin_return")
ACTION_FIELDS = ["arm_dx", "arm_dy", "arm_dz", "grip", "base_linear_m_s", "base_angular_rad_s", "hold_arm"]
MODELS = {
    "state_bc": "移动状态 BC",
    "visual_bc": "移动双相机 BC",
    "privileged_bc": "移动特权 BC（仿真诊断）",
    "smolvla": "SmolVLA · 移动动作头",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def datasets(lab, selected=None):
    root = Path(lab) / "datasets/transport"
    directories = (sorted(root.glob("transport_*")) if selected is None else
                   [contained_file(root, valid_id(rid)) for rid in selected])
    rows = []
    for directory in directories:
        try:
            manifest = read_json(directory / "manifest.json")
            spec = manifest["spec"]
            config_path = directory / "config.json"
            checksum = manifest["configuration"]["sha256"]
            if (manifest["id"] != directory.name or manifest["status"] != "succeeded"
                    or manifest.get("action_fields") != ACTION_FIELDS or manifest.get("frequency_hz") != 20
                    or spec.get("controller", "expert") != "expert"
                    or manifest.get("policy_identity", {}).get("learned_policy")
                    or file_sha256(config_path) != checksum):
                continue
            task = manifest.get("task_kind", "deliver")
            if task not in TASKS:
                continue
            frames = []
            for episode in manifest["episodes"]:
                index = episode["index"]
                if (type(index) is int and index >= 0 and episode["success"] is True
                        and not episode.get("interrupted") and episode["frames"] > 0
                        and (directory / "episodes" / f"ep{index:03d}.npz").is_file()):
                    frames.append({key: episode[key] for key in ("index", "seed", "frames")})
            if not frames:
                continue
            goal = read_json(config_path)["task"]["goal_xy"]
            rows.append({"id": directory.name, "task": task, "instruction": spec["instruction"],
                         "asset": spec["object_asset"], "goal": goal, "speed_scale": spec["speed_scale"],
                         "config_sha256": checksum, "episodes": frames})
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return rows


def split_episodes(rows):
    # Hash-identical configurations include repeat recordings with the same seed.
    seen, groups = set(), {}
    for row in rows:
        for episode in row["episodes"]:
            key = (row["task"], row["config_sha256"], episode["seed"])
            if key in seen:
                continue
            seen.add(key)
            groups.setdefault(row["task"], []).append({
                **{k: v for k, v in row.items() if k != "episodes"}, **episode,
            })
    train, validation = [], []
    for group in groups.values():
        group.sort(key=lambda episode: (episode["seed"], episode["id"]))
        train.extend(group[:-1] if len(group) > 1 else group)
        if len(group) > 1:
            validation.append(group[-1])
    if not train or not validation:
        raise ValueError("训练至少需要同一任务的两条不同示范，以保留完整验证轨迹。")
    return train, validation


def checkpoint_dir(lab, model_id):
    return contained_file(Path(lab) / "datasets/transport_models", valid_id(model_id))


def trained_models(lab):
    result = []
    for path in sorted((Path(lab) / "datasets/transport_models").glob("*/report.json"), reverse=True):
        try:
            row = read_json(path)
            if row["id"] == path.parent.name and row["model"] in MODELS and (path.parent / "weights.safetensors").is_file():
                result.append(row)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return sorted(result, key=lambda row: row["created"], reverse=True)
