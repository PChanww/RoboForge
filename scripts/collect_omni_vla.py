"""Collect unified multi-seed + two-object disambiguation dataset `vla_omni_v3`."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))

from vlab.collect import collect  # noqa: E402
from vlab.store import ConfigStore  # noqa: E402


def build_plan(default_cfg, hack_cfg) -> list[dict]:
    plan: list[dict] = []

    # Part A: 5 cups on `default` scene across seeds 0..5 (30 episodes)
    cups = [
        ("standard", "Pick up the standard cup and place it inside the tray."),
        ("tall", "Pick up the tall cup and place it inside the tray."),
        ("wide", "Pick up the wide cup and place it inside the tray."),
        ("imported/lux3d-blue-live-5887561", "Pick up the blue cup and place it inside the tray."),
        ("imported/cup-web", "Pick up the ceramic tumbler and place it inside the tray."),
    ]
    for asset, instr in cups:
        for seed in range(6):
            plan.append({
                "config": default_cfg,
                "asset": asset,
                "seed": seed,
                "active_target": "cup",
                "instruction": instr,
                "scene_tag": "default",
            })

    # Part B1: `hackathon_demo` two-object default poses across seeds 0..9 (20 episodes)
    for seed in range(10):
        plan.append({
            "config": hack_cfg,
            "asset": hack_cfg.task.asset,
            "seed": seed,
            "active_target": "cup",
            "instruction": "Pick up the blue cup and place it inside the tray.",
            "scene_tag": "hackathon_demo_default",
        })
        plan.append({
            "config": hack_cfg,
            "asset": hack_cfg.task.asset,
            "seed": seed,
            "active_target": "companion_cube",
            "instruction": "Pick up the orange cube and place it inside the tray.",
            "scene_tag": "hackathon_demo_default",
        })

    # Part B2: `hackathon_demo` two-object swapped/paired layouts (16 episodes, 8 pairs)
    pairs = [
        ([-0.14, -0.06], [-0.01, -0.06]),
        ([-0.01, -0.06], [-0.14, -0.06]),
        ([-0.12, -0.08], [-0.02, -0.02]),
        ([-0.02, -0.02], [-0.12, -0.08]),
        ([-0.13, -0.03], [-0.02, -0.08]),
        ([-0.02, -0.08], [-0.13, -0.03]),
        ([-0.11, -0.05], [-0.01, -0.04]),
        ([-0.01, -0.04], [-0.11, -0.05]),
    ]
    for idx, (cup_xy, cube_xy) in enumerate(pairs):
        oxy = {"cup": cup_xy, "companion_cube": cube_xy}
        plan.append({
            "config": hack_cfg,
            "asset": hack_cfg.task.asset,
            "seed": idx,
            "active_target": "cup",
            "object_xy": oxy,
            "instruction": "Pick up the blue cup and place it inside the tray.",
            "scene_tag": f"hackathon_demo_pair_{idx}",
        })
        plan.append({
            "config": hack_cfg,
            "asset": hack_cfg.task.asset,
            "seed": idx,
            "active_target": "companion_cube",
            "object_xy": oxy,
            "instruction": "Pick up the orange cube and place it inside the tray.",
            "scene_tag": f"hackathon_demo_pair_{idx}",
        })
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="vla_omni_v3")
    args = ap.parse_args()

    store = ConfigStore(LAB / "projects")
    default_cfg = store.load("default")
    hack_cfg = store.load("hackathon_demo").model_copy(deep=True)
    hack_cfg.scene.sensor_renderer = "mujoco"
    hack_cfg.scene.environment = None

    plan = build_plan(default_cfg, hack_cfg)
    assert len(plan) == 66, len(plan)
    out_dir = LAB / "datasets" / "collections" / args.name
    provenance = {
        "dataset_intent": "unified multi-seed 5-cup generalization + two-object same-table language disambiguation",
        "episodes_total": len(plan),
        "parts": {"default_5cups_x6seeds": 30, "hackathon_demo_default_10seeds_x2": 20, "hackathon_demo_swapped_8pairs_x2": 16},
        "render_mode": "mujoco",
    }
    manifest = collect(default_cfg, LAB, plan, out_dir, max_ticks=200, expert="servo", provenance=provenance)
    print("COLLECT_SUMMARY:", json.dumps(manifest["summary"]), flush=True)
    if manifest["summary"]["successes"] != manifest["summary"]["episodes"]:
        failed = [e for e in manifest["episodes"] if not e["success"]]
        raise SystemExit(f"Expert collection had {len(failed)} failures: {failed}")


if __name__ == "__main__":
    main()
