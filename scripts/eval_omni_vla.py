"""Closed-loop evaluation for `so100_smolvla_omni`:
- Suite 1: Two-object same-table language disambiguation on `hackathon_demo` (seen + unseen seeds + swapped XYs)
- Suite 2: Multi-seed generalization across 5 cups on `default` (seen seed=0 + unseen seeds=6,7)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))

from vlab.engine import PlatformEnv  # noqa: E402
from vlab.smolvla_infer import SmolVLAPolicyRunner  # noqa: E402
from vlab.store import ConfigStore  # noqa: E402


def _in_tray(env: PlatformEnv, target_id: str) -> bool:
    bname = "cup" if target_id == "cup" else f"node_{target_id}"
    bid = env.model.body(bname).id
    xy = env.data.xpos[bid][:2]
    gx, gy = env.config.task.goal_xy
    hx, hy = env.config.task.goal_half_extents or [0.074, 0.069]
    return bool(abs(float(xy[0]) - gx) < hx and abs(float(xy[1]) - gy) < hy)


def _rollout(env: PlatformEnv, runner: SmolVLAPolicyRunner, instruction: str, max_ticks: int = 160) -> dict:
    obs = env.observation()
    runner.reset()
    ok = False
    m = env.metrics()
    for _ in range(max_ticks):
        act = runner.act(obs, instruction=instruction)
        obs = env.step(act)
        m = env.metrics()
        if m.get("success"):
            ok = True
            break
    return {
        "success": ok,
        "ticks": int(m["tick"]),
        "grasped": bool(m.get("grasped")),
        "lifted": bool(m.get("lifted")),
        "tray_contact": bool(m.get("tray_contact")),
        "active_target": str(m.get("active_target")),
    }


def main():
    store = ConfigStore(LAB / "projects")
    default_cfg = store.load("default")
    hack_cfg = store.load("hackathon_demo").model_copy(deep=True)
    hack_cfg.scene.sensor_renderer = "mujoco"
    hack_cfg.scene.environment = None

    runner = SmolVLAPolicyRunner(LAB, "so100_smolvla_omni", shared=False)
    try:
        # Suite 1: Two-object same-table disambiguation
        suite1 = []
        env_h = PlatformEnv(hack_cfg, LAB)
        try:
            default_cases = [(s, s in (0, 1), None) for s in (0, 1, 10, 11)]
            swapped_cases = [
                (20, False, {"cup": [-0.01, -0.06], "companion_cube": [-0.14, -0.06]}),
                (21, False, {"cup": [-0.14, -0.06], "companion_cube": [-0.01, -0.06]}),
            ]
            for seed, seen, oxy in [*default_cases, *swapped_cases]:
                for tgt, instr in (
                    ("cup", "Pick up the blue cup and place it inside the tray."),
                    ("companion_cube", "Pick up the orange cube and place it inside the tray."),
                ):
                    env_h.reset(seed)
                    if oxy:
                        for k, xy in oxy.items():
                            env_h.set_object_xy(k, xy)
                    chosen = env_h.select_target_for_instruction(instr)
                    assert chosen == tgt, (chosen, tgt)
                    res = _rollout(env_h, runner, instr)
                    other = "companion_cube" if tgt == "cup" else "cup"
                    res.update({
                        "seed": seed,
                        "seen": seen,
                        "swapped": oxy is not None,
                        "object_xy": oxy,
                        "target": tgt,
                        "instruction": instr,
                        "other_object_in_tray": _in_tray(env_h, other),
                    })
                    suite1.append(res)
                    print("SUITE1", res, flush=True)
        finally:
            env_h.close()

        # Suite 2: Multi-seed 5-cup generalization on `default`
        suite2 = []
        cups = [
            ("standard", "Pick up the standard cup and place it inside the tray."),
            ("tall", "Pick up the tall cup and place it inside the tray."),
            ("wide", "Pick up the wide cup and place it inside the tray."),
            ("imported/lux3d-blue-live-5887561", "Pick up the blue cup and place it inside the tray."),
            ("imported/cup-web", "Pick up the ceramic tumbler and place it inside the tray."),
        ]
        for asset, instr in cups:
            c = default_cfg.model_copy(deep=True)
            c.task.asset = asset
            c.task.instruction = instr
            env_d = PlatformEnv(c, LAB)
            try:
                for seed, seen in ((0, True), (6, False), (7, False)):
                    env_d.reset(seed)
                    res = _rollout(env_d, runner, instr)
                    res.update({"asset": asset, "seed": seed, "seen": seen, "instruction": instr})
                    suite2.append(res)
                    print("SUITE2", res, flush=True)
            finally:
                env_d.close()
    finally:
        runner.close()

    def _rate(rows):
        n = len(rows)
        ok = sum(1 for r in rows if r["success"])
        return {"passed": ok, "total": n, "rate": round(ok / n, 4) if n else 0.0}

    summary = {
        "model_id": "so100_smolvla_omni",
        "two_object_disambiguation": {
            "all": _rate(suite1),
            "seen": _rate([r for r in suite1 if r["seen"]]),
            "unseen": _rate([r for r in suite1 if not r["seen"] and not r["swapped"]]),
            "swapped": _rate([r for r in suite1 if r["swapped"]]),
            "clean_single_target_in_tray": {
                "passed": sum(1 for r in suite1 if r["success"] and not r["other_object_in_tray"]),
                "total": len(suite1),
                "rate": round(sum(1 for r in suite1 if r["success"] and not r["other_object_in_tray"]) / len(suite1), 4) if suite1 else 0.0,
            },
        },
        "default_5cups_multiseed": {
            "all": _rate(suite2),
            "seen": _rate([r for r in suite2 if r["seen"]]),
            "unseen": _rate([r for r in suite2 if not r["seen"]]),
        },
    }
    report = {"summary": summary, "suite1_two_object": suite1, "suite2_5cups": suite2}
    out = LAB / "datasets" / "reports" / "eval_so100_smolvla_omni.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("EVAL_SUMMARY:", json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
