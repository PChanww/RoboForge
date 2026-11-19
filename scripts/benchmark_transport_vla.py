"""Measure the live VLA service using real simulator cameras and applied actions.

This is a latency benchmark, not a claim that the robot completed its task.
Run with the simulator .venv while the independent VLA service is ready.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vlab.store import ConfigStore
from vlab.transport import MobileTransportEnv
from vlab.transport_api import TransportSpec
from vlab.transport_policy import PolicyRegistry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=int, default=5, choices=range(2, 11))
    parser.add_argument("--output", type=Path, default=ROOT / "work/transport-vla-benchmark.json")
    args = parser.parse_args()
    spec = TransportSpec(controller="vla", policy_id="smolvla-libero", vla_replan_steps=5,
        instruction="从传送带抓起物品，然后放回原位。", task_kind="pick_return", speed_scale=1.8).model_dump()
    controller = PolicyRegistry(ROOT).create(spec)
    env = MobileTransportEnv(ConfigStore(ROOT / "projects").load(), ROOT, spec)
    try:
        env.reset(spec["seed"])
        controller.reset(spec["seed"], spec["instruction"])
        for _ in range(args.chunks*spec["vla_replan_steps"]):
            preview = env.snapshot(render=True)
            action = controller.action(env, env.state(), preview)
            env.step(action)
            controller.mark_applied()
        chunks = controller.diagnostics["chunks"]
        timings = [item["inference_s"] for item in chunks]
        report = {"kind": "real_vla_inference_latency_benchmark", "task_success_evaluated": False,
            "identity": controller.identity, "first_chunk_s": timings[0],
            "warm_median_s": statistics.median(timings[1:]), "all_chunk_s": timings,
            "peak_memory_mib": max((item.get("peak_memory_mib", 0.) for item in chunks), default=0.),
            "diagnostics": controller.diagnostics}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps({key: value for key, value in report.items() if key != "diagnostics"}, ensure_ascii=True), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
