"""P0-1 regression: bounded camera latency queue.

Checks capture<=available<=decision, delayed cameras serve legal historical frames
after warmup, reset clears to a waiting state, multi-camera, non-integer fps, and
bounded queue over long runs.
"""
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
from vlab.engine import PlatformEnv
from vlab.schema import ProjectConfig

TOL = 1e-4


def base_cfg():
    return ProjectConfig.model_validate_json((LAB / "configs" / "default.json").read_text(encoding="utf-8"))


def run_latency(latency, steps=60, fps=None):
    cfg = base_cfg()
    for s in cfg.sensors:
        s.latency_s = latency
        if fps:
            s.fps = fps
    env = PlatformEnv(cfg, LAB)
    try:
        obs = env.observation()
        if latency > 0:
            assert all(obs["images"][c] is None for c in obs["images"]), "reset should wait"
            assert all(obs["camera_status"][c]["state"] == "waiting" for c in obs["camera_status"])
        served = 0
        max_queue = 0
        for i in range(steps):
            obs = env.step([0, 0, 0, 1])
            t = obs["sim_time"]
            max_queue = max(max_queue, max(st["queue_len"] for st in obs["camera_status"].values()))
            for cid, f in obs["images"].items():
                st = obs["camera_status"][cid]
                if f is None:
                    assert t < latency + TOL or st["state"] == "waiting", (latency, t, cid)
                    continue
                served += 1
                assert f["capture_time"] <= f["available_time"] + TOL, (f,)
                assert f["available_time"] <= t + TOL, (f, t)
                age = t - f["capture_time"]
                cam_period = 1.0 / (fps or 30)
                assert latency - TOL <= age <= latency + cam_period + 0.05 + TOL, (latency, age)
        assert served > 0, f"no frames served at latency {latency}"
        assert max_queue <= 96, max_queue
        return {"latency": latency, "served": served, "max_queue": max_queue,
                "actual_fps": {c: st["actual_capture_fps"] for c, st in obs["camera_status"].items()}}
    finally:
        env.close()


def main():
    results = [run_latency(L) for L in (0.0, 0.05, 0.1, 0.2)]
    nonint = run_latency(0.1, steps=60, fps=17)
    # long run bound
    cfg = base_cfg()
    for s in cfg.sensors:
        s.latency_s = 0.1
    env = PlatformEnv(cfg, LAB)
    try:
        for _ in range(400):
            obs = env.step([0, 0, 0, 1])
        assert all(st["queue_len"] <= 96 for st in obs["camera_status"].values())
        long_queue = max(st["queue_len"] for st in obs["camera_status"].values())
    finally:
        env.close()
    print({"latencies": results, "non_integer_fps17": nonint["actual_fps"],
           "long_run_queue": long_queue})


if __name__ == "__main__":
    main()