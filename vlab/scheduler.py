"""Simulation-clock driven multi-rate scheduling.

Physics advances in fixed dt steps. Control ticks and camera captures are events
on the same sim clock; non-integer period ratios are handled by stepping to the
first physics tick at or after each scheduled event and recording the actual time.
"""
from __future__ import annotations

import math


class Scheduler:
    def __init__(self, physics_dt: float, control_hz: float, camera_fps: dict[str, float]):
        if physics_dt <= 0 or control_hz <= 0:
            raise ValueError("physics_dt and control_hz must be positive")
        self.dt = physics_dt
        self.control_period = 1.0 / control_hz
        self.camera_period = {cid: 1.0 / fps for cid, fps in camera_fps.items()}
        if any(p <= 0 for p in self.camera_period.values()):
            raise ValueError("camera fps must be positive")
        self.reset()

    def reset(self):
        self.tick = 0
        self.capture_count = {cid: 0 for cid in self.camera_period}
        self.last_capture = {cid: None for cid in self.camera_period}
        self.actual_intervals = {cid: [] for cid in self.camera_period}
        self.control_times = []

    def capture_times_in(self, t_start: float, t_end: float, cid: str):
        period = self.camera_period[cid]
        n0 = math.floor(t_start / period + 1e-9) + 1  # half-open (t_start, t_end]
        n1 = math.floor(t_end / period + 1e-9)
        return [n * period for n in range(max(n0, 1), n1 + 1)]

    def record_capture(self, cid: str, capture_time: float):
        prev = self.last_capture[cid]
        if prev is not None:
            self.actual_intervals[cid].append(round(capture_time - prev, 6))
            if len(self.actual_intervals[cid]) > 120:
                self.actual_intervals[cid].pop(0)
        self.last_capture[cid] = capture_time
        self.capture_count[cid] += 1

    def record_control(self, t: float):
        self.control_times.append(round(t, 6))
        self.tick += 1

    def timing_summary(self, decision_time: float, frames: dict):
        ages = {cid: round(decision_time - f["capture_time"], 6)
                for cid, f in frames.items() if f is not None}
        return {"decision_time": round(decision_time, 6),
                "image_age_s": ages,
                "capture_count": dict(self.capture_count),
                "actual_capture_intervals_s": {cid: iv[-5:] for cid, iv in self.actual_intervals.items()},
                "control_period_s": self.control_period,
                "camera_period_s": self.camera_period,
                "physics_dt_s": self.dt}