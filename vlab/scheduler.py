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


class CameraBuffer:
    """Bounded per-camera frame queue ordered by capture time.

    Keeps frames that are not yet available (available_time in the future) so a
    delayed camera still serves the latest *available* historical frame instead of
    overwriting it with an unreadable new one. History is evicted oldest-first and
    the queue length is capped to bound memory over long runs.
    """

    TOL = 1e-6

    def __init__(self, maxlen: int = 96):
        self.maxlen = maxlen
        self.reset()

    def reset(self):
        self.frames = []
        self.next_id = 0
        self.evicted = 0

    def push(self, frame: dict):
        f = dict(frame)
        f["frame_id"] = self.next_id
        self.next_id += 1
        self.frames.append(f)
        if len(self.frames) > self.maxlen:
            self.evicted += len(self.frames) - self.maxlen
            self.frames = self.frames[-self.maxlen:]

    def latest_available(self, decision_time: float):
        best = None
        for f in self.frames:  # ascending capture/available time
            if f["available_time"] <= decision_time + self.TOL:
                best = f
            else:
                break
        return best

    def pending(self, decision_time: float):
        return sum(1 for f in self.frames if f["available_time"] > decision_time + self.TOL)