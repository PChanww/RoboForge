"""T07: stale epoch/tick responses after reset are rejected and old frames do not leak.

A frame/action captured before reset carries the old epoch. After another reset the
session must reject it before touching physics, and camera buffers must contain only
frames from the new epoch/session.
"""
import sys
from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
from vlab.schema import ProjectConfig  # noqa: E402
from vlab.sessions import ControlConflict, Session  # noqa: E402


def cfg():
    return ProjectConfig.model_validate_json((LAB / "configs" / "default.json").read_text(encoding="utf-8"))


def test_stale_epoch_and_tick_rejected_after_reset():
    s = Session(cfg(), LAB, None, None)
    try:
        first = s.reset(0)
        e0 = first["epoch"]
        token = s.acquire("tester")
        tick0 = s.env.scheduler.tick
        zeros = [0.0] * s.env.controller.action_dim
        s.execute(token, zeros, tick=tick0, epoch=e0)
        old_frames = [f for b in s.env._buffers.values() for f in b.frames]
        assert old_frames and all(f["snapshot_epoch"] == e0 for f in old_frames)
        assert all(f["session_id"] == s.id for f in old_frames)

        second = s.reset(0)
        e1 = second["epoch"]
        assert e1 != e0
        qpos = s.env.data.qpos.copy()
        tick_after_reset = s.env.scheduler.tick

        with pytest.raises(ControlConflict, match="stale epoch"):
            s.execute(token, zeros, epoch=e0)
        with pytest.raises(ControlConflict, match="stale tick"):
            s.execute(token, zeros, tick=tick0 + 3, epoch=e1)
        assert s.env.scheduler.tick == tick_after_reset
        assert (s.env.data.qpos == qpos).all(), "rejected action must not change physics"

        s.execute(token, zeros, tick=tick_after_reset, epoch=e1)
        frames = [f for b in s.env._buffers.values() for f in b.frames]
        assert frames and all(f["snapshot_epoch"] == e1 for f in frames)
        assert all(f["session_id"] == s.id for f in frames)
        assert not any(f["snapshot_epoch"] == e0 for f in frames), "old-epoch frames leaked after reset"
    finally:
        s.close()


def main():
    test_stale_epoch_and_tick_rejected_after_reset()
    print({"stale_epoch_rejected": True, "stale_tick_rejected": True, "frames_new_epoch_only": True})


if __name__ == "__main__":
    main()
