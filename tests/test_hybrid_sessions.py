"""T08: hybrid scene cross-session isolation.

Two sessions render the same gaussian scene with different seeds. Backgrounds may
share the process-level cache (world-static), but frames, epochs and foreground must
stay isolated, and resetting one session must not touch the other.
"""
import base64
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
from vlab.schema import ProjectConfig, SceneInstance  # noqa: E402
from vlab.sessions import Session  # noqa: E402


def cfg():
    c = ProjectConfig.model_validate_json((LAB / "configs" / "default.json").read_text(encoding="utf-8"))
    c.scene.sensor_renderer = "hybrid"
    c.scene.environment = SceneInstance(asset_id="world_586c79650960", revision=1)
    return c


def frame_image(obs, key="front"):
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(obs["images"][key]["jpeg_b64"]))))


def frames_of(session):
    return [f for b in session.env._buffers.values() for f in b.frames]


def main():
    c = cfg()
    s1 = Session(c, LAB, None, None)
    s2 = Session(c, LAB, None, None)
    try:
        o1 = s1.reset(0)["observation"]
        o2 = s2.reset(7)["observation"]
        img1, img2 = frame_image(o1), frame_image(o2)
        f1, f2 = frames_of(s1), frames_of(s2)
        r = {"session_ids": [s1.id, s2.id],
             "s1_epoch": s1.epoch, "s2_epoch": s2.epoch,
             "s1_frames": len(f1), "s2_frames": len(f2),
             "s1_session_ids_ok": all(f["session_id"] == s1.id for f in f1),
             "s2_session_ids_ok": all(f["session_id"] == s2.id for f in f2),
             "s1_epochs": sorted({f["snapshot_epoch"] for f in f1}),
             "s2_epochs": sorted({f["snapshot_epoch"] for f in f2}),
             "image_mean_abs_diff": round(float(np.abs(img1.astype(int) - img2.astype(int)).mean()), 3),
             "background_cache_shared": bool(getattr(s2.env._hybrid, "from_cache", False)),
             "separate_hybrid_objects": s1.env._hybrid is not s2.env._hybrid}
        assert r["s1_session_ids_ok"] and r["s2_session_ids_ok"]
        assert s1.id != s2.id
        assert r["s1_epochs"] == [s1.epoch] and r["s2_epochs"] == [s2.epoch]
        assert r["image_mean_abs_diff"] > 0.1, "different seeds should change foreground pixels"

        e2, tick2 = s2.epoch, s2.env.scheduler.tick
        ids2 = [f["frame_id"] for f in s2.env._buffers["front"].frames]
        s1.reset(1)
        d = {"s2_epoch_after_s1_reset": s2.epoch, "s2_tick_after_s1_reset": s2.env.scheduler.tick,
             "s2_frame_ids_unchanged": [f["frame_id"] for f in s2.env._buffers["front"].frames] == ids2,
             "s2_frames_epoch": sorted({f["snapshot_epoch"] for f in frames_of(s2)})}
        r.update(d)
        assert d["s2_epoch_after_s1_reset"] == e2 and d["s2_tick_after_s1_reset"] == tick2
        assert d["s2_frame_ids_unchanged"] and d["s2_frames_epoch"] == [s2.epoch]
        print(r)
    finally:
        s1.close()
        s2.close()


def test_hybrid_sessions_isolated():
    main()


if __name__ == "__main__":
    main()
