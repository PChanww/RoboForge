"""P0-1: MuJoCo segmentation ids are geom objids (no +1) and objtype must be GEOM.

Minimal repro: floor geom=0, cup geom=1 with 168 cup pixels; the old fg={2} kept 0.
Also checks the real robot scene mask for cup/pads/links/tray/table.
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
from vlab.rendering.hybrid import foreground_ids, segmentation_mask  # noqa: E402
from vlab.schema import ProjectConfig, SceneInstance  # noqa: E402
from vlab.engine import PlatformEnv  # noqa: E402

GEOM = int(mujoco.mjtObj.mjOBJ_GEOM)


def render_seg(model, data, camera, w, h):
    r = mujoco.Renderer(model, height=h, width=w)
    r.enable_segmentation_rendering()
    r.update_scene(data, camera=camera)
    seg = r.render()
    r.close()
    return seg


def minimal():
    xml = '''<mujoco><visual><global offwidth="96" offheight="96"/></visual>
    <worldbody>
      <geom name="floor" type="plane" size="5 5 .01" pos="0 0 0"/>
      <geom name="cup" type="sphere" size=".1" pos="0 1.2 .1" rgba=".8 .2 .1 1"/>
      <body name="hand" pos=".3 1.2 .1"><geom name="pad" type="box" size=".05 .05 .05"/></body>
      <camera name="cam" pos="0 1.2 1.0" xyaxes="1 0 0 0 1 0" fovy="50"/>
    </worldbody></mujoco>'''
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    seg = render_seg(m, d, "cam", 96, 96)
    cup_id, pad_id = m.geom("cup").id, m.geom("pad").id
    assert cup_id == 1 and m.geom("floor").id == 0
    mask = segmentation_mask(seg, {cup_id, pad_id})
    cup_px = int((seg[:, :, 0] == cup_id).sum())
    pad_px = int((seg[:, :, 0] == pad_id).sum())
    floor_px = int((seg[:, :, 0] == 0).sum())
    assert cup_px > 100, f"cup should be visible, got {cup_px}"
    assert int(mask.sum()) == cup_px + pad_px, (mask.sum(), cup_px, pad_px)
    assert not mask[seg[:, :, 0] == 0].any()
    assert floor_px > 0 and not mask[seg[:, :, 0] == 0].any()
    assert (seg[mask][:, 1] == GEOM).all()
    # old +1 convention would target the pad and drop the cup
    old = segmentation_mask(seg, {cup_id + 1, pad_id + 1})
    assert not (old[seg[:, :, 0] == cup_id]).any()
    return {"cup_px": cup_px, "pad_px": pad_px, "floor_px": floor_px,
            "old_mask_pixels": int(old.sum()), "new_mask_pixels": int(mask.sum())}


def real_env():
    cfg = ProjectConfig.model_validate_json((LAB / "configs" / "default.json").read_text(encoding="utf-8"))
    cfg.scene.sensor_renderer = "hybrid"
    cfg.scene.environment = SceneInstance(asset_id="world_586c79650960", revision=1)
    env = PlatformEnv(cfg, LAB)
    try:
        env.reset(0)
        w, h = cfg.sensors[0].resolution
        seg = render_seg(env.model, env.data, "front", w, h)
        fg = foreground_ids(env)
        mask = segmentation_mask(seg, fg)
        by_name = {env.model.geom(i).name: i for i in fg if env.model.geom(i).name}
        out = {"fg_count": len(fg), "mask_pixels": int(mask.sum())}
        cup_ids = [i for i in fg if env.model.body(int(env.model.geom_bodyid[i])).name == "cup"]
        cup_px = int(np.isin(seg[:, :, 0], cup_ids).sum())
        assert cup_ids and cup_px > 0
        assert int(mask[np.isin(seg[:, :, 0], cup_ids)].sum()) == cup_px
        out["cup"] = {"ids": cup_ids, "pixels": cup_px}
        for name in ("left_pad", "right_pad", "tray"):
            gid = env.model.geom(name).id
            px = int((seg[:, :, 0] == gid).sum())
            kept = int(mask[seg[:, :, 0] == gid].sum())
            out[name] = {"id": gid, "pixels": px, "kept": kept, "in_fg": gid in fg}
            assert gid in fg and px > 0 and kept == px, (name, px, kept)
        # robot link geoms must be included and preserved
        link_ids = [i for i in fg if env.model.body(int(env.model.geom_bodyid[i])).name in {l.name for l in env.robot.links}]
        assert link_ids and all(int(mask[seg[:, :, 0] == i].sum()) == int((seg[:, :, 0] == i).sum()) for i in link_ids)
        # floor/table must not be foreground (tray is the one intentional world-body exception)
        for gid in range(env.model.ngeom):
            name = env.model.geom(gid).name
            body = env.model.body(int(env.model.geom_bodyid[gid])).name
            if body == "world" and name != "tray":
                assert gid not in fg, f"world geom {gid} ({name}) leaked into foreground"
        # compose: foreground preserved, background replaced
        from vlab.rendering.hybrid import HybridRenderer
        fake = HybridRenderer.__new__(HybridRenderer)
        fake._bg = np.zeros((h, w, 3), dtype=np.float32)
        rgb = np.full((h, w, 3), 200, dtype=np.uint8)
        comp, cmask = fake.compose(rgb, seg, fg)
        assert np.array_equal(cmask, mask)
        assert (comp[mask] == rgb[mask]).all()
        assert (comp[~mask] == 0).all()
        out["fg_names"] = sorted(n for n in by_name)
        return out
    finally:
        env.close()


def main():
    r = {"minimal": minimal(), "real": real_env()}
    print(r)
    return r


if __name__ == "__main__":
    main()
