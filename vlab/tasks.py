"""Cup pick-and-place task plugin: scene geometry, spawn and contact-based judging."""
from __future__ import annotations

from pathlib import Path

import numpy as np

import sim_env

from .schema import TaskConfig

TRAY_HALF = [.074, .069]


def asset_params(asset, lab_root=None):
    """Resolve builtin names, 'imported/<name>' ids, or a manifest path."""
    if asset in sim_env.BUILTINS:
        return sim_env._load_asset(asset)
    path = Path(asset)
    if not path.is_file():
        if lab_root is None:
            raise FileNotFoundError(f"asset {asset} is not a builtin and lab_root is unknown")
        path = Path(lab_root) / "assets" / asset / "manifest.json"
    return sim_env._load_asset(path)


def _tray_half(task: TaskConfig) -> list[float]:
    he = getattr(task, "goal_half_extents", None)
    if he and len(he) == 2:
        return [float(he[0]), float(he[1])]
    return list(TRAY_HALF)


def task_xml(task: TaskConfig, radius, height, mass, color, offset, handle, spawn, *, support_z: float = 0.0):
    """Task objects only (tray + dynamic cup). Environment comes from scene_builder."""
    gx, gy = task.goal_xy
    gs = task.goal_surface_m
    hx, hy = _tray_half(task)
    tray_xml = ""
    if getattr(task, "show_builtin_tray", True):
        thickness = max(0.004, gs - float(support_z))
        tray_cz = float(support_z) + thickness / 2.0
        lip_z = gs + 0.002
        lx = max(0.002, hx - 0.001)
        ly = max(0.002, hy - 0.001)
        tray_xml = f"""
        <geom name="tray" type="box" pos="{gx} {gy} {tray_cz}" size="{hx} {hy} {thickness/2}" rgba=".63 .69 .65 1"/>
        <geom type="box" pos="{gx - lx} {gy} {lip_z}" size=".002 {hy} .004" rgba=".63 .69 .65 1"/>
        <geom type="box" pos="{gx + lx} {gy} {lip_z}" size=".002 {hy} .004" rgba=".63 .69 .65 1"/>
        <geom type="box" pos="{gx} {gy - ly} {lip_z}" size="{lx} .002 .004" rgba=".63 .69 .65 1"/>
        <geom type="box" pos="{gx} {gy + ly} {lip_z}" size="{lx} .002 .004" rgba=".63 .69 .65 1"/>"""
    cup_z = float(support_z) + height / 2 + .001
    return f"""{tray_xml}
        <body name="cup" pos="{spawn[0]} {spawn[1]} {cup_z}">
          <freejoint name="cup_free"/>
          <geom name="cup_contact" type="cylinder" size="{radius} {height/2}" mass="{mass}"
                group="3" rgba=".2 .5 .6 0" friction="1.8 .005 .0001"/>
          <geom class="visual" type="mesh" mesh="cup_mesh" pos="0 0 {offset}" rgba="{color} 1"/>
          {handle}
        </body>"""


def spawn_xy(task: TaskConfig, seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(task.spawn_min, task.spawn_max)


class Judge:
    """Contact-based success: bilateral grasp, lift, upright rest in tray after release."""

    def __init__(
        self,
        model,
        data,
        task: TaskConfig,
        radius,
        height,
        *,
        body_name: str = "cup",
        geom_prefix: str = "cup_contact",
        freejoint_name: str = "cup_free",
        pad_geoms=("left_pad", "right_pad"),
    ):
        self.model, self.data, self.task = model, data, task
        self.radius, self.height = radius, height
        self.body_name = body_name
        self.cup = model.body(body_name).id
        self.cup_geoms = {
            gid for gid in range(model.ngeom)
            if (model.geom(gid).name == geom_prefix or model.geom(gid).name.startswith(f"{geom_prefix}_"))
        }
        if not self.cup_geoms:
            self.cup_geoms = {
                gid for gid in range(model.ngeom)
                if int(model.geom_bodyid[gid]) == self.cup and int(model.geom_group[gid]) != 2
            }
        self.cup_geom = sorted(self.cup_geoms)[0]
        tray_name = getattr(task, "goal_tray_collider_ref", None) or "tray"
        base_prefix = tray_name[:-7] if tray_name.endswith("_part_0") else tray_name
        matched = [
            gid for gid in range(model.ngeom)
            if (model.geom(gid).name == tray_name or model.geom(gid).name.startswith(f"{base_prefix}_part_"))
        ]
        if not matched:
            raise ValueError(f"goal tray collider geom {tray_name!r} not found in MuJoCo scene")
        self.tray_geoms = set(matched)
        self.tray_geom = matched[0]
        self.tray_half = _tray_half(task)
        self.cup_dofadr = int(model.joint(freejoint_name).dofadr[0])
        if len(set(pad_geoms)) != 2:
            raise ValueError("the tabletop grasping task requires two different contact pad geoms")
        self.pads = {model.geom(name).id for name in pad_geoms}
        self.start_xyz = data.xpos[self.cup].copy()
        self.reset()

    def reset(self):
        self.touched = self.grasped = self.lifted = self.success = False
        self.tray_contact = False
        self.contacts = 0
        self.stable_ticks = 0

    def _contacts(self):
        touching, on_tray = set(), False
        for c in self.data.contact:
            a, b = int(c.geom1), int(c.geom2)
            if a in self.cup_geoms or b in self.cup_geoms:
                touching.update(self.pads.intersection((a, b)))
                on_tray |= (a in self.tray_geoms or b in self.tray_geoms)
        return len(touching), on_tray

    def update(self, grip_target, open_m):
        cup = self.data.xpos[self.cup]
        contacts, self.tray_contact = self._contacts()
        self.contacts = contacts
        self.touched |= contacts > 0
        self.grasped |= contacts == 2
        self.lifted |= contacts == 2 and cup[2] > self.start_xyz[2] + self.task.lift_threshold_m
        gx, gy = self.task.goal_xy
        hx, hy = self.tray_half
        in_tray = np.all(np.abs(cup[:2] - [gx, gy]) + self.radius < [hx - .003, hy - .003])
        upright = self.data.xmat[self.cup, 8] > .93
        v_lin = self.data.qvel[self.cup_dofadr : self.cup_dofadr + 3]
        v_ang = self.data.qvel[self.cup_dofadr + 3 : self.cup_dofadr + 6]
        rest_offset = self.height / 2 if self.body_name == "cup" else 0.0
        resting = (abs(cup[2] - rest_offset - self.task.goal_surface_m) < .006
                   and np.linalg.norm(v_lin) < .018
                   and np.linalg.norm(v_ang) < .30)
        released = contacts == 0 and grip_target > open_m - .009
        self.stable_ticks = (self.stable_ticks + 1 if self.lifted and self.tray_contact
                             and in_tray and upright and resting and released else 0)
        self.success |= self.stable_ticks >= self.task.stable_ticks

    def metrics(self, tick, dt):
        return {"object_xyz": self.data.xpos[self.cup].tolist(),
                "goal_xyz": [*self.task.goal_xy, self.task.goal_surface_m],
                "touched": bool(self.touched), "grasped": bool(self.grasped),
                "lifted": bool(self.lifted), "success": bool(self.success),
                "contacts": self.contacts, "tray_contact": bool(self.tray_contact),
                "time_s": tick * dt, "tick": tick, "stable_ticks": self.stable_ticks}
