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


def task_xml(task: TaskConfig, radius, height, mass, color, offset, handle, spawn):
    """Task objects only (tray + dynamic cup). Environment comes from scene_builder."""
    gx, gy = task.goal_xy
    gs = task.goal_surface_m
    return f"""
        <geom name="tray" type="box" pos="{gx} {gy} {gs/2}" size="{TRAY_HALF[0]} {TRAY_HALF[1]} {gs/2}" rgba=".63 .69 .65 1"/>
        <geom type="box" pos="{gx-.073} {gy} .011" size=".002 .069 .004" rgba=".63 .69 .65 1"/>
        <geom type="box" pos="{gx+.073} {gy} .011" size=".002 .069 .004" rgba=".63 .69 .65 1"/>
        <geom type="box" pos="{gx} {gy-.068} .011" size=".073 .002 .004" rgba=".63 .69 .65 1"/>
        <geom type="box" pos="{gx} {gy+.068} .011" size=".073 .002 .004" rgba=".63 .69 .65 1"/>
        <body name="cup" pos="{spawn[0]} {spawn[1]} {height/2+.001}">
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

    def __init__(self, model, data, task: TaskConfig, radius, height):
        self.model, self.data, self.task = model, data, task
        self.radius, self.height = radius, height
        self.cup = model.body("cup").id
        self.cup_geom = model.geom("cup_contact").id
        self.tray_geom = model.geom("tray").id
        self.pads = {model.geom(n).id for n in ("left_pad", "right_pad")}
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
            if self.cup_geom in (a, b):
                touching.update(self.pads.intersection((a, b)))
                on_tray |= self.tray_geom in (a, b)
        return len(touching), on_tray

    def update(self, grip_target, open_m):
        cup = self.data.xpos[self.cup]
        contacts, self.tray_contact = self._contacts()
        self.contacts = contacts
        self.touched |= contacts > 0
        self.grasped |= contacts == 2
        self.lifted |= contacts == 2 and cup[2] > self.start_xyz[2] + self.task.lift_threshold_m
        gx, gy = self.task.goal_xy
        in_tray = np.all(np.abs(cup[:2] - [gx, gy]) + self.radius < [TRAY_HALF[0] - .003, TRAY_HALF[1] - .003])
        upright = self.data.xmat[self.cup, 8] > .93
        resting = (abs(cup[2] - self.height / 2 - self.task.goal_surface_m) < .006
                   and np.linalg.norm(self.data.qvel[-6:-3]) < .018
                   and np.linalg.norm(self.data.qvel[-3:]) < .30)
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