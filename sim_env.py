"""A deterministic MuJoCo desktop pick-and-place task. All lengths are metres."""

from __future__ import annotations

import json
import math
from pathlib import Path

import arm
import mujoco
import numpy as np


BUILTINS = {  # radius, height, mass, diffuse colour
    "standard": (.031, .078, .090, ".09 .48 .54"),
    "tall": (.026, .104, .085, ".82 .38 .16"),
    "wide": (.041, .062, .110, ".43 .48 .76"),
}
GOAL_XY = np.array([.133, .050])
GOAL_SURFACE = .009
MAX_STEP = .012
OPEN, CLOSED = .058, .006
CONTROL_STEPS = 25


def _cup_mesh(radius, height):
    """Open-top, double-walled procedural cup, origin at centre of mass proxy."""
    rings = [(radius * .79, -height / 2), (radius, height / 2),
             (radius - .003, height / 2), (radius * .79 - .003, -.44 * height)]
    vertices, faces = [], []
    n = 40
    for r, z in rings:
        for i in range(n):
            a = 2 * math.pi * i / n
            vertices.append(f"v {r * math.cos(a)} {r * math.sin(a)} {z}")
    for k in range(3):
        for i in range(n):
            a, b = k * n + i + 1, k * n + (i + 1) % n + 1
            faces.extend((f"f {a} {b} {b+n}", f"f {a} {b+n} {a+n}"))
    for i in range(1, n - 1):
        faces.extend((f"f 1 {i+2} {i+1}", f"f {3*n+1} {3*n+i+1} {3*n+i+2}"))
    return ("\n".join(vertices + faces) + "\n").encode()


def _load_asset(asset):
    if asset in BUILTINS:
        radius, height, mass, color = BUILTINS[asset]
        # The handle points along X; parallel jaws approach along Y.
        handle = []
        last = None
        for theta in np.linspace(-np.pi / 2, np.pi / 2, 11):
            point = (radius * .91 + radius * .57 * np.cos(theta),
                     0, height * .25 * np.sin(theta))
            if last is not None:
                endpoints = " ".join(f"{v:.6f}" for v in (*last, *point))
                handle.append(f'<geom class="visual" type="capsule" fromto="{endpoints}" size=".0035" rgba="{color} 1"/>')
            last = point
        return radius, height, mass, color, _cup_mesh(radius, height), 0., "".join(handle)
    path = Path(asset)
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    radius, height, mass = (float(manifest[k]) for k in ("radius_m", "height_m", "mass_kg"))
    if not (0 < radius < 1 and 0 < height < 1 and 0 < mass < 10):
        raise ValueError("Asset radius, height and mass are outside the desktop simulation range")
    visual = (path.parent / manifest.get("obj", "visual.obj")).resolve()
    if not visual.is_relative_to(path.parent.resolve()):
        raise ValueError("Visual file must be inside the asset package")
    rgb = np.asarray(manifest.get("color_rgb", [.20, .60, .63]), dtype=float)
    if rgb.shape != (3,) or not np.isfinite(rgb).all() or np.any((rgb < 0) | (rgb > 1)):
        raise ValueError("color_rgb must contain three finite colour channels in [0,1]")
    color = " ".join(f"{v:.6f}" for v in rgb)
    return radius, height, mass, color, visual.read_bytes(), -height / 2, ""


def _scene(radius, height, mass, color, offset, handle, spawn, spec):
    fingers = []
    for s, tag in ((1, "left"), (-1, "right")):
        fingers.append(f"""
        <body name="{tag}" pos="0 {s*.006} 0">
          <joint name="{tag}" type="slide" axis="0 {s} 0" range="0 .065" damping="4"/>
          <geom type="box" size=".014 .006 .022" material="rubber" mass=".035" friction="1.8 .005 .0001"
                name="{tag}_pad" solref=".004 1"/>
          <geom class="visual" type="box" pos="0 0 .036" size=".012 .005 .016" material="steel"/>
        </body>""")
    gx, gy = GOAL_XY
    return f"""
    <mujoco model="articulated tabletop">
      <compiler angle="radian" autolimits="true"/>
      <option timestep=".002" integrator="implicitfast" cone="elliptic" iterations="65" gravity="0 0 -9.81"/>
      <visual>
        <quality shadowsize="2048"/><global offwidth="1280" offheight="960"/>
        <map znear=".005" zfar="8"/>
        <headlight ambient=".32 .34 .37" diffuse=".49 .49 .48" specular=".1 .1 .1"/>
      </visual>
      <statistic center="0 0 .07" extent=".62"/>
      <default>
        <geom friction=".95 .005 .0001" solref=".006 1" solimp=".95 .99 .001"/>
        <default class="visual"><geom contype="0" conaffinity="0" group="2" mass="0"/></default>
      </default>
      <asset>
        {''.join(f'<mesh name="{n}" file="{n}.obj"/>' for n in ('base','upper','fore','wrist'))}
        <mesh name="cup_mesh" file="cup.obj"/>
        <material name="bench" rgba=".29 .37 .39 1" specular=".12" shininess=".2"/>
        <material name="steel" rgba=".69 .73 .74 1" specular=".45" shininess=".6"/>
        <material name="rubber" rgba=".13 .16 .18 1"/>
        <material name="accent" rgba=".86 .49 .16 1"/>
        <material name="tray" rgba=".63 .69 .65 1"/>
      </asset>
      <worldbody>
        <light pos="-.4 -.3 .85" dir=".25 .25 -1" diffuse=".69 .65 .59" castshadow="true"/>
        <light pos=".5 .4 .6" dir="-.6 -.3 -1" diffuse=".44 .48 .56" castshadow="false"/>
        <camera name="front" pos=".36 -.70 .46" xyaxes=".895 .445 0 -.16 .32 .93" fovy="40"/>
        <geom type="plane" size="2 2 .01" pos="0 0 -.061" rgba=".19 .23 .25 1"/>
        <geom name="table" type="box" pos="0 0 -.028" size=".32 .215 .028" material="bench"/>
        <geom name="tray" type="box" pos="{gx} {gy} {GOAL_SURFACE/2}" size=".074 .069 {GOAL_SURFACE/2}" material="tray"/>
        <geom type="box" pos="{gx-.073} {gy} .011" size=".002 .069 .004" material="tray"/>
        <geom type="box" pos="{gx+.073} {gy} .011" size=".002 .069 .004" material="tray"/>
        <geom type="box" pos="{gx} {gy-.068} .011" size=".073 .002 .004" material="tray"/>
        <geom type="box" pos="{gx} {gy+.068} .011" size=".073 .002 .004" material="tray"/>
        {arm.arm_xml(fingers,spec)}
        <body name="cup" pos="{spawn[0]} {spawn[1]} {height/2+.001}">
          <freejoint name="cup_free"/>
          <geom name="cup_contact" type="cylinder" size="{radius} {height/2}" mass="{mass}"
                group="3" rgba=".2 .5 .6 0" friction="1.8 .005 .0001"/>
          <geom class="visual" type="mesh" mesh="cup_mesh" pos="0 0 {offset}" rgba="{color} 1"/>
          {handle}
        </body>
      </worldbody>
      <actuator>
        {arm.actuators(spec)}
        <position name="left" joint="left" kp="350" kv="4" forcerange="-9 9"/>
        <position name="right" joint="right" kp="350" kv="4" forcerange="-9 9"/>
      </actuator>
    </mujoco>"""


class TabletopEnv:
    """Four actions: incremental X/Y/Z and an absolute parallel jaw opening."""

    def __init__(self, asset="standard", seed=0, candidate=None):
        self.spec = arm.parameters(candidate)
        self._renderer = None
        self._render_size = None
        self.asset, self.seed = str(asset), int(seed)
        self.reset()

    def reset(self, asset=None, seed=None):
        asset = self.asset if asset is None else str(asset)
        seed = self.seed if seed is None else int(seed)
        r, h, mass, color, mesh, offset, handle = _load_asset(asset)
        rng = np.random.default_rng(seed)
        spawn = rng.uniform([-.154, -.081], [-.085, -.026])
        new_model = mujoco.MjModel.from_xml_string(
            _scene(r, h, mass, color, offset, handle, spawn, self.spec), {"cup.obj": mesh, **arm.mesh_files()})
        self.close()
        self.model, self.data = new_model, mujoco.MjData(new_model)
        self.asset, self.seed, self.radius, self.height = asset, seed, r, h
        self._cup = self.model.body("cup").id
        self._eef = self.model.body("eef").id
        self._cup_geom = self.model.geom("cup_contact").id
        self._tray_geom = self.model.geom("tray").id
        self._pads = {self.model.geom(n).id for n in ("left_pad", "right_pad")}
        self._target = np.array([-.16, .095, .148])
        self._grip_target = OPEN
        self.data.qpos[:7] = [*arm.ik(self._target), OPEN, OPEN]
        self.data.ctrl[:] = self.data.qpos[:7]
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_step(self.model, self.data, nstep=80)
        self._start_xyz = self.data.xpos[self._cup].copy()
        self.tick, self._stable_ticks = 0, 0
        self._lifted, self._touched, self._grasped, self._success = False, False, False, False
        self._contacts, self._tray_contact = 0, False
        return self.observation()

    def observation(self):
        return {"robot": "lux3d_5r", "joints_rad": self.data.qpos[:5].tolist(), "eef": self.data.xpos[self._eef].tolist(),
                "gripper": float(np.clip((self.data.qpos[5:7].mean() - CLOSED) / (OPEN - CLOSED), 0, 1)),
                "tick": self.tick, "instruction": "Place the cup in the tray."}

    def _object_contacts(self):
        touching = set()
        on_tray = False
        for contact in self.data.contact:
            a, b = int(contact.geom1), int(contact.geom2)
            if self._cup_geom in (a, b):
                touching.update(self._pads.intersection((a, b)))
                on_tray |= self._tray_geom in (a, b)
        return len(touching), on_tray

    def step(self, action):
        a = np.asarray(action, dtype=float)
        if a.shape != (4,) or not np.isfinite(a).all() or np.any(np.abs(a) > 1.000001):
            raise ValueError("Expected four finite actions in [-1,1] (dx,dy,dz,open)")
        self._target = np.clip(self._target + MAX_STEP * a[:3], [-.28, -.17, .022], [.28, .17, .184])
        self._grip_target = CLOSED + (a[3] + 1) * .5 * (OPEN - CLOSED)
        self.data.ctrl[:] = [*arm.ik(self._target), self._grip_target, self._grip_target]
        mujoco.mj_step(self.model, self.data, nstep=CONTROL_STEPS)
        self.tick += 1
        cup = self.data.xpos[self._cup]
        contacts, self._tray_contact = self._object_contacts()
        self._contacts = contacts
        self._touched |= contacts > 0
        self._grasped |= contacts == 2
        self._lifted |= contacts == 2 and cup[2] > self._start_xyz[2] + .031
        in_tray = np.all(np.abs(cup[:2] - GOAL_XY) + self.radius < [.071, .066])
        upright = self.data.xmat[self._cup, 8] > .93
        resting = (abs(cup[2] - self.height / 2 - GOAL_SURFACE) < .006
                   and np.linalg.norm(self.data.qvel[-6:-3]) < .018
                   and np.linalg.norm(self.data.qvel[-3:]) < .30)
        released = contacts == 0 and self._grip_target > OPEN - .009
        self._stable_ticks = (self._stable_ticks + 1 if self._lifted and self._tray_contact
                              and in_tray and upright and resting and released else 0)
        # Episode success is latched once six consecutive settled control ticks pass.
        self._success |= self._stable_ticks >= 6
        return self.observation()

    def metrics(self):
        return {"object_xyz": self.data.xpos[self._cup].tolist(),
                "goal_xyz": [*GOAL_XY.tolist(), GOAL_SURFACE],
                "touched": self._touched, "grasped": self._grasped,
                "lifted": bool(self._lifted), "success": bool(self._success),
                "contacts": self._contacts, "tray_contact": self._tray_contact,
                "time_s": self.tick * CONTROL_STEPS * .002,
                "tick": self.tick, "stable_ticks": self._stable_ticks}

    def expert_action(self):
        """Privileged time-based baseline. It knows the initial object pose and goal."""
        source = self._start_xyz
        hover = min(.176, self.height / 2 + .090)
        grab = self.height / 2 + .002
        phases = ((24, source[:2], hover, 1), (44, source[:2], grab, 1),
                  (60, source[:2], grab, -1), (86, source[:2], hover, -1),
                  (126, GOAL_XY, hover, -1), (151, GOAL_XY, grab + GOAL_SURFACE, -1),
                  (165, GOAL_XY, grab + GOAL_SURFACE, 1))
        for end_tick, xy, z, grip in phases:
            if self.tick < end_tick:
                break
        else:
            xy, z, grip = GOAL_XY, hover, 1
        delta = np.clip((np.r_[xy, z] - self._target) / MAX_STEP, -1, 1)
        return [*delta.tolist(), grip]

    def render(self, width=640, height=480, camera="front"):
        width, height = int(width), int(height)
        if camera not in ("front", "wrist") or not (1 <= width <= 1280 and 1 <= height <= 960):
            raise ValueError("Expected front/wrist camera and render size up to 1280x960")
        if self._renderer is None or self._render_size != (width, height):
            self.close()
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
            self._render_size = width, height
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
            self._render_size = None
