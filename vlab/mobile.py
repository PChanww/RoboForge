"""Independent four-wheel contact-driven mobile robot, using existing Aholo assets.

Runtime motion only changes wheel actuator controls. Free-base pose is written
once at reset; no kinematic base animation or planar position servo is used.
"""
from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

from .duck_demo import CHASSIS, asset_info
from .engine import _jpeg_b64
from .mjcf import mesh_files, robot_xml

ARENA_HALF = 1.25
WHEEL_RADIUS = .028
TRACK_WIDTH = .156
WHEEL_NAMES = ("front_left", "rear_left", "front_right", "rear_right")
MAX_LINEAR = .22
MAX_ANGULAR = 1.8


class MobileEnv:
    def __init__(self, base_config, lab_root: Path, chassis_asset=CHASSIS):
        self.lab_root = Path(lab_root)
        self.asset = asset_info(self.lab_root, chassis_asset, "四轮底盘 · Aholo")
        if self.asset["source"] != "lux3d" or self.asset["kind"] != "robot_visual":
            raise ValueError("请选择 Aholo 机器人底盘外观模型。")
        self.robot = base_config.robots[0].model_copy(deep=True)
        self.robot.links = [l for l in self.robot.links if l.name != "mobile_chassis"]
        self.robot.joints = [j for j in self.robot.joints if j.child != "mobile_chassis"]
        for link in self.robot.links:
            link.gravcomp = False
            if link.name == self.robot.root_link:
                link.origin = [0, 0, .046]
        arm_body, arm_actuators, mesh_names, _ = robot_xml(self.robot, [])
        meshes = mesh_files(base_config, self.robot, mesh_names, self.lab_root)
        meshes["mobile_shell.obj"] = (self.lab_root / "assets" / chassis_asset / "visual.obj").read_bytes()
        bounds = np.asarray(self.asset["bounds_m"], dtype=float)
        extent = bounds[1] - bounds[0]
        if not np.isfinite(bounds).all() or min(extent) <= 0:
            raise ValueError("底盘模型尺寸无效。")
        shell_scale = min(.14 / max(extent[:2]), .065 / extent[2])
        center = bounds.mean(axis=0) * shell_scale
        shell_pos = [-center[0], -center[1], -.024 - bounds[0, 2]*shell_scale]
        mesh_decls = "".join(f'<mesh name="{n[:-4]}" file="{n}"' +
            (f' scale="{shell_scale} {shell_scale} {shell_scale}"' if n == "mobile_shell.obj" else "") + '/>' for n in meshes)
        wheels, drives = [], []
        for name, x, y in (("front_left", .047, TRACK_WIDTH/2), ("rear_left", -.047, TRACK_WIDTH/2),
                           ("front_right", .047, -TRACK_WIDTH/2), ("rear_right", -.047, -TRACK_WIDTH/2)):
            wheels.append(f'''<body name="wheel_{name}" pos="{x} {y} 0">
              <joint name="wheel_{name}" type="hinge" axis="0 1 0" damping=".0008" armature=".00008"/>
              <geom name="tire_{name}" type="cylinder" size="{WHEEL_RADIUS} .012" quat=".70710678 .70710678 0 0"
                mass=".10" rgba=".07 .09 .10 1" friction=".4 .001 .00002" contype="4" conaffinity="1" condim="4"/>
              <geom type="cylinder" size=".012 .0125" quat=".70710678 .70710678 0 0" class="visual" rgba=".55 .69 .65 1"/>
              <geom type="box" size=".024 .0127 .002" class="visual" rgba=".58 .72 .59 1"/>
            </body>''')
            drives.append(f'<velocity name="drive_{name}" joint="wheel_{name}" kv=".10" forcerange="-.25 .25" ctrllimited="true" ctrlrange="-15 15"/>')
        # Walls and all workcell obstacles are real colliders. The route itself
        # is a visual aid and exerts no forces on the robot.
        walls = []
        for name, pos, size in (("north", "0 1.27 .075", "1.29 .02 .075"),
                                ("south", "0 -1.27 .075", "1.29 .02 .075"),
                                ("east", "1.27 0 .075", ".02 1.25 .075"),
                                ("west", "-1.27 0 .075", ".02 1.25 .075")):
            walls.append(f'<geom name="wall_{name}" type="box" pos="{pos}" size="{size}" rgba=".20 .30 .28 1"/>')
        markers = []
        for x, y in ((0, 0), (.6, 0), (.6, .5), (0, .5)):
            markers.append(f'<geom type="cylinder" pos="{x} {y} .001" size=".055 .001" class="visual" rgba=".46 .67 .48 .8"/>')
        xml = f'''<mujoco model="RoboForge mobile contact demo">
          <compiler angle="radian" autolimits="true"/>
          <option timestep=".002" integrator="implicitfast" cone="elliptic" iterations="80" gravity="0 0 -9.81"/>
          <visual><global offwidth="800" offheight="600"/><quality shadowsize="2048"/>
            <map znear=".005" zfar="12"/><headlight ambient=".4 .43 .44" diffuse=".65 .65 .62"/></visual>
          <default><geom friction=".4 .001 .00002" solref=".008 1" solimp=".95 .99 .001"/>
            <default class="visual"><geom contype="0" conaffinity="0" mass="0" group="2"/></default></default>
          <asset>{mesh_decls}<texture name="grid" type="2d" builtin="checker" rgb1=".22 .29 .28" rgb2=".25 .32 .31" width="512" height="512"/>
            <material name="ground" texture="grid" texrepeat="12 12" reflectance=".08"/>
            <material name="rubber" rgba=".13 .16 .18 1"/><material name="steel" rgba=".69 .73 .74 1"/></asset>
          <worldbody><light pos="-.6 -.5 2.5" dir=".2 .2 -1" diffuse=".85 .85 .78"/>
            <geom name="floor" type="plane" size="1.5 1.5 .1" material="ground"/>{''.join(walls)}{''.join(markers)}
            <geom name="storage_box" type="box" pos="-.87 .85 .11" size=".19 .12 .11" rgba=".37 .53 .48 1"/>
            <geom name="storage_crate" type="box" pos=".91 .89 .08" size=".12 .15 .08" rgba=".64 .59 .43 1"/>
            <body name="mobile_base" pos="0 0 .030"><freejoint name="base_free"/>
              <geom name="chassis_contact" type="box" pos="0 0 .014" size=".054 .055 .016" mass="2.8"
                contype="4" conaffinity="1" rgba=".12 .25 .23 1"/>
              <geom name="aholo_shell" class="visual" type="mesh" mesh="mobile_shell" pos="{' '.join(map(str,shell_pos))}" rgba=".42 .54 .51 1"/>
              <geom class="visual" type="box" pos=".057 0 .032" size=".007 .022 .003" rgba=".72 .95 .60 1"/>
              <camera name="onboard" pos=".09 0 .11" xyaxes="0 -1 0 .25 0 .9682458" fovy="68"/>
              {''.join(wheels)}{arm_body}
            </body>
          </worldbody><actuator>{arm_actuators}{''.join(drives)}</actuator>
        </mujoco>'''
        self.model = mujoco.MjModel.from_xml_string(xml, meshes)
        self.data = mujoco.MjData(self.model)
        self._wheel_qpos = [int(self.model.joint("wheel_"+n).qposadr[0]) for n in WHEEL_NAMES]
        self._wheel_dof = [int(self.model.joint("wheel_"+n).dofadr[0]) for n in WHEEL_NAMES]
        self._wheel_ctrl = [self.model.actuator("drive_"+n).id for n in WHEEL_NAMES]
        self._arm_qpos = [int(self.model.joint(a.joint).qposadr[0]) for a in self.robot.actuators]
        self._renderers = {}
        self._images = {}
        self.reset()

    def reset(self):
        self.epoch = getattr(self, "epoch", 0) + 1
        mujoco.mj_resetData(self.model, self.data)
        # Compact transport posture. Gravity and actuator holding forces remain
        # active on the arm; no gravity compensation on the free mobile robot.
        posture = {"waist": 0., "shoulder": -.9, "elbow": 2.35, "wrist_pitch": -1.45, "wrist_yaw": 0., "left": .025, "right": .025}
        for a in self.robot.actuators:
            q = int(self.model.joint(a.joint).qposadr[0])
            joint = next(j for j in self.robot.joints if j.name == a.joint)
            value = posture.get(a.joint, joint.initial)
            self.data.qpos[q] = value
            self.data.ctrl[self.model.actuator(a.joint).id] = value
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_step(self.model, self.data, nstep=200)
        self._t0 = self.data.time
        self.tick = 0
        self._v = self._w = 0.
        self.distance_m = 0.
        self._previous_xy = self.data.qpos[:2].copy()
        self.trajectory = [{"x": float(self.data.qpos[0]), "y": float(self.data.qpos[1])}]
        self._images = {}
        return self.snapshot(render=False)

    def step(self, linear_m_s, angular_rad_s, dt=.05):
        if not np.isfinite([linear_m_s, angular_rad_s, dt]).all() or not .002 <= dt <= .1:
            raise ValueError("驾驶指令或步长无效。")
        wanted_v, wanted_w = np.clip(linear_m_s, -MAX_LINEAR, MAX_LINEAR), np.clip(angular_rad_s, -MAX_ANGULAR, MAX_ANGULAR)
        self._v += float(np.clip(wanted_v-self._v, -.35*dt, .35*dt))
        self._w += float(np.clip(wanted_w-self._w, -3.*dt, 3.*dt))
        left = (self._v-self._w*TRACK_WIDTH/2)/WHEEL_RADIUS
        right = (self._v+self._w*TRACK_WIDTH/2)/WHEEL_RADIUS
        self.data.ctrl[self._wheel_ctrl] = [left, left, right, right]
        mujoco.mj_step(self.model, self.data, nstep=round(dt/.002))
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError("移动机器人出现非有限物理状态。")
        self.tick += 1
        xy = self.data.qpos[:2].copy()
        self.distance_m += float(np.linalg.norm(xy-self._previous_xy))
        self._previous_xy = xy
        if self.tick % 3 == 0:
            self.trajectory.append({"x": float(xy[0]), "y": float(xy[1])})
            self.trajectory = self.trajectory[-1600:]
        return self.snapshot(render=False)

    def state(self):
        return {"time": float(self.data.time-self._t0), "base_pose": self.data.qpos[:7].copy(),
                "base_velocity": self.data.qvel[:6].copy(), "wheel_qpos": self.data.qpos[self._wheel_qpos].copy(),
                "wheel_qvel": self.data.qvel[self._wheel_dof].copy(), "arm_qpos": self.data.qpos[self._arm_qpos].copy()}

    def snapshot(self, render=True):
        q = self.data.qpos[:7]
        # MuJoCo stores freejoint quaternion as wxyz.
        qw, qx, qy, qz = q[3:7]
        yaw = math.atan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz))
        if render:
            self._render()
        forward = np.array([math.cos(yaw), math.sin(yaw)])
        return {"sim_time": round(float(self.data.time-self._t0), 4), "tick": self.tick, "epoch": self.epoch,
                "pose": {"x": float(q[0]), "y": float(q[1]), "z": float(q[2]), "yaw": yaw},
                "velocity": {"linear_m_s": float(np.dot(self.data.qvel[:2], forward)), "angular_rad_s": float(self.data.qvel[5])},
                "command": {"linear_m_s": self._v, "angular_rad_s": self._w},
                "wheel_velocity_rad_s": self.data.qvel[self._wheel_dof].tolist(),
                "trajectory": self.trajectory.copy(), "distance_m": self.distance_m, "images": self._images.copy(),
                "asset": self.asset, "bounds": {"half_extent_m": ARENA_HALF},
                "physics": "free base; four velocity-driven hinge wheels; contact friction; estimated drive geometry and inertia"}

    def _render(self):
        if "front" not in self._renderers:
            self._renderers["front"] = mujoco.Renderer(self.model, width=640, height=480)
            self._renderers["onboard"] = mujoco.Renderer(self.model, width=480, height=360)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = self.data.qpos[:3]+[0, 0, .14]
        cam.distance, cam.azimuth, cam.elevation = 1.25, 135, -31
        for name, camera in (("front", cam), ("onboard", "onboard")):
            renderer = self._renderers[name]
            renderer.update_scene(self.data, camera=camera)
            self._images[name] = "data:image/jpeg;base64,"+_jpeg_b64(renderer.render(), quality=82)

    def close(self):
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()
