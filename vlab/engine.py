"""Config-driven MuJoCo environment with multi-rate scheduling and versioned I/O.

Joints and actuators are addressed by name; observation/action dimensions come
from the controller config, never from a fixed array length.
"""
from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

import arm

from . import kinematics, scene_builder, tasks
from .mjcf import mesh_files, robot_xml
from .rendering.snapshot import capture_snapshot
from .scheduler import CameraBuffer, Scheduler
from .schema import ProjectConfig

SCHEMA_VERSION = "1.0"


def _jpeg_b64(frame, quality=86):
    buf = BytesIO()
    Image.fromarray(frame).save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class UnsupportedMode(ValueError):
    pass


class PlatformEnv:
    def __init__(self, config: ProjectConfig, lab_root: Path, robot_id=None, controller_id=None):
        self.config = config
        self.lab_root = Path(lab_root)
        self.robot = config.robots[0] if robot_id is None else next(r for r in config.robots if r.id == robot_id)
        self.controller = (config.controllers[0] if controller_id is None
                           else next(c for c in config.controllers if c.id == controller_id))
        self._renderers = {}
        self._seg_renderers = {}
        self._sensor_errors = {}
        self._hybrid = None
        self.model = self.data = None
        self._buffers = {}
        self._state = None
        self.epoch = 0  # single owner of the episode generation; bumped in reset()
        self.reset(config.scene.seed)

    # -- model construction -------------------------------------------------
    def _has_5r_chain(self):
        names = {j.name for j in self.robot.joints}
        return {"waist", "shoulder", "elbow", "wrist_pitch", "wrist_yaw"} <= names

    def _ik(self, point):
        if self.controller.mode != "cartesian_delta_ik":
            raise UnsupportedMode(f"robot {self.robot.id} has no analytic IK for {self.controller.mode}")
        if not self._has_5r_chain():
            raise UnsupportedMode(f"robot {self.robot.id} lacks the fixed-attitude 5R chain")
        return kinematics.analytic_ik(point, kinematics.chain_params(self.robot))

    def supported_modes(self):
        modes = ["joint_position"]
        if self._has_5r_chain():
            modes.append("cartesian_delta_ik")
        return modes

    def validate_kinematics(self, targets=None):
        if not self._has_5r_chain():
            return {"consistent": False, "reason": "no 5R chain"}
        if targets is None:
            targets = [[-.16, .095, .148], [-.11, -.066, .041], [.133, .05, .05],
                       [-.11, -.066, .129], [.5, .5, .3]]
        return kinematics.validate(self.model, self.data, self.robot, self._eef, targets, self._qposadr)

    def _build(self, seed):
        cfg = self.config
        if cfg.scene.environment and not cfg.scene.use_default_table and not any(c.enabled for c in cfg.scene.colliders):
            raise UnsupportedMode("Imported tabletop scene requires a default table or explicit static colliders; visual meshes do not provide contact")
        r, h, mass, color, mesh, offset, handle = tasks.asset_params(cfg.task.asset, self.lab_root)
        spawn = tasks.spawn_xy(cfg.task, seed)
        self._hybrid_ply = None
        if cfg.scene.sensor_renderer == "hybrid":
            from .rendering.hybrid import find_gaussian_ply
            if cfg.scene.environment is None:
                raise UnsupportedMode("hybrid renderer requires a bound gaussian scene environment")
            ply = find_gaussian_ply(self.lab_root, cfg.scene.environment)
            if ply is None:
                raise UnsupportedMode("hybrid renderer requires a gaussian (.ply) scene; mesh scenes use mujoco")
            if not (cfg.scene.use_default_table or any(c.enabled for c in cfg.scene.colliders)):
                raise UnsupportedMode("hybrid scene needs a default table or enabled static colliders for physics")
            self._hybrid_ply = ply
        body, actuators, mesh_names, world_cams = robot_xml(self.robot, cfg.sensors)
        env_xml, env_assets, self._env_info = scene_builder.environment_xml(cfg, self.lab_root)
        task_scene = tasks.task_xml(cfg.task, r, h, mass, color, offset, handle, spawn, support_z=cfg.scene.work_surface_z)
        mesh_assets = mesh_files(cfg, self.robot, mesh_names, self.lab_root)
        mesh_assets.update(env_assets)
        mesh_assets["cup.obj"] = mesh
        mesh_scales = self._env_info.get("mesh_scales", {})
        def _decl(n):
            if n == "scene_env.obj" and cfg.scene.environment:
                s = cfg.scene.environment.uniform_scale
                return f'<mesh name="scene_env" file="{n}" scale="{s:.6g} {s:.6g} {s:.6g}" inertia="shell"/>'
            if n in mesh_scales:
                s = mesh_scales[n]
                return f'<mesh name="{n[:-4]}" file="{n}" scale="{s:.6g} {s:.6g} {s:.6g}" inertia="shell"/>'
            return f'<mesh name="{n[:-4]}" file="{n}" inertia="shell"/>'
        mesh_decls = "".join(_decl(n) for n in mesh_assets if n != "cup.obj")
        mesh_decls += '<mesh name="cup_mesh" file="cup.obj" inertia="shell"/>'
        dt = cfg.scene.physics_dt
        grav = " ".join(str(v) for v in cfg.scene.gravity)
        xml = f"""
        <mujoco model="platform scene">
          <compiler angle="radian" autolimits="true"/>
          <option timestep="{dt}" integrator="implicitfast" cone="elliptic" iterations="65" gravity="{grav}"/>
          <visual><quality shadowsize="2048"/><global offwidth="1280" offheight="960"/>
            <map znear=".005" zfar="8"/>
            <headlight ambient=".32 .34 .37" diffuse=".49 .49 .48" specular=".1 .1 .1"/></visual>
          <statistic center="0 0 .07" extent=".62"/>
          <default>
            <geom friction=".95 .005 .0001" solref=".006 1" solimp=".95 .99 .001"/>
            <default class="visual"><geom contype="0" conaffinity="0" group="2" mass="0"/></default>
          </default>
          <asset>{mesh_decls}
            <material name="rubber" rgba=".13 .16 .18 1"/>
            <material name="steel" rgba=".69 .73 .74 1" specular=".45" shininess=".6"/>
          </asset>
          <worldbody>{world_cams}{env_xml}{task_scene}{body}</worldbody>
          <actuator>{actuators}</actuator>
        </mujoco>"""
        self._asset_radius, self._asset_height, self._cup_offset = r, h, offset
        return mujoco.MjModel.from_xml_string(xml, mesh_assets)

    # -- lifecycle ----------------------------------------------------------
    def reset(self, seed=None):
        seed = self.config.scene.seed if seed is None else int(seed)
        self.epoch = getattr(self, "epoch", 0) + 1  # before initial capture
        self.close()
        self.model = self._build(seed)
        self.data = mujoco.MjData(self.model)
        self.seed = seed
        self._joints = [a.joint for a in self.robot.actuators]
        self._ctrl_index = {a.joint: i for i, a in enumerate(self.robot.actuators)}
        self._qposadr = {n: int(self.model.joint(n).qposadr[0]) for n in self._joints}
        self._dofadr = {n: int(self.model.joint(n).dofadr[0]) for n in self._joints}
        self._eef = self.model.body(self.robot.end_effector).id
        g = self.robot.gripper
        self._open, self._closed = (g.open_m, g.closed_m) if g else (0, 0)
        self._target = np.array([-0.16, 0.095, 0.148])
        self._grip_target = self._open
        self._last_action = [0.0] * self.controller.action_dim
        self._servo_phase = 0
        self._servo_dwell = 0
        self._ik_error = None
        self._render_error = None
        if not hasattr(self, "_hybrid"):
            self._hybrid = None  # world-fixed gaussian background: cached across resets
        self._sensor_errors = {}
        self._seg_renderers = {}
        if self.controller.mode == "cartesian_delta_ik":
            ik = self._ik(self._target)
            for name, q in zip(("waist", "shoulder", "elbow", "wrist_pitch", "wrist_yaw"), ik):
                self.data.qpos[self._qposadr[name]] = q
        else:
            for j in self.robot.joints:
                if j.type != "fixed":
                    self.data.qpos[self._qposadr[j.name]] = j.initial
        if g:
            self.data.qpos[self._qposadr[g.left_joint]] = self._open
            self.data.qpos[self._qposadr[g.right_joint]] = self._open
        self.data.ctrl[:] = [self.data.qpos[self._qposadr[n]] for n in self._joints]
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_step(self.model, self.data, nstep=80)
        self._t0 = self.data.time
        fps = {s.id: s.fps for s in self.config.sensors}
        self.scheduler = Scheduler(self.config.scene.physics_dt, self.controller.frequency_hz, fps)
        self._latency = {s.id: s.latency_s for s in self.config.sensors}
        self._judge = tasks.Judge(self.model, self.data, self.config.task, self._asset_radius, self._asset_height)
        self._buffers = {s.id: CameraBuffer() for s in self.config.sensors}
        self._capture_all(0.0)
        self._sample_state(0.0)
        return self.observation()

    def episode_time(self):
        return self.data.time - self._t0

    def _step_to(self, episode_target):
        target = self._t0 + episode_target
        while self.data.time < target - 1e-9:
            mujoco.mj_step(self.model, self.data)

    def _render_camera(self, sensor, width=None, height=None):
        w, h = sensor.resolution if width is None else (width, height)
        key = (w, h)
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(self.model, height=h, width=w)
        renderer = self._renderers[key]
        renderer.update_scene(self.data, camera=sensor.id)
        rgb = renderer.render()
        if self.config.scene.sensor_renderer != "hybrid":
            return rgb
        if sensor.id != "front":
            raise UnsupportedMode("static_background_overlay only supports the world-fixed front camera")
        from .rendering.hybrid import foreground_ids
        hybrid = self._hybrid_renderer(sensor, w, h)
        skey = (w, h, "seg")
        if skey not in self._seg_renderers:
            r = mujoco.Renderer(self.model, height=h, width=w)
            r.enable_segmentation_rendering()
            self._seg_renderers[skey] = r
        segr = self._seg_renderers[skey]
        segr.update_scene(self.data, camera=sensor.id)
        seg = segr.render()
        comp, _ = hybrid.compose(rgb, seg, foreground_ids(self))
        return comp

    @property
    def render_mode(self):
        from .rendering.hybrid import HYBRID_MODE
        return HYBRID_MODE if self.config.scene.sensor_renderer == "hybrid" else "mujoco"

    def _hybrid_renderer(self, sensor, w, h):
        """Background renderer keyed by camera pose/intrinsics, size, asset hash,
        instance transform and renderer version. Rebuilt after reset/camera changes."""
        from .rendering.hybrid import HybridRenderer
        cam_id = self.model.camera(sensor.id).id
        k = sensor.intrinsics
        sx, sy = w / sensor.resolution[0], h / sensor.resolution[1]
        cam = {"world_pos": np.asarray(self.data.cam_xpos[cam_id]).copy(),
               "world_mat": np.asarray(self.data.cam_xmat[cam_id]).reshape(3, 3).copy(),
               "fx": k.fx * sx, "fy": k.fy * sy, "cx": k.cx * sx, "cy": k.cy * sy,
               "width": w, "height": h}
        env = self.config.scene.environment
        inst = {"position": env.position, "quat_xyzw": env.quaternion_xyzw,
                "uniform_scale": env.uniform_scale} if env else None
        from .rendering.hybrid import _cache_key
        key = _cache_key(self._hybrid_ply, cam, inst, w, h)
        if self._hybrid is None or getattr(self, "_hybrid_key", None) != key:
            self._hybrid = HybridRenderer(self._hybrid_ply, width=w, height=h, inst=inst, camera=cam)
            self._hybrid_key = key
        return self._hybrid

    def _push_frame(self, sensor, episode_time, snap=None):
        if self._render_error:
            return  # headless/no-GL: physics still runs, cameras report unavailable
        # Capture an immutable snapshot at the sampling instant, then render from it.
        if snap is None:
            snap = capture_snapshot(self, episode_time)
        cam = snap["cameras"][sensor.id]
        try:
            frame = self._render_camera(sensor)
        except UnsupportedMode as exc:  # e.g. wrist under hybrid: report, don't disable all
            self._sensor_errors[sensor.id] = str(exc)
            return
        except Exception as exc:  # GLFW/GL unavailable in a detached process
            self._render_error = f"{type(exc).__name__}: {exc}"
            return
        self._buffers[sensor.id].push({
            "capture_time": round(episode_time, 6),
            "available_time": round(episode_time + self._latency[sensor.id], 6),
            "width": sensor.resolution[0], "height": sensor.resolution[1],
            "request_id": snap["request_id"], "snapshot_epoch": snap["epoch"],
            "session_id": snap["session_id"],
            "camera_world_pos": list(cam["world_pos"]),
            "camera_world_mat": [list(r) for r in cam["world_mat"]],
            "jpeg_b64": _jpeg_b64(frame)})
        self.scheduler.record_capture(sensor.id, episode_time)

    def _capture_all(self, episode_time):
        snap = capture_snapshot(self, episode_time)
        for s in self.config.sensors:
            self._push_frame(s, episode_time, snap)

    def _sample_state(self, episode_time):
        pos = {n: float(self.data.qpos[self._qposadr[n]]) for n in self._joints}
        vel = {n: float(self.data.qvel[self._dofadr[n]]) for n in self._joints}
        eef_pos = self.data.xpos[self._eef].tolist()
        eef_mat = self.data.xmat[self._eef].reshape(3, 3).tolist()
        grip = None
        if self.robot.gripper:
            g = self.robot.gripper
            mean = (self.data.qpos[self._qposadr[g.left_joint]] + self.data.qpos[self._qposadr[g.right_joint]]) / 2
            grip = float(np.clip((mean - self._closed) / (self._open - self._closed), 0, 1))
        body_poses = {}
        for link in self.robot.links:
            bid = self.model.body(link.name).id
            body_poses[link.name] = {"position": self.data.xpos[bid].tolist(),
                                     "matrix": self.data.xmat[bid].reshape(3, 3).tolist()}
        cup = self.model.body("cup").id
        object_pose = {"position": self.data.xpos[cup].tolist(),
                       "matrix": self.data.xmat[cup].reshape(3, 3).tolist(),
                       "visual_offset": [0, 0, self._cup_offset],
                       "radius_m": float(self._asset_radius),
                       "height_m": float(self._asset_height)}
        dynamic_objects = {}
        cyl_type = int(mujoco.mjtGeom.mjGEOM_CYLINDER)
        for node in self.config.scene.asset_nodes:
            if node.role == "dynamic":
                nbid = self.model.body(f"node_{node.node_id}").id
                cols = []
                prefix = f"node_{node.node_id}_col"
                for gid in range(self.model.ngeom):
                    if int(self.model.geom_bodyid[gid]) != nbid:
                        continue
                    gname = self.model.geom(gid).name or ""
                    if not gname.startswith(prefix):
                        continue
                    gtype = int(self.model.geom_type[gid])
                    gs = self.model.geom_size[gid]
                    if gtype == cyl_type:
                        shape = "cylinder"
                        size_m = [float(gs[0] * 2), float(gs[0] * 2), float(gs[1] * 2)]
                    else:
                        shape = "box"
                        size_m = [float(gs[0] * 2), float(gs[1] * 2), float(gs[2] * 2)]
                    cols.append({
                        "shape": shape,
                        "size_m": size_m,
                        "position": self.model.geom_pos[gid].tolist(),
                    })
                dynamic_objects[node.node_id] = {
                    "position": self.data.xpos[nbid].tolist(),
                    "matrix": self.data.xmat[nbid].reshape(3, 3).tolist(),
                    "colliders": cols,
                }
        self._state = {"time": round(episode_time, 6), "joint_pos": pos, "joint_vel": vel,
                       "eef_position": eef_pos, "eef_matrix": eef_mat, "gripper": grip,
                       "body_poses": body_poses, "object_pose": object_pose,
                       "dynamic_objects": dynamic_objects}

    # -- action / step ------------------------------------------------------
    def validate_action(self, action):
        c = self.controller
        a = np.asarray(action, dtype=float)
        if a.shape != (c.action_dim,):
            raise ValueError(f"action dim {a.shape} != expected ({c.action_dim},) for mode {c.mode}")
        if not np.isfinite(a).all():
            raise ValueError("action contains non-finite values")
        for i, (lo, hi) in enumerate(c.action_range):
            if not lo - 1e-6 <= a[i] <= hi + 1e-6:
                raise ValueError(f"action[{i}]={a[i]} outside [{lo},{hi}] ({c.action_names[i]}, {c.units[i]})")
        return a

    def _apply(self, a):
        c = self.controller
        if c.mode == "cartesian_delta_ik":
            new_target = np.clip(self._target + c.max_delta_m * a[:3],
                                 c.workspace_min, c.workspace_max)
            self._grip_target = self._closed + (a[3] + 1) * .5 * (self._open - self._closed)
            try:
                ik = self._ik(new_target)
                self._target = new_target
                self._ik_error = None
            except kinematics.Unreachable as exc:
                self._ik_error = exc.reason  # keep last valid control target
                ik = self._ik(self._target)
            for name, q in zip(("waist", "shoulder", "elbow", "wrist_pitch", "wrist_yaw"), ik):
                self.data.ctrl[self._ctrl_index[name]] = q
            if self.robot.gripper:
                g = self.robot.gripper
                self.data.ctrl[self._ctrl_index[g.left_joint]] = self._grip_target
                self.data.ctrl[self._ctrl_index[g.right_joint]] = self._grip_target
        elif c.mode == "joint_position":
            for name, q in zip(c.action_names[:-1], a[:-1]):
                self.data.ctrl[self._ctrl_index[name]] = q
            if self.robot.gripper:
                g = self.robot.gripper
                self._grip_target = float(a[-1])
                self.data.ctrl[self._ctrl_index[g.left_joint]] = self._grip_target
                self.data.ctrl[self._ctrl_index[g.right_joint]] = self._grip_target
        else:
            raise UnsupportedMode(c.mode)

    def step(self, action):
        a = self.validate_action(action)
        self._last_action = a.tolist()
        self._apply(a)
        t_start = self.episode_time()
        t_end = t_start + self.scheduler.control_period
        events = []
        for s in self.config.sensors:
            events.extend((t, s) for t in self.scheduler.capture_times_in(t_start, t_end, s.id))
        for t in sorted({t for t, _ in events}):
            self._step_to(t)
            snap = capture_snapshot(self, self.episode_time())  # shared by same-instant cameras
            for _, s in [e for e in events if e[0] == t]:
                self._push_frame(s, self.episode_time(), snap)
        self._step_to(t_end)
        self._sample_state(self.episode_time())
        self.scheduler.record_control(self.episode_time())
        self._judge.update(self._grip_target, self._open)
        return self.observation()

    # -- observation / metrics ---------------------------------------------
    def observation(self, decision_time=None):
        t = self.episode_time() if decision_time is None else decision_time
        frames, status = {}, {}
        elapsed = max(t, 1e-6)
        for s in self.config.sensors:
            buf = self._buffers[s.id]
            f = buf.latest_available(t)
            frames[s.id] = f
            captures = self.scheduler.capture_count[s.id]
            serr = self._sensor_errors.get(s.id) or self._render_error
            status[s.id] = {
                "state": ("render_unavailable" if serr
                          else "ok" if f else ("waiting" if buf.frames else "no_frame")),
                "render_error": serr,
                "frame_id": f["frame_id"] if f else None,
                "capture_time": f["capture_time"] if f else None,
                "available_time": f["available_time"] if f else None,
                "age_s": round(t - f["capture_time"], 6) if f else None,
                "configured_fps": s.fps,
                "actual_capture_fps": round(captures / elapsed, 3) if t > 1e-6 else None,
                "pending_frames": buf.pending(t),
                "evicted_frames": buf.evicted,
                "queue_len": len(buf.frames),
            }
        return {"schema_version": SCHEMA_VERSION, "robot": self.robot.id,
                "controller": self.controller.id, "control_mode": self.controller.mode,
                "render_mode": self.render_mode,
                "config_version": self.config.version, "tick": self.scheduler.tick,
                "sim_time": round(t, 6), "seed": self.seed,
                "state": self._state, "images": frames, "camera_status": status,
                "timing": self.scheduler.timing_summary(t, frames),
                "instruction": self.config.task.instruction}

    def metrics(self):
        m = self._judge.metrics(self.scheduler.tick, self.scheduler.control_period)
        m["terminated"] = bool(m["success"])
        m["truncated"] = self.episode_time() >= self.config.task.time_limit_s
        return m

    def expert_action(self):
        """Privileged scripted baseline (reads initial object pose and goal)."""
        if self.controller.mode != "cartesian_delta_ik":
            raise UnsupportedMode("expert baseline requires cartesian_delta_ik")
        source = self._judge.start_xyz
        goal = np.array(self.config.task.goal_xy)
        ws_z = float(self.config.scene.work_surface_z)
        hover = min(.176, ws_z + self._asset_height / 2 + .090)
        grab = ws_z + self._asset_height / 2 + .002
        gs = self.config.task.goal_surface_m
        drop = gs + self._asset_height / 2 + .002
        hover_goal = min(.176, max(hover, gs + self._asset_height / 2 + .040))
        phases = ((24, source[:2], hover, 1), (44, source[:2], grab, 1),
                  (60, source[:2], grab, -1), (86, source[:2], hover_goal, -1),
                  (126, goal, hover_goal, -1), (151, goal, drop, -1),
                  (165, goal, drop, 1))
        for end_tick, xy, z, grip in phases:
            if self.scheduler.tick < end_tick:
                break
        else:
            xy, z, grip = goal, hover, 1
        delta = np.clip((np.r_[xy, z] - self._target) / self.controller.max_delta_m, -1, 1)
        return [*delta.tolist(), grip]

    def servo_expert_action(self):
        """Closed-loop position-servo expert: actions are a function of the MEASURED
        end-effector pose (and object pose), with phases switched by position/contact
        rather than by tick. This makes the demonstration imitable in closed loop."""
        if self.controller.mode != "cartesian_delta_ik":
            raise UnsupportedMode("servo expert requires cartesian_delta_ik")
        eef = self.data.xpos[self._eef]
        cup = self.data.xpos[self._judge.cup]
        goal = np.asarray(self.config.task.goal_xy)
        ws_z = float(self.config.scene.work_surface_z)
        h, gs = self._asset_height, self.config.task.goal_surface_m
        hover, grab = min(.176, ws_z + h / 2 + .09), ws_z + h / 2 + .002
        lift, drop = min(.176, max(ws_z + h / 2 + .10, gs + h / 2 + .04)), gs + h / 2 + .002
        tol = .005
        g = self.robot.gripper
        mean = (self.data.qpos[self._qposadr[g.left_joint]] + self.data.qpos[self._qposadr[g.right_joint]]) / 2
        grip_norm = float(np.clip((mean - self._closed) / (self._open - self._closed), 0, 1))
        p = self._servo_phase
        waypoints = [(cup[0], cup[1], hover), (cup[0], cup[1], grab), (cup[0], cup[1], grab),
                     (cup[0], cup[1], lift), (goal[0], goal[1], lift), (goal[0], goal[1], drop),
                     (goal[0], goal[1], drop), (goal[0], goal[1], lift)]
        grips = [1, 1, -1, -1, -1, -1, 1, 1]
        wp, grip = np.asarray(waypoints[p]), grips[p]
        if p in (0, 1, 3, 4, 5) and np.linalg.norm(eef - wp) < tol:
            self._servo_phase = p + 1
        elif p == 2:
            self._servo_dwell = self._servo_dwell + 1 if self._judge.contacts == 2 else 0
            if self._servo_dwell >= 3:
                self._servo_phase, self._servo_dwell = 3, 0
        elif p == 6:
            self._servo_dwell = self._servo_dwell + 1 if grip_norm > .8 else 0
            if self._servo_dwell >= 3:
                self._servo_phase, self._servo_dwell = 7, 0
        delta = np.clip((wp - eef) / self.controller.max_delta_m, -1, 1)
        return [*delta.tolist(), grip]

    def render(self, camera="front", width=800, height=540):
        """Preview rendering: uses exactly the same backend/scene source as observation."""
        if camera not in self._buffers:
            raise ValueError(f"unknown camera {camera}")
        sensor = next(s for s in self.config.sensors if s.id == camera)
        return self._render_camera(sensor, width, height)

    def close(self):
        for r in self._renderers.values():
            r.close()
        self._renderers = {}
        for r in getattr(self, "_seg_renderers", {}).values():
            r.close()
        self._seg_renderers = {}
