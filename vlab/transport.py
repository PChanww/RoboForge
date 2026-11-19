"""Real conveyor pick, wheel-driven navigation and open-bin placement.

Reuses the validated conveyor, mesh contact, gripper and state/camera pipeline.
The mobile base and task object remain unconstrained free rigid bodies.
"""
from __future__ import annotations

import math
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from . import kinematics, scene_builder, tasks
from .duck_conveyor import BIN_FLOOR, ROLLERS, scene_xml
from .duck_demo import CHASSIS, DUCK, asset_info, make_config
from .engine import PlatformEnv
from .mjcf import mesh_files, robot_xml
from .transport_geometry import prepare_object, bin_geometry
from .transport_navigation import plan_navigation
from .transport_tasks import EXTRA_STAGES, TASK_KINDS, TransportProgram

STAGES = ("持续输送", "追踪目标", "跟随接近", "跟随下降", "跟随夹紧", "确认抬升",
          "收拢准备搬运", "导航搬运", "到位转向并停车", "越过目标箱沿", "箱内下降", "松爪释放", "撤离并检查", "抓取或搬运失效")
STAGES += EXTRA_STAGES
WHEELS = ("front_left", "rear_left", "front_right", "rear_right")
RADIUS = .028


class MobileTransportEnv(PlatformEnv):
    def __init__(self, base_config, lab_root: Path, spec: dict, *, render=True):
        self.spec = dict(spec)
        self.task_kind = spec.get("task_kind", "deliver")
        if self.task_kind not in TASK_KINDS:
            raise ValueError("任务支持入箱、拿起放回和入箱后取回。")
        self.speed_scale = float(spec.get("speed_scale", 1.))
        if not math.isfinite(self.speed_scale) or not 1 <= self.speed_scale <= 2:
            raise ValueError("速度倍率支持1～2。")
        self.navigation = None
        if spec.get("target_xy") is not None:
            self.navigation = plan_navigation(spec["target_xy"], spec.get("bin_size_m", [.21,.22,.056]))
            if not self.navigation["ready"]:
                raise ValueError("；".join(self.navigation["issues"]))
        self.retreat_m = float(spec.get("retreat_distance_m", .4))
        if not .30 <= self.retreat_m <= .55:
            raise ValueError("后退距离支持0.30～0.55 m。")
        cfg = make_config(base_config, Path(lab_root), int(spec.get("seed", 3)), spec.get("object_asset", DUCK), spec.get("chassis_asset", CHASSIS))
        cfg.task.duck_conveyor.belt_speed_m_s = float(spec.get("belt_speed_m_s", .025))
        cfg.task.duck_conveyor.continuous = True
        cfg.task.duck_conveyor.random_pose = bool(spec.get("random_pose", True))
        cfg.task.goal_xy = list(spec["target_xy"]) if self.navigation else [.12, .205+self.retreat_m+.235]
        cfg.task.time_limit_s = 180. if self.task_kind != "deliver" else 90.
        cfg.task.instruction = spec.get("instruction", "从不停的传送带上抓取物品，向后移动，放入后方箱子。")
        cfg.scene.physics_dt = .002
        cfg.scene.gravity = [0, 0, -9.81]
        cfg.provenance.update({
            "transport.task": "single contact-only pick, wheel-driven workcell navigation or straight retreat, open-bin placement",
            "transport.base": "6 kg chassis plus arm and four 0.12 kg wheels; unconstrained free rigid body",
            "transport.wheels": "radius 0.028 m; axle distance 0.200 m; track 0.156 m; friction 0.95 0.001 0.00002",
            "transport.drive": "max target 0.065 m/s; acceleration limit 0.12 m/s^2; wheel velocity kv 0.3, torque limit 0.4 Nm",
            "transport.grip": "finger kp 350 N/m, force limit 9 N per side; 0.010 m initial preload; slip >0.003 m closes 0.0005 m/control tick down to 0.006 m; pad friction 2.0 0.005 0.0001",
            "transport.coordinates": "world XY metres; forward -Y and retreat +Y; explicit bin target or legacy rear layout",
            "transport.physics": "MuJoCo implicitfast; dt 0.002 s; elliptic cone; 80 iterations; estimated uncalibrated parameters",
            "transport.expert": "privileged simulation-state controller; contact-gated travel; known static-workcell route planner",
            "robot.base": "unconstrained mobile rigid body driven by four wheel-ground contacts; Aholo chassis visual",
            "transport.control": "fixed 20 Hz state/action collection and control; 2 ms physics substeps",
        })
        cfg.controllers[0].frequency_hz = 20.
        cfg.controllers[0].workspace_min = [-1.5, -.85, .005]
        cfg.controllers[0].workspace_max = [1.5, 1.8, .29]
        for link in cfg.robots[0].links:
            link.gravcomp = False
            if link.name == cfg.robots[0].root_link:
                link.origin = [0, 0, .033]
        for sensor in cfg.sensors:
            sensor.latency_s = sensor.noise_std = 0.
            if sensor.id == "front":
                goal = np.array(cfg.task.goal_xy)
                center = (goal+np.array([-.1,.05]))/2
                extent = max(.65, float(np.linalg.norm(goal-[0,.205])))
                sensor.position = [center[0]-extent, center[1]-extent, extent*.95]
                z = np.array(sensor.position)-np.array([*center, .06])
                z /= np.linalg.norm(z)
                x = np.cross([0, 0, 1], z); x /= np.linalg.norm(x)
                sensor.xyaxes = [*x, *np.cross(z, x)]
                sensor.resolution = [480, 360]
                sensor.intrinsics.fx = sensor.intrinsics.fy = 380
                sensor.intrinsics.cx, sensor.intrinsics.cy = 240, 180
        if not render:
            cfg.sensors = []
        self._holding = False
        self._base_v = 0.
        self.phase = 0
        self._images = {}
        super().__init__(cfg, Path(lab_root))
        self.asset = asset_info(Path(lab_root), cfg.task.asset, "操作物品 · Aholo")
        self.chassis_asset = asset_info(Path(lab_root), cfg.task.duck_conveyor.chassis_asset, "四轮底盘 · Aholo")

    def _build(self, seed):
        cfg = self.config
        r, h, mass, color, mesh, offset, self.object_parameters = prepare_object(self.lab_root, cfg.task.asset, self.spec.get("object_parameters"))
        for key, source in self.spec.get("object_parameter_sources", {}).items():
            if key in self.object_parameters["parameter_sources"] and source in ("user_explicit", "model_inferred", "rules_estimate"):
                self.object_parameters["parameter_sources"][key] = source
        body, actuators, names, cams = robot_xml(self.robot, cfg.sensors)
        env_xml, env_assets, self._env_info = scene_builder.environment_xml(cfg, self.lab_root)
        spawn = tasks.spawn_xy(cfg.task, seed)
        task_xml = scene_xml(cfg.task, offset, color, spawn, seed, mesh, mass)
        scene = ET.fromstring("<root>"+task_xml+"</root>")
        for geom in list(scene):
            if geom.get("name") == "tray" or geom.get("name", "").startswith("bin_wall_"):
                scene.remove(geom)
        bin_asset = self.spec.get("bin_asset") if self.spec.get("bin_mode", "builtin") != "builtin" else None
        bin_xml, bin_assets, self.bin_metadata = bin_geometry(cfg.task.goal_xy, self.spec.get("bin_size_m", [.21,.22,.056]),
            self.spec.get("bin_color", "#2E8F85"), self.lab_root, bin_asset)
        task_xml = "".join(ET.tostring(node, encoding="unicode") for node in scene)+bin_xml
        task_xml += '<geom name="conveyor_frame_contact" type="box" pos="-.1325 -.07 -.036" size=".178 .065 .015" contype="1" conaffinity="4" rgba=".15 .20 .24 1"/>'
        cfg.provenance.update({"transport.object_parameters": json.dumps(self.object_parameters, ensure_ascii=False),
            "transport.bin": json.dumps(self.bin_metadata, ensure_ascii=False),
            "transport.navigation": json.dumps(self.navigation, ensure_ascii=False),
            "transport.speed_scale": str(self.speed_scale)})
        cfg.provenance.update({
            "transport.task": "contact-only pick, waypoint navigation or legacy retreat, custom open-bin placement",
            "transport.wheels": f"radius .028 m; axle .200 m; track .156 m; sliding mu {'.15' if self.navigation else '.95'}",
            "transport.drive": f"max target {.065*self.speed_scale:g} m/s; acceleration {.12*min(self.speed_scale,1.5):g} m/s^2; angular .8 rad/s, angular acceleration 1.2 rad/s^2; wheel differential demand 4x skid compensation; kv .3 torque limit .4 Nm",
            "transport.coordinates": "world XY targets; body forward -Y; differential skid steering; endpoint heading retained",
            "transport.expert": "privileged simulation-state controller; static inflated-obstacle grid route; contact-gated transport",
            "transport.placement_heading": "hold arm during wheel-driven alignment to face the bin; stop before reaching; retain physical joint ranges",
            "transport.ground": f"z=-.061 m; sliding mu {'.15' if self.navigation else '.95'}; uncalibrated demonstration values",
        })
        actuators += "".join(f'<velocity name="belt_drive_{i}" joint="roller_joint_{i}" kv=".3" forcerange="-1 1"/>' for i in range(ROLLERS))
        wheels = []
        for name, x, y in (("front_left", -.078, -.100), ("rear_left", -.078, .100),
                           ("front_right", .078, -.100), ("rear_right", .078, .100)):
            wheels.append(f'''<body name="transport_wheel_{name}" pos="{x} {y} 0">
              <joint name="transport_wheel_{name}" type="hinge" axis="1 0 0" damping=".0008" armature=".00008"/>
              <geom name="transport_tire_{name}" type="cylinder" quat=".70710678 0 .70710678 0" size="{RADIUS} .012"
                mass=".12" friction="{.15 if self.navigation else .95} .001 .00002" contype="4" conaffinity="1" rgba=".06 .09 .10 1"/>
              <geom class="visual" type="box" size=".0125 .023 .002" rgba=".56 .72 .57 1"/>
            </body>''')
            actuators += f'<velocity name="transport_drive_{name}" joint="transport_wheel_{name}" kv=".3" forcerange="-.4 .4"/>'
        body = f'''<body name="transport_base" pos="0 .205 -.033"><freejoint name="transport_free"/>
            <geom name="transport_chassis" type="box" pos="0 0 .010" size=".055 .095 .020" mass="6.0"
              contype="4" conaffinity="1" rgba=".14 .29 .27 1"/>{''.join(wheels)}{body}</body>'''
        meshes = mesh_files(cfg, self.robot, names, self.lab_root)
        meshes.update(env_assets); meshes.update(bin_assets); meshes["cup.obj"] = mesh
        declarations = "".join(f'<mesh name="{n[:-4]}" file="{n}"/>' for n in meshes if n != "cup.obj")+'<mesh name="cup_mesh" file="cup.obj"/>'
        xml = f'''<mujoco model="RoboForge mobile conveyor transport"><compiler angle="radian" autolimits="true"/>
          <option timestep=".002" integrator="implicitfast" cone="elliptic" iterations="80" gravity="0 0 -9.81"/>
          <visual><global offwidth="800" offheight="600"/><quality shadowsize="2048"/><map znear=".005" zfar="8"/>
            <headlight ambient=".32 .34 .37" diffuse=".49 .49 .48" specular=".1 .1 .1"/></visual>
          <default><geom friction=".95 .005 .0001" solref=".006 1" solimp=".95 .99 .001"/>
            <default class="visual"><geom contype="0" conaffinity="0" group="2" mass="0"/></default></default>
          <asset>{declarations}<material name="rubber" rgba=".13 .16 .18 1"/><material name="steel" rgba=".69 .73 .74 1"/></asset>
          <worldbody>{cams}{env_xml}{task_xml}{body}</worldbody><actuator>{actuators}</actuator></mujoco>'''
        self._asset_radius, self._asset_height, self._cup_offset = r, h, offset
        model = mujoco.MjModel.from_xml_string(xml, meshes)
        if self.navigation:
            for gid in range(model.ngeom):
                if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_PLANE:
                    model.geom_friction[gid,0] = .15
        model.geom_friction[model.geom("cup_contact").id,0] = self.object_parameters["friction"]
        self._base_q = int(model.joint("transport_free").qposadr[0])
        self._base_d = int(model.joint("transport_free").dofadr[0])
        self._wheel_q = [int(model.joint("transport_wheel_"+n).qposadr[0]) for n in WHEELS]
        self._wheel_d = [int(model.joint("transport_wheel_"+n).dofadr[0]) for n in WHEELS]
        self._wheel_ctrl = [model.actuator("transport_drive_"+n).id for n in WHEELS]
        return model

    def _frame(self, point):
        """World point -> (rotation, body root, point in the chassis frame)."""
        base = self.data.qpos[self._base_q:self._base_q+7]
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, base[3:])
        rotation = rotation.reshape(3, 3)
        root = self.model.body(self.robot.root_link).pos
        return rotation, root, rotation.T @ (np.asarray(point, dtype=float)-base[:3])-root

    def _ik(self, point):
        return kinematics.analytic_ik(self._frame(point)[2], kinematics.chain_params(self.robot))

    def _reach_project(self, point):
        """Project in the chassis frame: the arm rides a free base.

        A world target reachable at one moment becomes unreachable once the
        wheels move, because the local offset grows. Projecting in the frame the
        analytic solver actually uses, then mapping back to world, keeps the
        stored target executable instead of raising a millimetre overshoot.
        """
        rotation, root, local = self._frame(point)
        projected, info = kinematics.clamp_to_reach(local, kinematics.chain_params(self.robot))
        if info is None:
            return np.asarray(point, dtype=float), None
        base = self.data.qpos[self._base_q:self._base_q+7]
        return rotation @ (projected+root)+base[:3], dict(info, frame="chassis_local")

    def reset(self, seed=None):
        self._holding, self._base_v, self._base_w = False, 0., 0.
        self.phase = 0
        super().reset(seed)
        self._base_start = self.data.qpos[self._base_q:self._base_q+7].copy()
        half = np.asarray(self.bin_metadata["inner_half_m"])
        def inside_custom_bin():
            low, high = self._judge.bounds()
            center = np.asarray(self.config.task.goal_xy)
            return bool(np.all(low[:2] > center-half) and np.all(high[:2] < center+half))
        self._judge.inside = inside_custom_bin
        self._route = self.navigation["planned_route"] if self.navigation else [[0.,.205],[0.,.205+self.retreat_m]]
        self._route_index, self._travelled = 1, 0.
        self._previous_xy = self._base_start[:2].copy()
        self._trajectory = [self._previous_xy.tolist()]
        self._lost = self._dwell = 0
        self._transported = False
        self._applied = [0., 0., 0., 1., 0., 0., 0.]
        self._applied_constraint = None
        self._release_pose = np.zeros(3)
        self._grasp_reference_z = None
        self._images = {}
        self._program = TransportProgram(self)
        return self.snapshot(render=False)

    def _apply(self, action):
        if self._holding:
            # Wheels carry the entire held arm. A fixed world Cartesian target
            # would incorrectly counteract the base motion and stretch the arm.
            self._target = self.data.xpos[self._eef].copy()
            self._servo_velocity[:] = 0
            self._grip_target = self._closed+(action[3]+1)*.5*(self._open-self._closed)
            g = self.robot.gripper
            self.data.ctrl[self._ctrl_index[g.left_joint]] = self._grip_target
            self.data.ctrl[self._ctrl_index[g.right_joint]] = self._grip_target
            # Held arm: no IK target is commanded, so the effective target is
            # the measured tool position. Still recorded so every step carries
            # the same three-layer evidence.
            measured = [float(v) for v in self.data.xpos[self._eef]]
            self._constraint = {"requested_target": measured, "target": measured,
                                "limited": False, "detail": None}
        else:
            super()._apply(action)

    def _fail(self):
        self.phase = 13
        self._duck_task.failed = True
        self._judge.delivery_failed = True

    def _yaw(self):
        w,x,y,z = self.data.qpos[self._base_q+3:self._base_q+7]
        return math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))

    def _drive_route(self):
        xy = self.data.qpos[self._base_q:self._base_q+2]
        point = np.asarray(self._route[self._route_index])
        delta = point-xy
        distance = float(np.linalg.norm(delta))
        last = self._route_index == len(self._route)-1
        if distance < (.012 if last else .045):
            if last:
                self.phase = 8
                return 0., 0.
            self._route_index += 1
            delta = np.asarray(self._route[self._route_index])-xy
            distance = float(np.linalg.norm(delta))
        desired_yaw = math.atan2(delta[1],delta[0])+math.pi/2
        error = math.atan2(math.sin(desired_yaw-self._yaw()), math.cos(desired_yaw-self._yaw()))
        sign = 1.
        if abs(error) > math.pi/2:
            error = math.atan2(math.sin(error+math.pi),math.cos(error+math.pi))
            sign = -1.
        angular = float(np.clip(error*2., -.8, .8))
        v = min(.065*self.speed_scale, distance*1.3)*max(0.,math.cos(error))
        if abs(error) > .30:
            v = 0.
        return sign*v, angular

    def expert_action(self):
        self._program.before_expert()
        if self.phase >= 14:
            return self._program.expert_action()
        eef = self.data.xpos[self._eef]
        cup = self.data.xpos[self._judge.cup]
        grip = self._duck_task.hold_width or self._closed
        hold = False
        drive = 0.
        turn = 0.
        if 5 <= self.phase <= 10 and self._grasp_reference_z is not None:
            # Respond to measured slip by gradually increasing finger preload.
            # Only the motor target changes; contact forces retain the object.
            slip = self._grasp_reference_z-float(cup[2]-eef[2])
            if slip > .003 and self._judge.contacts == 2:
                self._duck_task.hold_width = max(self._closed, grip-.0005)
                grip = self._duck_task.hold_width
        if self.phase < 6:
            task, judge, p = self._duck_task, self._judge, self.phase
            velocity = np.clip(self.data.qvel[judge.velocity:judge.velocity+3], -.06, .06).copy()
            velocity[2] = 0
            follow = cup+velocity*.08
            grab_z = max(float(cup[2]), .031)
            waypoint = np.array([follow[0], follow[1], .14 if p < 3 else grab_z])
            if p == 0:
                self.phase = 1
            elif p == 1:
                self.phase = 2
            elif p == 2 and np.linalg.norm(eef[:2]-cup[:2]) < .009 and abs(eef[2]-.14) < .005:
                self.phase = 3
            elif p == 3 and np.linalg.norm(eef-waypoint) < .008:
                self.phase = 4
            elif p == 4:
                g = self.robot.gripper
                mean = (self.data.qpos[self._qposadr[g.left_joint]]+self.data.qpos[self._qposadr[g.right_joint]])/2
                if judge.contacts == 2 and task.hold_width is None:
                    # Maintain spring preload through the longer mobile carry.
                    # With the 350 N/m finger servos this requests about 3.5 N
                    # per side, still bounded by their physical force limits.
                    task.hold_width = max(self._closed, mean-.010)
                self._dwell = self._dwell+1 if judge.contacts == 2 else 0
                if self._dwell >= 5:
                    task.grasp_eef, task.grasp_offset = eef.copy(), cup-eef
                    self._grasp_reference_z = float(task.grasp_offset[2])
                    self.phase, self._dwell = 5, 0
            elif p == 5:
                waypoint = np.r_[task.grasp_eef[:2], .18]
                self._lost = self._lost+1 if judge.contacts < 2 else 0
                if self._lost >= 3:
                    self._fail()
                elif np.linalg.norm(eef-waypoint) < .005 and np.linalg.norm(self._servo_velocity) < .025:
                    self.phase = 6
            task.phase = self.phase
            grip = (task.hold_width or self._closed) if p >= 4 else self._open
            action = self._smooth_expert_action(waypoint, (.15,.22,.22,.09,.07,.10)[p]*self.speed_scale, grip,
                                               feedforward=velocity if p <= 4 else None)
            return {"arm": action, "base_linear_m_s": 0., "hold_arm": False}
        if 6 <= self.phase <= 10:
            self._lost = self._lost+1 if self._judge.contacts < 2 else 0
            if self._lost >= 3:
                self._fail()
        if 6 <= self.phase <= 10 and self._judge.contacts == 2:
            self._duck_task.grasp_offset = .85*self._duck_task.grasp_offset+.15*(cup-eef)
        base = self.data.qpos[self._base_q:self._base_q+3]
        base_speed = np.linalg.norm(self.data.qvel[self._base_d:self._base_d+3])
        offset = self._duck_task.grasp_offset
        drop = max(BIN_FLOOR+(cup[2]-self._judge.bounds()[0][2])+.002-offset[2], .038)
        dest = np.array([*self.config.task.goal_xy, drop])-np.r_[offset[:2], 0.]
        waypoint, speed = self._target.copy(), .08
        if self.phase == 6:
            waypoint = base+np.array([.10, -.11, .233])
            if np.linalg.norm(eef-waypoint) < .005 and np.linalg.norm(self._servo_velocity) < .025:
                self.phase, self._dwell = 7, 0
                self._target = eef.copy()
        elif self.phase == 7:
            hold = True
            if self.navigation:
                drive, turn = self._drive_route()
            else:
                remaining = self._base_start[1]+self.retreat_m-base[1]
                drive = -min(.065*self.speed_scale, max(0., remaining*1.2))
                if remaining < .009:
                    self.phase = 8
            self._transported = True
        elif self.phase == 8:
            hold = True
            aligned = True
            if self.navigation:
                # Keep the box in the arm's front workspace before extending.
                # Reaching sideways behind the shoulder can cross atan2's
                # +/-pi branch and demand an entire turn from a limited joint.
                delta = np.asarray(self.config.task.goal_xy)-base[:2]
                desired_yaw = math.atan2(delta[1], delta[0])+math.pi/2
                error = math.atan2(math.sin(desired_yaw-self._yaw()), math.cos(desired_yaw-self._yaw()))
                aligned = abs(error) < .025
                turn = float(np.clip(error*1.3, -.55, .55)) if not aligned else 0.
            self._dwell = self._dwell+1 if aligned and base_speed < .008 and abs(self.data.qvel[self._base_d+5]) < .04 else 0
            if self._dwell >= 5:
                self.phase = 9
                self._target, self._servo_velocity = eef.copy(), np.zeros(3)
        elif self.phase == 9:
            waypoint = np.r_[dest[:2], .19]
            speed = .10
            if np.linalg.norm(eef-waypoint) < .005 and np.linalg.norm(self._servo_velocity) < .025:
                self.phase = 10
        elif self.phase == 10:
            waypoint, speed = dest, .04
            arrived = np.linalg.norm(eef-waypoint) < .005 and np.linalg.norm(self._servo_velocity) < .02
            if (arrived or self._judge.tray_contact) and self._judge.contacts == 2 and self._judge.inside():
                if self._judge.bounds()[0][2] > BIN_FLOOR-.001:
                    self._judge.controlled_release = True
                    self._release_pose = self._target.copy()
                    self.phase, self._dwell = 11, 0
        elif self.phase == 11:
            waypoint, grip, speed = self._release_pose, self._open, .025
            self._dwell += 1
            if self._dwell > 18:
                self.phase = 12
        elif self.phase == 12:
            waypoint, grip, speed = np.r_[self._release_pose[:2], .19], self._open, .10
        else:
            hold, drive = True, 0.
        action = self._smooth_expert_action(waypoint, speed*self.speed_scale, grip)
        return {"arm": action, "base_linear_m_s": drive, "base_angular_rad_s": turn, "hold_arm": hold}

    def step(self, action):
        arm = self.validate_action(action["arm"])
        requested = float(action.get("base_linear_m_s", 0.))
        requested_w = float(action.get("base_angular_rad_s",0.))
        if not math.isfinite(requested) or not math.isfinite(requested_w):
            raise ValueError("底盘指令含非有限数值。")
        desired = float(np.clip(requested, -.065*self.speed_scale, .065*self.speed_scale))
        desired_w = float(np.clip(requested_w,-.8,.8))
        if self.phase == 13:
            desired = 0.
            desired_w = 0.
        acceleration = .12*min(self.speed_scale,1.5)
        self._base_v += float(np.clip(desired-self._base_v, -acceleration*.05, acceleration*.05))
        self._base_w += float(np.clip(desired_w-self._base_w,-1.2*.05,1.2*.05))
        # Four fixed wheels must skid laterally to turn. Compensate the
        # estimated skid loss with wheel-speed demand, retaining torque limits.
        steering = self._base_w*4.0
        left = (self._base_v+steering*.156/2)/RADIUS
        right = (self._base_v-steering*.156/2)/RADIUS
        self.data.ctrl[self._wheel_ctrl] = [left,left,right,right]
        self._holding = bool(action.get("hold_arm", False))
        self._program.before_step(arm)
        super().step(arm)
        self._program.observe()
        self._applied = [*arm.tolist(), self._base_v, self._base_w, float(self._holding)]
        self._applied_constraint = self._constraint
        xy = self.data.qpos[self._base_q:self._base_q+2].copy()
        self._travelled += float(np.linalg.norm(xy-self._previous_xy))
        self._previous_xy = xy
        if self.scheduler.tick % 3 == 0:
            self._trajectory.append(xy.tolist())
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError("移动抓取出现非有限物理状态。")
        return self.snapshot(render=False)

    def metrics(self):
        result = super().metrics()
        distance = float(self.data.qpos[self._base_q+1]-self._base_start[1])
        result.update(retreated_distance_m=distance, failed=self.phase == 13,
                      travelled_distance_m=self._travelled,
                      goal_distance_m=float(np.linalg.norm(self.data.qpos[self._base_q:self._base_q+2]-self._route[-1])),
                      phase=self.phase, phase_label=STAGES[self.phase], task="mobile_conveyor_transport")
        moved = result["goal_distance_m"] < .06 if self.navigation else distance >= self.retreat_m-.03
        result["success"] = bool(result["success"] and self._transported and moved and not result["delivery_failed"])
        if self.task_kind != "deliver":
            result.update(self._program.metrics())
            result["success"] = bool(self._program.complete and not result["delivery_failed"])
        else:
            result.update(self._program.metrics())
            result["program_complete"] = result["success"]
            result["delivered_to_bin"] = result["success"]
        result["terminated"] = result["success"] or result["failed"]
        result["reach_limited_steps"] = self._reach_limited
        return result

    def state(self):
        oq = int(self.model.joint("cup_free").qposadr[0])
        od = int(self.model.joint("cup_free").dofadr[0])
        return {"time": self.episode_time(), "base_pose": self.data.qpos[self._base_q:self._base_q+7].copy(),
                "base_velocity": self.data.qvel[self._base_d:self._base_d+6].copy(),
                "wheel_qpos": self.data.qpos[self._wheel_q].copy(), "wheel_qvel": self.data.qvel[self._wheel_d].copy(),
                "arm_qpos": np.array([self.data.qpos[self._qposadr[n]] for n in self._joints]),
                "object_pose": self.data.qpos[oq:oq+7].copy(), "object_velocity": self.data.qvel[od:od+6].copy(),
                "belt_speed_m_s": self._duck_task.speed, "phase": self.phase, "contacts": self._judge.contacts}

    def snapshot(self, render=True):
        observation = self.observation()
        images = {name: "data:image/jpeg;base64,"+row["jpeg_b64"] for name, row in observation["images"].items() if row is not None}
        base = self.data.qpos[self._base_q:self._base_q+7]
        w, x, y, z = base[3:]
        metrics = self.metrics()
        return {"tick": self.scheduler.tick, "sim_time": round(self.episode_time(), 5), "epoch": self.epoch,
                "images": images, "phase": self.phase, "phase_label": STAGES[self.phase], "metrics": metrics,
                "pose": {"x": float(base[0]), "y": float(base[1]), "z": float(base[2]), "yaw": math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))},
                "base_velocity": self.data.qvel[self._base_d:self._base_d+6].tolist(),
                "transport_distance_m": metrics["retreated_distance_m"], "belt_speed_m_s": self._duck_task.speed,
                "planned_route": self._route, "target_xy": self.config.task.goal_xy, "trajectory": self._trajectory[-1800:],
                "distance_m": self._travelled, "goal_distance_m": metrics["goal_distance_m"], "speed_scale": self.speed_scale,
                "applied_action": self._applied, "camera_status": observation["camera_status"],
                "arm_constraint": self._applied_constraint}
