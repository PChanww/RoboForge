"""Contact-driven duck sorting with a kinematic slat belt and a free rigid duck.

Only belt mocap bodies and robot actuator targets are commanded during a run.
The duck's freejoint pose is set once at reset and thereafter owned by MuJoCo.
Collision uses the generated mesh's convex hull; mass/friction remain estimates.
"""
from __future__ import annotations

import math
import mujoco
import numpy as np

SLATS = 40
ROLLERS = 26
LEFT, RIGHT, ROLLER = -.29, .025, .018
LOOP = 2 * (RIGHT - LEFT) + 2 * math.pi * ROLLER
BIN_HALF = np.array([.099, .104])
BIN_FLOOR, BIN_RIM = .009, .065
STAGES = ("持续输送", "追踪目标", "跟随接近", "跟随下降", "跟随夹紧", "确认抬升", "越过箱沿", "箱内下降", "箱内松爪", "撤离", "抓取失效")


def initial_pose(seed, random_pose=True):
    rng = np.random.default_rng(seed)
    yaw = float(rng.uniform(-math.pi, math.pi)) if random_pose else 0.
    roll = (0., .45, -.45, math.pi / 2, -math.pi / 2)[seed % 5] if random_pose else 0.
    pitch = float(rng.uniform(-.18, .18)) if random_pose else 0.
    q = np.empty(4)
    mujoco.mju_euler2Quat(q, np.array([roll, pitch, yaw]), "xyz")
    return q


def scene_xml(task, mesh_offset, color, spawn, seed, mesh, mass):
    gx, gy = task.goal_xy
    q = initial_pose(seed, task.duck_conveyor.random_pose)
    vertices = np.array([[float(v) for v in line.split()[1:4]]
                         for line in mesh.decode().splitlines() if line.startswith("v ")])
    vertices[:, 2] += mesh_offset
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, q)
    z = -float(np.min(vertices @ rotation.reshape(3, 3)[2])) + .003
    fmt = lambda v: " ".join(f"{x:.7g}" for x in v)
    parts = [f'''<body name="cup" pos="{spawn[0]} {spawn[1]} {z}" quat="{fmt(q)}">
      <freejoint name="cup_free"/>
      <geom name="cup_contact" type="mesh" mesh="cup_mesh" pos="0 0 {mesh_offset}" mass="{mass}" group="3" rgba="1 .8 0 0" condim="4" friction="1.2 .003 .0001" solref=".008 1"/>
      <geom class="visual" type="mesh" mesh="cup_mesh" pos="0 0 {mesh_offset}" rgba="{color} 1"/>
    </body>''']
    # Real open box: floor and four side walls, no invisible grasp attachment.
    parts.append(f'<geom name="tray" type="box" pos="{gx} {gy} .0045" size=".105 .110 .0045" rgba=".14 .48 .45 1"/>')
    for i, (dx, dy, sx, sy) in enumerate(((-.102, 0, .003, .110), (.102, 0, .003, .110), (0, -.107, .105, .003), (0, .107, .105, .003))):
        parts.append(f'<geom name="bin_wall_{i}" type="box" pos="{gx+dx} {gy+dy} .037" size="{sx} {sy} .028" rgba=".18 .56 .52 1" solref=".008 1"/>')
    for dx in (-.093, .093):
        for dy in (-.098, .098):
            parts.append(f'<geom type="box" pos="{gx+dx} {gy+dy} -.0305" size=".007 .007 .0305" rgba=".22 .29 .30 1"/>')
    parts.append('<geom type="box" pos="-.1325 -.07 -.036" size=".178 .065 .015" rgba=".15 .20 .24 1" contype="0" conaffinity="0"/>')
    for x in (-.26, -.005):
        for y in (-.115, -.025):
            parts.append(f'<geom type="box" pos="{x} {y} -.052" size=".009 .009 .020" rgba=".45 .51 .55 1" contype="0" conaffinity="0"/>')
    for i in range(SLATS):
        parts.append(f'<body name="belt_{i}" mocap="true"><geom name="belt_surface_{i}" type="box" size="{LOOP/SLATS/2+.0005} .056 .002" contype="0" conaffinity="0" mass="0" rgba="{(".22 .30 .34" if i%5 else ".35 .50 .50")} 1"/></body>')
    # A dense driven-roller contact proxy supplies actual tangential velocity.
    # Mocap visual slats alone have no frictional velocity in MuJoCo.
    for i, x in enumerate(np.linspace(LEFT, RIGHT, ROLLERS)):
        parts.append(f'<body name="roller_{i}" pos="{x} -.07 -.010"><joint name="roller_joint_{i}" type="hinge" axis="0 1 0" damping=".002" armature=".0001"/><geom name="roller_contact_{i}" type="cylinder" size=".010 .056" quat=".7071068 .7071068 0 0" mass=".05" contype="2" conaffinity="1" group="3" rgba="0 0 0 0" friction="1.2 .0001 .00001" solref=".015 1"/></body>')
    return "".join(parts)


class DuckTask:
    def __init__(self, env):
        self.env = env
        self.phase = 0
        self.distance = self.speed = 0.
        self.dwell = 0
        self.hold_width = None
        self.grasp_eef = np.zeros(3)
        self.grasp_offset = np.zeros(3)
        self.release_pose = np.zeros(3)
        self.failed = False
        self.lost_ticks = 0
        self.mocap = [env.model.body(f"belt_{i}").mocapid[0] for i in range(SLATS)]
        self.drives = [env.model.actuator(f"belt_drive_{i}").id for i in range(ROLLERS)]
        self.initial_quat = initial_pose(env.seed, env.config.task.duck_conveyor.random_pose)
        self.move_belt(0.)

    def move_belt(self, dt):
        cfg = self.env.config.task.duck_conveyor
        desired = cfg.belt_speed_m_s if cfg.continuous or self.phase == 0 else 0.
        self.speed += float(np.clip(desired - self.speed, -.35 * dt, .35 * dt))
        self.distance += self.speed * dt
        self.env.data.ctrl[self.drives] = self.speed / .010
        straight = RIGHT - LEFT
        for i, mid in enumerate(self.mocap):
            s = (i * LOOP / SLATS + self.distance) % LOOP
            if s < straight:
                x, z, angle = LEFT + s, -.002, 0.
            elif s < straight + math.pi * ROLLER:
                angle = (s - straight) / ROLLER
                x, z = RIGHT + ROLLER * math.sin(angle), -.002 - ROLLER + ROLLER * math.cos(angle)
            elif s < 2 * straight + math.pi * ROLLER:
                x, z, angle = RIGHT - (s - straight - math.pi * ROLLER), -.002 - 2 * ROLLER, math.pi
            else:
                angle = math.pi + (s - 2 * straight - math.pi * ROLLER) / ROLLER
                x, z = LEFT + ROLLER * math.sin(angle), -.002 - ROLLER + ROLLER * math.cos(angle)
            self.env.data.mocap_pos[mid] = [x, -.07, z]
            self.env.data.mocap_quat[mid] = [math.cos(angle/2), 0, math.sin(angle/2), 0]

    def expert_action(self):
        e = self.env
        judge = e._judge
        cup = e.data.xpos[judge.cup]
        eef = e.data.xpos[e._eef]
        gx, gy = e.config.task.goal_xy
        g = e.robot.gripper
        mean = float((e.data.qpos[e._qposadr[g.left_joint]] + e.data.qpos[e._qposadr[g.right_joint]]) / 2)
        p = self.phase
        if 5 <= p <= 7 and judge.contacts == 2:
            self.grasp_offset = .85 * self.grasp_offset + .15 * (cup - eef)
        velocity = np.clip(e.data.qvel[judge.velocity:judge.velocity + 3], -.06, .06).copy()
        velocity[2] = 0
        follow = cup + velocity * .08
        low, high = judge.bounds()
        # Open inside the box with the finger pads above the floor. The object
        # settles the last few millimetres under gravity, never by pose writes.
        drop = max(BIN_FLOOR + (cup[2] - low[2]) + .002 - self.grasp_offset[2], .038)
        dest = np.array([gx, gy, drop])
        dest[:2] -= self.grasp_offset[:2]
        waypoints = [e._target, [follow[0], follow[1], .14],
                     [follow[0], follow[1], .14], [follow[0], follow[1], cup[2]],
                     [follow[0], follow[1], cup[2]], [*self.grasp_eef[:2], .17],
                     [*dest[:2], .17], dest, self.release_pose, [*self.release_pose[:2], .17], e._target]
        wp = np.array(waypoints[p], dtype=float)
        arrived = np.linalg.norm(eef - wp) < .004 and np.linalg.norm(e._servo_velocity) < .025
        if 5 <= p <= 7:
            self.lost_ticks = self.lost_ticks + 1 if judge.contacts < 2 else 0
            if self.lost_ticks >= 3:
                self.failed, self.phase = True, 10
                judge.delivery_failed = True
                return e._smooth_expert_action(e._target, .04, self.hold_width or e._closed)
        if p == 0:
            self.phase = 1
        elif p == 1:
            self.phase = 2
        elif p == 2 and np.linalg.norm(eef[:2] - cup[:2]) < .009 and abs(eef[2] - .14) < .005:
            self.phase = 3
        elif p == 3 and np.linalg.norm(eef - cup) < .007:
            self.phase = 4
        elif p == 4:
            if judge.contacts == 2 and self.hold_width is None:
                self.hold_width = max(e._closed, mean - .003)
            self.dwell = self.dwell + 1 if judge.contacts == 2 else 0
            if self.dwell >= 5:
                self.grasp_eef = eef.copy()
                self.grasp_offset = cup - eef
                self.phase, self.dwell = 5, 0
        elif p in (5, 6) and arrived:
            self.phase += 1
        elif p == 7 and (arrived or judge.tray_contact) and judge.contacts == 2:
            if judge.inside() and low[2] > BIN_FLOOR - .001:
                judge.controlled_release = True
                self.release_pose = e._target.copy()
                self.phase = 8
        elif p == 8:
            self.dwell = self.dwell + 1 if mean > e._open - .005 else 0
            if self.dwell >= 3:
                self.phase, self.dwell = 9, 0
        grip = (self.hold_width if self.hold_width is not None else e._closed) if 4 <= p <= 7 or p == 10 else e._open
        return e._smooth_expert_action(wp, (.15, .22, .22, .10, .08, .12, .12, .045, .025, .12, .02)[p],
                                      grip, feedforward=velocity if 1 <= p <= 4 else None)


class DuckJudge:
    def __init__(self, model, data, task, radius, height):
        self.model, self.data, self.task = model, data, task
        self.cup = model.body("cup").id
        self.object_geoms = {model.geom("cup_contact").id}
        gid = model.geom("cup_contact").id
        mesh = model.geom_dataid[gid]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        self.vertices = model.mesh_vert[start:start + count].copy()
        self.geom = gid
        self.controlled_release = self.delivery_failed = False
        self.pads = {model.geom(n).id for n in ("left_pad", "right_pad")}
        self.bottom = model.geom("tray").id
        self.velocity = model.joint("cup_free").dofadr[0]
        self.start_xyz = data.xpos[self.cup].copy()
        self.start_quat = data.xquat[self.cup].copy()
        self.contacts = self.stable_ticks = 0
        self.touched = self.grasped = self.lifted = self.success = self.tray_contact = False

    def bounds(self):
        points = self.vertices @ self.data.geom_xmat[self.geom].reshape(3, 3).T + self.data.geom_xpos[self.geom]
        return points.min(axis=0), points.max(axis=0)

    def inside(self):
        low, high = self.bounds()
        center = np.asarray(self.task.goal_xy)
        return bool(np.all(low[:2] > center - BIN_HALF) and np.all(high[:2] < center + BIN_HALF))

    def update(self, grip_target, open_m):
        pads, bottom = set(), False
        for contact in self.data.contact:
            a, b = int(contact.geom1), int(contact.geom2)
            if self.object_geoms.intersection((a, b)):
                pads.update(self.pads.intersection((a, b)))
                bottom |= self.bottom in (a, b)
        self.contacts, self.tray_contact = len(pads), bottom
        cup = self.data.xpos[self.cup]
        self.touched |= self.contacts > 0
        self.grasped |= self.contacts == 2
        self.lifted |= self.contacts == 2 and cup[2] > self.start_xyz[2] + self.task.lift_threshold_m
        in_box = self.inside()
        stationary = np.linalg.norm(self.data.qvel[self.velocity:self.velocity+6]) < .08
        released = self.contacts == 0 and grip_target > open_m - .009
        self.stable_ticks = self.stable_ticks+1 if self.lifted and self.controlled_release and not self.delivery_failed and bottom and in_box and stationary and released else 0
        self.success |= self.stable_ticks >= self.task.stable_ticks

    def metrics(self, tick, dt):
        return {"object_xyz": self.data.xpos[self.cup].tolist(), "goal_xyz": [*self.task.goal_xy, .009],
                "touched": bool(self.touched), "grasped": bool(self.grasped), "lifted": bool(self.lifted),
                "success": bool(self.success), "contacts": self.contacts, "tray_contact": bool(self.tray_contact),
                "stable_ticks": self.stable_ticks, "controlled_release": self.controlled_release, "delivery_failed": self.delivery_failed, "time_s": tick*dt, "tick": tick}
