"""Contact-only extensions for lift/return and deliver/retrieve/return tasks.

The program changes actuator goals only. Task completion is observed from the
free object's contacts and motion after physical release onto the live belt.
"""
from __future__ import annotations

import math
import numpy as np


EXTRA_STAGES = (
    "入箱后抬爪", "箱内物品上方", "箱内下降接近", "箱内二次夹紧", "箱内二次抬升",
    "收拢准备返程", "沿原路线返程", "传送带前转向停车", "原抓取点上方", "放回传送带下降",
    "传送带上松爪", "撤离并检查返还", "全部子任务完成",
)
TASK_KINDS = {"deliver", "pick_return", "bin_return", "bin_retrieve"}


class TransportProgram:
    def __init__(self, env):
        self.env = env
        self.kind = env.spec.get("task_kind", "deliver")
        self.vla = env.spec.get("controller", "expert") == "vla"
        if self.kind not in TASK_KINDS:
            raise ValueError("任务支持入箱、拿起放回和入箱后取回。")
        self.events = []
        self.pickup_xy = None
        self.delivered = self.regrasped = self.returned = self.complete = False
        self.retrieved = False
        self.dwell = self.lost = self.stable = 0
        self.release = None
        self.release_armed = False
        self.release_error = None
        self.release_belt_distance = None
        self.route = [list(p) for p in reversed(env._route)]
        self.route_index = min(1, len(self.route)-1)
        self.route_errors = []
        self.roller_ids = {env.model.geom(f"roller_contact_{i}").id for i in range(26)}

    def event(self, stage, **evidence):
        if any(row["stage"] == stage for row in self.events):
            return
        e = self.env
        self.events.append({"stage": stage, "tick": e.scheduler.tick,
                            "time_s": round(e.episode_time(), 6), **evidence})

    def observe(self):
        e, j = self.env, self.env._judge
        cup = e.data.xpos[j.cup]
        if self.pickup_xy is None and j.contacts == 2 and (e.phase >= 5 or self.vla):
            self.pickup_xy = cup[:2].copy()
            self.event("pick_from_belt", contacts=int(j.contacts), object_xy=cup[:2].tolist())
        if j.lifted:
            self.event("first_lift", object_z=float(cup[2]))
        if self.vla:
            self._observe_policy_progress()
        if self.kind != "pick_return" and e._transported and np.linalg.norm(e.data.qpos[e._base_q:e._base_q+2]-e._route[-1]) < .06:
            self.event("transport_to_bin", base_xy=e.data.qpos[e._base_q:e._base_q+2].tolist())
        if self.kind != "pick_return" and not self.delivered and j.success:
            self.delivered = True
            self.event("place_in_bin", stable_ticks=int(j.stable_ticks), tray_contact=bool(j.tray_contact),
                       controlled_release=bool(j.controlled_release), object_xy=cup[:2].tolist())
            if self.kind in ("bin_return", "bin_retrieve"):
                e.phase = 14
            self.dwell = self.lost = 0
        if self.kind == "deliver":
            return
        if self.kind == "bin_retrieve":
            velocity = e.data.qvel[j.velocity:j.velocity+6]
            above = j.bounds()[0][2] > e.bin_metadata["rim_z_m"]+.015
            held = j.contacts == 2 and any(row["stage"] == "regrasp_from_bin" for row in self.events)
            steady = np.linalg.norm(velocity[:3]) < .025 and abs(velocity[2]) < .015 and np.linalg.norm(velocity[3:]) < .4
            self.stable = self.stable+1 if self.delivered and above and held and steady else 0
            if self.stable >= 5:
                self.regrasped = self.retrieved = self.complete = True
                self.event("lifted_from_bin", contacts=int(j.contacts), stable_ticks=self.stable,
                           object_low_z=float(j.bounds()[0][2]), rim_z_m=e.bin_metadata["rim_z_m"],
                           linear_speed_m_s=float(np.linalg.norm(velocity[:3])),
                           vertical_speed_m_s=float(abs(velocity[2])),
                           angular_speed_rad_s=float(np.linalg.norm(velocity[3:])))
                e.phase = 26
            return
        if e.phase >= 24 and e.phase != 26:
            contact = any(j.geom in (int(c.geom1), int(c.geom2)) and
                          bool(self.roller_ids.intersection((int(c.geom1), int(c.geom2))))
                          for c in e.data.contact)
            low, high = j.bounds()
            # An irregular mesh can overhang a roller edge while remaining
            # physically supported. Require its centre over the live surface,
            # its footprint within the conveyor's length and a real contact.
            on_belt = low[0] > -.29 and high[0] < .025 and -.126 < cup[1] < -.014
            velocity = e.data.qvel[j.velocity:j.velocity+6]
            released = j.contacts == 0 and e._grip_target > e._open-.009
            stable = contact and on_belt and released and abs(velocity[2]) < .025 and np.linalg.norm(velocity[3:]) < .7
            self.stable = self.stable+1 if stable else 0
            if self.stable >= 6 and e.phase == 25 and e.data.xpos[e._eef][2] > .12:
                self.returned = self.complete = True
                self.event("return_to_belt", roller_contact=True, stable_ticks=self.stable,
                           contacts=int(j.contacts), controlled_release=True, object_xy=cup[:2].tolist(),
                           source_xy=self.pickup_xy.tolist())
                e.phase = 26

    def before_step(self, arm):
        """Record release intent plus physical prerequisites, never actions.

        A learned controller does not call the expert's finite state machine.
        The same judge still needs evidence that opening followed a real grasp
        and lift at a supported low release position.
        """
        if not self.vla:
            return
        e, j = self.env, self.env._judge
        requested_width = e._closed+(float(arm[3])+1)*.5*(e._open-e._closed)
        opening = requested_width > e._grip_target+1e-6
        if j.contacts != 2 or not j.lifted or not opening:
            return
        low, _ = j.bounds()
        cup = e.data.xpos[j.cup]
        if j.inside() and .007 <= low[2] <= .027:
            j.controlled_release = True
        allowed = self.kind == "pick_return" or self.kind == "bin_return" and self.regrasped
        if allowed and self.pickup_xy is not None and np.linalg.norm(cup[:2]-self.pickup_xy) < .025 and -.002 <= low[2] <= .018:
            self.release_armed = True
            self.release_error = float(np.linalg.norm(cup[:2]-self.pickup_xy))
            self.event("released_on_belt", object_xy=cup[:2].tolist(), source_xy=self.pickup_xy.tolist(),
                       release_error_m=self.release_error, contacts=int(j.contacts))
            e.phase = 24

    def _observe_policy_progress(self):
        e, j = self.env, self.env._judge
        cup = e.data.xpos[j.cup]
        if self.complete:
            e.phase = 26
            return
        if self.release_armed:
            e.phase = 25 if j.contacts == 0 else 24
            return
        if self.kind in ("bin_return", "bin_retrieve") and self.delivered:
            if j.contacts == 2:
                self.event("regrasp_from_bin", contacts=2, object_z=float(cup[2]))
                e.phase = 18
                if j.bounds()[0][2] > e.bin_metadata["rim_z_m"]+.015:
                    self.regrasped = True
                    if self.kind == "bin_retrieve":
                        return
                    self.event("lifted_from_bin", contacts=2, object_low_z=float(j.bounds()[0][2]))
                    e.phase = 20
                    xy = e.data.qpos[e._base_q:e._base_q+2]
                    distance = float(np.linalg.norm(xy-e._base_start[:2]))
                    if distance < .06 and abs(e._yaw()) < .1 and np.linalg.norm(e.data.qvel[e._base_d:e._base_d+3]) < .02:
                        self.event("return_along_route", base_xy=xy.tolist(), heading_error_rad=abs(e._yaw()))
                        e.phase = 22
            return
        if j.contacts == 2:
            e.phase = 5 if j.lifted else 4
            if j.lifted:
                distance = np.linalg.norm(e.data.qpos[e._base_q:e._base_q+2]-e._base_start[:2])
                e._transported |= bool(distance > .025)
                e.phase = 7 if distance > .025 else 6
            if j.inside() and j.bounds()[0][2] < .03:
                e.phase = 10
        elif j.controlled_release:
            e.phase = 12
        elif self.pickup_xy is None:
            e.phase = 2

    def before_expert(self):
        if self.kind == "pick_return" and self.env.phase == 6:
            self.env.phase = 22
            self.dwell = self.lost = 0

    def _drive_back(self):
        e = self.env
        xy = e.data.qpos[e._base_q:e._base_q+2]
        point = np.asarray(self.route[self.route_index])
        delta = point-xy
        distance = float(np.linalg.norm(delta))
        last = self.route_index == len(self.route)-1
        if distance < (.012 if last else .045):
            if last:
                e.phase = 21
                return 0., 0.
            self.route_index += 1
            delta = np.asarray(self.route[self.route_index])-xy
            distance = float(np.linalg.norm(delta))
        desired = math.atan2(delta[1], delta[0])+math.pi/2
        error = math.atan2(math.sin(desired-e._yaw()), math.cos(desired-e._yaw()))
        sign = 1.
        if abs(error) > math.pi/2:
            error = math.atan2(math.sin(error+math.pi), math.cos(error+math.pi))
            sign = -1.
        angular = float(np.clip(error*2, -.8, .8))
        speed = min(.065*e.speed_scale, distance*1.3)*max(0., math.cos(error)) if abs(error) < .30 else 0.
        errors = []
        for a, b in zip(self.route, self.route[1:]):
            a, b = np.asarray(a), np.asarray(b)
            d = b-a
            t = np.clip(np.dot(xy-a, d)/max(np.dot(d,d), 1e-12), 0, 1)
            errors.append(np.linalg.norm(xy-a-t*d))
        self.route_errors.append(float(min(errors, default=0.)))
        return sign*speed, angular

    def expert_action(self):
        e, j, task = self.env, self.env._judge, self.env._duck_task
        p = e.phase
        eef, cup = e.data.xpos[e._eef], e.data.xpos[j.cup]
        base = e.data.qpos[e._base_q:e._base_q+3]
        grip, hold, drive, turn, speed = task.hold_width or e._closed, False, 0., 0., .08
        waypoint = e._target.copy()
        if 18 <= p <= 23:
            self.lost = self.lost+1 if j.contacts < 2 else 0
            if self.lost >= 3:
                e._fail()
            if j.contacts == 2:
                task.grasp_offset = .85*task.grasp_offset+.15*(cup-eef)
                slip = e._grasp_reference_z-float(cup[2]-eef[2])
                if slip > .003:
                    task.hold_width = max(e._closed, grip-.0005)
                    grip = task.hold_width
        arrived = lambda point: np.linalg.norm(eef-point) < .005 and np.linalg.norm(e._servo_velocity) < .025
        if p == 14:
            grip = e._open
            waypoint = np.r_[e._release_pose[:2], .19]
            if arrived(waypoint):
                e.phase = 15
        elif p == 15:
            grip = e._open
            waypoint = np.r_[cup[:2], .19]
            if arrived(waypoint):
                e.phase = 16
        elif p in (16, 17):
            waypoint = np.r_[cup[:2], max(float(cup[2]), .040)]
            speed = .045
            if p == 16:
                grip = e._open
                if arrived(waypoint):
                    e.phase = 17
                    task.hold_width = None
            else:
                if j.contacts == 2 and task.hold_width is None:
                    g = e.robot.gripper
                    mean = (e.data.qpos[e._qposadr[g.left_joint]]+e.data.qpos[e._qposadr[g.right_joint]])/2
                    task.hold_width = max(e._closed, mean-.010)
                grip = task.hold_width or e._closed
                self.dwell = self.dwell+1 if j.contacts == 2 else 0
                if self.dwell >= 5:
                    task.grasp_eef, task.grasp_offset = eef.copy(), cup-eef
                    e._grasp_reference_z = float(task.grasp_offset[2])
                    self.lost = self.dwell = 0
                    e.phase = 18
                    self.event("regrasp_from_bin", contacts=2, object_z=float(cup[2]))
        elif p == 18:
            waypoint = np.r_[task.grasp_eef[:2], .19]
            if arrived(waypoint) and j.bounds()[0][2] > e.bin_metadata["rim_z_m"]+.015:
                self.regrasped = True
                if self.kind != "bin_retrieve":
                    self.event("lifted_from_bin", contacts=int(j.contacts), object_low_z=float(j.bounds()[0][2]))
                    e.phase = 19
        elif p == 19:
            yaw = e._yaw()
            c, s = math.cos(yaw), math.sin(yaw)
            waypoint = base+np.array([c*.10+s*.11, s*.10-c*.11, .233])
            if arrived(waypoint):
                e.phase = 20
                e._target, e._servo_velocity = eef.copy(), np.zeros(3)
        elif p == 20:
            hold = True
            drive, turn = self._drive_back()
        elif p == 21:
            hold = True
            error = math.atan2(math.sin(-e._yaw()), math.cos(-e._yaw()))
            aligned = abs(error) < .025
            turn = float(np.clip(error*1.3, -.55, .55)) if not aligned else 0.
            stable = np.linalg.norm(e.data.qvel[e._base_d:e._base_d+3]) < .008 and abs(e.data.qvel[e._base_d+5]) < .04
            self.dwell = self.dwell+1 if aligned and stable else 0
            if self.dwell >= 5:
                self.event("return_along_route", base_xy=base[:2].tolist(), heading_error_rad=abs(error))
                e.phase = 22
                e._target, e._servo_velocity = eef.copy(), np.zeros(3)
        elif p in (22, 23):
            offset = task.grasp_offset
            source = self.pickup_xy-offset[:2]
            drop = max((cup[2]-j.bounds()[0][2])+.002-offset[2], .031)
            waypoint = np.r_[source, .18 if p == 22 else drop]
            speed = .09 if p == 22 else .035
            if p == 22 and arrived(waypoint):
                e.phase = 23
            elif p == 23 and arrived(waypoint) and j.contacts == 2 and j.bounds()[0][2] > -.002:
                self.release = e._target.copy()
                self.release_belt_distance = task.distance
                self.release_armed = True
                self.release_error = float(np.linalg.norm(cup[:2]-self.pickup_xy))
                self.event("released_on_belt", object_xy=cup[:2].tolist(), source_xy=self.pickup_xy.tolist(),
                           release_error_m=self.release_error, contacts=int(j.contacts))
                self.dwell = 0
                e.phase = 24
        elif p == 24:
            waypoint = self.release.copy()
            waypoint[0] += task.distance-self.release_belt_distance
            # Release the elastic finger preload gently while following the
            # live surface, instead of dragging the supported mesh against it.
            grip, speed = min(e._open, e._servo_grip+.025*.05), .045
            g = e.robot.gripper
            mean = (e.data.qpos[e._qposadr[g.left_joint]]+e.data.qpos[e._qposadr[g.right_joint]])/2
            self.dwell = self.dwell+1 if mean > e._open-.005 else 0
            if self.dwell >= 3:
                self.release = e._target.copy()
                e.phase = 25
        elif p == 25:
            waypoint, grip, speed = np.r_[self.release[:2], .18], e._open, .09
        else:
            hold, grip = True, (task.hold_width or e._closed) if self.kind == "bin_retrieve" else e._open
        return {"arm": e._smooth_expert_action(waypoint, speed*e.speed_scale, grip),
                "base_linear_m_s": drive, "base_angular_rad_s": turn, "hold_arm": hold}

    def metrics(self):
        return {"task_kind": self.kind, "program_complete": self.complete,
                "task_events": [dict(row) for row in self.events], "delivered_to_bin": self.delivered,
                "regrasped_from_bin": self.regrasped, "returned_to_belt": self.returned,
                "retrieved_from_bin": self.retrieved, "retrieve_stable_ticks": self.stable if self.kind == "bin_retrieve" else 0,
                "return_stable_ticks": self.stable if self.kind != "bin_retrieve" else 0,
                "return_release_error_m": self.release_error,
                "return_path_rmse_m": float(np.sqrt(np.mean(np.square(self.route_errors)))) if self.route_errors else None,
                "return_route": self.route if self.kind == "bin_return" else [],
                "return_source_xy": self.pickup_xy.tolist() if self.pickup_xy is not None else None}
