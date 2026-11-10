"""Generate configs/default.json: the validated Lux3D 5R arm as a versioned config."""
import json
import math
from pathlib import Path

from .schema import (ActuatorConfig, CameraIntrinsics, CollisionGeom, ControllerConfig,
                     GripperConfig, JointConfig, LinkConfig, ProjectConfig, RobotConfig,
                     SceneConfig, SensorConfig, TaskConfig, VisualGeom)

LAB = Path(__file__).resolve().parents[1]


def _fovy_intrinsics(fovy_deg, w=256, h=256):
    fy = (h / 2) / math.tan(math.radians(fovy_deg) / 2)
    return CameraIntrinsics(fx=fy, fy=fy, cx=w / 2, cy=h / 2)


def build() -> ProjectConfig:
    rubber = [.13, .16, .18, 1]
    steel = [.69, .73, .74, 1]

    def hub():
        return CollisionGeom(type="sphere", size=[.027], mass_kg=.08, contype=0, conaffinity=0,
                             rgba=rubber, group=0)

    def rod(length, mass):
        return CollisionGeom(type="capsule", fromto=[0, 0, .02, 0, 0, length - .07], size=[.017],
                             mass_kg=mass, contype=2, conaffinity=1, rgba=[0, 0, 0, 0], group=3)

    def shell(mesh):
        return VisualGeom(mesh=mesh)

    def finger(tag, s):
        return LinkConfig(
            name=tag, parent="eef", origin=[0, s * .006, 0], gravcomp=False,
            collisions=[
                CollisionGeom(name=f"{tag}_pad", type="box", size=[.014, .006, .022], mass_kg=.035,
                              friction=[1.8, .005, .0001], rgba=rubber, group=0, solref=[.004, 1]),
                CollisionGeom(type="box", pos=[0, 0, .036], size=[.012, .005, .016], mass_kg=0,
                              contype=0, conaffinity=0, rgba=steel, group=2)])

    robot = RobotConfig(
        id="lux3d_5r", root_link="pedestal", end_effector="eef", tool_offset_m=.075,
        mesh_dir="assets/robot",
        links=[
            LinkConfig(name="pedestal", origin=[0, .205, 0], gravcomp=False, visuals=[shell("base")]),
            LinkConfig(name="turret", parent="pedestal", origin=[0, 0, .08], collisions=[hub()]),
            LinkConfig(name="upper_arm", parent="turret", origin=[0, 0, 0],
                       collisions=[rod(.23, .45)], visuals=[shell("upper")]),
            LinkConfig(name="forearm", parent="upper_arm", origin=[0, 0, .23],
                       collisions=[rod(.23, .32), hub()], visuals=[shell("fore")]),
            LinkConfig(name="wrist", parent="forearm", origin=[0, 0, .23], collisions=[hub()]),
            LinkConfig(name="tool", parent="wrist", origin=[0, 0, 0], visuals=[shell("wrist")],
                       collisions=[CollisionGeom(type="sphere", size=[.018], mass_kg=.06,
                                                 contype=0, conaffinity=0, rgba=steel, group=0)]),
            LinkConfig(name="eef", parent="tool", origin=[0, 0, -.075],
                       collisions=[CollisionGeom(type="box", pos=[0, 0, .060], size=[.016, .040, .008],
                                                 mass_kg=.09, rgba=rubber, group=0)]),
            finger("left", 1), finger("right", -1),
        ],
        joints=[
            JointConfig(name="waist", child="turret", axis=[0, 0, 1], range=[-3.14, 3.14]),
            JointConfig(name="shoulder", child="upper_arm", axis=[0, 1, 0], range=[-1.7, 2.7]),
            JointConfig(name="elbow", child="forearm", axis=[0, 1, 0], range=[.05, 3.08], initial=1.5),
            JointConfig(name="wrist_pitch", child="wrist", axis=[0, 1, 0], range=[-3.45, 1.7]),
            JointConfig(name="wrist_yaw", child="tool", axis=[0, 0, 1], range=[-3.14, 3.14]),
            JointConfig(name="eef_fixed", child="eef", type="fixed"),
            JointConfig(name="left", child="left", type="prismatic", axis=[0, 1, 0], range=[0, .065],
                        damping=4, armature=0),
            JointConfig(name="right", child="right", type="prismatic", axis=[0, -1, 0], range=[0, .065],
                        damping=4, armature=0),
        ],
        actuators=[
            ActuatorConfig(joint="waist", kp=320, kv=15, force_limit=32),
            ActuatorConfig(joint="shoulder", kp=370, kv=17, force_limit=32),
            ActuatorConfig(joint="elbow", kp=290, kv=12, force_limit=32),
            ActuatorConfig(joint="wrist_pitch", kp=150, kv=6, force_limit=8),
            ActuatorConfig(joint="wrist_yaw", kp=100, kv=4, force_limit=5),
            ActuatorConfig(joint="left", kp=350, kv=4, force_limit=9),
            ActuatorConfig(joint="right", kp=350, kv=4, force_limit=9),
        ],
        gripper=GripperConfig(left_joint="left", right_joint="right", open_m=.058, closed_m=.006),
    )
    sensors = [
        SensorConfig(id="front", parent="world", position=[.36, -.70, .46],
                     xyaxes=[.895, .445, 0, -.16, .32, .93], intrinsics=_fovy_intrinsics(40),
                     resolution=[256, 256], fps=30),
        SensorConfig(id="wrist", parent="eef", position=[.062, 0, .075],
                     xyaxes=[0, 1, 0, -1, 0, .72], intrinsics=_fovy_intrinsics(87),
                     resolution=[256, 256], fps=30),
    ]
    controllers = [
        ControllerConfig(id="cartesian", mode="cartesian_delta_ik",
                         action_names=["dx", "dy", "dz", "grip"], action_dim=4,
                         units=["normalized", "normalized", "normalized", "normalized"],
                         frame="world", absolute=False, action_range=[[-1, 1]] * 4,
                         frequency_hz=20, max_delta_m=.012,
                         task_space_note="5-DOF arm holds a fixed downward tool attitude; "
                                         "arbitrary 6D end-effector poses are not trackable.",
                         required_observations=["state", "front", "wrist"]),
        ControllerConfig(id="joints", mode="joint_position",
                         action_names=["waist", "shoulder", "elbow", "wrist_pitch", "wrist_yaw", "gripper"],
                         action_dim=6, units=["rad"] * 5 + ["m"], frame="joint", absolute=True,
                         action_range=[[-3.14, 3.14], [-1.7, 2.7], [.05, 3.08], [-3.45, 1.7],
                                       [-3.14, 3.14], [.006, .058]],
                         frequency_hz=20, max_delta_m=None,
                         required_observations=["state"]),
    ]
    from .make_second_config import build as _second
    gantry = _second().robots[0]
    return ProjectConfig(project_id="default", version=1, validation_status="validated",
                         scene=SceneConfig(seed=0), robots=[robot, gantry], sensors=sensors,
                         controllers=controllers, task=TaskConfig(),
                         provenance={"robot.links.visuals": "model_estimate",
                                     "robot.links.collisions.mass_kg": "user",
                                     "robot.actuators.force_limit": "user"})


if __name__ == "__main__":
    cfg = build()
    out = LAB / "configs" / "default.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(cfg.model_dump_json(indent=2), encoding="utf-8")
    print("wrote", out)