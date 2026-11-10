"""Generate configs/second_robot.json: a 3-DOF prismatic gantry (different DOF/motion
from the 5R arm) to prove the interfaces carry no fixed joint-count assumptions."""
import json
from pathlib import Path

from .make_default_config import _fovy_intrinsics
from .schema import (ActuatorConfig, CollisionGeom, ControllerConfig, GripperConfig,
                     JointConfig, LinkConfig, ProjectConfig, RobotConfig, SceneConfig,
                     SensorConfig, TaskConfig)

LAB = Path(__file__).resolve().parents[1]


def _box(size, mass, color, pos=(0, 0, 0), friction=(0.95, 0.005, 0.0001)):
    return CollisionGeom(type="box", size=list(size), pos=list(pos), mass_kg=mass,
                         friction=list(friction), rgba=color, group=0)


def build() -> ProjectConfig:
    steel = [.62, .68, .70, 1]
    rubber = [.13, .16, .18, 1]
    accent = [.86, .49, .16, 1]

    def finger(tag, s):
        return LinkConfig(name=tag, parent="eef", origin=[0, s * .006, 0], gravcomp=False,
                          collisions=[
                              CollisionGeom(name=f"{tag}_pad", type="box", size=[.012, .005, .018],
                                            mass_kg=.03, friction=[1.8, .005, .0001], rgba=rubber,
                                            group=0, solref=[.004, 1])])

    robot = RobotConfig(
        id="cartesian_3dof", root_link="base", end_effector="eef", tool_offset_m=.03,
        mesh_dir="assets/robot",
        links=[
            LinkConfig(name="base", origin=[0, .205, 0], gravcomp=False,
                       collisions=[_box([.02, .02, .012], .6, steel, pos=[0, 0, .012])]),
            LinkConfig(name="x_slide", parent="base", origin=[0, 0, .12], collisions=[
                _box([.014, .014, .014], .2, accent),
                _box([.004, .004, .004], 0, steel, pos=[0, 0, .01])]),
            LinkConfig(name="y_slide", parent="x_slide", collisions=[
                _box([.012, .012, .012], .15, steel)]),
            LinkConfig(name="z_slide", parent="y_slide", collisions=[
                _box([.010, .010, .010], .12, accent)]),
            LinkConfig(name="eef", parent="z_slide", origin=[0, 0, -.03], collisions=[
                CollisionGeom(type="box", pos=[0, 0, .006], size=[.014, .030, .006], mass_kg=.05,
                              rgba=rubber, group=0)]),
            finger("left", 1), finger("right", -1),
        ],
        joints=[
            JointConfig(name="x", child="x_slide", type="prismatic", axis=[1, 0, 0], range=[-.30, .30]),
            JointConfig(name="y", child="y_slide", type="prismatic", axis=[0, 1, 0], range=[-.32, .20]),
            JointConfig(name="z", child="z_slide", type="prismatic", axis=[0, 0, 1], range=[-.12, .06]),
            JointConfig(name="eef_fixed", child="eef", type="fixed"),
            JointConfig(name="left", child="left", type="prismatic", axis=[0, 1, 0], range=[0, .065],
                        damping=4, armature=0),
            JointConfig(name="right", child="right", type="prismatic", axis=[0, -1, 0], range=[0, .065],
                        damping=4, armature=0),
        ],
        actuators=[
            ActuatorConfig(joint="x", kp=300, kv=15, force_limit=50),
            ActuatorConfig(joint="y", kp=300, kv=15, force_limit=50),
            ActuatorConfig(joint="z", kp=300, kv=15, force_limit=50),
            ActuatorConfig(joint="left", kp=350, kv=4, force_limit=9),
            ActuatorConfig(joint="right", kp=350, kv=4, force_limit=9),
        ],
        gripper=GripperConfig(left_joint="left", right_joint="right", open_m=.058, closed_m=.006),
    )
    sensors = [
        SensorConfig(id="front", parent="world", position=[.36, -.70, .46],
                     xyaxes=[.895, .445, 0, -.16, .32, .93], intrinsics=_fovy_intrinsics(40),
                     resolution=[256, 256], fps=15),
        SensorConfig(id="wrist", parent="eef", position=[.05, 0, .04],
                     xyaxes=[0, 1, 0, -1, 0, .9], intrinsics=_fovy_intrinsics(80),
                     resolution=[256, 256], fps=15),
    ]
    controllers = [
        ControllerConfig(id="joints", mode="joint_position",
                         action_names=["x", "y", "z", "gripper"], action_dim=4,
                         units=["m", "m", "m", "m"], frame="joint", absolute=True,
                         action_range=[[-.30, .30], [-.32, .20], [-.12, .06], [.006, .058]],
                         frequency_hz=20, max_delta_m=None, required_observations=["state"]),
    ]
    return ProjectConfig(project_id="second_robot", version=1, validation_status="validated",
                         scene=SceneConfig(seed=0), robots=[robot], sensors=sensors,
                         controllers=controllers, task=TaskConfig(),
                         provenance={"robot.links": "synthetic cartesian gantry"})


if __name__ == "__main__":
    out = LAB / "configs" / "second_robot.json"
    out.write_text(build().model_dump_json(indent=2), encoding="utf-8")
    print("wrote", out)