"""Versioned, validated platform configuration (schema_version 1.0).

All lengths are metres, angles radians, times seconds. Entities reference each
other by name; the engine resolves names, never fixed array indices.
"""
from __future__ import annotations

from enum import Enum
import math
from typing import Literal

from pydantic import BaseModel, Field, model_validator

SCHEMA_VERSION = "1.0"


class ParamSource(str, Enum):
    user = "user"
    vendor = "vendor"
    model_estimate = "model_estimate"
    measured = "measured"


class Vec3(BaseModel):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_list(self):
        return [self.x, self.y, self.z]


class CollisionGeom(BaseModel):
    name: str | None = None
    type: Literal["capsule", "box", "sphere", "cylinder", "mesh"]
    size: list[float]
    fromto: list[float] | None = None
    pos: list[float] = [0, 0, 0]
    mass_kg: float = 0.0
    friction: list[float] = [0.95, 0.005, 0.0001]
    contype: int = 1
    conaffinity: int = 1
    rgba: list[float] = [0, 0, 0, 0]
    group: int = 3
    mesh: str | None = None
    solref: list[float] = [0.006, 1]


class VisualGeom(BaseModel):
    mesh: str
    pos: list[float] = [0, 0, 0]
    quat: list[float] = [1, 0, 0, 0]
    scale: list[float] = [1, 1, 1]
    rgba: list[float] = [0.84, 0.73, 0.52, 1]


class LinkConfig(BaseModel):
    name: str
    parent: str | None = None
    origin: list[float] = [0, 0, 0]
    gravcomp: bool = True
    visuals: list[VisualGeom] = []
    collisions: list[CollisionGeom] = []
    com: list[float] | None = None
    inertia_diag: list[float] | None = None

    @model_validator(mode="after")
    def _inertia_positive(self):
        if self.inertia_diag is not None:
            a, b, c = self.inertia_diag
            if min(a, b, c) <= 0 or a + b < c or a + c < b or b + c < a:
                raise ValueError(f"link {self.name}: inertia must be positive definite (triangle inequality)")
        return self


class JointConfig(BaseModel):
    name: str
    child: str
    type: Literal["fixed", "revolute", "prismatic"] = "revolute"
    axis: list[float] = [0, 0, 1]
    range: list[float] | None = None
    damping: float = 0.8
    armature: float = 0.012
    initial: float = 0.0

    @model_validator(mode="after")
    def _checks(self):
        if self.type != "fixed":
            if sum(v * v for v in self.axis) < 1e-12:
                raise ValueError(f"joint {self.name}: zero-length axis")
            if self.range is not None and self.range[0] > self.range[1]:
                raise ValueError(f"joint {self.name}: inverted range")
            if self.range is not None and not self.range[0] <= self.initial <= self.range[1]:
                raise ValueError(f"joint {self.name}: initial pose outside range")
        return self


class ActuatorConfig(BaseModel):
    joint: str
    kind: Literal["position"] = "position"
    kp: float
    kv: float
    force_limit: float


class GripperConfig(BaseModel):
    left_joint: str
    right_joint: str
    open_m: float = 0.058
    closed_m: float = 0.006
    pad_geoms: list[str] = ["left_pad", "right_pad"]


class RobotConfig(BaseModel):
    id: str
    root_link: str
    links: list[LinkConfig]
    joints: list[JointConfig]
    actuators: list[ActuatorConfig]
    end_effector: str
    tool_offset_m: float = 0.075
    gripper: GripperConfig | None = None
    reference_pose: dict[str, float] = {}
    mesh_dir: str = "assets/robot"

    @model_validator(mode="after")
    def _tree(self):
        names = [l.name for l in self.links]
        if len(names) != len(set(names)):
            raise ValueError("duplicate link names")
        jnames = [j.name for j in self.joints]
        if len(jnames) != len(set(jnames)):
            raise ValueError("duplicate joint names")
        by_name = {l.name: l for l in self.links}
        if self.root_link not in by_name:
            raise ValueError("root_link not in links")
        if self.end_effector not in by_name:
            raise ValueError("end_effector not in links")
        for j in self.joints:
            if j.child not in by_name:
                raise ValueError(f"joint {j.name}: child {j.child} not in links")
        for l in self.links:
            if l.name != self.root_link and l.parent not in by_name:
                raise ValueError(f"link {l.name}: parent {l.parent} not in links (broken chain)")
        # every non-root link must be the child of exactly one joint (no loops)
        children = [j.child for j in self.joints]
        if len(children) != len(set(children)):
            raise ValueError("a link has multiple joints (closed loop)")
        for l in self.links:
            if l.name != self.root_link and l.name not in children:
                raise ValueError(f"link {l.name}: disconnected from joint tree")
        for a in self.actuators:
            if a.joint not in jnames:
                raise ValueError(f"actuator references unknown joint {a.joint}")
        return self


class CameraIntrinsics(BaseModel):
    fx: float
    fy: float
    cx: float
    cy: float

    @model_validator(mode="after")
    def _positive(self):
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("focal lengths must be positive")
        return self


class SensorConfig(BaseModel):
    id: str
    type: Literal["rgb_camera"] = "rgb_camera"
    parent: str = "world"
    position: list[float] = [0, 0, 0]
    xyaxes: list[float] = [1, 0, 0, 0, 1, 0]
    intrinsics: CameraIntrinsics
    resolution: list[int] = [256, 256]
    fps: float = 30.0
    latency_s: float = 0.0
    noise_std: float = 0.0

    @model_validator(mode="after")
    def _checks(self):
        w, h = self.resolution
        if not (1 <= w <= 1280 and 1 <= h <= 960):
            raise ValueError(f"sensor {self.id}: resolution out of range")
        if self.fps <= 0:
            raise ValueError(f"sensor {self.id}: fps must be positive")
        if self.latency_s < 0 or self.noise_std < 0:
            raise ValueError(f"sensor {self.id}: latency/noise must be non-negative")
        if len(self.xyaxes) != 6:
            raise ValueError(f"sensor {self.id}: xyaxes needs 6 numbers")
        return self


class ControllerConfig(BaseModel):
    id: str
    mode: Literal["cartesian_delta_ik", "joint_position"]
    action_names: list[str]
    action_dim: int
    units: list[str]
    frame: str = "world"
    absolute: bool = False
    action_range: list[list[float]]
    frequency_hz: float = 20.0
    max_delta_m: float | None = 0.012
    workspace_min: list[float] = [-0.28, -0.17, 0.022]
    workspace_max: list[float] = [0.28, 0.17, 0.184]
    task_space_note: str = ""
    required_observations: list[str] = []

    @model_validator(mode="after")
    def _checks(self):
        if len(self.action_names) != self.action_dim or len(self.units) != self.action_dim:
            raise ValueError("action_names/units must match action_dim")
        if len(self.action_range) != self.action_dim:
            raise ValueError("action_range must match action_dim")
        for lo, hi in self.action_range:
            if lo > hi:
                raise ValueError("inverted action range")
        if self.frequency_hz <= 0:
            raise ValueError("frequency_hz must be positive")
        return self


class DuckConveyorConfig(BaseModel):
    belt_speed_m_s: float = Field(default=.025, gt=0, le=.06)
    pick_x: float = Field(default=-.115, ge=-.15, le=-.06)
    random_pose: bool = True
    continuous: bool = True
    chassis_asset: str = "imported/four-wheel-robot-chassis-82e8e5"


class TaskConfig(BaseModel):
    id: str = "cup_pick_place"
    plugin: str = "cup_pick_place"
    asset: str = "standard"
    goal_xy: list[float] = [0.133, 0.050]
    goal_surface_m: float = 0.009
    time_limit_s: float = 9.0
    lift_threshold_m: float = 0.031
    stable_ticks: int = 6
    spawn_min: list[float] = [-0.154, -0.081]
    spawn_max: list[float] = [-0.085, -0.026]
    instruction: str = "Place the cup in the tray."
    object_color_rgb: list[float] | None = None
    duck_conveyor: DuckConveyorConfig = Field(default_factory=DuckConveyorConfig)

    @model_validator(mode="after")
    def _color(self):
        if self.object_color_rgb is not None and (len(self.object_color_rgb) != 3 or
                any(not math.isfinite(v) or not 0 <= v <= 1 for v in self.object_color_rgb)):
            raise ValueError("object_color_rgb requires three channels in [0,1]")
        return self


def _scene_pose(position, quaternion):
    if len(position) != 3 or not all(math.isfinite(v) for v in position):
        raise ValueError("Scene position needs three finite values")
    if len(quaternion) != 4 or not all(math.isfinite(v) for v in quaternion) or sum(v*v for v in quaternion) < 1e-12:
        raise ValueError("Scene quaternion needs four finite, non-zero components")
    norm = math.sqrt(sum(v*v for v in quaternion))
    quaternion[:] = [v / norm for v in quaternion]


class SceneCollider(BaseModel):
    """Static collision proxy. full_size_m is the FULL extent (MuJoCo half-sizes derived on export)."""
    name: str
    shape: Literal["box", "plane", "cylinder"] = "box"
    full_size_m: list[float] = [0.64, 0.43, 0.056]
    position: list[float] = [0, 0, 0]
    quaternion_xyzw: list[float] = [0, 0, 0, 1]
    friction: list[float] = [0.95, 0.005, 0.0001]
    enabled: bool = True
    contype: int = 1
    conaffinity: int = 1
    tag: str = "environment"

    @model_validator(mode="after")
    def _size(self):
        from .scene_assets import valid_id
        valid_id(self.name)
        if len(self.full_size_m) != 3 or any(not math.isfinite(v) or v <= 0 for v in self.full_size_m):
            raise ValueError(f"collider {self.name}: full_size_m needs three positive finite dimensions")
        _scene_pose(self.position, self.quaternion_xyzw)
        if self.shape == "cylinder" and abs(self.full_size_m[0] - self.full_size_m[1]) > 1e-9:
            raise ValueError(f"collider {self.name}: cylinder needs equal x/y (no elliptical support)")
        if len(self.friction) != 3 or any(not math.isfinite(v) or v < 0 for v in self.friction):
            raise ValueError("friction requires three finite non-negative values")
        return self


class SceneInstance(BaseModel):
    asset_id: str
    revision: int = Field(default=1, ge=1)
    position: list[float] = [0, 0, 0]
    quaternion_xyzw: list[float] = [0, 0, 0, 1]
    uniform_scale: float = 1.0

    @model_validator(mode="after")
    def _scale(self):
        from .scene_assets import valid_id
        valid_id(self.asset_id)
        _scene_pose(self.position, self.quaternion_xyzw)
        if not math.isfinite(self.uniform_scale) or self.uniform_scale <= 0:
            raise ValueError("uniform_scale must be positive (non-uniform scale not supported)")
        return self


class SceneProp(BaseModel):
    """Imported static scene mesh, with an optional explicitly estimated box proxy."""
    name: str
    asset: str
    position: list[float] = [0, 0, 0]
    quaternion_xyzw: list[float] = [0, 0, 0, 1]
    uniform_scale: float = 1.0
    collision: bool = False

    @model_validator(mode="after")
    def _checks(self):
        from .scene_assets import valid_id
        valid_id(self.name)
        if not self.asset.startswith("imported/"):
            raise ValueError("Scene props must reference imported/<asset_id>")
        valid_id(self.asset.removeprefix("imported/"))
        _scene_pose(self.position, self.quaternion_xyzw)
        if (not all(math.isfinite(v) for v in self.quaternion_xyzw)
                or sum(v*v for v in self.quaternion_xyzw) < 1e-12):
            raise ValueError("Scene prop quaternion must normalize to a finite rotation")
        if not math.isfinite(self.uniform_scale) or self.uniform_scale <= 0:
            raise ValueError("Scene prop uniform_scale must be positive and finite")
        return self


class SceneAssetManifest(BaseModel):
    asset_id: str
    revision: int = 1
    display_name: str = ""
    category: Literal["scene"] = "scene"
    provider: Literal["local", "aholo_world"] = "local"
    source_kind: Literal["generation", "reconstruction", "import"] = "import"
    remote_task_id: str | None = None
    source_detail: str = ""
    created: str = ""
    files: dict[str, str] = {}
    format: str = "obj"
    up_axis: str = "Z"
    unit_scale: float = 1.0
    calibrated: bool = False
    visual_asset: str | None = "processed/visual.obj"
    collision_asset: str | None = None
    calibration: dict = {}
    bbox: list | None = None
    status: str = "ready"


class SceneConfig(BaseModel):
    physics_backend: Literal["mujoco"] = "mujoco"
    physics_dt: float = 0.002
    gravity: list[float] = [0, 0, -9.81]
    seed: int = 0
    coordinate_convention: str = "z_up_right_handed"
    environment: SceneInstance | None = None
    colliders: list[SceneCollider] = []
    props: list[SceneProp] = Field(default_factory=list)
    sensor_renderer: Literal["mujoco", "hybrid"] = "mujoco"
    use_default_table: bool = True
    work_surface_z: float = 0.0
    table_color_rgb: list[float] = [.29, .37, .39]

    @model_validator(mode="after")
    def _dt(self):
        if len(self.table_color_rgb) != 3 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in self.table_color_rgb):
            raise ValueError("table_color_rgb requires three channels in [0,1]")
        if not 0 < self.physics_dt <= 0.01:
            raise ValueError("physics_dt must be in (0, 0.01]")
        names = [c.name for c in self.colliders]
        if len(names) != len(set(names)):
            raise ValueError("Scene collider names must be unique")
        prop_names = [p.name for p in self.props]
        if len(prop_names) != len(set(prop_names)):
            raise ValueError("Scene prop names must be unique")
        generated_names = {f"scene_prop_{p.name}_{suffix}"
                           for p in self.props for suffix in ("visual", "contact")}
        if (set(names) & set(prop_names)) or (set(names) & generated_names):
            raise ValueError("Scene prop names must not conflict with collider names")
        if self.work_surface_z != 0:
            raise ValueError("Current tabletop task only supports work_surface_z=0; task-frame adaptation is not implemented")
        return self


class PolicyConfig(BaseModel):
    id: str
    architecture: str
    weights_path: str | None = None
    weights_sha256: str | None = None
    observation_contract: str = "1.0"
    action_contract: str = "1.0"
    run_mode: Literal["external_http", "local", "remote"] = "external_http"
    status: Literal["unverified", "verified", "failed"] = "unverified"
    notes: str = ""


class ProjectConfig(BaseModel):
    schema_version: str = SCHEMA_VERSION
    project_id: str = "default"
    version: int = 1
    validation_status: Literal["candidate", "validated"] = "candidate"
    scene: SceneConfig = SceneConfig()
    robots: list[RobotConfig]
    sensors: list[SensorConfig] = []
    controllers: list[ControllerConfig]
    task: TaskConfig = TaskConfig()
    policies: list[PolicyConfig] = []
    provenance: dict[str, str] = {}

    @model_validator(mode="after")
    def _refs(self):
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version {self.schema_version}")
        rids = [r.id for r in self.robots]
        if len(rids) != len(set(rids)):
            raise ValueError("duplicate robot ids")
        sids = [s.id for s in self.sensors]
        if len(sids) != len(set(sids)):
            raise ValueError("duplicate sensor ids")
        cids = [c.id for c in self.controllers]
        if len(cids) != len(set(cids)):
            raise ValueError("duplicate controller ids")
        bodies = {l.name for r in self.robots for l in r.links} | {"world"}
        for s in self.sensors:
            if s.parent not in bodies:
                raise ValueError(f"sensor {s.id}: parent {s.parent} is not a known body")
        return self
