"""Versioned data models for the generation loop.

Rules encoded here:
- Every numeric field carries provenance; a model estimate is never a measured
  fact. ``confidence`` is suggestion strength only.
- ``None`` means "user did not say / unknown", distinct from an accepted default
  (provenance.confirmation == accepted).
- AssetSpec (appearance) and PhysicsSpec (mass/friction/proxies) are separate;
  unknown dimensions keep an asset uncalibrated and block physics binding.
"""
from __future__ import annotations

import math
import re
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

GENERATION_SCHEMA_VERSION = "1.0"


class FieldSource(str, Enum):
    user_explicit = "user_explicit"
    project_config = "project_config"
    measured = "measured"
    model_inferred = "model_inferred"
    default = "default"
    provider_predicted = "provider_predicted"


class Confirmation(str, Enum):
    unconfirmed = "unconfirmed"
    accepted = "accepted"


class Verification(str, Enum):
    unverified = "unverified"
    validated = "validated"
    measured = "measured"


class FieldProvenance(BaseModel):
    source: FieldSource = FieldSource.default
    confirmation: Confirmation = Confirmation.unconfirmed
    verification: Verification = Verification.unverified
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    note: str = ""

    @model_validator(mode="after")
    def _measured_requires_source(self):
        if self.verification == Verification.measured and self.source != FieldSource.measured:
            raise ValueError("verification=measured requires source=measured")
        return self


def _positive_finite(v: float | None, label: str):
    if v is not None and (not math.isfinite(v) or v <= 0):
        raise ValueError(f"{label} must be positive and finite, or null (unknown)")
    return v


class Dimension(BaseModel):
    """A length in metres. value_m=None marks an unknown dimension."""
    value_m: float | None = None
    provenance: FieldProvenance = Field(default_factory=FieldProvenance)

    @field_validator("value_m")
    @classmethod
    def _check(cls, v):
        return _positive_finite(v, "dimension")


class Mass(BaseModel):
    value_kg: float | None = None
    provenance: FieldProvenance = Field(default_factory=FieldProvenance)

    @field_validator("value_kg")
    @classmethod
    def _check(cls, v):
        return _positive_finite(v, "mass")


class AssetKind(str, Enum):
    task_object = "task_object"
    static_prop = "static_prop"
    robot_visual = "robot_visual"
    environment_splat = "environment_splat"
    environment_mesh = "environment_mesh"  # reserved: online generation unsupported


class AssetRole(str, Enum):
    visual_only = "visual_only"
    static = "static"
    dynamic = "dynamic"


_DATA_IMAGE_RE = re.compile(r"^data:image/(png|jpeg|jpg|webp);base64,[A-Za-z0-9+/=\s]+$", re.IGNORECASE)


def validate_reference_image(v: str | None) -> str | None:
    if v is None:
        return None
    if not isinstance(v, str):
        raise ValueError("reference_image must be a string URL or data:image URI")
    s = v.strip()
    if not s:
        return None
    if s.lower().startswith(("http://", "https://")):
        if len(s) > 2048 or any(ch.isspace() for ch in s):
            raise ValueError("reference_image URL must be <= 2048 chars without whitespace")
        return s
    if s.lower().startswith("data:image/"):
        if len(s) > 6_000_000:
            raise ValueError("reference_image data URI exceeds 6 MB limit")
        if not _DATA_IMAGE_RE.match(s):
            raise ValueError("reference_image data URI must be base64 png/jpeg/webp")
        header, _, b64 = s.partition(",")
        return f"{header},{re.sub(r'\\s+', '', b64)}"
    raise ValueError("reference_image must be an http(s) URL or data:image/(png|jpeg|webp);base64 URI")


class AssetSpec(BaseModel):
    schema_version: str = GENERATION_SCHEMA_VERSION
    kind: AssetKind = AssetKind.task_object
    role: AssetRole = AssetRole.dynamic
    name: str = ""
    description_zh: str = ""
    user_words: str = ""  # original user text; treated as data, never instructions
    height: Dimension = Field(default_factory=Dimension)
    width: Dimension = Field(default_factory=Dimension)
    depth: Dimension = Field(default_factory=Dimension)
    radius: Dimension = Field(default_factory=Dimension)
    appearance: str = ""
    reference_image: str | None = None
    has_handle: bool | None = None  # None = user did not say
    open_top: bool | None = None
    provenance: dict[str, FieldProvenance] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)

    @field_validator("reference_image")
    @classmethod
    def _ref_image(cls, v):
        return validate_reference_image(v)

    @model_validator(mode="after")
    def _role_kind(self):
        if self.kind == AssetKind.robot_visual and self.role == AssetRole.dynamic:
            raise ValueError("robot_visual is appearance only; role must be visual_only or static")
        if self.kind == AssetKind.environment_splat and self.role != AssetRole.visual_only:
            raise ValueError("environment_splat is a visual background; role must be visual_only")
        return self

    def unknown_dims(self) -> list[str]:
        return [n for n in ("height", "width", "depth", "radius")
                if getattr(self, n).value_m is None]


class ProxyShape(str, Enum):
    box = "box"
    cylinder = "cylinder"
    plane = "plane"


class CollisionProxy(BaseModel):
    """Full extents in metres: box [x,y,z]; cylinder [d,d,h]; plane [x,y,thickness]."""
    shape: ProxyShape = ProxyShape.box
    size_m: list[float]
    position: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    quaternion_xyzw: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])
    basis: Literal["user_explicit", "proxy_estimate", "measured"] = "proxy_estimate"

    @model_validator(mode="after")
    def _size(self):
        if len(self.size_m) != 3 or any(not math.isfinite(v) or v <= 0 for v in self.size_m):
            raise ValueError("proxy size_m needs three positive finite full extents")
        if self.shape == ProxyShape.cylinder and abs(self.size_m[0] - self.size_m[1]) > 1e-9:
            raise ValueError("cylinder proxy needs equal x/y diameters")
        if len(self.position) != 3 or len(self.quaternion_xyzw) != 4:
            raise ValueError("proxy pose needs position[3] and quaternion_xyzw[4]")
        return self


class PhysicsSpec(BaseModel):
    """Separate from appearance. Proxy-based inertia is an estimate, never measured."""
    role: AssetRole = AssetRole.dynamic
    mass: Mass = Field(default_factory=Mass)
    friction: list[float] | None = None
    friction_provenance: FieldProvenance = Field(default_factory=FieldProvenance)
    collision_candidates: list[CollisionProxy] = Field(default_factory=list)
    com: list[float] | None = None
    inertia_basis: Literal["proxy_estimate", "user_explicit", "unknown"] = "unknown"
    contact_notes: str = ""
    graspable: bool | None = None  # None = unknown; never auto-true
    graspable_provenance: FieldProvenance = Field(default_factory=FieldProvenance)
    validation: list[str] = Field(default_factory=list)

    @field_validator("friction")
    @classmethod
    def _friction(cls, v):
        if v is not None and (len(v) != 3 or any(not math.isfinite(x) or x < 0 for x in v)):
            raise ValueError("friction needs three finite non-negative values")
        return v


class SceneMode(str, Enum):
    background_scene = "background_scene"
    composed_scene = "composed_scene"


class AssetNode(BaseModel):
    node_id: str
    asset_ref: str  # "imported/<slug>" | "scenes/<asset_id>/<rev>" | "builtin/<name>"
    role: AssetRole = AssetRole.static
    position: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    quaternion_xyzw: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])
    uniform_scale: float = 1.0
    collider_ref: str | None = None
    depends_on: list[str] = Field(default_factory=list)
    mass_kg: float | None = None
    friction: list[float] | None = None
    collision_mode: Literal["auto", "primitive", "compound"] = "auto"

    @field_validator("uniform_scale")
    @classmethod
    def _scale(cls, v):
        return _positive_finite(v, "uniform_scale")

    @field_validator("mass_kg")
    @classmethod
    def _mass(cls, v):
        return _positive_finite(v, "mass_kg")


class SceneRecipe(BaseModel):
    schema_version: str = GENERATION_SCHEMA_VERSION
    recipe_id: str
    task_goal: str = ""
    scene_mode: SceneMode = SceneMode.composed_scene
    robot_ref: str | None = None
    task_asset_ref: str | None = None
    support_plane_z: float = 0.0
    background_ref: str | None = None  # "scenes/<asset_id>/<rev>"
    asset_nodes: list[AssetNode] = Field(default_factory=list)
    colliders: list[dict] = Field(default_factory=list)  # SceneCollider dicts
    use_default_table: bool = True
    sensor_renderer: Literal["mujoco", "hybrid"] = "mujoco"
    layout_constraints: list[str] = Field(default_factory=list)
    calibration: dict = Field(default_factory=dict)
    goal_tray_node_id: str | None = None
    goal_xy: list[float] | None = None
    goal_surface_m: float | None = None
    goal_half_extents: list[float] | None = None
    versions: dict = Field(default_factory=dict)
    manifest_hash: str = ""

    @model_validator(mode="after")
    def _checks(self):
        if not math.isfinite(self.support_plane_z) or not -0.10 <= self.support_plane_z <= 0.12:
            raise ValueError("support_plane_z must be finite and within [-0.10, 0.12] m")
        ids = [n.node_id for n in self.asset_nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate asset node ids")
        if self.scene_mode == SceneMode.background_scene and not self.background_ref:
            raise ValueError("background_scene requires a background_ref")
        if self.goal_tray_node_id is not None:
            node = next((n for n in self.asset_nodes if n.node_id == self.goal_tray_node_id), None)
            if node is None:
                raise ValueError(f"goal_tray_node_id {self.goal_tray_node_id!r} not in asset_nodes")
            if node.role != AssetRole.static:
                raise ValueError(f"goal_tray_node_id {self.goal_tray_node_id!r} must have role='static'")
        if self.goal_xy is not None and (len(self.goal_xy) != 2 or any(not math.isfinite(v) for v in self.goal_xy)):
            raise ValueError("goal_xy requires two finite floats [x, y]")
        if self.goal_surface_m is not None and (not math.isfinite(self.goal_surface_m) or self.goal_surface_m <= 0):
            raise ValueError("goal_surface_m must be finite and positive")
        if self.goal_half_extents is not None and (
            len(self.goal_half_extents) != 2 or any(not math.isfinite(v) or v <= 0 for v in self.goal_half_extents)
        ):
            raise ValueError("goal_half_extents requires two positive finite floats [hx, hy]")
        return self


class CompiledGenerationRequest(BaseModel):
    compiler_version: str
    spec_id: str
    spec_revision: int
    base_spec_hash: str = ""
    spec_hash: str
    request_hash: str = ""
    provider: Literal["lux3d", "aholo_world", "local_import"]
    template: Literal["single_object", "environment_workspace", "composed_workspace"]
    prompt_en: str
    prompt_en_original: str
    spec_zh: str
    user_words: str
    payload: dict = Field(default_factory=dict)
    excluded_or_postprocess_fields: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    edited: bool = False


class SessionState(str, Enum):
    draft = "draft"
    clarifying = "clarifying"
    ready = "ready"
    frozen = "frozen"
    generating = "generating"
    review = "review"
    ready_to_apply = "ready_to_apply"
    applied = "applied"


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    ts: str
    kind: Literal["text", "question", "status", "error"] = "text"


class PlannerQuestion(BaseModel):
    id: str
    text: str
    field: str | None = None
    options: list[str] = Field(default_factory=list)
    answer: str | None = None
    status: Literal["open", "answered", "skipped"] = "open"


class SessionBranch(BaseModel):
    branch_id: str
    parent_revision: int
    reason: str
    created: str
    spec_snapshot: dict


class GenerationSession(BaseModel):
    schema_version: str = GENERATION_SCHEMA_VERSION
    session_id: str
    state: SessionState = SessionState.draft
    created: str
    updated: str
    revision: int = 1
    project_id: str | None = None
    messages: list[ChatMessage] = Field(default_factory=list)
    questions: list[PlannerQuestion] = Field(default_factory=list)
    asset_spec: AssetSpec = Field(default_factory=AssetSpec)
    physics_spec: PhysicsSpec = Field(default_factory=PhysicsSpec)
    branches: list[SessionBranch] = Field(default_factory=list)
    compiled: CompiledGenerationRequest | None = None
    job_ids: list[str] = Field(default_factory=list)
    recipe_id: str | None = None
    capability_snapshot: dict = Field(default_factory=dict)
    aux_state: str = "waiting_for_configuration"


class PlannerReply(BaseModel):
    """Strict JSON contract for the LLM. Anything else is rejected/repaired once."""
    reply_zh: str = ""
    spec_patch: dict = Field(default_factory=dict)
    physics_patch: dict = Field(default_factory=dict)
    questions: list[PlannerQuestion] = Field(default_factory=list)
    complete: bool = False
    warnings: list[str] = Field(default_factory=list)

    @field_validator("questions")
    @classmethod
    def _max_questions(cls, v):
        if len(v) > 3:
            raise ValueError("at most 3 clarification questions per round")
        return v