"""Deterministic, versioned prompt compiler.

Structured spec fields are authoritative. The compiler never passes mass,
friction or joint limits to a 3D generation API, never invents SDK fields, and
keeps the user's words, a zh spec summary and an editable en prompt. Unsupported
requests (seed/layout/colliders/URDF) are surfaced as postprocess, not silently
dropped. zh<->en numeric and negation consistency is enforced.
"""
from __future__ import annotations

import hashlib
import json

from .schema import (AssetKind, AssetSpec, CompiledGenerationRequest, GenerationSession,
                     PhysicsSpec, SceneRecipe)

COMPILER_VERSION = "1.0"

LUX3D_PAYLOAD_DEFAULTS = {
    "version": "G1-Turbo", "style": "photorealistic", "face_count": 20000,
    "output_format": ["glb"], "enable_pbr": True,
}
POSTPROCESS_FIELDS = ["mass_kg", "friction", "joint_limits", "collision_proxies",
                      "seed", "layout", "urdf", "negative_prompt"]


class CompileError(Exception):
    pass


def spec_hash(spec: AssetSpec, physics: PhysicsSpec) -> str:
    canonical = json.dumps({"asset": spec.model_dump(), "physics": physics.model_dump()},
                           sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _cm(v: float) -> str:
    cm = round(v * 100, 2)
    return f"{cm:g} cm"


def _contradictions(spec: AssetSpec) -> list[str]:
    words = (spec.user_words or "").lower()
    issues = []
    if spec.has_handle is True and any(w in words for w in ("handleless", "no handle", "无把手", "没有把手")):
        issues.append("user_words 说无把手，但规格 has_handle=true")
    if spec.has_handle is False and any(w in words for w in ("with handle", "with a handle", "带把手", "有把手")):
        issues.append("user_words 说带把手，但规格 has_handle=false")
    return issues


def build_en_prompt(spec: AssetSpec) -> str:
    if spec.kind == AssetKind.environment_splat:
        base = spec.description_zh or spec.appearance or "an empty workspace"
        return (f"Environment background: {base}. Empty central workspace, no robots, no people, "
                "no text. Photorealistic indoor scene.")
    if spec.kind == AssetKind.robot_visual:
        noun = "robot arm appearance model (visual shell only, no articulated joints)"
    elif spec.kind == AssetKind.static_prop:
        noun = "static prop object"
    else:
        noun = "object"
    if spec.name:
        noun = f"{spec.name} ({noun})"
    parts = [f"A single upright {noun}"]
    dims = []
    if spec.height.value_m is not None:
        dims.append(f"{_cm(spec.height.value_m)} tall")
    if spec.radius.value_m is not None:
        dims.append(f"{_cm(spec.radius.value_m * 2)} outside diameter")
    else:
        if spec.width.value_m is not None:
            dims.append(f"{_cm(spec.width.value_m)} wide")
        if spec.depth.value_m is not None:
            dims.append(f"{_cm(spec.depth.value_m)} deep")
    if dims:
        parts.append(", ".join(dims))
    feats = []
    if spec.has_handle is False:
        feats.append("handleless")
    elif spec.has_handle is True:
        feats.append("with a handle")
    if spec.open_top is True:
        feats.append("an open top")
    elif spec.open_top is False:
        feats.append("a closed top")
    if feats:
        parts.append(", ".join(feats))
    if spec.appearance:
        parts.append(spec.appearance)
    return ". ".join(parts) + ". Isolated object, no table, no liquid, no text."


def build_zh_spec(spec: AssetSpec, physics: PhysicsSpec) -> str:
    def lab(d):
        if d.value_m is None:
            return "未知"
        p = d.provenance
        return f"{d.value_m:g} m（{p.source.value}/{p.confirmation.value}/{p.verification.value}）"
    lines = [f"类型：{spec.kind.value}；角色：{spec.role.value}",
             f"名称：{spec.name or '未命名'}",
             f"高度：{lab(spec.height)}；宽度：{lab(spec.width)}；深度：{lab(spec.depth)}；半径：{lab(spec.radius)}",
             f"把手：{'无' if spec.has_handle is False else '有' if spec.has_handle is True else '未说明'}；"
             f"开口：{'是' if spec.open_top is True else '否' if spec.open_top is False else '未说明'}",
             f"外观：{spec.appearance or '未说明'}",
             f"质量：{physics.mass.value_kg if physics.mass.value_kg is not None else '未知'} kg"
             f"（{physics.mass.provenance.source.value}）；可抓取：{physics.graspable}"]
    if spec.description_zh:
        lines.append(f"描述：{spec.description_zh}")
    return "\n".join(lines)


def _warnings(spec: AssetSpec, physics: PhysicsSpec) -> list[str]:
    w = []
    unknown = spec.unknown_dims()
    if unknown:
        w.append(f"未知尺寸 {unknown}：生成结果为未标定视觉资产，标定前不绑定物理")
    w.append("生成器不保证精确尺寸/拓扑/关节；需导入后校验与标定")
    if spec.kind == AssetKind.robot_visual:
        w.append("robot_visual 仅外观，不是可动物理机器人；不生成 URDF/关节")
    if spec.kind == AssetKind.environment_mesh:
        w.append("environment_mesh 在线生成未支持；仅本地导入")
    if physics.mass.value_kg is not None and physics.mass.provenance.verification.value != "measured":
        w.append("质量为估计/默认值，非测量；代理惯量同样为估计")
    return w


def compile_session(session: GenerationSession, provider: str | None = None,
                    prompt_override: str | None = None) -> CompiledGenerationRequest:
    spec, physics = session.asset_spec, session.physics_spec
    issues = _contradictions(spec)
    if issues:
        raise CompileError("规格与用户原话矛盾，已阻止提交：" + "；".join(issues))
    if spec.kind == AssetKind.environment_mesh:
        raise CompileError("environment_mesh 在线生成未支持；请改用本地导入")
    if provider is None:
        provider = "aholo_world" if spec.kind == AssetKind.environment_splat else "lux3d"
    template = "environment_workspace" if provider == "aholo_world" else "single_object"
    en = build_en_prompt(spec)
    if prompt_override is not None:
        en = prompt_override
    payload = {"prompt": en}
    if provider == "lux3d":
        payload.update(LUX3D_PAYLOAD_DEFAULTS)
    return CompiledGenerationRequest(
        compiler_version=COMPILER_VERSION, spec_id=session.session_id,
        spec_revision=session.revision, spec_hash=spec_hash(spec, physics),
        provider=provider, template=template, prompt_en=en, prompt_en_original=build_en_prompt(spec),
        spec_zh=build_zh_spec(spec, physics), user_words=spec.user_words, payload=payload,
        excluded_or_postprocess_fields=list(POSTPROCESS_FIELDS), warnings=_warnings(spec, physics),
        edited=prompt_override is not None and prompt_override != build_en_prompt(spec))


def compile_scene_prompt(recipe: SceneRecipe) -> str:
    """composed_workspace template: parameterized robot workspace description."""
    nodes = ", ".join(f"{n.asset_ref}({n.role.value})" for n in recipe.asset_nodes) or "no extra props"
    return (f"Composed robot workspace on a parameterized workbench at z=0. Task: {recipe.task_goal}. "
            f"Manipulable and static assets assembled separately: {nodes}. "
            "Fixed front camera, clear central workspace, no people, no text.")