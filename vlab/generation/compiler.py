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
import re

from .schema import (AssetKind, AssetSpec, CompiledGenerationRequest, GenerationSession,
                     PhysicsSpec, SceneRecipe, validate_reference_image)

COMPILER_VERSION = "1.1"

LUX3D_PAYLOAD_DEFAULTS = {
    "version": "G1-Turbo", "style": "photorealistic", "face_count": 20000,
    "output_format": ["glb"], "enable_pbr": True,
}
POSTPROCESS_FIELDS = ["mass_kg", "friction", "joint_limits", "collision_proxies",
                      "seed", "layout", "urdf", "negative_prompt"]


class CompileError(Exception):
    pass


def spec_hash(spec: AssetSpec, physics: PhysicsSpec) -> str:
    asset_dump = spec.model_dump()
    if asset_dump.get("reference_image") is None:
        asset_dump.pop("reference_image", None)
    canonical = json.dumps({"asset": asset_dump, "physics": physics.model_dump()},
                           sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def compiled_request_hash(*, compiler_version: str, base_spec_hash: str, provider: str,
                          template: str, prompt_en: str, payload: dict) -> str:
    """Hash covering both the structured spec and the compiled provider payload/prompt override."""
    canonical = json.dumps({
        "compiler_version": compiler_version,
        "base_spec_hash": base_spec_hash,
        "provider": provider,
        "template": template,
        "prompt_en": prompt_en,
        "payload": payload,
    }, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _cm(v: float) -> str:
    cm = round(v * 100, 2)
    return f"{cm:g} cm"


_NO_HANDLE_RE = re.compile(r"(?:\bhandleless\b|\bno\s+handle\b|\bwithout\s+(?:a\s+)?handle\b|无把手|没有把手)")
_WITH_HANDLE_RE = re.compile(r"(?:\bwith\s+(?:a\s+)?handle\b|带把手|有把手)")
_OPEN_TOP_RE = re.compile(r"(?:\bopen[\s-]+top\b|开口)")
_CLOSED_TOP_RE = re.compile(r"(?:\bclosed[\s-]+top\b|\bsealed[\s-]+top\b|闭口|封口)")
_VESSEL_FEATURE_KINDS = {AssetKind.task_object, AssetKind.static_prop}

_UNIT_SCALE = {"m": 1.0, "米": 1.0, "cm": 0.01, "厘米": 0.01, "mm": 0.001, "毫米": 0.001}
_DIM_PATTERNS = [
    ("height", re.compile(r"(?:(\d+(?:\.\d+)?)\s*(cm|mm|m|厘米|毫米|米)\s*(?:tall|high|height)\b|(?:height|tall|high|高度|高)\s*[:=：]?\s*(\d+(?:\.\d+)?)\s*(cm|mm|m|厘米|毫米|米)\b)")),
    ("width", re.compile(r"(?:(\d+(?:\.\d+)?)\s*(cm|mm|m|厘米|毫米|米)\s*(?:wide|width)\b|(?:width|wide|宽度|宽)\s*[:=：]?\s*(\d+(?:\.\d+)?)\s*(cm|mm|m|厘米|毫米|米)\b)")),
    ("depth", re.compile(r"(?:(\d+(?:\.\d+)?)\s*(cm|mm|m|厘米|毫米|米)\s*(?:deep|depth)\b|(?:depth|deep|深度|深)\s*[:=：]?\s*(\d+(?:\.\d+)?)\s*(cm|mm|m|厘米|毫米|米)\b)")),
    ("diameter", re.compile(r"(?:(\d+(?:\.\d+)?)\s*(cm|mm|m|厘米|毫米|米)\s*(?:outside\s+diameter|diameter)\b|(?:outside\s+diameter|diameter|直径)\s*[:=：]?\s*(\d+(?:\.\d+)?)\s*(cm|mm|m|厘米|毫米|米)\b)")),
    ("radius", re.compile(r"(?:(\d+(?:\.\d+)?)\s*(cm|mm|m|厘米|毫米|米)\s*radius\b|(?:radius|半径)\s*[:=：]?\s*(\d+(?:\.\d+)?)\s*(cm|mm|m|厘米|毫米|米)\b)")),
]


def _contradictions(spec: AssetSpec) -> list[str]:
    if spec.kind not in _VESSEL_FEATURE_KINDS:
        return []
    words = (spec.user_words or "").lower()
    issues = []
    if spec.has_handle is True and _NO_HANDLE_RE.search(words):
        issues.append("user_words 说无把手，但规格 has_handle=true")
    if spec.has_handle is False and _WITH_HANDLE_RE.search(words):
        issues.append("user_words 说带把手，但规格 has_handle=false")
    return issues


def validate_prompt_against_spec(spec: AssetSpec, prompt_en: str) -> list[str]:
    """Validate an edited/compiled prompt against hard negation and dimension constraints."""
    if not isinstance(prompt_en, str) or not prompt_en.strip():
        raise CompileError("prompt_en 不能为空")
    text = prompt_en.lower()
    issues = []
    if spec.kind in _VESSEL_FEATURE_KINDS:
        if spec.has_handle is True and _NO_HANDLE_RE.search(text):
            issues.append("提示词包含无把手表述，与规格 has_handle=true 冲突")
        if spec.has_handle is False and _WITH_HANDLE_RE.search(text):
            issues.append("提示词包含带把手表述，与规格 has_handle=false 冲突")
        if spec.open_top is True and _CLOSED_TOP_RE.search(text):
            issues.append("提示词包含闭口/封口表述，与规格 open_top=true 冲突")
        if spec.open_top is False and _OPEN_TOP_RE.search(text):
            issues.append("提示词包含开口表述，与规格 open_top=false 冲突")
    for axis, pattern in _DIM_PATTERNS:
        target_dim = spec.radius if axis in ("diameter", "radius") else getattr(spec, axis)
        if target_dim.value_m is None:
            continue
        expected_m = target_dim.value_m * (2.0 if axis == "diameter" else 1.0)
        for m in pattern.finditer(text):
            num_s, unit = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
            val_m = float(num_s) * _UNIT_SCALE[unit]
            tol = max(0.002, 0.05 * expected_m)
            if abs(val_m - expected_m) > tol:
                issues.append(f"提示词 {axis}={val_m:g} m 与规格 {expected_m:g} m 冲突")
    return issues


def build_en_prompt(spec: AssetSpec) -> str:
    if spec.kind == AssetKind.environment_splat:
        base = spec.description_zh or spec.appearance or "an empty workspace"
        return (f"Environment background: {base}. Empty central workspace, no robots, no people, "
                "no text. Photorealistic indoor scene.")
    if spec.kind == AssetKind.environment_mesh:
        base = spec.name or spec.description_zh or spec.appearance or "modern robotics workbench tabletop"
        dims = []
        if spec.width.value_m is not None:
            dims.append(f"{_cm(spec.width.value_m)} wide")
        if spec.depth.value_m is not None:
            dims.append(f"{_cm(spec.depth.value_m)} deep")
        if spec.height.value_m is not None:
            dims.append(f"{_cm(spec.height.value_m)} tall")
        dtxt = f" ({', '.join(dims)})" if dims else ""
        return (f"3D environment mesh of {base}{dtxt}, flat horizontal worktop, "
                "empty central placement zone, no robots, no people, no text.")
    if spec.kind == AssetKind.robot_visual:
        noun = "robot arm or component mesh (static visual model, no articulated joints)"
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
    if spec.kind in _VESSEL_FEATURE_KINDS:
        if spec.has_handle is not None:
            parts.append("with a handle" if spec.has_handle else "handleless")
        if spec.open_top is not None:
            parts.append("an open top" if spec.open_top else "a closed top")
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
             f"外观：{spec.appearance or '未说明'}",
             f"质量：{physics.mass.value_kg if physics.mass.value_kg is not None else '未知'} kg"
             f"（{physics.mass.provenance.source.value}）；可抓取：{physics.graspable}"]
    if spec.kind in _VESSEL_FEATURE_KINDS:
        if spec.has_handle is not None:
            lines.append(f"把手：{'有' if spec.has_handle else '无'}")
        if spec.open_top is not None:
            lines.append(f"开口：{'是' if spec.open_top else '否'}")
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
        w.append("environment_mesh 在线生成走 Lux3D 网格，按规格尺度标定并自动分解静态碰撞盒（非高斯全景）")
    if physics.mass.value_kg is not None and physics.mass.provenance.verification.value != "measured":
        w.append("质量为估计/默认值，非测量；代理惯量同样为估计")
    return w


def compile_session(session: GenerationSession, provider: str | None = None,
                    prompt_override: str | None = None,
                    reference_image_override: str | None = None,
                    generation_mode: str = "auto") -> CompiledGenerationRequest:
    spec, physics = session.asset_spec, session.physics_spec
    issues = _contradictions(spec)
    if issues:
        raise CompileError("规格与用户原话矛盾，已阻止提交：" + "；".join(issues))
    if generation_mode not in {"auto", "text_to_3d", "img_to_3d"}:
        raise CompileError(f"不支持的 generation_mode: {generation_mode!r}")
    try:
        ref_img = validate_reference_image(
            reference_image_override if reference_image_override is not None else spec.reference_image
        )
    except ValueError as exc:
        raise CompileError(str(exc)) from exc
    if provider is None:
        provider = "aholo_world" if spec.kind == AssetKind.environment_splat else "lux3d"
    if spec.kind == AssetKind.environment_mesh and provider == "aholo_world":
        raise CompileError("environment_mesh 需输出三角网格，请使用 lux3d 在线生成或本地导入")
    template = "environment_workspace" if (provider == "aholo_world" or spec.kind == AssetKind.environment_mesh) else "single_object"
    original_en = build_en_prompt(spec)
    en = original_en
    if prompt_override is not None:
        prompt_issues = validate_prompt_against_spec(spec, prompt_override)
        if prompt_issues:
            raise CompileError("提示词修改与硬约束冲突，已阻止提交：" + "；".join(prompt_issues))
        en = prompt_override
    warnings = _warnings(spec, physics)
    payload = {"prompt": en}
    if provider == "lux3d":
        payload.update(LUX3D_PAYLOAD_DEFAULTS)
        if ref_img:
            mode = "img_to_3d" if generation_mode == "auto" else generation_mode
            payload["generation_mode"] = mode
            payload["img"] = ref_img
            if mode == "img_to_3d":
                payload.pop("style", None)
                warnings.append("img_to_3d 以参考图为主驱动生成，风格/英文文本不传给 Lux3D；仍需导入后按规格尺寸标定")
            else:
                payload["use_text_with_img"] = True
        elif generation_mode == "img_to_3d":
            raise CompileError("img_to_3d 需要先在规格或预览中提供参考图 (reference_image)")
    elif provider == "aholo_world" and ref_img:
        if not ref_img.lower().startswith(("http://", "https://")):
            raise CompileError("Aholo World ref_image 仅支持 http(s) URL，暂不支持本地 data:image 上传")
        payload["ref_image"] = ref_img
    base_hash = spec_hash(spec, physics)
    req_hash = compiled_request_hash(
        compiler_version=COMPILER_VERSION, base_spec_hash=base_hash,
        provider=provider, template=template, prompt_en=en, payload=payload)
    return CompiledGenerationRequest(
        compiler_version=COMPILER_VERSION, spec_id=session.session_id,
        spec_revision=session.revision, base_spec_hash=base_hash,
        spec_hash=req_hash, request_hash=req_hash,
        provider=provider, template=template, prompt_en=en, prompt_en_original=original_en,
        spec_zh=build_zh_spec(spec, physics), user_words=spec.user_words, payload=payload,
        excluded_or_postprocess_fields=list(POSTPROCESS_FIELDS), warnings=warnings,
        edited=prompt_override is not None and prompt_override != original_en)


def compile_scene_prompt(recipe: SceneRecipe) -> str:
    """composed_workspace template: parameterized robot workspace description."""
    nodes = ", ".join(f"{n.asset_ref}({n.role.value})" for n in recipe.asset_nodes) or "no extra props"
    return (f"Composed robot workspace on a parameterized workbench at z=0. Task: {recipe.task_goal}. "
            f"Manipulable and static assets assembled separately: {nodes}. "
            "Fixed front camera, clear central workspace, no people, no text.")
