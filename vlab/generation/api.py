"""HTTP routes for the generation loop, unified under /api/v1.

Credentials are read from backend env only; nothing here returns API keys to the
browser, message history or error stacks. Model output never becomes a path or
shell string. LLM replies never auto-trigger paid generation (confirm is a
separate, explicit product interaction handled by the orchestrator).
"""
from __future__ import annotations

from pathlib import Path
import re

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError

from ..aux_model import AuxModel
from .capabilities import capability_registry
from .compiler import CompileError, compile_session, spec_hash
from .object_provider import ObjectJobStore
from .orchestrator import Orchestrator
from .planner import Planner, PlannerError, apply_reply
from .scene_recipe import (RecipeStore, apply_mutate, build_recipe_from_session,
                           validate_recipe)
from .schema import (AssetKind, AssetRole, AssetSpec, ChatMessage, Confirmation, Dimension,
                     FieldProvenance, FieldSource, GenerationSession, Mass, PhysicsSpec,
                     PlannerQuestion, SceneRecipe, SessionBranch, SessionState, Verification)
from .sessions import RevisionConflict, SessionDataError, SessionStore, _now, new_id


class CreateSessionBody(BaseModel):
    project_id: str | None = None
    initial_text: str | None = None
    kind: str | None = None
    role: str | None = None


class MessageBody(BaseModel):
    content: str
    expected_revision: int


class SpecPatchBody(BaseModel):
    expected_revision: int
    kind: str | None = None
    role: str | None = None
    name: str | None = None
    description_zh: str | None = None
    appearance: str | None = None
    reference_image: str | None = None
    user_words: str | None = None
    has_handle: bool | None = None
    open_top: bool | None = None
    height_m: float | None = None
    width_m: float | None = None
    depth_m: float | None = None
    radius_m: float | None = None
    mass_kg: float | None = None
    friction: list[float] | None = None
    graspable: bool | None = None
    tags: list[str] | None = None
    accept_defaults: list[str] = []


class CompileBody(BaseModel):
    expected_revision: int | None = None
    provider: str | None = None
    prompt_override: str | None = None
    reference_image: str | None = None
    generation_mode: str = "auto"


class AnswerBody(BaseModel):
    question_id: str
    answer: str
    expected_revision: int


class BranchBody(BaseModel):
    reason: str
    expected_revision: int


class ConfirmBody(BaseModel):
    client_request_id: str | None = None
    expected_revision: int | None = None
    expected_compiled_revision: int | None = None
    expected_request_hash: str | None = None
    provider: str | None = None
    remote_id: str | None = None


class RetryBody(BaseModel):
    node: str


class RecoverBody(BaseModel):
    remote_id: str
    provider: str | None = None


class ImportRemoteBody(BaseModel):
    remote_id: str
    provider: str | None = None
    client_request_id: str | None = None
    expected_revision: int | None = None


class RecipeBody(BaseModel):
    recipe_id: str | None = None
    task_goal: str | None = None
    scene_mode: str = "composed_scene"
    robot_ref: str | None = None
    task_asset_ref: str | None = None
    background_ref: str | None = None
    use_default_table: bool = True
    sensor_renderer: str = "mujoco"
    colliders: list[dict] = []
    asset_nodes: list[dict] | None = None
    layout_constraints: list[str] = []
    calibration: dict = {}
    use_as_goal_tray: bool = False
    goal_tray_node_id: str | None = None
    goal_xy: list[float] | None = None
    goal_surface_m: float | None = None
    goal_half_extents: list[float] | None = None
    support_plane_z: float = 0.0


class PartSplitBody(BaseModel):
    segments: int = 4
    axis: str = "auto"
    remote: bool = False


class RecipeApplyBody(BaseModel):
    project_id: str = "default"
    expected_project_version: int


def _user_prov() -> FieldProvenance:
    return FieldProvenance(source=FieldSource.user_explicit, confirmation=Confirmation.accepted,
                           verification=Verification.unverified)


_DEFAULT_ROLE_BY_KIND = {
    "task_object": AssetRole.dynamic.value,
    "static_prop": AssetRole.static.value,
    "robot_visual": AssetRole.visual_only.value,
    "environment_splat": AssetRole.visual_only.value,
    "environment_mesh": AssetRole.visual_only.value,
}


def _is_user_hard_prov(prov_dict: dict | None) -> bool:
    return bool(
        prov_dict
        and prov_dict.get("source") == FieldSource.user_explicit.value
        and prov_dict.get("confirmation") == Confirmation.accepted.value
    )


def apply_user_patch(session: GenerationSession, body: SpecPatchBody) -> None:
    provided = body.model_fields_set
    spec = session.asset_spec
    data = spec.model_dump()
    for key in ("kind", "role", "name", "description_zh", "appearance", "reference_image",
                "user_words", "has_handle", "open_top", "tags"):
        if key in provided:
            data[key] = getattr(body, key)
            data.setdefault("provenance", {})[key] = _user_prov().model_dump()
    if "kind" in provided and "role" not in provided:
        role_prov = data.get("provenance", {}).get("role")
        if not _is_user_hard_prov(role_prov):
            if data["kind"] == AssetKind.environment_splat.value:
                data["role"] = AssetRole.visual_only.value
            elif data["kind"] == AssetKind.robot_visual.value and data.get("role") == AssetRole.dynamic.value:
                data["role"] = AssetRole.visual_only.value
    for key, dim in (("height_m", "height"), ("width_m", "width"), ("depth_m", "depth"), ("radius_m", "radius")):
        if key in provided:
            val = getattr(body, key)
            data[dim] = {"value_m": val, "provenance": _user_prov().model_dump()}
    try:
        session.asset_spec = AssetSpec.model_validate(data)
    except Exception as exc:
        raise ValueError(f"asset_spec invalid: {exc}")
    phys = session.physics_spec.model_dump()
    phys["role"] = session.asset_spec.role.value
    if "mass_kg" in provided:
        phys["mass"] = {"value_kg": body.mass_kg, "provenance": _user_prov().model_dump()}
    if "friction" in provided:
        phys["friction"] = body.friction
        phys["friction_provenance"] = _user_prov().model_dump()
    if "graspable" in provided:
        phys["graspable"] = body.graspable
        phys["graspable_provenance"] = _user_prov().model_dump()
    try:
        session.physics_spec = PhysicsSpec.model_validate(phys)
    except Exception as exc:
        raise ValueError(f"physics_spec invalid: {exc}")
    for field in body.accept_defaults:
        prov = session.asset_spec.provenance.get(field)
        if prov:
            prov.confirmation = Confirmation.accepted


_LENGTH_RE = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*(cm|mm|m|厘米|毫米|米)?\s*$", re.IGNORECASE)
_MASS_RE = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*(kg|g|千克|公斤|克)?\s*$", re.IGNORECASE)
_LEN_UNITS = {"m": 1.0, "米": 1.0, "cm": 0.01, "厘米": 0.01, "mm": 0.001, "毫米": 0.001}
_FIELD_ALIASES = {
    "height": "height_m", "height_m": "height_m",
    "width": "width_m", "width_m": "width_m",
    "depth": "depth_m", "depth_m": "depth_m",
    "radius": "radius_m", "radius_m": "radius_m",
    "diameter": "diameter_m", "diameter_m": "diameter_m",
    "mass": "mass_kg", "mass_kg": "mass_kg",
    "has_handle": "has_handle", "handle": "has_handle",
    "open_top": "open_top",
    "graspable": "graspable",
    "friction": "friction",
    "appearance": "appearance", "color": "appearance",
    "name": "name", "kind": "kind", "role": "role",
}
_TEXT_FIELD_HINTS = [
    ("height_m", ("高度", "多高", "height", "tall")),
    ("width_m", ("宽度", "多宽", "width", "wide")),
    ("depth_m", ("深度", "多深", "depth", "deep")),
    ("diameter_m", ("直径", "diameter")),
    ("radius_m", ("半径", "radius")),
    ("mass_kg", ("质量", "重量", "多重", "mass", "weight")),
    ("has_handle", ("把手", "handle")),
    ("open_top", ("开口", "open top", "open_top")),
    ("graspable", ("可抓取", "graspable")),
    ("friction", ("摩擦", "friction")),
    ("appearance", ("外观", "颜色", "材质", "appearance")),
    ("name", ("名称", "名字", "叫什么")),
]


def _resolve_question_field(q: PlannerQuestion) -> str | None:
    if q.field:
        return _FIELD_ALIASES.get(q.field.strip().lower(), q.field.strip().lower())
    qid = (q.id or "").strip().lower()
    if qid in _FIELD_ALIASES:
        return _FIELD_ALIASES[qid]
    text = (q.text or "").lower()
    matches = [field for field, cues in _TEXT_FIELD_HINTS if any(c in text for c in cues)]
    if len(matches) == 1:
        return matches[0]
    return None


def _parse_answer_patch(q: PlannerQuestion, answer: str) -> tuple[dict, str | None]:
    ans = answer.strip()
    if not ans:
        return {}, "回答不能为空"
    field = _resolve_question_field(q)
    if not field:
        return {}, None
    if ans.lower() in {"未知", "不知道", "不确定", "跳过", "skip", "unknown", "n/a"}:
        return {}, None
    if field in {"height_m", "width_m", "depth_m", "radius_m", "diameter_m"}:
        m = _LENGTH_RE.match(ans)
        if not m:
            return {}, f"无法将回答 {ans!r} 解析为长度（示例：20 cm 或 0.2 m）"
        val = float(m.group(1)) * _LEN_UNITS[(m.group(2) or "m").lower()]
        if field == "diameter_m":
            return {"radius_m": val / 2.0}, None
        return {field: val}, None
    if field == "mass_kg":
        m = _MASS_RE.match(ans)
        if not m:
            return {}, f"无法将回答 {ans!r} 解析为质量（示例：0.15 kg 或 150 g）"
        unit = (m.group(2) or "kg").lower()
        val = float(m.group(1)) * (0.001 if unit in {"g", "克"} else 1.0)
        return {"mass_kg": val}, None
    if field in {"has_handle", "open_top", "graspable"}:
        low = ans.lower()
        if low in {"是", "有", "yes", "true", "带把手", "有把手", "开口", "可抓取"}:
            return {field: True}, None
        if low in {"否", "无", "没有", "no", "false", "无把手", "没有把手", "闭口", "封闭", "不可抓取"}:
            return {field: False}, None
        return {}, f"无法将回答 {ans!r} 解析为是/否布尔值"
    if field == "friction":
        parts = [p for p in re.split(r"[\s,，]+", ans) if p]
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            return {}, f"无法将回答 {ans!r} 解析为 3 个摩擦数值"
        if len(vals) != 3:
            return {}, "摩擦需要 3 个数值 [slide, torsion, rolling]"
        return {"friction": vals}, None
    if field in {"appearance", "name", "kind", "role"}:
        return {field: ans}, None
    return {}, None


def _spec_has_meaningful_content(s: GenerationSession) -> bool:
    sp = s.asset_spec
    return bool(
        sp.name or sp.description_zh or sp.appearance
        or sp.height.value_m is not None or sp.width.value_m is not None
        or sp.depth.value_m is not None or sp.radius.value_m is not None
    )


def register_generation_api(app: FastAPI, lab_root: Path, store=None, world_store=None, commit_scene=None):
    sessions = SessionStore(lab_root / "datasets" / "generation_sessions")
    planner = Planner()
    object_store = ObjectJobStore(lab_root / "datasets" / "generation_jobs", lab_root=lab_root)
    orchestrator = Orchestrator(sessions, object_store, world_store, config_store=store)
    recipes = RecipeStore(lab_root / "datasets" / "scene_recipes")

    def _registry():
        return capability_registry(planner.status())

    def _get(sid) -> GenerationSession:
        try:
            s = sessions.get(sid)
        except SessionDataError as exc:
            raise HTTPException(422, str(exc))
        if s is None:
            raise HTTPException(404, f"unknown generation session {sid}")
        return s

    def _check_rev(s, expected):
        if s.revision != expected:
            raise HTTPException(409, {"error": f"revision conflict: expected {expected}, current {s.revision}",
                                      "current_revision": s.revision, "expected_revision": expected})

    def _save(s, expected):
        try:
            return sessions.save(s, expected)
        except RevisionConflict as exc:
            raise HTTPException(409, {"error": str(exc), "current_revision": exc.current,
                                      "expected_revision": exc.expected})

    def _session_out(s: GenerationSession) -> dict:
        data = s.model_dump()
        if s.compiled is not None:
            cur_base = spec_hash(s.asset_spec, s.physics_spec)
            expected_base = s.compiled.base_spec_hash or s.compiled.spec_hash
            data["compiled_stale"] = (expected_base != cur_base)
        else:
            data["compiled_stale"] = False
        return data

    @app.get("/api/v1/generation/capabilities")
    def generation_capabilities():
        return _registry()

    @app.get("/api/v1/generation/remote-tasks")
    def generation_remote_tasks(provider: str = "lux3d", page: int = 1, page_size: int = 20):
        try:
            return orchestrator.list_remote_tasks(provider, page=page, page_size=page_size)
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @app.post("/api/v1/generation/sessions")
    def create_session(body: CreateSessionBody):
        kind_str = body.kind or AssetKind.task_object.value
        if kind_str not in _DEFAULT_ROLE_BY_KIND:
            raise HTTPException(400, f"unknown asset kind: {kind_str!r}")
        role_str = body.role or _DEFAULT_ROLE_BY_KIND[kind_str]
        prov: dict = {}
        if body.kind is not None:
            prov["kind"] = _user_prov().model_dump()
        if body.role is not None:
            prov["role"] = _user_prov().model_dump()
        spec_data = {
            "kind": kind_str,
            "role": role_str,
            "user_words": body.initial_text or "",
            "provenance": prov,
        }
        try:
            spec = AssetSpec.model_validate(spec_data)
            phys = PhysicsSpec.model_validate({"role": spec.role.value})
        except ValidationError as exc:
            raise HTTPException(400, f"invalid initial asset_spec: {exc}")
        msgs = []
        if body.initial_text:
            msgs.append(ChatMessage(role="user", content=body.initial_text, ts=_now()))
        s = sessions.create(
            project_id=body.project_id,
            capability_snapshot=_registry(),
            aux_state=planner.status()["state"],
            asset_spec=spec,
            physics_spec=phys,
            messages=msgs,
        )
        return _session_out(s)

    @app.get("/api/v1/generation/sessions")
    def list_sessions():
        return {"sessions": sessions.list()}

    @app.get("/api/v1/generation/sessions/{sid}")
    def get_session(sid: str):
        return _session_out(_get(sid))

    @app.post("/api/v1/generation/sessions/{sid}/messages")
    def post_message(sid: str, body: MessageBody):
        with sessions.session_lock(sid):
            s = _get(sid)
            _check_rev(s, body.expected_revision)
            s.messages.append(ChatMessage(role="user", content=body.content, ts=_now()))
            if not s.asset_spec.user_words:
                s.asset_spec.user_words = body.content
            if planner.configured():
                try:
                    reply = planner.plan(s, _registry())
                    warnings = apply_reply(s, reply)
                    if reply.reply_zh:
                        s.messages.append(ChatMessage(role="assistant", content=reply.reply_zh, ts=_now()))
                    for w in warnings:
                        s.messages.append(ChatMessage(role="system", content=w, ts=_now(), kind="status"))
                    s.aux_state = "ready"
                except Exception as exc:
                    s.messages.append(ChatMessage(role="system", content=f"规划器不可用：{exc}。可继续用表单/JSON 编辑。",
                                                  ts=_now(), kind="error"))
                    s.aux_state = planner.status()["state"]
            else:
                s.aux_state = "waiting_for_configuration"
                s.messages.append(ChatMessage(
                    role="system", ts=_now(), kind="status",
                    content="waiting_for_configuration：未配置 AUX_MODEL_BASE_URL/AUX_MODEL_API_KEY，"
                            "对话规划不可用；表单/JSON/本地导入仍可使用。"))
            return _session_out(_save(s, body.expected_revision))

    @app.patch("/api/v1/generation/sessions/{sid}/spec")
    def patch_spec(sid: str, body: SpecPatchBody):
        with sessions.session_lock(sid):
            s = _get(sid)
            _check_rev(s, body.expected_revision)
            try:
                apply_user_patch(s, body)
            except ValueError as exc:
                raise HTTPException(400, str(exc))
            if s.state == SessionState.draft:
                s.state = SessionState.clarifying
            return _session_out(_save(s, body.expected_revision))

    @app.post("/api/v1/generation/sessions/{sid}/answer")
    def answer_question(sid: str, body: AnswerBody):
        with sessions.session_lock(sid):
            s = _get(sid)
            _check_rev(s, body.expected_revision)
            q = next((q for q in s.questions if q.id == body.question_id), None)
            if q is None:
                raise HTTPException(404, "unknown question")
            ans = (body.answer or "").strip()
            if not ans:
                raise HTTPException(400, "answer cannot be empty")
            patch_dict, parse_err = _parse_answer_patch(q, ans)
            if parse_err:
                raise HTTPException(400, parse_err)
            if patch_dict:
                try:
                    apply_user_patch(s, SpecPatchBody(expected_revision=body.expected_revision, **patch_dict))
                except ValueError as exc:
                    raise HTTPException(400, str(exc))
            q.answer, q.status = ans, "answered"
            s.messages.append(ChatMessage(
                role="user",
                content=f"回答问题 [{q.id}]（{q.text}）：{ans}",
                ts=_now(),
                kind="text",
            ))
            open_left = any(item.status == "open" for item in s.questions)
            if planner.configured():
                try:
                    reply = planner.plan(s, _registry())
                    warnings = apply_reply(s, reply)
                    if reply.reply_zh:
                        s.messages.append(ChatMessage(role="assistant", content=reply.reply_zh, ts=_now()))
                    for w in warnings:
                        s.messages.append(ChatMessage(role="system", content=w, ts=_now(), kind="status"))
                    s.aux_state = "ready"
                except Exception as exc:
                    s.messages.append(ChatMessage(
                        role="system",
                        content=f"已记录回答，但规划器不可用：{exc}。可继续用表单/JSON 编辑。",
                        ts=_now(),
                        kind="error",
                    ))
                    s.aux_state = planner.status()["state"]
                    s.state = (SessionState.ready if (not open_left and _spec_has_meaningful_content(s))
                               else SessionState.clarifying)
            else:
                if not patch_dict:
                    s.messages.append(ChatMessage(
                        role="system",
                        content="已记录回答到对话上下文（waiting_for_configuration：辅助模型未配置，若规格字段未自动映射请在右侧表单补充）。",
                        ts=_now(),
                        kind="status",
                    ))
                s.state = (SessionState.ready if (not open_left and _spec_has_meaningful_content(s))
                           else SessionState.clarifying)
            return _session_out(_save(s, body.expected_revision))

    @app.post("/api/v1/generation/sessions/{sid}/compile")
    def compile_preview(sid: str, body: CompileBody):
        with sessions.session_lock(sid):
            s = _get(sid)
            if body.expected_revision is not None:
                _check_rev(s, body.expected_revision)
            try:
                compiled = compile_session(
                    s,
                    provider=body.provider,
                    prompt_override=body.prompt_override,
                    reference_image_override=body.reference_image,
                    generation_mode=body.generation_mode,
                )
            except CompileError as exc:
                raise HTTPException(400, str(exc))
            s.compiled = compiled
            return _session_out(_save(s, s.revision))

    @app.post("/api/v1/generation/sessions/{sid}/branch")
    def create_branch(sid: str, body: BranchBody):
        with sessions.session_lock(sid):
            s = _get(sid)
            _check_rev(s, body.expected_revision)
            s.branches.append(SessionBranch(
                branch_id=new_id("br"), parent_revision=s.revision, reason=body.reason, created=_now(),
                spec_snapshot={"asset_spec": s.asset_spec.model_dump(), "physics_spec": s.physics_spec.model_dump()}))
            return _session_out(_save(s, body.expected_revision))

    @app.post("/api/v1/generation/sessions/{sid}/confirm")
    def confirm_generation(sid: str, body: ConfirmBody):
        _get(sid)
        try:
            result = orchestrator.confirm(
                sid,
                client_request_id=body.client_request_id,
                expected_revision=body.expected_revision,
                expected_compiled_revision=body.expected_compiled_revision,
                expected_request_hash=body.expected_request_hash,
                provider=body.provider,
                remote_id=body.remote_id,
            )
        except RevisionConflict as exc:
            raise HTTPException(409, {"error": str(exc), "current_revision": exc.current,
                                      "expected_revision": exc.expected})
        except CompileError as exc:
            raise HTTPException(400, str(exc))
        except KeyError:
            raise HTTPException(404, "unknown generation session")
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        if isinstance(result, dict) and "session" in result:
            sess = _get(sid)
            result["session"] = _session_out(sess)
        return result

    @app.get("/api/v1/generation/sessions/{sid}/tasks")
    def session_tasks(sid: str):
        try:
            return orchestrator.tasks(sid)
        except KeyError:
            raise HTTPException(404, "unknown generation session")

    @app.post("/api/v1/generation/sessions/{sid}/tasks/{jid}/poll")
    def poll_task(sid: str, jid: str):
        try:
            return orchestrator.poll(sid, jid)
        except KeyError:
            raise HTTPException(404, "job not linked to session")

    @app.post("/api/v1/generation/sessions/{sid}/tasks/{jid}/retry")
    def retry_task(sid: str, jid: str, body: RetryBody):
        try:
            return orchestrator.retry_node(sid, jid, body.node)
        except KeyError:
            raise HTTPException(404, "job not linked to session")
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @app.post("/api/v1/generation/sessions/{sid}/tasks/{jid}/recover")
    def recover_task(sid: str, jid: str, body: RecoverBody):
        try:
            res = orchestrator.recover(sid, jid, body.remote_id, provider=body.provider)
            res["session"] = _session_out(_get(sid))
            return res
        except KeyError:
            raise HTTPException(404, "job not linked to session")
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @app.post("/api/v1/generation/sessions/{sid}/import-remote")
    def import_remote_task(sid: str, body: ImportRemoteBody):
        _get(sid)
        try:
            res = orchestrator.import_remote(
                sid,
                body.remote_id,
                provider=body.provider,
                client_request_id=body.client_request_id,
                expected_revision=body.expected_revision,
            )
        except RevisionConflict as exc:
            raise HTTPException(409, {"error": str(exc), "current_revision": exc.current,
                                      "expected_revision": exc.expected})
        except CompileError as exc:
            raise HTTPException(400, str(exc))
        except KeyError:
            raise HTTPException(404, "unknown generation session")
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        if isinstance(res, dict) and "session" in res:
            res["session"] = _session_out(_get(sid))
        return res

    @app.post("/api/v1/generation/sessions/{sid}/recipe")
    def recipe_from_session(sid: str, body: RecipeBody):
        with sessions.session_lock(sid):
            s = _get(sid)
            try:
                recipe = build_recipe_from_session(s, lab_root, overrides=body.model_dump(exclude_none=True))
            except ValueError as exc:
                raise HTTPException(400, str(exc))
            recipe = recipes.put(recipe)
            s.recipe_id = recipe.recipe_id
            if s.state in {SessionState.review, SessionState.generating, SessionState.ready}:
                s.state = SessionState.ready_to_apply
            _save(s, s.revision)
            return recipe.model_dump()

    @app.get("/api/v1/scene-recipes")
    def list_recipes():
        return {"recipes": recipes.list()}

    @app.get("/api/v1/scene-recipes/{rid}")
    def get_recipe(rid: str):
        r = recipes.get(rid)
        if r is None:
            raise HTTPException(404, f"unknown recipe {rid}")
        return r.model_dump()

    @app.post("/api/v1/scene-recipes/{rid}/validate")
    def validate_recipe_ep(rid: str, project_id: str = "default"):
        r = recipes.get(rid)
        if r is None:
            raise HTTPException(404, f"unknown recipe {rid}")
        if store is None:
            raise HTTPException(409, "config store unavailable")
        return validate_recipe(r, lab_root, store, project_id)

    @app.post("/api/v1/scene-recipes/{rid}/apply")
    def apply_recipe_ep(rid: str, body: RecipeApplyBody):
        r = recipes.get(rid)
        if r is None:
            raise HTTPException(404, f"unknown recipe {rid}")
        if store is None or commit_scene is None:
            raise HTTPException(409, "atomic scene commit unavailable")
        report = validate_recipe(r, lab_root, store, body.project_id)
        if not report["valid"]:
            raise HTTPException(400, {"error": "recipe failed validation; project unchanged",
                                      "issues": report["issues"], "warnings": report["warnings"]})
        saved = commit_scene(body.project_id, apply_mutate(r, lab_root=lab_root), body.expected_project_version)
        return {"version": saved.version, "project_id": saved.project_id,
                "validation_status": saved.validation_status,
                "scene": saved.scene.model_dump(), "task_asset": saved.task.asset,
                "task": saved.task.model_dump(),
                "recipe_id": r.recipe_id, "manifest_hash": report["manifest_hash"],
                "read_back": {"environment": saved.scene.environment.model_dump() if saved.scene.environment else None,
                              "colliders": [c.model_dump() for c in saved.scene.colliders],
                              "asset_nodes": [n.model_dump() for n in saved.scene.asset_nodes],
                              "use_default_table": saved.scene.use_default_table,
                              "work_surface_z": saved.scene.work_surface_z,
                              "sensor_renderer": saved.scene.sensor_renderer,
                              "task": saved.task.model_dump()}}

    @app.post("/api/v1/assets/{slug}/part-split")
    def split_imported_asset(slug: str, body: PartSplitBody):
        from ..scene_assets import contained_file, valid_id
        from asset_pipeline.part_split import split_mesh_file
        import json as _json

        try:
            clean_slug = valid_id(slug)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        asset_dir = (lab_root / "assets" / "imported" / clean_slug).resolve()
        if not (asset_dir / "manifest.json").is_file():
            raise HTTPException(404, f"unknown imported asset {clean_slug!r}")
        mf = _json.loads((asset_dir / "manifest.json").read_text(encoding="utf-8"))

        if body.remote:
            if not object_store.provider.configured():
                raise HTTPException(409, "AHOLO_API_KEY not configured for Lux3D cloud part_split")
            gid = mf.get("generation_id")
            if not gid:
                raise HTTPException(400, "asset has no remote Lux3D generation_id; use local split (remote=false)")
            try:
                st = object_store.provider.get_status(str(gid))
                urls = [o.get("content") for o in (st.get("outputs") or [])
                        if isinstance(o.get("content"), str) and o["content"].startswith("https://")]
                if not urls:
                    raise ValueError("no remote GLB output URL found on generation task")
                remote_split_id = object_store.provider.submit_part_split(urls[0])
                return {"mode": "lux3d_cloud", "slug": clean_slug, "remote_task_id": remote_split_id, "source_task_id": str(gid)}
            except Exception as exc:
                raise HTTPException(400, f"Lux3D part_split failed: {exc}")

        src_rel = mf.get("source_mesh") if (asset_dir / str(mf.get("source_mesh", ""))).is_file() else mf.get("obj", "visual.obj")
        try:
            src_path = contained_file(asset_dir, src_rel)
            parts_dir = asset_dir / "parts"
            if parts_dir.exists():
                import shutil as _shutil
                _shutil.rmtree(parts_dir)
            result = split_mesh_file(
                src_path,
                parts_dir,
                up="y" if src_path.suffix.lower() in (".glb", ".gltf") else "z",
                segments=max(0, min(12, int(body.segments))),
                axis=body.axis if body.axis in ("auto", "x", "y", "z") else "auto",
            )
            mf["part_split"] = result
            (asset_dir / "manifest.json").write_text(_json.dumps(mf, ensure_ascii=False, indent=2), encoding="utf-8")
            return {"mode": "local", "slug": clean_slug, "parts_dir": f"assets/imported/{clean_slug}/parts", **result}
        except Exception as exc:
            raise HTTPException(400, f"local part split failed: {exc}")

    return {"sessions": sessions, "planner": planner, "registry": _registry,
            "objects": object_store, "orchestrator": orchestrator, "recipes": recipes}