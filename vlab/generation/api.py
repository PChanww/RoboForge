"""HTTP routes for the generation loop, unified under /api/v1.

Credentials are read from backend env only; nothing here returns API keys to the
browser, message history or error stacks. Model output never becomes a path or
shell string. LLM replies never auto-trigger paid generation (confirm is a
separate, explicit product interaction handled by the orchestrator).
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ..aux_model import AuxModel
from .capabilities import capability_registry
from .compiler import CompileError, compile_session
from .object_provider import ObjectJobStore
from .orchestrator import Orchestrator
from .planner import Planner, PlannerError, apply_reply
from .scene_recipe import (RecipeStore, apply_mutate, build_recipe_from_session,
                           validate_recipe)
from .schema import (AssetSpec, ChatMessage, Confirmation, Dimension, FieldProvenance,
                     FieldSource, GenerationSession, Mass, PhysicsSpec, SceneRecipe,
                     SessionBranch, SessionState, Verification)
from .sessions import RevisionConflict, SessionStore, _now, new_id


class CreateSessionBody(BaseModel):
    project_id: str | None = None
    initial_text: str | None = None
    kind: str | None = None


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
    provider: str | None = None
    remote_id: str | None = None


class RetryBody(BaseModel):
    node: str


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
    asset_nodes: list[dict] = []
    layout_constraints: list[str] = []
    calibration: dict = {}


class RecipeApplyBody(BaseModel):
    project_id: str = "default"
    expected_project_version: int


def _user_prov() -> FieldProvenance:
    return FieldProvenance(source=FieldSource.user_explicit, confirmation=Confirmation.accepted,
                           verification=Verification.unverified)


def apply_user_patch(session: GenerationSession, body: SpecPatchBody) -> None:
    provided = body.model_fields_set
    spec = session.asset_spec
    data = spec.model_dump()
    for key in ("kind", "role", "name", "description_zh", "appearance", "user_words",
                "has_handle", "open_top", "tags"):
        if key in provided:
            data[key] = getattr(body, key)
            data.setdefault("provenance", {})[key] = _user_prov().model_dump()
    for key, dim in (("height_m", "height"), ("width_m", "width"), ("depth_m", "depth"), ("radius_m", "radius")):
        if key in provided:
            val = getattr(body, key)
            data[dim] = {"value_m": val, "provenance": _user_prov().model_dump()}
    try:
        session.asset_spec = AssetSpec.model_validate(data)
    except Exception as exc:
        raise ValueError(f"asset_spec invalid: {exc}")
    phys = session.physics_spec.model_dump()
    if "mass_kg" in provided:
        phys["mass"] = {"value_kg": body.mass_kg, "provenance": _user_prov().model_dump()}
    if "friction" in provided:
        phys["friction"] = body.friction
        phys["friction_provenance"] = _user_prov().model_dump()
    if "graspable" in provided:
        phys["graspable"] = body.graspable
    try:
        session.physics_spec = PhysicsSpec.model_validate(phys)
    except Exception as exc:
        raise ValueError(f"physics_spec invalid: {exc}")
    for field in body.accept_defaults:
        prov = session.asset_spec.provenance.get(field)
        if prov:
            prov.confirmation = Confirmation.accepted


def register_generation_api(app: FastAPI, lab_root: Path, store=None, world_store=None, commit_scene=None):
    sessions = SessionStore(lab_root / "datasets" / "generation_sessions")
    planner = Planner()
    object_store = ObjectJobStore(lab_root / "datasets" / "generation_jobs", lab_root=lab_root)
    orchestrator = Orchestrator(sessions, object_store, world_store, config_store=store)
    recipes = RecipeStore(lab_root / "datasets" / "scene_recipes")

    def _registry():
        return capability_registry(planner.status())

    def _get(sid) -> GenerationSession:
        s = sessions.get(sid)
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

    @app.get("/api/v1/generation/capabilities")
    def generation_capabilities():
        return _registry()

    @app.post("/api/v1/generation/sessions")
    def create_session(body: CreateSessionBody):
        s = sessions.create(project_id=body.project_id, capability_snapshot=_registry(),
                            aux_state=planner.status()["state"])
        if body.kind:
            s.asset_spec.kind = body.kind
        if body.initial_text:
            s.asset_spec.user_words = body.initial_text
            s.messages.append(ChatMessage(role="user", content=body.initial_text, ts=_now()))
            s = sessions.save(s, s.revision)
        return s.model_dump()

    @app.get("/api/v1/generation/sessions")
    def list_sessions():
        return {"sessions": sessions.list()}

    @app.get("/api/v1/generation/sessions/{sid}")
    def get_session(sid: str):
        return _get(sid).model_dump()

    @app.post("/api/v1/generation/sessions/{sid}/messages")
    def post_message(sid: str, body: MessageBody):
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
            except PlannerError as exc:
                s.messages.append(ChatMessage(role="system", content=f"规划器不可用：{exc}。可继续用表单/JSON 编辑。",
                                              ts=_now(), kind="error"))
                s.aux_state = planner.status()["state"]
        else:
            s.aux_state = "waiting_for_configuration"
            s.messages.append(ChatMessage(
                role="system", ts=_now(), kind="status",
                content="waiting_for_configuration：未配置 AUX_MODEL_BASE_URL/AUX_MODEL_API_KEY，"
                        "对话规划不可用；表单/JSON/本地导入仍可使用。"))
        return _save(s, body.expected_revision).model_dump()

    @app.patch("/api/v1/generation/sessions/{sid}/spec")
    def patch_spec(sid: str, body: SpecPatchBody):
        s = _get(sid)
        _check_rev(s, body.expected_revision)
        try:
            apply_user_patch(s, body)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        if s.state == SessionState.draft:
            s.state = SessionState.clarifying
        return _save(s, body.expected_revision).model_dump()

    @app.post("/api/v1/generation/sessions/{sid}/answer")
    def answer_question(sid: str, body: AnswerBody):
        s = _get(sid)
        _check_rev(s, body.expected_revision)
        q = next((q for q in s.questions if q.id == body.question_id), None)
        if q is None:
            raise HTTPException(404, "unknown question")
        q.answer, q.status = body.answer, "answered"
        if not any(q.status == "open" for q in s.questions):
            s.state = SessionState.ready
        return _save(s, body.expected_revision).model_dump()

    @app.post("/api/v1/generation/sessions/{sid}/compile")
    def compile_preview(sid: str, body: CompileBody):
        s = _get(sid)
        if body.expected_revision is not None:
            _check_rev(s, body.expected_revision)
        try:
            compiled = compile_session(s, provider=body.provider, prompt_override=body.prompt_override)
        except CompileError as exc:
            raise HTTPException(400, str(exc))
        s.compiled = compiled
        return _save(s, s.revision).model_dump()

    @app.post("/api/v1/generation/sessions/{sid}/branch")
    def create_branch(sid: str, body: BranchBody):
        s = _get(sid)
        _check_rev(s, body.expected_revision)
        s.branches.append(SessionBranch(
            branch_id=new_id("br"), parent_revision=s.revision, reason=body.reason, created=_now(),
            spec_snapshot={"asset_spec": s.asset_spec.model_dump(), "physics_spec": s.physics_spec.model_dump()}))
        return _save(s, body.expected_revision).model_dump()

    @app.post("/api/v1/generation/sessions/{sid}/confirm")
    def confirm_generation(sid: str, body: ConfirmBody):
        _get(sid)
        try:
            result = orchestrator.confirm(sid, client_request_id=body.client_request_id,
                                          expected_revision=body.expected_revision,
                                          provider=body.provider, remote_id=body.remote_id)
        except RevisionConflict as exc:
            raise HTTPException(409, {"error": str(exc), "current_revision": exc.current,
                                      "expected_revision": exc.expected})
        except CompileError as exc:
            raise HTTPException(400, str(exc))
        except KeyError:
            raise HTTPException(404, "unknown generation session")
        except ValueError as exc:
            raise HTTPException(400, str(exc))
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

    @app.post("/api/v1/generation/sessions/{sid}/recipe")
    def recipe_from_session(sid: str, body: RecipeBody):
        s = _get(sid)
        try:
            recipe = build_recipe_from_session(s, lab_root, overrides=body.model_dump(exclude_none=True))
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        recipe = recipes.put(recipe)
        s.recipe_id = recipe.recipe_id
        if s.state in {SessionState.review, SessionState.generating, SessionState.ready}:
            s.state = SessionState.ready_to_apply
        sessions.save(s, s.revision)
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
        saved = commit_scene(body.project_id, apply_mutate(r), body.expected_project_version)
        return {"version": saved.version, "project_id": saved.project_id,
                "validation_status": saved.validation_status,
                "scene": saved.scene.model_dump(), "task_asset": saved.task.asset,
                "recipe_id": r.recipe_id, "manifest_hash": report["manifest_hash"],
                "read_back": {"environment": saved.scene.environment.model_dump() if saved.scene.environment else None,
                              "colliders": [c.model_dump() for c in saved.scene.colliders],
                              "use_default_table": saved.scene.use_default_table,
                              "sensor_renderer": saved.scene.sensor_renderer}}

    return {"sessions": sessions, "planner": planner, "registry": _registry,
            "objects": object_store, "orchestrator": orchestrator, "recipes": recipes}