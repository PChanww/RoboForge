"""Internal orchestration: confirm -> freeze -> job, with idempotency and recovery.

No HTTP self-loop: calls ObjectJobStore / GenerationJobStore in-process. Confirm
binds an immutable spec_revision + compiled payload hash + provider + project +
config_version + client_request_id. Duplicate confirms return the same local job.
A known remote_id is reused for recovery (no new paid task); an unknown submit is
never blindly resubmitted. LLM replies never reach this path automatically.
"""
from __future__ import annotations

import re
import threading
import uuid
from pathlib import Path

from .compiler import CompileError, compile_session, spec_hash, validate_prompt_against_spec
from .schema import SessionState
from ..scene_assets import valid_id


class Orchestrator:
    def __init__(self, sessions, object_store, world_store, config_store=None):
        self.sessions = sessions
        self.objects = object_store
        self.world = world_store
        self.configs = config_store
        self._confirm_lock = threading.RLock()

    def _config_version(self, project_id):
        if self.configs is None or not project_id:
            return None
        try:
            return self.configs.load(project_id).version if self.configs.exists(project_id) else None
        except Exception:
            return None

    def _resolve_compiled(self, s, *, provider=None, expected_compiled_revision=None, expected_request_hash=None):
        current_base = spec_hash(s.asset_spec, s.physics_spec)
        if s.compiled is not None:
            compiled = s.compiled
            if compiled.base_spec_hash:
                if compiled.base_spec_hash != current_base:
                    raise CompileError("规格在编译后已变更，请先重新编译预览后再确认生成")
            elif compiled.spec_hash != current_base:
                raise CompileError("规格或旧版编译已过期，请先重新编译预览后再确认生成")
            if provider is not None and provider != compiled.provider:
                raise CompileError(
                    f"已编译预览针对 {compiled.provider}，与请求 provider={provider} 不一致；请重新编译预览"
                )
            if expected_compiled_revision is not None and compiled.spec_revision != expected_compiled_revision:
                raise CompileError(
                    f"编译版本不一致：期望 {expected_compiled_revision}，当前 {compiled.spec_revision}；请重新编译预览"
                )
            req_hash = compiled.request_hash or compiled.spec_hash
            if expected_request_hash is not None and req_hash != expected_request_hash:
                raise CompileError("已编译请求哈希不一致，请先重新编译预览后再确认生成")
            issues = validate_prompt_against_spec(s.asset_spec, compiled.prompt_en)
            if issues:
                raise CompileError("已编译提示词与硬约束冲突，已阻止提交：" + "；".join(issues))
            return compiled
        if expected_compiled_revision is not None or expected_request_hash is not None:
            raise CompileError("尚无已编译预览，请先编译预览后再确认生成")
        return compile_session(s, provider=provider)

    def confirm(self, sid, *, client_request_id=None, expected_revision=None,
                expected_compiled_revision=None, expected_request_hash=None,
                provider=None, remote_id=None, background=True):
        session_lock = self.sessions.session_lock(sid) if hasattr(self.sessions, "session_lock") else self._confirm_lock
        with self._confirm_lock, session_lock:
            s = self.sessions.get(sid)
            if s is None:
                raise KeyError(sid)
            compiled = self._resolve_compiled(
                s,
                provider=provider,
                expected_compiled_revision=expected_compiled_revision,
                expected_request_hash=expected_request_hash,
            )
            provider = compiled.provider
            if provider == "aholo_world" and self.world is None:
                raise ValueError("aholo_world job store is unavailable")
            h = compiled.spec_hash
            req_hash = compiled.request_hash or h
            crid = client_request_id or f"{s.session_id}:{req_hash}:{provider}"
            project_id = s.project_id
            cfg_version = self._config_version(project_id)

            if provider == "lux3d":
                existing = self.objects.find_by_request(crid)
            else:
                existing = self.world.find_by_request(crid) if hasattr(self.world, "find_by_request") else next(
                    (j for j in self.world.list() if j.get("params", {}).get("client_request_id") == crid),
                    None,
                )

            if existing is not None:
                ep = existing.get("params", {})
                if existing.get("spec_id", ep.get("spec_id")) not in (None, s.session_id):
                    raise ValueError(f"client_request_id {crid!r} belongs to another session")
                if ep.get("spec_hash") != h or ep.get("prompt") != compiled.prompt_en or (
                    provider == "lux3d" and ep.get("payload") != compiled.payload
                ):
                    raise ValueError(f"client_request_id {crid!r} already bound to a different request payload")
                if expected_revision is not None and s.revision != expected_revision:
                    # Allow idempotent response-loss replay when this exact confirmation already bumped revision.
                    already_linked = (
                        existing["id"] in s.job_ids
                        and expected_revision in (ep.get("spec_revision"), ep.get("session_revision"))
                    )
                    if not already_linked:
                        from .sessions import RevisionConflict
                        raise RevisionConflict(expected_revision, s.revision)
                if remote_id:
                    if existing.get("remote_id"):
                        if existing["remote_id"] != str(remote_id).strip():
                            raise ValueError(f"job {existing['id']} already bound to remote_id {existing['remote_id']}")
                        job = existing
                    else:
                        store = self.objects if provider == "lux3d" else self.world
                        job = store.attach_remote(existing["id"], remote_id, provider=provider)
                else:
                    job = existing
            else:
                if expected_revision is not None and s.revision != expected_revision:
                    from .sessions import RevisionConflict
                    raise RevisionConflict(expected_revision, s.revision)
                if provider == "lux3d":
                    spec = s.asset_spec
                    params = {
                        "prompt": compiled.prompt_en, "payload": compiled.payload,
                        "slug": self._slug(s, compiled),
                        "kind": spec.kind.value, "role": spec.role.value,
                        "height_m": spec.height.value_m, "width_m": spec.width.value_m,
                        "depth_m": spec.depth.value_m, "radius_m": spec.radius.value_m,
                        "mass_kg": s.physics_spec.mass.value_kg, "friction": s.physics_spec.friction,
                        "spec_id": s.session_id, "spec_revision": compiled.spec_revision,
                        "session_revision": s.revision,
                        "spec_hash": h, "request_hash": req_hash,
                        "compiler_version": compiled.compiler_version, "client_request_id": crid,
                        "provider": provider, "project_id": project_id, "config_version": cfg_version,
                    }
                    if remote_id:
                        params["remote_id"] = remote_id
                    job = self.objects.submit(params, background=background)
                else:  # aholo_world
                    params = {
                        "prompt": compiled.prompt_en, "spec_hash": h, "request_hash": req_hash,
                        "client_request_id": crid,
                        "spec_id": s.session_id, "spec_revision": compiled.spec_revision,
                        "session_revision": s.revision,
                        "project_id": project_id, "config_version": cfg_version,
                        "name": "RoboForge background",
                    }
                    if compiled.payload.get("ref_image"):
                        params["ref_image"] = compiled.payload["ref_image"]
                    if remote_id:
                        params["remote_id"] = remote_id
                    job = self.world.submit("generation", params, background=background)

            needs_save = (
                s.compiled != compiled
                or s.state != SessionState.generating
                or job["id"] not in s.job_ids
            )
            if needs_save:
                s.compiled = compiled
                s.state = SessionState.generating
                if job["id"] not in s.job_ids:
                    s.job_ids.append(job["id"])
                s = self.sessions.save(s, s.revision)
            return {
                "session": s.model_dump(),
                "job": job,
                "client_request_id": crid,
                "binding": {
                    "spec_revision": compiled.spec_revision,
                    "spec_hash": h,
                    "request_hash": req_hash,
                    "base_spec_hash": compiled.base_spec_hash,
                    "provider": provider,
                    "project_id": project_id,
                    "config_version": cfg_version,
                },
            }

    def _slug(self, s, compiled=None):
        base = s.asset_spec.name or s.asset_spec.kind.value
        slug = re.sub(r"[^a-zA-Z0-9_-]", "-", base).strip("-").lower()[:28] or "asset"
        rev = compiled.spec_revision if compiled is not None else s.revision
        seq = len(s.job_ids) + 1
        return f"{slug}-{s.session_id[-6:]}-r{rev}-n{seq}-{uuid.uuid4().hex[:6]}"

    def tasks(self, sid):
        s = self.sessions.get(sid)
        if s is None:
            raise KeyError(sid)
        out = []
        for jid in s.job_ids:
            j = self.objects.get(jid) or (self.world.get(jid) if self.world is not None else None)
            if j:
                out.append(j)
        return {"session_id": sid, "state": s.state.value, "tasks": out}

    def retry_node(self, sid, jid, node, *, background=True):
        s = self.sessions.get(sid)
        if s is None or jid not in s.job_ids:
            raise KeyError("job not linked to session")
        if self.objects.get(jid):
            return self.objects.retry_node(jid, node, background=background)
        if self.world is None:
            raise KeyError("world job store unavailable")
        # World jobs recover via poll (download/import reuse the known remote_id).
        return self.world.poll(jid, background=background)

    def poll(self, sid, jid, *, background=True):
        s = self.sessions.get(sid)
        if s is None or jid not in s.job_ids:
            raise KeyError("job not linked to session")
        if self.objects.get(jid):
            return self.objects.poll(jid, background=background)
        if self.world is None:
            raise KeyError("world job store unavailable")
        return self.world.poll(jid, background=background)

    def recover(self, sid, jid, remote_id: str, *, provider: str | None = None):
        s = self.sessions.get(sid)
        if s is None or jid not in s.job_ids:
            raise KeyError("job not linked to session")
        obj_job = self.objects.get(jid)
        if obj_job is not None:
            if s.asset_spec.kind.value == "environment_splat":
                raise ValueError("session kind environment_splat expects aholo_world, not lux3d")
            job = self.objects.attach_remote(jid, remote_id, provider=provider)
            return {"session": s.model_dump(), "job": job}
        if self.world is not None and self.world.get(jid) is not None:
            if s.asset_spec.kind.value != "environment_splat":
                raise ValueError(f"session kind {s.asset_spec.kind.value} expects lux3d, not aholo_world")
            job = self.world.attach_remote(jid, remote_id, provider=provider)
            return {"session": s.model_dump(), "job": job}
        raise KeyError("job not found")

    def list_remote_tasks(self, provider: str = "lux3d", *, page: int = 1, page_size: int = 20) -> dict:
        if provider not in {"lux3d", "aholo_world"}:
            raise ValueError(f"unsupported remote provider: {provider!r}")
        prov = self.objects.provider if provider == "lux3d" else (self.world.provider if self.world is not None else None)
        if prov is None or not hasattr(prov, "configured") or not prov.configured():
            return {"provider": provider, "configured": False, "total": 0, "page": page,
                    "page_size": page_size, "items": [], "error": f"{provider} 未配置 API Key"}
        if not hasattr(prov, "list_remote_tasks"):
            return {"provider": provider, "configured": True, "total": 0, "page": page,
                    "page_size": page_size, "items": [], "error": f"{provider} 不支持远端任务列表"}
        try:
            res = (prov.list_remote_tasks(page=page, page_size=page_size) if provider == "lux3d"
                   else prov.list_remote_tasks(page_num=max(0, page - 1), page_size=page_size))
            res.setdefault("configured", True)
            res.setdefault("error", None)
            return res
        except Exception as exc:
            return {"provider": provider, "configured": True, "total": 0, "page": page,
                    "page_size": page_size, "items": [], "error": str(exc)}

    def import_remote(self, sid: str, remote_id: str, *, provider: str | None = None,
                      client_request_id: str | None = None, expected_revision: int | None = None,
                      background: bool = True) -> dict:
        rid = valid_id(str(remote_id or "").strip())
        session_lock = self.sessions.session_lock(sid) if hasattr(self.sessions, "session_lock") else self._confirm_lock
        with self._confirm_lock, session_lock:
            s = self.sessions.get(sid)
            if s is None:
                raise KeyError(sid)
            if expected_revision is not None and s.revision != expected_revision:
                from .sessions import RevisionConflict
                raise RevisionConflict(expected_revision, s.revision)
            inferred = "aholo_world" if s.asset_spec.kind.value == "environment_splat" else "lux3d"
            target_provider = provider or inferred
            if target_provider not in {"lux3d", "aholo_world"}:
                raise ValueError(f"unsupported remote provider: {target_provider!r}")
            if target_provider != inferred:
                raise ValueError(f"session kind {s.asset_spec.kind.value} expects {inferred}, not {target_provider}")
            store = self.objects if target_provider == "lux3d" else self.world
            if store is None or not store.provider.configured():
                raise ValueError(f"{target_provider} 未配置 API Key，无法校验并导入远端任务")
            # Idempotent hit if this session already bound this remote_id
            for jid in reversed(s.job_ids):
                existing = store.get(jid)
                if existing and existing.get("remote_id") == rid:
                    job = store.poll(jid, background=background)
                    return {"session": self.sessions.get(sid).model_dump(), "job": job, "imported_remote": True}
            try:
                st = store.provider.get_status(rid)
            except Exception as exc:
                raise ValueError(f"远端任务 {rid!r} 查询失败: {exc}") from exc
            if not isinstance(st, dict) or st.get("remote_id") not in (None, rid):
                raise ValueError(f"远端任务 {rid!r} 响应不匹配")
            if st.get("state") == "failed":
                raise ValueError(f"远端任务 {rid!r} 在服务端已失败: {st.get('error') or st.get('provider_status')}")
            cur_base = spec_hash(s.asset_spec, s.physics_spec)
            need_recompile = (
                s.compiled is None
                or (s.compiled.base_spec_hash and s.compiled.base_spec_hash != cur_base)
                or (not s.compiled.base_spec_hash and s.compiled.spec_hash != cur_base)
                or s.compiled.provider != target_provider
            )
            if need_recompile:
                s.compiled = compile_session(s, provider=target_provider)
                s = self.sessions.save(s, s.revision)
            crid = client_request_id or f"{s.session_id}:import:{target_provider}:{rid}"
            res = self.confirm(
                sid,
                client_request_id=crid,
                expected_revision=s.revision,
                provider=target_provider,
                remote_id=rid,
                background=False,
            )
            job = self.poll(sid, res["job"]["id"], background=background)
            latest_s = self.sessions.get(sid)
            return {
                "session": latest_s.model_dump(),
                "job": job,
                "client_request_id": res["client_request_id"],
                "binding": res["binding"],
                "imported_remote": True,
            }