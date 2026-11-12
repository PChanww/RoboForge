"""Internal orchestration: confirm -> freeze -> job, with idempotency and recovery.

No HTTP self-loop: calls ObjectJobStore / GenerationJobStore in-process. Confirm
binds an immutable spec_revision + compiled payload hash + provider + project +
config_version + client_request_id. Duplicate confirms return the same local job.
A known remote_id is reused for recovery (no new paid task); an unknown submit is
never blindly resubmitted. LLM replies never reach this path automatically.
"""
from __future__ import annotations

from pathlib import Path

from .compiler import compile_session, spec_hash
from .schema import SessionState


class Orchestrator:
    def __init__(self, sessions, object_store, world_store, config_store=None):
        self.sessions = sessions
        self.objects = object_store
        self.world = world_store
        self.configs = config_store

    def _config_version(self, project_id):
        if self.configs is None or not project_id:
            return None
        try:
            return self.configs.load(project_id).version if self.configs.exists(project_id) else None
        except Exception:
            return None

    def confirm(self, sid, *, client_request_id=None, expected_revision=None,
                provider=None, remote_id=None, background=True):
        s = self.sessions.get(sid)
        if s is None:
            raise KeyError(sid)
        if expected_revision is not None and s.revision != expected_revision:
            from .sessions import RevisionConflict
            raise RevisionConflict(expected_revision, s.revision)
        compiled = compile_session(s, provider=provider)
        provider = compiled.provider
        h = compiled.spec_hash
        crid = client_request_id or f"{s.session_id}:{h}:{provider}"
        project_id = s.project_id
        cfg_version = self._config_version(project_id)

        if provider == "lux3d":
            existing = self.objects.find_by_request(crid)
            if existing:
                job = existing
            else:
                spec = s.asset_spec
                params = {
                    "prompt": compiled.prompt_en, "payload": compiled.payload, "slug": self._slug(s),
                    "kind": spec.kind.value, "role": spec.role.value,
                    "height_m": spec.height.value_m, "width_m": spec.width.value_m,
                    "depth_m": spec.depth.value_m, "radius_m": spec.radius.value_m,
                    "mass_kg": s.physics_spec.mass.value_kg, "friction": s.physics_spec.friction,
                    "spec_id": s.session_id, "spec_revision": s.revision, "spec_hash": h,
                    "compiler_version": compiled.compiler_version, "client_request_id": crid,
                    "provider": provider, "project_id": project_id, "config_version": cfg_version,
                }
                if remote_id:
                    params["remote_id"] = remote_id
                job = self.objects.submit(params, background=background)
        else:  # aholo_world
            job = None
            for j in self.world.list():
                if j.get("params", {}).get("client_request_id") == crid:
                    job = j
                    break
            if job is None:
                params = {"prompt": compiled.prompt_en, "spec_hash": h, "client_request_id": crid,
                          "spec_id": s.session_id, "spec_revision": s.revision,
                          "project_id": project_id, "config_version": cfg_version,
                          "name": "RoboForge background"}
                if remote_id:
                    params["remote_id"] = remote_id
                job = self.world.submit("generation", params, background=background)

        s.compiled = compiled
        s.state = SessionState.generating
        if job["id"] not in s.job_ids:
            s.job_ids.append(job["id"])
        s = self.sessions.save(s, s.revision)
        return {"session": s.model_dump(), "job": job, "client_request_id": crid,
                "binding": {"spec_revision": compiled.spec_revision, "spec_hash": h,
                            "provider": provider, "project_id": project_id, "config_version": cfg_version}}

    def _slug(self, s):
        import re
        base = s.asset_spec.name or s.asset_spec.kind.value
        slug = re.sub(r"[^a-zA-Z0-9_-]", "-", base).strip("-").lower()[:48] or "asset"
        return f"{slug}-{s.session_id[-6:]}"

    def tasks(self, sid):
        s = self.sessions.get(sid)
        if s is None:
            raise KeyError(sid)
        out = []
        for jid in s.job_ids:
            j = self.objects.get(jid) or self.world.get(jid)
            if j:
                out.append(j)
        return {"session_id": sid, "state": s.state.value, "tasks": out}

    def retry_node(self, sid, jid, node, *, background=True):
        s = self.sessions.get(sid)
        if s is None or jid not in s.job_ids:
            raise KeyError("job not linked to session")
        if self.objects.get(jid):
            return self.objects.retry_node(jid, node, background=background)
        # World jobs recover via poll (download/import reuse the known remote_id).
        return self.world.poll(jid, background=background)

    def poll(self, sid, jid, *, background=True):
        s = self.sessions.get(sid)
        if s is None or jid not in s.job_ids:
            raise KeyError("job not linked to session")
        if self.objects.get(jid):
            return self.objects.poll(jid, background=background)
        return self.world.poll(jid, background=background)