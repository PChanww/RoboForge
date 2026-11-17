"""SQLite-backed generation session persistence (stdlib only).

Process boundary: the single uvicorn backend process owns this store. SQLite WAL
gives durability across refresh/restart; optimistic revision checks give 409 on
concurrent edits instead of silent overwrite. This does not rewrite the existing
World GenerationJobStore or JobManager.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from .schema import AssetSpec, ChatMessage, GenerationSession, PhysicsSpec, SessionState


class RevisionConflict(Exception):
    def __init__(self, expected: int, current: int):
        self.expected, self.current = expected, current
        super().__init__(f"revision conflict: expected {expected}, current {current}")


class SessionDataError(ValueError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class SessionStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "sessions.db"
        self._lock = threading.RLock()
        self._session_locks: dict[str, threading.RLock] = {}
        with self._conn_ctx() as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("""CREATE TABLE IF NOT EXISTS sessions(
                session_id TEXT PRIMARY KEY, revision INTEGER NOT NULL,
                state TEXT NOT NULL, data TEXT NOT NULL,
                created TEXT NOT NULL, updated TEXT NOT NULL)""")

    def session_lock(self, sid: str) -> threading.RLock:
        with self._lock:
            lk = self._session_locks.get(sid)
            if lk is None:
                lk = threading.RLock()
                self._session_locks[sid] = lk
            return lk

    def _conn(self):
        return sqlite3.connect(self.path, timeout=30)

    @contextmanager
    def _conn_ctx(self):
        conn = self._conn()
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def create(self, project_id: str | None = None, capability_snapshot: dict | None = None,
               aux_state: str = "waiting_for_configuration",
               asset_spec: AssetSpec | None = None,
               physics_spec: PhysicsSpec | None = None,
               messages: list[ChatMessage] | None = None) -> GenerationSession:
        now = _now()
        s = GenerationSession(
            session_id=new_id("gs"), created=now, updated=now,
            project_id=project_id, capability_snapshot=capability_snapshot or {},
            aux_state=aux_state,
            asset_spec=asset_spec if asset_spec is not None else AssetSpec(),
            physics_spec=physics_spec if physics_spec is not None else PhysicsSpec(),
            messages=list(messages or []),
        )
        with self._lock, self._conn_ctx() as c:
            c.execute("INSERT INTO sessions VALUES(?,?,?,?,?,?)",
                      (s.session_id, s.revision, s.state.value, s.model_dump_json(), now, now))
        return s

    @staticmethod
    def _repair_legacy_payload(raw_json: str) -> GenerationSession | None:
        try:
            data = json.loads(raw_json)
        except Exception:
            return None
        spec = data.get("asset_spec")
        if not isinstance(spec, dict):
            return None
        kind = spec.get("kind")
        role = spec.get("role")
        role_prov = (spec.get("provenance") or {}).get("role") or {}
        if role_prov.get("source") == "user_explicit":
            return None
        changed = False
        if kind == "environment_splat" and role != "visual_only":
            spec["role"] = "visual_only"
            changed = True
        elif kind == "robot_visual" and role == "dynamic":
            spec["role"] = "visual_only"
            changed = True
        if not changed:
            return None
        phys = data.get("physics_spec")
        if isinstance(phys, dict):
            phys["role"] = spec["role"]
        try:
            return GenerationSession.model_validate(data)
        except ValidationError:
            return None

    def get(self, sid: str) -> GenerationSession | None:
        with self._lock, self._conn_ctx() as c:
            row = c.execute("SELECT data FROM sessions WHERE session_id=?", (sid,)).fetchone()
        if not row:
            return None
        try:
            return GenerationSession.model_validate_json(row[0])
        except ValidationError as exc:
            repaired = self._repair_legacy_payload(row[0])
            if repaired is not None:
                return repaired
            raise SessionDataError(f"session {sid} stored data is invalid: {exc}") from exc

    def list(self) -> list[dict]:
        with self._lock, self._conn_ctx() as c:
            rows = c.execute("SELECT session_id, revision, state, created, updated FROM sessions "
                             "ORDER BY updated DESC").fetchall()
        return [{"session_id": r[0], "revision": r[1], "state": r[2], "created": r[3], "updated": r[4]}
                for r in rows]

    def save(self, session: GenerationSession, expected_revision: int) -> GenerationSession:
        """Persist a mutated session; bump revision. Raises RevisionConflict on mismatch."""
        with self._lock, self._conn_ctx() as c:
            row = c.execute("SELECT revision FROM sessions WHERE session_id=?",
                            (session.session_id,)).fetchone()
            if row is None:
                raise KeyError(session.session_id)
            if row[0] != expected_revision:
                raise RevisionConflict(expected_revision, row[0])
            session.revision = expected_revision + 1
            session.updated = _now()
            c.execute("UPDATE sessions SET revision=?, state=?, data=?, updated=? WHERE session_id=?",
                      (session.revision, session.state.value, session.model_dump_json(),
                       session.updated, session.session_id))
        return session

    def mutate(self, sid: str, expected_revision: int, fn) -> GenerationSession:
        s = self.get(sid)
        if s is None:
            raise KeyError(sid)
        fn(s)
        return self.save(s, expected_revision)

    def set_state(self, sid: str, state: SessionState) -> GenerationSession:
        s = self.get(sid)
        if s is None:
            raise KeyError(sid)
        s.state = state
        return self.save(s, s.revision)