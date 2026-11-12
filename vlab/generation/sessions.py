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

from .schema import GenerationSession, SessionState


class RevisionConflict(Exception):
    def __init__(self, expected: int, current: int):
        self.expected, self.current = expected, current
        super().__init__(f"revision conflict: expected {expected}, current {current}")


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
        with self._conn_ctx() as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("""CREATE TABLE IF NOT EXISTS sessions(
                session_id TEXT PRIMARY KEY, revision INTEGER NOT NULL,
                state TEXT NOT NULL, data TEXT NOT NULL,
                created TEXT NOT NULL, updated TEXT NOT NULL)""")

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
               aux_state: str = "waiting_for_configuration") -> GenerationSession:
        now = _now()
        s = GenerationSession(session_id=new_id("gs"), created=now, updated=now,
                              project_id=project_id, capability_snapshot=capability_snapshot or {},
                              aux_state=aux_state)
        with self._lock, self._conn_ctx() as c:
            c.execute("INSERT INTO sessions VALUES(?,?,?,?,?,?)",
                      (s.session_id, s.revision, s.state.value, s.model_dump_json(), now, now))
        return s

    def get(self, sid: str) -> GenerationSession | None:
        with self._lock, self._conn_ctx() as c:
            row = c.execute("SELECT data FROM sessions WHERE session_id=?", (sid,)).fetchone()
        return GenerationSession.model_validate_json(row[0]) if row else None

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