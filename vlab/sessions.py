"""Isolated sessions: one worker thread (OpenGL context) and one env per session.

A session has a single control owner; conflicting controllers get a clear 409.
"""
from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

from .engine import PlatformEnv
from .schema import ProjectConfig


class ControlConflict(Exception):
    pass


class Session:
    def __init__(self, config: ProjectConfig, lab_root, robot_id, controller_id):
        self.id = uuid.uuid4().hex
        self.config = config
        self.robot_id = robot_id
        self.controller_id = controller_id
        self._lock = threading.Lock()
        self._owner = None
        self._token = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"sim-{self.id[:6]}")
        self.env = self._executor.submit(PlatformEnv, config, lab_root, robot_id, controller_id).result()

    def submit(self, fn, *args, **kwargs):
        return self._executor.submit(fn, *args, **kwargs).result()

    def acquire(self, owner: str) -> str:
        with self._lock:
            if self._owner is not None and self._owner != owner:
                raise ControlConflict(f"session owned by '{self._owner}'; release before '{owner}' acquires")
            self._owner = owner
            self._token = uuid.uuid4().hex
            return self._token

    def release(self, token: str):
        with self._lock:
            if token != self._token:
                raise ControlConflict("invalid control token")
            self._owner = self._token = None

    def check_control(self, token: str):
        with self._lock:
            if self._owner is None:
                raise ControlConflict("no control owner; acquire control first")
            if token != self._token:
                raise ControlConflict(f"control token mismatch (owner '{self._owner}')")

    def owner(self):
        with self._lock:
            return self._owner

    def close(self):
        self._executor.submit(self.env.close).result()
        self._executor.shutdown(wait=True)


class SessionManager:
    def __init__(self):
        self._sessions = {}
        self._lock = threading.Lock()

    def create(self, config, lab_root, robot_id=None, controller_id=None) -> Session:
        s = Session(config, lab_root, robot_id, controller_id)
        with self._lock:
            self._sessions[s.id] = s
        return s

    def get(self, session_id) -> Session:
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(session_id)
            return self._sessions[session_id]

    def list(self):
        with self._lock:
            return [{"id": s.id, "robot": s.env.robot.id, "controller": s.env.controller.id,
                     "owner": s.owner(), "config_version": s.config.version}
                    for s in self._sessions.values()]

    def close(self, session_id):
        with self._lock:
            s = self._sessions.pop(session_id, None)
        if s:
            s.close()

    def close_all(self):
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for s in sessions:
            s.close()