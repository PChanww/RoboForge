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
        self.epoch = 0
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"sim-{self.id[:6]}")
        self.env = self._executor.submit(PlatformEnv, config, lab_root, robot_id, controller_id).result()
        self.env.session_id = self.id
        self.epoch = self.env.epoch

    def submit(self, fn, *args, **kwargs):
        return self._executor.submit(fn, *args, **kwargs).result()

    def reset(self, seed=None):
        def _run():
            obs = self.env.reset(seed)
            self.epoch = self.env.epoch  # env owns epoch; session mirrors it
            return {"observation": obs, "epoch": self.epoch}
        return self.submit(_run)

    def _guard(self, token, tick, epoch):
        self.check_control(token)
        if epoch is not None and epoch != self.epoch:
            raise ControlConflict(f"stale epoch {epoch}; current {self.epoch} (reset occurred)")
        if tick is not None and tick != self.env.scheduler.tick:
            raise ControlConflict(f"stale tick {tick}; current {self.env.scheduler.tick}")

    def execute(self, token, action, tick=None, epoch=None):
        # Validation and execution share the session's single serial worker.
        def _run():
            self._guard(token, tick, epoch)
            return self.env.step(action)
        return self.submit(_run)

    def execute_block(self, token, actions, epoch=None):
        def _run():
            self.check_control(token)
            if epoch is not None and epoch != self.epoch:
                raise ControlConflict(f"stale epoch {epoch}; current {self.epoch}")
            for a in actions:  # prevalidate every action before executing any
                self.env.validate_action(a)
            obs, executed = None, 0
            for a in actions:
                try:
                    obs = self.env.step(a)
                    executed += 1
                except ValueError as exc:
                    return {"observation": obs, "executed": executed, "error": str(exc)}
            return {"observation": obs, "executed": executed}
        return self.submit(_run)

    def expert(self, token, epoch=None, steps: int = 1):
        n = max(1, min(int(steps or 1), 50))
        def _run():
            self._guard(token, None, epoch)
            obs = None
            for _ in range(n):
                obs = self.env.step(self.env.servo_expert_action())
                if self.env.metrics().get("success"):
                    break
            return {**obs, "metrics": self.env.metrics()}
        return self.submit(_run)

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