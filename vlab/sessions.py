"""Isolated sessions: one worker thread (OpenGL context) and one env per session.

A session has a single control owner; conflicting controllers get a clear 409.
"""
from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

from .engine import PlatformEnv
from .omni_instruction import MODEL_ID as OMNI_VLA, UnsupportedInstruction, validate_target
from .scene_assets import valid_id
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
        try:
            self.env = self._executor.submit(PlatformEnv, config, lab_root, robot_id, controller_id).result()
        except Exception:
            self._executor.shutdown(wait=True)
            raise
        self.env.session_id = self.id
        self.epoch = self.env.epoch
        self._policy_runners = {}
        self._policy_context = None

    def submit(self, fn, *args, **kwargs):
        return self._executor.submit(fn, *args, **kwargs).result()

    def reset(self, seed=None):
        def _run():
            obs = self.env.reset(seed)
            self.epoch = self.env.epoch  # env owns epoch; session mirrors it
            self._policy_context = None
            for runner in self._policy_runners.values():
                runner.reset()
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

    def policy(self, token, model_id: str = "hackathon_vla_bc", epoch=None, steps: int = 1,
               instruction: str | None = None, start: bool = False):
        model_id = valid_id(model_id)
        n = max(1, min(int(steps or 1), 50))
        def _run():
            self._guard(token, None, epoch)
            tick = self.env.scheduler.tick
            new_run = start or self._policy_context != (model_id, instruction, tick)
            target = None
            if model_id == OMNI_VLA and new_run:
                try:
                    target = validate_target(
                        instruction,
                        cup_asset=self.env.config.task.asset,
                        dynamic_ids={node.node_id for node in self.env.config.scene.asset_nodes if node.role == "dynamic"},
                    )
                except UnsupportedInstruction as exc:
                    self._policy_context = None
                    for _ in range(n):
                        obs = self.env.advance()
                    return {**obs, "metrics": self.env.metrics(), "policy_model": model_id,
                            "last_action": None, "instruction_warning": str(exc)}
            runner = self._policy_runners.get(model_id)
            if runner is None:
                smolvla_stats = self.env.lab_root / "datasets" / "models" / model_id / "so100_stats.json"
                if smolvla_stats.is_file():
                    import json as _json
                    st = _json.loads(smolvla_stats.read_text(encoding="utf-8"))
                    controller = self.env.controller
                    if controller.mode != "cartesian_delta_ik" or controller.action_dim != 4:
                        raise ValueError("SmolVLA requires four-channel Cartesian delta IK")
                    obs0 = self.env.observation()
                    missing = sorted(set(st.get("state_joints") or []) - obs0["state"]["joint_pos"].keys())
                    if missing:
                        raise ValueError(f"SmolVLA requires joint state for: {', '.join(missing)}")
                    imgs0 = obs0.get("images") or {}
                    for cam in (st.get("camera_keys") or ["front", "wrist"]):
                        frame = imgs0.get(cam)
                        if not frame or not frame.get("jpeg_b64"):
                            raise ValueError(f"Camera {cam!r} frame unavailable for SmolVLA")
                    from .smolvla_infer import SmolVLAPolicyRunner
                    runner = SmolVLAPolicyRunner(self.env.lab_root, model_id, shared=True)
                else:
                    from .visual_infer import VisualPolicyRunner
                    runner = VisualPolicyRunner(self.env.lab_root, model_id)
                self._policy_runners[model_id] = runner
            if target is not None:
                runner.reset()
                metrics = self.env.metrics()
                if metrics.get("active_target") != target or metrics.get("success"):
                    self.env.set_active_target(target)
            elif tick == 0:
                self.env.select_target_for_instruction(instruction)
                runner.reset()
            obs = None
            last_action = None
            for _ in range(n):
                action = runner.act(self.env.observation(), instruction=instruction)
                last_action = action
                obs = self.env.step(action)
                if self.env.metrics().get("success"):
                    break
            self._policy_context = (model_id, instruction, self.env.scheduler.tick)
            return {**obs, "metrics": self.env.metrics(), "policy_model": model_id, "last_action": last_action}
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
        for runner in self._policy_runners.values():
            if hasattr(runner, "close"):
                runner.close()
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
        from .smolvla_infer import close_all_smolvla_workers
        close_all_smolvla_workers()
