"""Explicit, fail-closed policy connection for mobile task experiments.

Model input includes language, cameras and proprioception. Ground-truth object
poses, task stages, contact counts and expert actions are not sent to a VLA.
"""
from collections import deque
import copy
import json
import math
import os
from pathlib import Path
import threading
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener
import uuid

import numpy as np

from .transport_training import MODELS, STATE_FIELDS, trained_models

SCHEMA = "mobile7_v1"
_OPENER = build_opener(ProxyHandler({}))
_HEALTH_CACHE = {}


class PolicyCancelled(RuntimeError):
    """The owner stopped waiting; a late HTTP result must never move the robot."""


def validate_camera_alignment(preview, decision_time):
    """Accept only current successful captures, including after a renderer fault."""
    if not isinstance(decision_time, (int, float)) or not math.isfinite(decision_time):
        raise ValueError("双相机决策时刻必须为有限秒数。")
    stamp = preview.get("sim_time")
    if (not isinstance(stamp, (int, float)) or not math.isfinite(stamp)
            or not math.isclose(stamp, decision_time, abs_tol=1e-6, rel_tol=0)):
        raise ValueError("双相机预览时刻与当前机器人状态不一致。")
    statuses, images = preview.get("camera_status", {}), preview.get("images", {})
    for name in ("front", "wrist"):
        status = statuses.get(name, {})
        captured, available = status.get("capture_time"), status.get("available_time")
        age = status.get("age_s")
        if (status.get("state") != "ok" or status.get("render_error")
                or not isinstance(images.get(name), str) or not images[name]
                or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in (captured, available, age))
                or not math.isclose(captured, decision_time, abs_tol=1e-6, rel_tol=0)
                or available > decision_time+1e-6 or not math.isclose(age, 0., abs_tol=1e-6, rel_tol=0)):
            raise ValueError(f"{name}相机没有本决策时刻成功采集的画面，停止采集，拒绝使用过期图像。")


def _json_call(url, body=None, timeout=2.):
    request = Request(url, data=None if body is None else json.dumps(body, allow_nan=False).encode(),
                      headers={"Content-Type": "application/json", "Accept": "application/json"})
    try:
        with _OPENER.open(request, timeout=timeout) as response:
            payload = response.read(8 * 1024 * 1024 + 1)
            if len(payload) > 8 * 1024 * 1024:
                raise ValueError("VLA响应超过协议允许大小。")
            return json.loads(payload)
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"VLA推理服务不可用：{type(exc).__name__}；未切换到专家。") from exc


def _identity(health, policy_id):
    identity = copy.deepcopy(health.get("identity", {}))
    if (identity.get("action_schema") != SCHEMA or identity.get("action_dim") != 7
            or identity.get("learned_policy") is not True
            or not isinstance(identity.get("weights_sha256"), str)
            or len(identity["weights_sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in identity["weights_sha256"])):
        raise ValueError("VLA服务未声明可审计的权重与mobile7_v1动作契约。")
    kind = "learned_bc" if identity.get("kind") == "learned_bc" else "vla"
    identity.update(id=policy_id, kind=kind)
    return identity


class ExpertController:
    identity = {"id": "expert", "kind": "privileged_expert", "label": "仿真状态专家基线",
                "learned_policy": False, "action_dim": 7, "action_schema": SCHEMA,
                "privileged_observations": True}
    diagnostics = {"inference_count": 0, "executed_model_actions": 0}

    def reset(self, seed, instruction):
        pass

    def action(self, env, state, preview):
        return env.expert_action()


class VLAController:
    def __init__(self, endpoint, identity, spec, training_model=None):
        self.endpoint = endpoint
        self.training_model = training_model
        self.identity = identity
        self.spec = spec
        self.queue = deque()
        self.diagnostics = {"inference_count": 0, "inference_s": 0., "last_inference_s": None,
                            "executed_model_actions": 0, "raw_action_schema": identity.get("raw_action_schema"),
                            "inference_pending": False, "inference_attempts": 0, "cancelled_inferences": 0,
                            "chunks": []}
        self._cancel_event = None
        self._diagnostics_callback = None

    def bind_runtime(self, cancel_event, diagnostics_callback=None):
        self._cancel_event = cancel_event
        self._diagnostics_callback = diagnostics_callback

    def _notify_diagnostics(self):
        if self._diagnostics_callback is not None:
            self._diagnostics_callback(self.diagnostics)

    def _check_cancelled(self):
        if self._cancel_event is not None and self._cancel_event.is_set():
            raise PolicyCancelled("VLA等待已取消；未执行迟到的模型动作。")

    def _infer(self, packet):
        self._check_cancelled()
        if self._cancel_event is None:
            return _json_call(self.endpoint + "/infer", packet,
                              timeout=float(self.spec.get("vla_max_inference_s", 60.)))
        # urllib's blocking read cannot be interrupted by Event. Only this
        # daemon owns the HTTP request; the physics owner can stop immediately.
        # A late result is discarded and this thread never touches the env.
        done, result = threading.Event(), {}
        def request():
            try:
                result["value"] = _json_call(self.endpoint + "/infer", packet,
                    timeout=float(self.spec.get("vla_max_inference_s", 60.)))
            except Exception as exc:
                result["error"] = exc
            finally:
                done.set()
        threading.Thread(target=request, daemon=True, name="transport-vla-http").start()
        while not done.wait(.05):
            self._check_cancelled()
        self._check_cancelled()
        if "error" in result:
            raise result["error"]
        return result["value"]

    def mark_applied(self):
        """Called by the physics owner after env.step actually applied a row."""
        self.diagnostics["executed_model_actions"] += 1

    def reset(self, seed, instruction):
        self.queue.clear()
        self.session = uuid.uuid4().hex
        self.seed, self.instruction = int(seed), instruction
        self.diagnostics.update(episode_seed=seed, session_id=self.session)

    def _packet(self, env, state, preview):
        validate_camera_alignment(preview, state.get("time"))
        # EEF is measured from robot kinematics; no object information is used.
        base = np.asarray(state["base_pose"], dtype=float)
        w, x, y, z = base[3:]
        rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                             [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                             [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        eef = np.asarray(env.data.xpos[env._eef], dtype=float)
        local_eef = rotation.T @ (eef-base[:3])
        gripper = env.robot.gripper
        jaw = float(np.mean([env.data.qpos[env._qposadr[name]]
                             for name in (gripper.left_joint, gripper.right_joint)]))
        grip = float(np.clip((jaw-env._closed)/(env._open-env._closed), 0., 1.))
        proprio = np.concatenate([np.asarray(state[key], dtype=float).ravel()
                                  for key in STATE_FIELDS])
        if proprio.shape != (28,) or not np.isfinite(proprio).all():
            raise ValueError("VLA机器人状态必须为28维有限数值。")
        images = {key: preview.get("images", {}).get(key) for key in ("front", "wrist")}
        if not all(isinstance(value, str) and value for value in images.values()):
            raise ValueError("VLA必须使用本控制时刻的双相机画面。")
        packet = {"request_id": uuid.uuid4().hex, "session_id": self.session, "tick": preview["tick"],
                "seed": self.seed, "instruction": self.instruction, "task_kind": self.spec.get("task_kind", "deliver"),
                "observation": {"state": proprio.tolist(), "eef_position_base": local_eef.tolist(),
                                "gripper_open_fraction": grip, "images": images},
                "max_steps": int(self.spec.get("vla_replan_steps", 5)), "action_schema": SCHEMA}
        if self.training_model:
            packet["max_steps"] = 1
            packet["mobile_context"] = {"goal": preview["target_xy"], "time": preview["sim_time"],
                                        "belt_speed": state["belt_speed_m_s"], "speed_scale": preview["speed_scale"]}
            if self.training_model == "privileged_bc":
                packet["observation"]["privileged_state"] = [
                    *np.asarray(state["object_pose"], dtype=float).tolist(),
                    *np.asarray(state["object_velocity"], dtype=float).tolist()]
        return packet

    def action(self, env, state, preview):
        self._check_cancelled()
        if not self.queue:
            packet = self._packet(env, state, preview)
            started = time.perf_counter()
            self.diagnostics.update(inference_pending=True,
                inference_attempts=self.diagnostics["inference_attempts"]+1)
            self._notify_diagnostics()
            try:
                result = self._infer(packet)
            except PolicyCancelled:
                self.diagnostics["cancelled_inferences"] += 1
                raise
            finally:
                self.diagnostics["inference_pending"] = False
                self._notify_diagnostics()
            elapsed = time.perf_counter()-started
            if (result.get("request_id") != packet["request_id"] or result.get("session_id") != self.session
                    or result.get("tick") != packet["tick"] or result.get("action_schema") != SCHEMA
                    or result.get("weights_sha256") != self.identity["weights_sha256"]):
                raise ValueError("VLA返回的实验、时刻、权重或动作契约不匹配。")
            actions = np.asarray(result.get("actions"), dtype=float)
            if (actions.ndim != 2 or actions.shape[1] != 7 or not 1 <= len(actions) <= packet["max_steps"]
                    or not np.isfinite(actions).all() or np.any(np.abs(actions[:, :4]) > 1.+1e-7)
                    or np.any(np.abs(actions[:, 4]) > .065*self.spec.get("speed_scale", 1.6)+1e-7)
                    or np.any(np.abs(actions[:, 5]) > .8+1e-7) or not np.isin(actions[:, 6], [0., 1.]).all()):
                raise ValueError("VLA动作形状、有限数值或幅度不符合7维契约。")
            raw = np.asarray(result.get("raw_actions", actions), dtype=float)
            if raw.ndim != 2 or len(raw) != len(actions) or not np.isfinite(raw).all():
                raise ValueError("VLA原始动作证据缺失或无效。")
            self.queue.extend(actions.tolist())
            self.diagnostics["inference_count"] += 1
            self.diagnostics["inference_s"] += elapsed
            self.diagnostics["last_inference_s"] = elapsed
            self.diagnostics["chunks"].append({"tick": packet["tick"], "seed": self.seed,
                "request_id": packet["request_id"], "inference_s": elapsed,
                "raw_actions": raw.tolist(), "adapted_actions": actions.tolist(),
                **{key: result[key] for key in ("device", "model_inference_s", "peak_memory_mib") if result.get(key) is not None}})
        row = self.queue.popleft()
        return {"arm": row[:4], "base_linear_m_s": row[4], "base_angular_rad_s": row[5], "hold_arm": bool(row[6])}


class PolicyRegistry:
    def __init__(self, lab_root):
        self.lab = Path(lab_root)
        self.entries = {"smolvla-libero": {"label": "SmolVLA · LIBERO 实验适配",
            "endpoint": os.environ.get("VLAB_TRANSPORT_VLA_URL", "http://127.0.0.1:8772").rstrip("/"),
            "limitations": ["预训练于LIBERO机械臂；当前相机与机器人不同，跨机器人效果需要实测。",
                             "本适配只执行模型的XYZ与夹爪动作，旋转动作保留为证据；底盘指令为零。",
                             "需要具备移动动作的模型才能验证完整移动VLA；不会自动用专家补动作。"]}}
        path = self.lab / "work/transport-policies.json"
        if path.is_file():
            rows = json.loads(path.read_text(encoding="utf8"))
            for key, value in rows.items():
                if not isinstance(key, str) or not isinstance(value, dict) or not isinstance(value.get("endpoint"), str):
                    raise ValueError("本地VLA连接配置无效。")
                self.entries[key] = {field: value[field] for field in
                                     ("endpoint", "label", "limitations") if field in value}
        # Privileged diagnostics are only sent to the bundled loopback service.
        for report in trained_models(self.lab):
            job, model = report["id"], report["model"]
            self.entries[f"mobile-{job}"] = {
                "label": f"{MODELS[model]} · {job[:6]}",
                "endpoint": f"http://127.0.0.1:8772/mobile/{job}",
                "training_model": model,
                "limitations": {
                    "privileged_bc": ["读取物体真实位姿和速度，仅用于仿真诊断。"],
                    "smolvla": ["冻结 SmolVLA 编码，使用成功专家示范训练本机七维动作头。"],
                }.get(model, []) + ["请使用真实物理仿真评估任务完成率。"],
            }

    def _health(self, key, *, fresh=False):
        if key not in self.entries:
            raise ValueError("未注册所选VLA策略。")
        endpoint = self.entries[key]["endpoint"].rstrip("/")
        cached = _HEALTH_CACHE.get(endpoint)
        if not fresh and cached and time.monotonic()-cached[0] < 2.:
            return copy.deepcopy(cached[1])
        try:
            value = _json_call(endpoint+"/health", timeout=1.5)
            if value.get("ready"):
                value["identity"] = _identity(value, key)
        except ValueError as exc:
            value = {"ready": False, "state": "offline", "error": str(exc)}
        _HEALTH_CACHE[endpoint] = (time.monotonic(), value)
        return value

    def status(self):
        rows = []
        for key, entry in self.entries.items():
            health = self._health(key)
            kind = "learned_bc" if entry.get("training_model") in ("state_bc", "visual_bc", "privileged_bc") else "vla"
            rows.append({"id": key, "kind": kind, "label": entry.get("label", key),
                         "ready": bool(health.get("ready")), "state": health.get("state", "ready" if health.get("ready") else "offline"),
                         "action_schema": SCHEMA, "action_dim": 7, "identity": health.get("identity"),
                         "adapter": health.get("identity", {}).get("adapter"),
                         "limitations": entry.get("limitations", []), "error": health.get("error")})
        return {"vla_connected": any(row["ready"] for row in rows), "policies": rows,
                "current_controller": "selectable", "default_policy_id": "smolvla-libero",
                "notes": ["策略由实验显式选择；VLA服务失败会停止并保留数据，绝不回退专家。"]}

    def validate(self, spec):
        if spec.get("controller", "expert") == "expert":
            return copy.deepcopy(ExpertController.identity)
        if spec.get("controller") != "vla":
            raise ValueError("未知的任务控制策略。")
        key = spec.get("policy_id") or "smolvla-libero"
        health = self._health(key, fresh=True)
        if not health.get("ready"):
            raise ValueError("所选VLA服务尚未就绪，请在策略模块检查连接；没有启动仿真或收费生成。")
        return _identity(health, key)

    def create(self, spec):
        identity = self.validate(spec)
        if spec.get("controller", "expert") == "expert":
            return ExpertController()
        key = spec.get("policy_id") or "smolvla-libero"
        entry = self.entries[key]
        return VLAController(entry["endpoint"].rstrip("/"), identity, spec, entry.get("training_model"))
