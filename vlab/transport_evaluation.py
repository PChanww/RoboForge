"""Evidence-based transport reports and explicit policy/data contracts.

Reports read immutable saved episodes. They never relabel an expert rollout as
VLA inference, infer a real-world success rate, or modify the source dataset.
"""
from __future__ import annotations

import hashlib
import json
import threading
import zipfile
from pathlib import Path

import numpy as np
from fastapi import HTTPException
from fastapi.responses import FileResponse, JSONResponse

from .scene_assets import contained_file, valid_id

ACTION_FIELDS = ["arm_dx", "arm_dy", "arm_dz", "grip", "base_linear_m_s", "base_angular_rad_s", "hold_arm"]
PHASES = ["持续输送", "追踪目标", "跟随接近", "跟随下降", "跟随夹紧", "确认抬升", "收拢准备搬运",
          "导航搬运", "到位朝向与停车", "越过目标箱沿", "箱内下降", "松爪释放", "撤离并检查", "抓取或搬运失效",
          "箱内结果确认后抬爪", "箱内物品上方", "箱内下降接近", "二次夹紧", "二次抬升", "收拢返程",
          "原路线返程", "传送带前转向停车", "返回抓取点上方", "放回传送带下降", "放回松爪", "撤离并检查返还", "全程序完成"]
LEGACY_PHASE_COUNT = 14
TASK_SEQUENCES = {
    "deliver": ["pick_from_belt", "transport_to_bin", "place_in_bin"],
    "pick_return": ["pick_from_belt", "return_to_belt"],
    "bin_retrieve": ["pick_from_belt", "transport_to_bin", "place_in_bin", "regrasp_from_bin", "lifted_from_bin"],
    "bin_return": ["pick_from_belt", "transport_to_bin", "place_in_bin", "regrasp_from_bin", "return_along_route", "return_to_belt"],
}
TASK_LABELS = {"pick_from_belt": "传送带抓取", "transport_to_bin": "移动到箱旁", "place_in_bin": "稳定入箱",
               "regrasp_from_bin": "箱内再抓取", "return_along_route": "沿原路返回", "return_to_belt": "放回传送带",
               "lifted_from_bin": "抬出箱沿并稳定夹持"}
TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}


def _action_fields(row):
    return ACTION_FIELDS.copy() if "speed_scale" in row["spec"] else [n for n in ACTION_FIELDS if n != "base_angular_rad_s"]


def _file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def policy_capabilities(lab_root):
    lab = Path(lab_root)
    result = {
        "vla_connected": False, "transport_action_dim": 7, "legacy_action_dim": 4,
        "current_controller": "privileged_expert",
        "notes": [
            "当前移动搬运由读取仿真状态的专家基线执行；结果不代表VLA策略成绩。",
            "后端已有SmolVLA历史接入、视觉BC训练评估及策略契约检查。",
            "历史SmolVLA针对旧龙门机构，视觉BC为4维机械臂动作；均未验证当前7维移动任务。",
            "移动VLA需适配双相机、任务语言、机器人状态与线／角速度动作，禁止直接截断或填零冒充兼容。",
        ],
        "backend": {"smolvla_integration": (lab / "integrations/smolvla.py").is_file(),
                    "visual_bc_training": (lab / "scripts/train_image_bc.py").is_file(),
                    "contract_checks": (lab / "vlab/compat.py").is_file(),
                    "local_visual_checkpoints": len(list((lab / "datasets/models").glob("visual*.pt")))},
        "links": {"training": "/training", "evaluations": "/evaluations", "projects": "/projects"},
    }
    from .transport_policy import PolicyRegistry
    status = PolicyRegistry(lab).status()
    result.update(status)
    # Available registry choices do not change any historical run's identity.
    result["notes"] = status.get("notes", []) + [
        "每次实验记录实际策略身份、权重哈希、适配器和推理统计；请求VLA失败时不会自动切换专家。",
        "策略可连接不代表当前机器人或多阶段任务已经训练验证，成绩以该次数据和完整程序判定为准。",
    ]
    return result


def _controller_identity(row, config):
    identity = row.get("policy_identity")
    if not identity:
        stored = config.get("provenance", {}).get("transport.policy_identity")
        try:
            identity = json.loads(stored) if isinstance(stored, str) else stored
        except (ValueError, TypeError):
            identity = None
    if isinstance(identity, dict) and identity:
        result = dict(identity)
        result.setdefault("learned_policy", result.get("kind") == "vla")
        result.setdefault("label", "VLA策略" if result["learned_policy"] else "仿真状态专家基线")
        return result
    if row["spec"].get("controller", "expert") == "vla":
        return {"kind": "unverified_vla", "label": "VLA尚未开始执行", "learned_policy": False,
                "requested_policy_id": row["spec"].get("policy_id")}
    return {"kind": "privileged_expert", "label": "仿真状态专家基线", "learned_policy": False, "id": "expert"}


def _task_sequence(spec):
    return TASK_SEQUENCES.get(spec.get("task_kind", "deliver"), TASK_SEQUENCES["deliver"]).copy()


def _program_evidence(spec, metrics, frames, duration, success):
    """Check saved physical events without deriving success from phase labels."""
    sequence = _task_sequence(spec)
    events = metrics.get("task_events", [])
    issues = []
    if not isinstance(events, list):
        events = []
        issues.append("子任务事件必须是有序列表。")
    accepted = []
    previous = (-1., -1)
    for event in events:
        if not isinstance(event, dict):
            issues.append("子任务事件缺少阶段、时刻或物理证据。")
            continue
        stage, tick, stamp = event.get("stage"), event.get("tick"), event.get("time_s")
        if (not isinstance(stage, str) or not isinstance(tick, int) or isinstance(tick, bool)
                or not isinstance(stamp, (int, float)) or isinstance(stamp, bool)
                or not np.isfinite(stamp) or not 0 <= tick <= frames
                or not 0 <= stamp <= duration+1e-6 or not np.isclose(stamp, tick*.05, atol=1e-6, rtol=0)
                or stamp < previous[0] or tick < previous[1]):
            issues.append("子任务事件时刻或顺序不符合本次20Hz采集记录。")
            continue
        previous = (stamp, tick)
        accepted.append(event)
    canonical = [event["stage"] for event in accepted if event["stage"] in sequence]
    if len(canonical) != len(set(canonical)):
        issues.append("同一子任务被重复标记完成。")
    if success and canonical != sequence:
        issues.append("成功样本缺少按任务顺序记录的完整子任务事件。")
    kind = spec.get("task_kind", "deliver")
    if success and not metrics.get("program_complete"):
        issues.append("样本成功标签未得到完整任务程序完成判定支持。")
    if success and kind in ("pick_return", "bin_return"):
        required = ["program_complete", "returned_to_belt"]
        if kind == "bin_return":
            required += ["delivered_to_bin", "regrasped_from_bin"]
        returned = next((event for event in accepted if event["stage"] == "return_to_belt"), {})
        if (not all(metrics.get(key) for key in required)
                or not all(returned.get(key) for key in ("roller_contact", "controlled_release"))
                or returned.get("contacts") != 0 or returned.get("stable_ticks", 0) < 6
                or metrics.get("delivery_failed")):
            issues.append("成功样本缺少完整返还程序和松爪后传送带稳定支撑的物理证据。")
        if kind == "bin_return":
            placed = next((event for event in accepted if event["stage"] == "place_in_bin"), {})
            lifted = next((event for event in accepted if event["stage"] == "lifted_from_bin"), {})
            if (not all(placed.get(key) for key in ("tray_contact", "controlled_release"))
                    or placed.get("stable_ticks", 0) < 6 or lifted.get("contacts") != 2):
                issues.append("入箱取回成功样本缺少稳定入箱和二次夹持抬出箱沿的证据。")
    if success and kind == "bin_retrieve":
        placed = next((event for event in accepted if event["stage"] == "place_in_bin"), {})
        lifted = next((event for event in accepted if event["stage"] == "lifted_from_bin"), {})
        clearance = lifted.get("object_low_z", -1.) - lifted.get("rim_z_m", 0.)
        if (not all(metrics.get(key) for key in ("program_complete", "delivered_to_bin", "regrasped_from_bin", "retrieved_from_bin"))
                or not all(placed.get(key) for key in ("tray_contact", "controlled_release"))
                or placed.get("stable_ticks", 0) < 6 or lifted.get("contacts") != 2
                or lifted.get("stable_ticks", 0) < 5 or not np.isfinite(clearance) or clearance <= .015
                or metrics.get("contacts") != 2 or metrics.get("delivery_failed")):
            issues.append("箱内取出成功样本缺少稳定入箱、二次夹持和抬过箱沿后稳定保持的物理证据。")
    completed = {event["stage"] for event in accepted}
    subtasks = [{"stage": stage, "label": TASK_LABELS[stage], "completed": stage in completed,
                 "evidence": next((event for event in accepted if event["stage"] == stage), None)}
                for stage in sequence]
    return events, subtasks, issues


def _npy_header(archive, name):
    with archive.open(name + ".npy") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError("Unsupported NPY header version")
        return tuple(shape), dtype


class TransportEvaluation:
    def __init__(self, lab_root):
        self.lab = Path(lab_root)
        self.root = self.lab / "datasets/transport"
        self._cache = {}
        self._lock = threading.RLock()

    def directory(self, rid):
        valid_id(rid)
        if not rid.startswith("transport_"):
            raise ValueError("无效的移动实验ID。")
        directory = contained_file(self.root.resolve(), rid)
        if not (directory / "run.json").is_file():
            raise KeyError(rid)
        return directory

    def read(self, rid):
        return json.loads((self.directory(rid) / "run.json").read_text(encoding="utf-8"))

    def contract(self, rid):
        row = self.read(rid)
        fields = _action_fields(row)
        legacy = len(fields) != 7
        path = self.directory(rid) / "config.json"
        cfg = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        controller = (cfg.get("controllers") or [{}])[0]
        identity = _controller_identity(row, cfg)
        evaluated = bool(identity.get("learned_policy") and any(ep.get("frames", 0) > 0 for ep in row.get("episodes", [])))
        model = identity.get("training_model")
        privileged = model == "privileged_bc"
        conditioning = {}
        if model:
            conditioning["task_parameters"] = ["target_xy", "task_kind", "simulation_time", "belt_speed_m_s", "speed_scale"]
            if privileged:
                conditioning["privileged_state"] = {"object_pose": 7, "object_velocity": 6, "scope": "simulation_only"}
        return {
            "schema_version": "transport-policy-contract-1.0", "run_id": rid,
            "experiment_name": row["spec"].get("experiment_name", ""),
            "controller_source": identity["kind"], "controller_identity": identity,
            "learned_policy_evaluated": evaluated, "vla_evaluated": evaluated and identity["kind"] == "vla",
            "policy_diagnostics": row.get("policy_diagnostics", {}),
            "task_kind": row["spec"].get("task_kind", "deliver"),
            "task_sequence": _task_sequence(row["spec"]),
            "task": row["spec"]["instruction"], "fps": 20.,
            "observation": {
                "task": {"dtype": "string", "meaning": "task language; retained verbatim"},
                "state": {"fields": {"arm_qpos": [7], "base_pose": [7], "base_velocity": [6],
                                     "wheel_qpos": [4], "wheel_qvel": [4]},
                          "dim": 28, "quaternion_order": "wxyz", "frame": "world",
                          "note": "Real deployment requires matching joint sensing and base localization."},
                "images": {s["id"]: {"shape": [s["resolution"][1], s["resolution"][0], 3], "dtype": "uint8", "color": "RGB"}
                           for s in cfg.get("sensors", []) if s["id"] in ("front", "wrist")},
                **conditioning,
                "annotations_only": ([] if privileged else ["object_pose", "object_velocity"]) + ["phase", "contacts", "success"],
            },
            "action": {"dim": len(fields), "names": fields,
                       "arm_range": [-1., 1.], "arm_max_delta_m": controller.get("max_delta_m"),
                       "arm_frame": controller.get("frame", "world"), "grip": "-1 closed, +1 open",
                       "linear_unit": "m/s", "angular_unit": "rad/s",
                       "hold_arm": "0 or 1; hold arm joints while still applying grip command",
                       "applied_limits": {"linear_abs": .065 * row["spec"].get("speed_scale", 1.), "angular_abs": .8},
                       "source_field": "action", "unclamped_request_field": "requested_action"},
            "alignment": "S_t and RGB at time[t], followed by recorded A_t applied for 0.05 simulation seconds",
            "normalization": "not fitted; training must fit statistics on training split only",
            "compatibility": {"historical_4d_policies": False, "current_mobile_7d": not legacy,
                              "legacy_6d_recording": legacy,
                              "reason": "This historical recording has six fields and no angular action; it is unsupported by the current seven-field mobile policy contract. Do not pad or truncate actions." if legacy else "Current mobile action has seven fields; old 4D and historical 6D policies cannot drive this task without an explicit adapter."},
            "config_sha256": row.get("configuration", {}).get("sha256"),
        }

    def _episode(self, directory, row, fields, config, spec):
        issues, timeline = [], []
        n = int(row["frames"])
        last_contact, failure = None, None
        path = directory / "episodes" / f"ep{row['index']:03d}.npz"
        try:
            if n <= 0:
                issues.append("采集样本没有有效帧，不能作为已完成验证。")
            image_shapes = {"image_"+s["id"]: (n, s["resolution"][1], s["resolution"][0], 3)
                            for s in config.get("sensors", []) if s["id"] in ("front", "wrist")}
            if set(image_shapes) != {"image_front", "image_wrist"}:
                issues.append("有效配置缺少双相机分辨率，无法核对图像契约。")
            with zipfile.ZipFile(path) as archive:
                for name in ("image_front", "image_wrist"):
                    actual, dtype = _npy_header(archive, name)
                    if actual != image_shapes.get(name) or dtype != np.dtype("uint8"):
                        issues.append(f"{name}的形状或RGB数据类型与相机契约不一致。")
            with np.load(path, allow_pickle=False) as z:
                required = {"time": (n,), "action": (n, len(fields)), "requested_action": (n, len(fields)),
                            "arm_qpos": (n, 7), "base_pose": (n, 7), "base_velocity": (n, 6),
                            "wheel_qpos": (n, 4), "wheel_qvel": (n, 4), "object_pose": (n, 7),
                            "object_velocity": (n, 6), "phase": (n,), "contacts": (n,)}
                for name, shape in required.items():
                    a = z[name]
                    if a.shape != shape or not np.issubdtype(a.dtype, np.number) or not np.isfinite(a).all():
                        issues.append(f"{name}的形状或有限数值检查未通过。")
                action = z["action"]
                if action.shape == (n, len(fields)) and np.issubdtype(action.dtype, np.number) and np.isfinite(action).all():
                    if np.any(np.abs(action[:, :4]) > 1+1e-7):
                        issues.append("实际机械臂／夹爪动作超出[-1,1]归一化范围。")
                    if np.any(np.abs(action[:, fields.index("base_linear_m_s")]) > .065*spec.get("speed_scale", 1.)+1e-7):
                        issues.append("实际底盘线速度动作超出本次速度倍率上限。")
                    if "base_angular_rad_s" in fields and np.any(np.abs(action[:, fields.index("base_angular_rad_s")]) > .8+1e-7):
                        issues.append("实际底盘角速度动作超出0.8 rad/s上限。")
                    if np.any(~np.isin(action[:, fields.index("hold_arm")], [0., 1.])):
                        issues.append("hold_arm实际动作必须为0或1。")
                ts, phase, contacts = z["time"], z["phase"], z["contacts"]
                if len(ts) > 1 and not np.allclose(np.diff(ts), .05, rtol=0, atol=1e-6):
                    issues.append("采样时间未严格按20Hz递增，不能声明50ms动作对齐。")
                if ts.shape == (n,) and n and not np.isclose(ts[0], 0., rtol=0, atol=1e-6):
                    issues.append("样本首帧时间不是重置后的0秒。")
                if not np.isclose(float(row.get("duration_s", -1)), n*.05, rtol=0, atol=1e-6):
                    issues.append("样本记录时长与20Hz帧数不一致。")
                phase_count = len(PHASES) if "task_kind" in spec else LEGACY_PHASE_COUNT
                if np.any(~np.isin(phase, np.arange(phase_count))) or np.any(~np.isin(contacts, [0,1,2])):
                    issues.append("执行阶段或夹持接触数量超出有效离散范围。")
                if n and ts.shape == phase.shape == contacts.shape == (n,) and all(np.isfinite(a).all() for a in (ts, phase, contacts)):
                    boundaries = np.r_[0, np.flatnonzero(np.diff(phase)) + 1, n]
                    for start, end in zip(boundaries, boundaries[1:]):
                        p = int(phase[start])
                        timeline.append({"phase": p, "label": PHASES[p] if 0 <= p < len(PHASES) else str(p),
                                         "start_s": round(float(ts[start]), 6), "end_s": round(float(ts[end-1])+.05, 6),
                                         "frames": int(end-start)})
                    both = np.flatnonzero(contacts == 2)
                    last_contact = round(float(ts[both[-1]]), 6) if len(both) else None
                    p = int(phase[-1])
                    if not row.get("success"):
                        metrics = row.get("metrics", {})
                        if row.get("interrupted"):
                            reason = "采集被停止或中断，不能据此判断任务不可行。"
                        elif spec.get("task_kind") == "bin_return" and metrics.get("delivered_to_bin"):
                            reason = ("已完成入箱和二次抓取，但尚未完成沿原路返还与传送带稳定支撑。"
                                      if metrics.get("regrasped_from_bin") else "已完成稳定入箱，但尚未完成箱内再抓取和返还程序。")
                        elif spec.get("task_kind") == "bin_retrieve" and metrics.get("delivered_to_bin"):
                            reason = "已完成稳定入箱，但尚未完成二次夹持并稳定抬出箱沿。"
                        elif spec.get("task_kind") == "pick_return" and metrics.get("lifted"):
                            reason = "已经抓取并抬升，但尚未完成放回传送带后的松爪和稳定支撑判定。"
                        elif metrics.get("delivery_failed") and metrics.get("lifted") and len(both) and contacts[-1] < 2:
                            reason = "已经抓取并抬升，但搬运或放置时失去双侧夹持；未完成受控释放与稳定入箱。"
                        elif not metrics.get("lifted"):
                            reason = "本次未完成稳定抓取与抬升。"
                        elif metrics.get("controlled_release") and not metrics.get("success"):
                            reason = "已经释放，但箱内接触或低速稳定条件未通过。"
                        else:
                            reason = "本次在执行阶段结束前未满足完整任务的成功条件。"
                        failure = {"phase": p, "stage": PHASES[p] if 0 <= p < len(PHASES) else str(p),
                                   "reason": reason, "last_bilateral_contact_s": last_contact}
                metrics = row.get("metrics", {})
                expected_success = bool(metrics.get("success")) and not row.get("interrupted", False)
                if bool(row.get("success")) != expected_success:
                    issues.append("样本成功标签与物理判定结果不一致。")
                if spec.get("task_kind", "deliver") == "deliver" and row.get("success") and (not all(metrics.get(key) for key in ("grasped", "lifted", "controlled_release", "tray_contact"))
                                           or metrics.get("contacts") != 0 or metrics.get("stable_ticks", 0) < 6
                                           or metrics.get("delivery_failed")):
                    issues.append("成功样本缺少双侧抓取、抬升、受控释放与箱底稳定接触的完整物理判定证据。")
        except (OSError, KeyError, TypeError, ValueError, IndexError, EOFError, zipfile.BadZipFile) as exc:
            issues.append(f"采集文件未完整通过读取检查（{type(exc).__name__}）。")
        metrics = row.get("metrics", {})
        events, subtasks, program_issues = _program_evidence(spec, metrics, n, float(row.get("duration_s", 0)), bool(row.get("success")))
        # Historical datasets predate explicit task programs and remain readable.
        if "task_kind" in spec:
            issues.extend(program_issues)
        return {"index": row["index"], "seed": row.get("seed"), "success": bool(row.get("success")),
                "frames": n, "duration_s": row.get("duration_s"), "metrics": row.get("metrics", {}),
                "task_kind": spec.get("task_kind", "deliver"), "task_sequence": _task_sequence(spec),
                "program_complete": metrics.get("program_complete", bool(row.get("success")) if "task_kind" not in spec else False),
                "task_events": events, "subtasks": subtasks,
                "reach_evidence": row.get("reach_evidence", "unavailable"), "reach_limit": row.get("reach_limit"),
                "phase_timeline": timeline, "failure": failure, "data_quality": {"valid": not issues, "issues": issues,
                    "image_validation": "NPY shape/dtype headers only; pixels are not decoded or semantically scored"}}

    def report(self, rid):
        directory = self.directory(rid)
        row = self.read(rid)
        files = [directory / name for name in ("run.json", "config.json", "manifest.json")] + sorted((directory / "episodes").glob("*.npz"))
        signature = tuple((str(p.name), p.stat().st_mtime_ns, p.stat().st_size) for p in files if p.is_file())
        with self._lock:
            if rid in self._cache and self._cache[rid][0] == signature:
                return self._cache[rid][1]
            fields = _action_fields(row)
            config_path, manifest_path = directory / "config.json", directory / "manifest.json"
            document_issues = []
            config = {}
            for p in (config_path, manifest_path):
                if p.is_file():
                    try:
                        document = json.loads(p.read_text(encoding="utf-8"))
                        if not isinstance(document, dict):
                            raise ValueError("configuration and manifest must be JSON objects")
                        if p == config_path:
                            config = document
                        elif document.get("action_fields") != fields or document.get("frequency_hz") != 20:
                            document_issues.append("数据清单动作字段或20Hz频率与本实验契约不一致。")
                    except (OSError, ValueError, TypeError):
                        document_issues.append(f"{p.name}无法作为有效JSON配置读取。")
                elif row.get("episodes") or row["status"] == "succeeded":
                    document_issues.append(f"完整实验缺少{p.name}。")
            if config and (config.get("controllers") or [{}])[0].get("frequency_hz") != 20:
                document_issues.append("有效控制配置不是20Hz，与动作采集声明不一致。")
            episodes = [self._episode(directory, ep, fields, config, row["spec"]) for ep in row.get("episodes", [])]
            issues = [f"Episode {ep['index']+1}: {issue}" for ep in episodes for issue in ep["data_quality"]["issues"]]
            issues.extend(document_issues)
            if row["status"] == "succeeded" and not episodes:
                issues.append("实验标记完成但没有已保存的采集帧。")
            if row["status"] == "succeeded" and (any(not ep["success"] for ep in episodes) or
                    (row["spec"].get("episodes") is not None and len(episodes) != row["spec"]["episodes"])):
                issues.append("实验完成状态与已保存的成功样本数量不一致。")
            if row.get("configuration"):
                p = directory / "config.json"
                if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != row["configuration"]["sha256"]:
                    issues.append("有效配置文件与采集时记录的SHA-256不一致。")
            status = row["status"]
            successes = sum(ep["success"] for ep in episodes)
            identity = _controller_identity(row, config)
            learned = bool(identity.get("learned_policy"))
            task_kind = row["spec"].get("task_kind", "deliver")
            sequence = _task_sequence(row["spec"])
            if issues:
                code, label, reason = "invalid_data", "数据校验未通过", "数据契约检查存在问题，请先修复数据再用于训练。"
            elif status == "succeeded" and episodes and successes == len(episodes):
                code = "passed"
                label = {"vla": "本次VLA验证通过", "learned_bc": "本次BC策略验证通过"}.get(
                    identity["kind"], "本次专家基线通过")
                reason = "已按顺序完成"+"、".join(TASK_LABELS[stage] for stage in sequence)+"；仅代表本次参数与种子的结果。"
            elif status == "failed":
                code, label = "failed", "本次验证未通过"
                error = row.get("error")
                last_failure = next((ep["failure"] for ep in reversed(episodes) if ep["failure"]), None)
                if error and any(ep.get("interrupted") for ep in row.get("episodes", [])):
                    reason = error
                    if last_failure:
                        last_failure["reason"] = error
                else:
                    reason = last_failure["reason"] if last_failure else error or "场景未完成。"
            elif status in ("interrupted", "cancelled"):
                code, label, reason = "incomplete", "验证未完成", "任务被中断或停止，已有数据保留。"
            else:
                code, label, reason = "pending", "等待模型确认" if status == "awaiting_asset_review" else "验证中" if status == "running" else "尚未完成验证", "生成、用户确认或仿真尚未全部完成。"
            policy = policy_capabilities(self.lab)
            scope = ("该权重与适配器在当前仿真、资产、参数和种子下的实际模型执行；不是实物或其他任务的泛化结论"
                     if learned else "当前仿真、资产、参数与种子的专家基线可行性；不是实物或VLA泛化结论")
            result = {"schema_version": "transport-evaluation-1.1", "run_id": rid, "status": status,
                      "experiment_name": row["spec"].get("experiment_name", ""),
                      "action_contract": {"dim": len(fields), "fields": fields, "current_mobile_7d_supported": len(fields) == 7},
                      "verdict": {"code": code, "label": label, "reason": reason, "scope": scope},
                      "controller": identity, "policy_diagnostics": row.get("policy_diagnostics", {}),
                      "task_kind": task_kind, "task_sequence": sequence,
                      "learned_policy_evaluated": bool(learned and any(ep["frames"] > 0 for ep in episodes)),
                      "vla_evaluated": bool(learned and identity["kind"] == "vla" and any(ep["frames"] > 0 for ep in episodes)),
                      "summary": {"episodes": len(episodes), "successes": successes,
                                  "frames": sum(ep["frames"] for ep in episodes),
                                  "success_rate": successes / len(episodes) if episodes else None},
                      "episodes": episodes, "data_quality": {"valid": not issues, "issues": issues}, "policy": policy,
                      "artifacts": [{"file": str(p.relative_to(directory)).replace("\\", "/"), "bytes": p.stat().st_size,
                                     "sha256": _file_hash(p)} for p in files if p.is_file()],
                      "downloads": {"report_url": f"/api/v1/transport/runs/{rid}/evaluation/download",
                                    "contract_url": f"/api/v1/transport/runs/{rid}/policy-contract",
                                    "archive_url": f"/api/v1/transport/runs/{rid}/archive"},
                      "limitations": ["物性与碰撞代理为估计，未经实物标定。", "成功和失败数据均保留，训练前需按结果与阶段筛选。",
                                      "历史6维移动数据不兼容当前7维策略契约，不自动填充角速度动作。" if len(fields) == 6 else "当前7维移动动作不兼容原4维视觉BC训练器，导出包保留完整契约，不自动截断动作。"]}
            self._cache[rid] = (signature, result)
            return result

    def archive(self, rid):
        directory = self.directory(rid)
        row = self.read(rid)
        if row["status"] not in TERMINAL or not row.get("episodes") or not any(ep.get("frames",0)>0 for ep in row["episodes"]):
            raise ValueError("请在实验结束并保存数据后导出完整实验包。")
        report, contract = self.report(rid), self.contract(rid)
        files = [directory / name for name in ("run.json", "config.json", "manifest.json")]
        files += [directory / "episodes" / f"ep{ep['index']:03d}.npz" for ep in row["episodes"]]
        if any(not path.is_file() for path in files):
            raise ValueError("完整实验包所需配置、清单或样本文件缺失，无法导出完整原始数据。")
        files = list(dict.fromkeys([*files, *sorted((directory / "episodes").glob("*.npz"))]))
        digest = hashlib.sha256(json.dumps(report, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]
        exports = self.lab / "datasets/transport_exports"
        exports.mkdir(parents=True, exist_ok=True)
        target = exports / f"{rid}_{digest}.zip"
        with self._lock:
            if not target.is_file():
                temp = target.with_suffix(".tmp")
                with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_STORED) as archive:
                    for path in files:
                        if path.is_file():
                            archive.write(path, str(path.relative_to(directory)).replace("\\", "/"))
                    archive.writestr("evaluation.json", json.dumps(report, ensure_ascii=False, indent=2))
                    archive.writestr("policy-contract.json", json.dumps(contract, ensure_ascii=False, indent=2))
                temp.replace(target)
        return target


def register_transport_evaluation(app, lab_root):
    evaluator = TransportEvaluation(lab_root)

    def invoke(fn, *args):
        try:
            return fn(*args)
        except (KeyError, FileNotFoundError):
            raise HTTPException(404, "实验或采集数据不存在。")
        except ValueError as exc:
            raise HTTPException(409, str(exc))

    @app.get("/api/v1/transport/policy-capabilities")
    def policies():
        return policy_capabilities(lab_root)

    @app.get("/api/v1/transport/history")
    def history():
        rows = []
        for path in sorted(evaluator.root.glob("*/run.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:100]:
            row = json.loads(path.read_text(encoding="utf-8"))
            item = {k: row.get(k) for k in ("id", "status", "summary", "created", "updated", "error")}
            item.update(task_kind=row.get("spec", {}).get("task_kind", "deliver"),
                        experiment_name=row.get("spec", {}).get("experiment_name", ""),
                        instruction=row.get("spec", {}).get("instruction", ""),
                        policy_identity=_controller_identity(row, {}))
            rows.append(item)
        return {"runs": rows}

    @app.get("/api/v1/transport/runs/{rid}/evaluation")
    def report(rid: str):
        return invoke(evaluator.report, rid)

    @app.get("/api/v1/transport/runs/{rid}/evaluation/download")
    def report_download(rid: str, inline: bool = False):
        headers = {} if inline else {"Content-Disposition": f'attachment; filename="{rid}_evaluation.json"'}
        return JSONResponse(invoke(evaluator.report, rid), headers=headers)

    @app.get("/api/v1/transport/runs/{rid}/policy-contract")
    def contract(rid: str, inline: bool = False):
        headers = {} if inline else {"Content-Disposition": f'attachment; filename="{rid}_policy-contract.json"'}
        return JSONResponse(invoke(evaluator.contract, rid), headers=headers)

    @app.get("/api/v1/transport/runs/{rid}/archive")
    def archive(rid: str):
        return FileResponse(invoke(evaluator.archive, rid), media_type="application/zip", filename=f"{rid}_experiment.zip")
