"""Persistent conveyor grasp -> wheel retreat -> rear-bin collection runs.

One worker owns physics and both OpenGL cameras. HTTP reads immutable cached
previews; a browser is not required to advance an accepted run.
"""
from __future__ import annotations

import base64
import copy
import errno
import hashlib
import io
import json
import platform
import re
import shutil
import threading
import time
import uuid
from contextlib import nullcontext
from typing import Literal
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import HTTPException
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .duck_demo import CHASSIS, DUCK, asset_info
from .scene_assets import valid_id
from .studio import _number
from .transport_geometry import prepare_object, bin_geometry
from .transport_asset_review import AssetCatalog
from .transport_generation import (check_parameters, explicit_parameters, prepare_assets,
                                   suggest_parameters, GenerationBlocked)

DT = .05
MAX_TICKS = 1800
TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}
TASK_SEQUENCES = {
    "deliver": ["pick_from_belt", "transport_to_bin", "place_in_bin"],
    "pick_return": ["pick_from_belt", "return_to_belt"],
    "bin_retrieve": ["pick_from_belt", "transport_to_bin", "place_in_bin", "regrasp_from_bin", "lifted_from_bin"],
    "bin_return": ["pick_from_belt", "transport_to_bin", "place_in_bin", "regrasp_from_bin", "return_along_route", "return_to_belt"],
}
TASK_LABELS = {"pick_from_belt": "从移动传送带抓取并抬升", "transport_to_bin": "移动到目标箱旁",
               "place_in_bin": "受控释放并确认稳定入箱", "regrasp_from_bin": "从箱内重新抓取并抬升",
               "return_along_route": "沿原路线返回传送带旁", "return_to_belt": "受控放回传送带并确认返还",
               "lifted_from_bin": "抬出箱沿并保持稳定夹持"}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _plain(value):
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _atomic(path, data):
    temp = path.with_suffix(".tmp")
    try:
        temp.write_text(json.dumps(data, ensure_ascii=False, indent=1, allow_nan=False), encoding="utf-8")
        _replace(temp, path)
    except OSError:
        _discard_temp(temp)
        raise


def _replace(temp, path):
    # Windows may temporarily deny replacement while a reader has the file open.
    for attempt in range(4):
        try:
            temp.replace(path)
            return
        except PermissionError as exc:
            if getattr(exc, "winerror", None) not in (5, 32, 33) or attempt == 3:
                raise
            time.sleep(.025 * (attempt + 1))


def _discard_temp(path):
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _storage_status(exc):
    disk_full = exc.errno in {errno.ENOSPC, getattr(errno, "EDQUOT", errno.ENOSPC)}
    return 507 if disk_full or getattr(exc, "winerror", None) in (39, 112) else 503


def _storage_message(exc):
    if _storage_status(exc) == 507:
        return "磁盘空间不足，无法写入搬运实验数据。请释放项目所在磁盘的空间后重试。"
    return "无法写入搬运实验数据。请检查项目目录的写入权限与文件占用。"


class TransportSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    instruction: str = Field(default="从不停移动的传送带上抓取物品，然后向后移动，放到后方箱子里。", min_length=4, max_length=1500)
    object_asset: str = DUCK
    chassis_asset: str = CHASSIS
    episodes: int = Field(default=1, ge=1, le=5)
    seed: int = Field(default=3, ge=0, le=999999)
    belt_speed_m_s: float = Field(default=.025, ge=.01, le=.03)
    retreat_distance_m: float = Field(default=.40, ge=.30, le=.55)
    random_pose: bool = True
    target_xy: list[float] | None = None
    speed_scale: float = Field(default=1.6, ge=1., le=2.)
    task_kind: Literal["deliver", "pick_return", "bin_return", "bin_retrieve"] = "deliver"
    task_sequence: list[str] = Field(default_factory=list, max_length=6)
    controller: Literal["expert", "vla"] = "expert"
    policy_id: str | None = Field(default=None, min_length=1, max_length=128)
    vla_replan_steps: int = Field(default=5, ge=1, le=10)
    vla_max_inference_s: float = Field(default=60., ge=1., le=120.)
    object_mode: Literal["reuse", "generate"] = "reuse"
    object_description: str = Field(default="黄色塑料小鸭，圆润的单个刚体。", max_length=1500)
    object_parameters: dict[str, float | None] = Field(default_factory=dict)
    object_parameter_sources: dict[str, Literal["user_explicit", "model_inferred", "rules_estimate"]] = Field(default_factory=dict)
    bin_mode: Literal["builtin", "generate", "reuse"] = "builtin"
    bin_description: str = Field(default="青绿色敞口塑料收纳箱，薄直壁，无盖。", max_length=1500)
    bin_asset: str | None = None
    bin_size_m: list[float] = Field(default_factory=lambda: [.21, .22, .056])
    bin_color: str = Field(default="#2E8F85", pattern=r"^#[0-9a-fA-F]{6}$")
    experiment_name: str = Field(default="", max_length=80)
    object_name: str = Field(default="", max_length=80)
    bin_name: str = Field(default="", max_length=80)

    @field_validator("experiment_name", "object_name", "bin_name", mode="before")
    @classmethod
    def clean_name(cls, value):
        if isinstance(value, str):
            value = value.strip()
            if any(ord(char) < 32 or 127 <= ord(char) < 160 or char in "\u2028\u2029" for char in value):
                raise ValueError("名称只能填写一行。")
        return value

    @field_validator("target_xy")
    @classmethod
    def target_valid(cls, value):
        if value is not None and (len(value) != 2 or not np.isfinite(value).all()):
            raise ValueError("目标位置需要两个有限世界坐标x,y，单位米。")
        return value

    @field_validator("bin_size_m")
    @classmethod
    def bin_valid(cls, value):
        if (len(value) != 3 or not np.isfinite(value).all() or
            not (.16 <= value[0] <= .30 and .16 <= value[1] <= .30 and .04 <= value[2] <= .09)):
            raise ValueError("箱体宽／深支持16～30 cm，墙高4～9 cm。")
        return value

    @field_validator("object_parameters")
    @classmethod
    def object_valid(cls, value):
        return check_parameters(value)


def plan_transport(spec: TransportSpec):
    spec = spec.model_copy(deep=True)
    words = spec.instruction.lower()
    issues = []
    returns = bool(re.search(r"原路.*?(?:放回|返还)|(?:放回|送回|还回|返回|返还).*?(?:传送带|输送带|原处|原位|原位置)|(?:拿起|抬起|抓起).*?(?:放回|归位)|return.*?(?:belt|conveyor|original)|put.*?back", words))
    takes_from_bin = bool(re.search(r"(?:拿出|取出|拿出来|取出来|再抓|重新抓|二次抓|regrasp)|(?:放|送).*?(?:箱|bin|box).*?(?:再|然后|接着|随后|之后|then).*?(?:放回|送回|返回|返还|back)", words))
    if returns:
        spec.task_kind = "bin_return" if any(w in words for w in ("箱", "bin", "box")) and takes_from_bin else "pick_return"
    elif takes_from_bin and any(w in words for w in ("箱", "bin", "box")):
        spec.task_kind = "bin_retrieve"
    spec.task_sequence = TASK_SEQUENCES[spec.task_kind].copy()
    if spec.task_kind == "deliver" and not any(w in words for w in ("传送带", "输送带", "conveyor")):
        issues.append("请明确从传送带上抓取物品。")
    if not any(w in words for w in ("抓", "夹", "拾", "拿起", "抬起", "拿出", "取出", "grasp", "pick")):
        issues.append("当前组合任务需要抓取物品。")
    if spec.task_kind != "pick_return" and not any(w in words for w in ("箱", "bin", "box")):
        issues.append("请明确把物品放到后方箱内。")
    if returns and any(w in words for w in ("箱", "bin", "box")) and not takes_from_bin:
        issues.append("请明确是否先放入箱内、再取出放回传送带；当前描述的任务顺序不完整。")
    for pattern, issue in (
        (r"巡检|巡航|跨楼层|楼梯", "本版为平面场区内的单目标搬运，不执行巡检或跨楼层任务。"),
        (r"倒水|液体|装配|拧|开门|双臂|两台机器人|多个机器人|同时抓|连续来料|堆叠|循环|重复执行|反复|另一个箱|第二个箱|桌子|桌面|pour|assembl|repeat|loop", "当前支持单机器人单物体的一次入箱、拿起返放或入箱再取回程序；复杂追加动作与循环尚未支持。"),
        (r"停带|暂停传送带|停止传送带", "本版组合任务全程不停带，请保留传送带连续运行。"),
    ):
        # The supported word 不停带 must not be interpreted as 停带.
        if re.search(pattern, words.replace("不停带", "持续输送")):
            issues.append(issue)
    count = re.search(r"(?:采集|收集|记录|录制)\s*([\d零一二三四五六七八九十两]+)\s*(?:次|条|组)", words)
    if count:
        value = _number(count[1])
        if 1 <= value <= 5:
            spec.episodes = value
        else:
            issues.append("组合任务单次支持采集1～5条。")
    seed = re.search(r"(?:seed|种子)\s*[:：=]?\s*(\d+)", words)
    if seed:
        if int(seed[1]) <= 999999:
            spec.seed = int(seed[1])
        else:
            issues.append("随机种子需要在0～999999之间。")
    speed = re.search(r"(?:带速|传送带速度)\s*[:：=为]?\s*(\d+(?:\.\d+)?)\s*(cm/s|m/s|厘米每秒|米每秒|厘米/秒|米/秒)", words)
    if speed:
        value = float(speed[1]) / (100 if speed[2] in ("cm/s", "厘米每秒", "厘米/秒") else 1)
        if .01 <= value <= .03:
            spec.belt_speed_m_s = value
        else:
            issues.append("本版组合任务带速支持1～3 cm/s。")
    elif re.search(r"带速|传送带速度", words):
        issues.append("带速请使用明确单位，例如“带速2.5 cm/s”。")
    distance = re.search(r"(?:后退|向后移动|退后)\s*(\d+(?:\.\d+)?)\s*(厘米|cm|米|m)", words)
    if distance:
        value = float(distance[1]) / (100 if distance[2] in ("厘米", "cm") else 1)
        if .30 <= value <= .55:
            spec.retreat_distance_m = value
        else:
            issues.append("后退距离支持30～55 cm；箱子位置将随之布置。")
    if any(w in words for w in ("固定姿态", "不要随机", "不随机", "直立")):
        spec.random_pose = False
    elif any(w in words for w in ("随机姿态", "不同姿态")):
        spec.random_pose = True
    coord = re.search(r"(?:坐标|位置|箱子|箱体|目标)\s*[:：为]?\s*[（(]\s*(-?\d+(?:\.\d+)?)\s*[,，]\s*(-?\d+(?:\.\d+)?)\s*[）)]", words)
    xy = re.search(r"x\s*[:=：]\s*(-?\d+(?:\.\d+)?)\s*[,，;；\s]+y\s*[:=：]\s*(-?\d+(?:\.\d+)?)", words)
    if coord or xy:
        parsed = coord or xy
        spec.target_xy = [float(parsed[1]), float(parsed[2])]
    elif re.search(r"坐标|x\s*[:=]|y\s*[:=]", words) and spec.target_xy is None:
        issues.append("目标坐标请填写x、y，或写成“箱子坐标(0.6,0.8)”，单位米。")
    explicit = explicit_parameters(spec.object_description)
    explicit.update(spec.object_parameters)
    try:
        spec.object_parameters = check_parameters(explicit)
    except ValueError as exc:
        issues.append(str(exc))
    if spec.object_mode == "generate" and not spec.object_description.strip():
        issues.append("请描述需要生成的抓取物品。")
    if spec.bin_mode == "generate" and not spec.bin_description.strip():
        issues.append("请描述需要生成的敞口箱外观。")
    if spec.bin_mode == "reuse" and not spec.bin_asset:
        issues.append("请选择已导入的箱子外观模型。")
    nav = None
    if spec.target_xy is not None and spec.task_kind != "pick_return":
        from .transport_navigation import plan_navigation
        nav = plan_navigation(spec.target_xy, spec.bin_size_m)
        issues.extend(nav.get("issues", []))
    count_generation = int(spec.object_mode == "generate")+int(spec.bin_mode == "generate")
    target_label = f"移动到箱子坐标({spec.target_xy[0]:g}, {spec.target_xy[1]:g}) m" if spec.target_xy is not None else f"四轮底盘后退{spec.retreat_distance_m*100:g} cm"
    program_label = {"deliver": f"{target_label}，放入自定义箱", "pick_return": "拿起后原地受控放回传送带",
                     "bin_return": f"{target_label}，入箱后再取出，沿原路返回并放回传送带",
                     "bin_retrieve": f"{target_label}，入箱后再取出，在箱上方保持稳定夹持"}[spec.task_kind]
    steps = ["生成或复用Aholo物品和箱子外观，应用物品参数"]+[TASK_LABELS[stage] for stage in spec.task_sequence]+["保存完整状态、动作、双相机与每阶段判定"]
    return {"ready": not issues, "spec": spec.model_dump(),
            "summary": f"不停带{spec.belt_speed_m_s*100:g} cm/s抓取，{program_label}，{spec.speed_scale:g}倍速度，采集{spec.episodes}条。",
            "steps": steps, "task_kind": spec.task_kind, "task_sequence": spec.task_sequence,
            "issues": list(dict.fromkeys(issues)), "warnings": [
                "生成模式使用Aholo额度；已有模型的外形保持原样，尺寸与物性覆盖按本次仿真配置应用。",
                "可指定有限平面场区内的位置，规划会绕开传送带和箱体；场外、重叠或无法停靠的位置会被拒绝。",
                "箱子生成网格用于外观，敞口箱底和薄壁作为碰撞代理；生成不保证精确拓扑。",
                "夹爪靠接触保持物品，底盘靠轮地摩擦移动；质量、摩擦和轮组几何为估计值，失败按实际结果保存。",
                "每条示范重置一个物品；状态专家与VLA策略按实际配置记录来源，不自动替换策略。",
            ], "generation_count": count_generation, "parser": "transport_rules_v4", "navigation": nav}


def _action_array(action):
    if isinstance(action, dict):
        arm = action.get("arm", [0, 0, 0, 0])
        return np.asarray([*arm, action.get("base_linear_m_s", 0), action.get("base_angular_rad_s", 0), float(action.get("hold_arm", False))], dtype=float)
    return np.asarray(action, dtype=float)


def _image(encoded):
    if not encoded:
        raise ValueError("双相机画面不可用，未将缺失画面作为有效采集。")
    raw = base64.b64decode(encoded.split(",", 1)[-1])
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)


class TransportStore:
    def __init__(self, configs, lab_root, *, env_factory=None, realtime=True, generation=None):
        self.configs, self.lab = configs, Path(lab_root)
        self.root = self.lab / "datasets/transport"
        self.root.mkdir(parents=True, exist_ok=True)
        self.env_factory, self.realtime = env_factory, realtime
        self.generation = generation
        self.catalog = AssetCatalog(self.lab)
        self._lock = threading.RLock()
        self._rows, self._workers, self._cancel = {}, {}, {}
        for path in self.root.glob("transport_*/run.json"):
            row = json.loads(path.read_text(encoding="utf-8"))
            if row["status"] == "awaiting_asset_review":
                self._cancel[row["id"]] = threading.Event()
                self._refresh_reviews(row)
                if any(item["status"] == "rejected" for item in row.get("asset_reviews", [])):
                    row.update(status="cancelled", error="用户放弃了生成模型；缓存保留，未开始仿真。")
                self._finish(row)
            elif row["status"] not in TERMINAL:
                row.update(status="interrupted", error="后端重启，原仿真已中断；未自动重复采集。", updated=_now())
                self._finish(row)
            self._rows[row["id"]] = row
        # Resume only a run whose approvals were already durably recorded before
        # a crash. Unanswered review items always remain paused.
        for row in list(self._rows.values()):
            if (row["status"] == "awaiting_asset_review" and row.get("asset_reviews")
                    and all(item["status"] == "accepted" for item in row["asset_reviews"])):
                row = copy.deepcopy(row)
                row.update(status="queued", error=None)
                try:
                    self._publish(row, durable=True)
                except OSError as exc:
                    row["status"] = "interrupted"
                    self._note_storage_error(row, exc)
                    self._finish(row)
                    continue
                self._launch(row)

    def _policies(self):
        from .transport_policy import PolicyRegistry
        return PolicyRegistry(self.lab)

    def _launch(self, row, previous=None):
        def run():
            if previous is not None:
                previous.join()
            self._run(copy.deepcopy(row))
        worker = threading.Thread(target=run, daemon=True, name=row["id"])
        self._workers[row["id"]] = worker
        worker.start()

    def _refresh_reviews(self, row):
        if row.get("asset_review_required"):
            row["asset_reviews"] = [self.catalog.review_item(row, step)
                for step in row.get("asset_steps", [])
                if step.get("job_id") and step.get("status") == "succeeded" and step.get("asset_id")]

    def review_asset(self, rid, role, decision, client_request_id):
        with self._lock:
            row = self.get(rid)
            self._refresh_reviews(row)
            item = next((item for item in row.get("asset_reviews", []) if item["role"] == role), None)
            if item is None:
                raise KeyError(role)
            if item["status"] == "pending" and row["status"] != "awaiting_asset_review":
                raise ValueError("任务当前不在模型审阅阶段。")
            self.catalog.review(item["asset_id"], rid, role, decision, client_request_id)
            self._refresh_reviews(row)
            if decision == "reject":
                self._cancel.setdefault(rid, threading.Event()).set()
                row.update(status="cancelled", error="用户放弃了生成模型；缓存保留，未开始仿真。")
                self._publish(row, durable=True, final=True)
            elif row["status"] == "awaiting_asset_review" and all(
                    review["status"] == "accepted" for review in row["asset_reviews"]):
                row.update(status="queued", error=None)
                self._publish(row, durable=True)
                self._launch(row, self._workers.get(rid))
            else:
                self._publish(row, durable=True)
            return self.get(rid)

    def review_item(self, rid, role):
        row = self.get(rid)
        self._refresh_reviews(row)
        item = next((item for item in row.get("asset_reviews", []) if item["role"] == role), None)
        if item is None:
            raise KeyError(role)
        return item

    def _directory(self, rid):
        valid_id(rid)
        if not rid.startswith("transport_"):
            raise ValueError("无效的组合任务ID。")
        return self.root / rid

    def get(self, rid, compact=False):
        valid_id(rid)
        with self._lock:
            if rid not in self._rows:
                raise KeyError(rid)
            row = self._rows[rid]
            diagnostics = row.get("policy_diagnostics")
            if compact and isinstance(diagnostics, dict) and "chunks" in diagnostics:
                row = {**row, "policy_diagnostics": {key: value for key, value in diagnostics.items() if key != "chunks"}}
            return copy.deepcopy(row)

    def _note_storage_error(self, row, exc):
        message = _storage_message(exc)
        row["storage_error"] = message
        if row["status"] == "succeeded":
            row["status"] = "failed"
        if row["status"] in TERMINAL and message not in (row.get("error") or ""):
            row["error"] = " ".join(filter(None, (row.get("error"), message)))

    def _publish(self, row, *, durable=False, final=False, tolerate_storage_error=False):
        with self._lock:
            cancelled = self._cancel.get(row["id"])
            if cancelled is not None and cancelled.is_set():
                if final:
                    saved = row.get("summary", {}).get("frames", 0)
                    message = row.get("storage_error") or ("已停止并保存已采集的完整帧。" if saved else "实验已停止，尚未保存回合。")
                    row.update(stop_requested=True, status="cancelled", error=message)
                else:
                    row.update(stop_requested=True, status="stopping", error="正在停止组合任务并保存已采集的完整帧。")
            row["updated"] = _now()
            if durable:
                try:
                    _atomic(self._directory(row["id"]) / "run.json", row)
                except OSError as exc:
                    if not tolerate_storage_error:
                        raise
                    self._note_storage_error(row, exc)
            self._rows[row["id"]] = copy.deepcopy(row)

    def _finish(self, row):
        # A paused run must accept the final model approval while the previous
        # worker finishes its manifest. Terminal runs publish their saved data
        # before HTTP readers see the final state.
        with nullcontext() if row["status"] == "awaiting_asset_review" else self._lock:
            self._publish(row, durable=True, final=True, tolerate_storage_error=True)
            try:
                self._manifest(row)
            except OSError as exc:
                with self._lock:
                    if self._rows[row["id"]]["updated"] == row["updated"]:
                        self._note_storage_error(row, exc)
                        self._publish(row, durable=True, final=True, tolerate_storage_error=True)

    def _record_configuration(self, row, env):
        """Freeze the effective physical config, including generated task edits."""
        snapshot = env.config.model_dump(mode="json")
        provenance = snapshot.setdefault("provenance", {})
        provenance.update({"transport.policy_identity": json.dumps(row.get("policy_identity", {}), ensure_ascii=False),
                           "transport.task_kind": row["spec"].get("task_kind", "deliver"),
                           "transport.task_sequence": json.dumps(row.get("task_sequence", []), ensure_ascii=False)})
        sensors = {s["id"]: s for s in snapshot["sensors"]}
        if not {"front", "wrist"}.issubset(sensors):
            raise ValueError("组合采集必须包含前视和腕部相机配置与校准。")
        path = self._directory(row["id"]) / "config.json"
        _atomic(path, snapshot)
        row["configuration"] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                "file": "config.json", "encoding": "utf-8",
                                "hash_scope": "exact config.json file bytes",
                                "url": f"/api/v1/transport/datasets/{row['id']}/config"}
        row["camera_configs"] = {name: sensors[name] for name in ("front", "wrist")}
        import mujoco
        row["runtime"] = {"python": platform.python_version(), "numpy": np.__version__, "mujoco": mujoco.__version__}
        self._publish(row, durable=True)
        self._manifest(row)

    def _validate_assets(self, spec, *, pending=False):
        records = self.catalog.snapshot()["records"]
        object_name = records.get(spec.object_asset, {}).get("name") or "抓取物品 · Aholo"
        obj = None if pending and spec.object_mode == "generate" else asset_info(self.lab, spec.object_asset, object_name)
        chassis = asset_info(self.lab, spec.chassis_asset, "四轮底盘 · Aholo")
        if obj is not None:
            bounds = np.asarray(obj.get("bounds_m"), dtype=float)
            if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
                raise ValueError("物品模型尺寸无效。")
            ext = bounds[1] - bounds[0]
            source_height = obj.get("height_m")
            if (not isinstance(source_height, (int, float)) or not np.isfinite(source_height)
                    or not 0 < source_height <= .075 or min(ext) <= 0 or max(ext[:2]) > .085):
                # make_config retains the original demo's source-asset envelope.
                # Reject before generating a different role could incur a charge,
                # even when this run requests smaller effective dimensions.
                raise ValueError("当前组合任务要求原始Aholo物品高≤7.5 cm、水平尺寸≤8.5 cm；本次尺寸覆盖不能绕过原模型限制，请重新生成适配的小物品。")
            requested = spec.object_parameters
            effective = [requested.get("width_m", ext[0]), requested.get("depth_m", ext[1]), requested.get("height_m", ext[2])]
            if (obj["source"] != "lux3d" or obj["kind"] != "task_object" or obj["role"] != "dynamic"
                    or not 0 < effective[2] <= .075 or min(effective) <= 0 or max(effective[:2]) > .085):
                raise ValueError("请选择高≤7.5 cm、水平尺寸≤8.5 cm的Aholo动态小物体，或提供本次尺寸覆盖。")
            # Validate real mesh, mass and friction before another role could
            # submit a paid generation job. Bounds metadata alone is not enough.
            prepare_object(self.lab, spec.object_asset, spec.object_parameters)
        if chassis["source"] != "lux3d" or chassis["kind"] != "robot_visual":
            raise ValueError("请选择Aholo机器人底盘外观。")
        box = None
        if spec.bin_mode == "reuse" or (not pending and spec.bin_mode == "generate"):
            box = asset_info(self.lab, spec.bin_asset, records.get(spec.bin_asset, {}).get("name") or "箱子外观 · Aholo")
            if box["source"] != "lux3d" or box["kind"] != "static_prop" or box["role"] not in ("static", "visual_only"):
                raise ValueError("箱子外观需要Aholo静态道具；碰撞由敞口箱底与薄壁代理构成。")
            bin_geometry(spec.target_xy or [0., spec.retreat_distance_m], spec.bin_size_m,
                         spec.bin_color, self.lab, spec.bin_asset)
        return {"object": obj, "chassis": chassis, "bin": box}

    def create(self, spec: TransportSpec, client_request_id):
        if not client_request_id or len(client_request_id) > 100:
            raise ValueError("需要长度1～100的稳定提交ID。")
        plan = plan_transport(spec)
        if not plan["ready"]:
            raise ValueError("；".join(plan["issues"]))
        # Omit empty new name fields to preserve idempotent retries of older requests.
        empty_names = {field for field in ("experiment_name", "object_name", "bin_name") if not getattr(spec, field)}
        request_hash = hashlib.sha256(spec.model_dump_json(exclude=empty_names).encode()).hexdigest()
        with self._lock:
            existing = next((r for r in self._rows.values() if r["client_request_id"] == client_request_id), None)
            if existing:
                if existing["request_hash"] != request_hash:
                    raise ValueError("该提交ID已绑定其他要求，请创建新请求。")
                return copy.deepcopy(existing)
            if (any(r["status"] not in TERMINAL for r in self._rows.values())
                    or any(worker.is_alive() for worker in self._workers.values())):
                raise ValueError("已有组合任务正在运行，请先等待或停止。")
            if shutil.disk_usage(self.root).free == 0:
                raise OSError(errno.ENOSPC, "No space for transport data")
            spec = TransportSpec.model_validate(plan["spec"])
            # Check policy availability before any role can submit a paid mesh
            # request; VLA failures must never silently select the expert.
            self._policies().validate(spec.model_dump())
            reuse = [spec.chassis_asset]
            if spec.object_mode == "reuse":
                reuse.append(spec.object_asset)
            if spec.bin_mode == "reuse":
                reuse.append(spec.bin_asset)
            if any(not self.catalog.is_reusable(ref) for ref in reuse):
                raise ValueError("所选缓存模型尚未保存到本地资产库；仅本次使用的模型不能用于新的任务。")
            if plan["generation_count"] and not (self.generation and self.generation["objects"].provider.configured()):
                raise ValueError("后端尚未配置Aholo，无法提交新模型生成；可复用已有模型。")
            assets = self._validate_assets(spec, pending=True)
            rid = "transport_" + uuid.uuid4().hex[:14]
            directory = self._directory(rid)
            row = {"id": rid, "status": "queued", "spec": spec.model_dump(), "plan": plan,
                   "task_kind": spec.task_kind, "task_sequence": spec.task_sequence,
                   "client_request_id": client_request_id, "request_hash": request_hash,
                   "created": _now(), "updated": _now(), "assets": assets,
                   "current_episode": 0, "preview": None, "episodes": [], "error": None, "stop_requested": False,
                   "asset_review_required": bool(plan["generation_count"]), "asset_reviews": [],
                   "summary": {"episodes": 0, "successes": 0, "frames": 0},
                   "dataset_id": rid, "manifest_url": f"/api/v1/transport/datasets/{rid}/manifest",
                   "dataset": {"id": rid, "manifest_url": f"/api/v1/transport/datasets/{rid}/manifest", "episodes": []}}
            row["asset_steps"] = [
                {"role": "object", "status": "pending" if spec.object_mode == "generate" else "reused", "asset_id": None if spec.object_mode == "generate" else spec.object_asset},
                {"role": "bin", "status": "pending" if spec.bin_mode == "generate" else "reused" if spec.bin_mode == "reuse" else "builtin", "asset_id": spec.bin_asset if spec.bin_mode == "reuse" else None},
            ]
            row["parameter_provenance"] = {key: {"source": spec.object_parameter_sources.get(key, "user_explicit"), "verification": "unverified"}
                                           for key in spec.object_parameters}
            directory.mkdir()
            try:
                (directory / "episodes").mkdir()
                self._publish(row, durable=True)
            except OSError:
                for path in (directory / "episodes", directory):
                    try:
                        path.rmdir()
                    except OSError:
                        pass
                raise
            self._cancel[rid] = threading.Event()
            self._launch(row)
            return self.get(rid)

    def retry(self, rid):
        """Recover asset processing only. Never automatically repeat physics."""
        with self._lock:
            row = self.get(rid)
            if row["status"] not in ("failed", "interrupted") or row.get("configuration") or row["episodes"]:
                raise ValueError("仅可恢复进入仿真前中断的模型处理；已有仿真不会自动重复。")
            if not self.generation:
                raise ValueError("生成服务不可用。")
            if (any(r["id"] != rid and r["status"] not in TERMINAL for r in self._rows.values())
                    or any(worker.is_alive() for worker in self._workers.values())):
                raise ValueError("已有组合任务正在运行，请先等待或停止。")
            for step in row.get("asset_steps", []):
                if not step.get("job_id"):
                    continue
                job = self.generation["objects"].get(step["job_id"])
                if not job or job["state"] in ("submit_unknown", "unconfigured"):
                    raise ValueError("远端提交结果未知，不能盲目重新收费提交；请先恢复原生成任务。")
                if job["state"] in ("failed", "retryable_error"):
                    if not job.get("remote_id") or job.get("node") not in ("remote", "download", "import", "validate"):
                        raise ValueError("没有可恢复的已知远端任务，不会重新收费提交。")
                    self.generation["objects"].retry_node(job["id"], job["node"], background=True)
            self._cancel[rid] = threading.Event()
            row.update(status="generating", error=None, stop_requested=False)
            self._publish(row, durable=True)
            worker = threading.Thread(target=self._run, args=(copy.deepcopy(row),), daemon=True, name=rid)
            self._workers[rid] = worker
            worker.start()
            return self.get(rid)

    def stop(self, rid):
        with self._lock:
            row = self.get(rid)
            if row["status"] in TERMINAL:
                return row
            # Publish the request immediately; the final terminal state follows
            # once the owner thread has saved partial data and closed OpenGL.
            self._cancel.setdefault(rid, threading.Event()).set()
            try:
                (self._directory(rid) / "cancel").write_text(_now(), encoding="ascii")
            except OSError as exc:
                self._note_storage_error(row, exc)
            worker = self._workers.get(rid)
            if row["status"] == "awaiting_asset_review" or worker is None or not worker.is_alive():
                self._finish(row)
            else:
                self._publish(row, durable=True, tolerate_storage_error=True)
            return row

    def _is_cancelled(self, rid):
        return self._cancel[rid].is_set() or (self._directory(rid) / "cancel").exists()

    def _manifest(self, row):
        manifest = {k: copy.deepcopy(row[k]) for k in ("id", "spec", "assets", "episodes", "summary", "created", "updated", "status", "error")}
        manifest["experiment_name"] = row["spec"].get("experiment_name", "")
        legacy = "speed_scale" not in row["spec"]
        manifest.update(schema_version="3.0", kind="mobile_conveyor_transport", frequency_hz=20,
                        executor_version="mobile_conveyor_transport_v3", parser=row.get("plan", {}).get("parser", "transport_rules_v3"),
                        action_alignment="S_t is sampled before the recorded actual A_t is applied for 50 ms",
                        action_fields=["arm_dx", "arm_dy", "arm_dz", "grip", "base_linear_m_s", "base_angular_rad_s", "hold_arm"],
                        asset_steps=copy.deepcopy(row.get("asset_steps", [])),
                        asset_reviews=copy.deepcopy(row.get("asset_reviews", [])),
                        parameter_provenance=copy.deepcopy(row.get("parameter_provenance", {})),
                        parameter_suggestion=copy.deepcopy(row.get("parameter_suggestion", {})),
                        task_kind=row["spec"].get("task_kind", "deliver"),
                        task_sequence=copy.deepcopy(row.get("task_sequence", row["spec"].get("task_sequence", []))),
                        policy_identity=copy.deepcopy(row.get("policy_identity", {})),
                        policy_diagnostics=copy.deepcopy(row.get("policy_diagnostics", {})),
                        camera_fields=["image_front", "image_wrist"], images_recorded=True,
                        camera_configs=copy.deepcopy(row.get("camera_configs", {})),
                        image_storage="uint8 RGB arrays decoded from simulation camera JPEGs; no resizing or cropping",
                        runtime=copy.deepcopy(row.get("runtime", {})),
                        physics={"base": "free rigid body, wheel-ground contact drive; no runtime base teleportation",
                                 "grasp": "two-finger physical contact; no weld or object teleportation",
                                 "conveyor": "continuous driven roller contact",
                                 "parameters": "estimated mass, friction, chassis geometry; not real-world calibrated",
                                 "controller": "privileged simulation state expert; bounded planar waypoint navigation around known static geometry"})
        if row.get("policy_identity"):
            manifest["physics"]["controller"] = row["policy_identity"].get("kind", "unknown")
        if row["spec"].get("controller", "expert") == "expert":
            manifest["expert_version"] = "mobile_conveyor_transport_v3"
        if legacy:
            manifest.update(schema_version="1.0", expert_version="mobile_conveyor_transport_v1", parser="transport_rules_v1")
            manifest["action_fields"].remove("base_angular_rad_s")
        if row.get("configuration"):
            manifest["configuration"] = copy.deepcopy(row["configuration"])
            manifest["configuration"]["snapshot"] = json.loads((self._directory(row["id"]) / "config.json").read_text(encoding="utf-8"))
        _atomic(self._directory(row["id"]) / "manifest.json", manifest)

    def _save_episode(self, row, states, actions, requested, frames, seed, metrics, *, interrupted=False,
                      constraints=None):
        if not states:
            return
        directory = self._directory(row["id"]) / "episodes"
        available = shutil.disk_usage(directory).free
        if available == 0:
            raise OSError(errno.ENOSPC, "No space for transport episode")
        arrays = {key: np.asarray([s[key] for s in states]) for key in states[0]}
        arrays.update(action=np.asarray(actions, dtype=float), requested_action=np.asarray(requested, dtype=float),
                      image_front=np.stack(frames["front"]), image_wrist=np.stack(frames["wrist"]))
        # Three-layer arm evidence: what the policy asked for, the target the
        # constraint layer allowed, and the action actually applied. Only
        # written when every step carries it; never filled with fabricated
        # zeros, because a missing constraint must stay visibly missing.
        reach_evidence = "unavailable"
        reach_limit = None
        if constraints and len(constraints) == len(states) and all(
                item and item.get("target") and item.get("requested_target") for item in constraints):
            arrays.update(
                arm_target=np.asarray([item["target"] for item in constraints], dtype=float),
                arm_target_requested=np.asarray([item["requested_target"] for item in constraints], dtype=float),
                reach_limited=np.asarray([bool(item.get("limited")) for item in constraints], dtype=bool))
            reach_evidence = "recorded"
            limited = arrays["reach_limited"]
            if limited.any():
                deviation = np.linalg.norm(arrays["arm_target"] - arrays["arm_target_requested"], axis=1)
                reach_limit = {"steps": int(limited.sum()),
                               "first_frame": int(np.argmax(limited)),
                               "max_deviation_m": round(float(deviation[limited].max()), 6)}
        for key, value in arrays.items():
            if value.dtype.kind in "fiu" and not np.isfinite(value).all():
                raise ValueError(f"记录字段{key}含非有限数值。")
        index = len(row["episodes"])
        file = directory / f"ep{index:03d}.npz"
        temp = file.with_suffix(".tmp")
        # Prefer a fast raw flush on stop if there is room for the arrays and
        # NumPy/ZIP metadata. Compression can salvage frames on a tighter disk.
        raw = interrupted and sum(value.nbytes for value in arrays.values()) + 65536 <= available
        try:
            with temp.open("wb") as stream:
                (np.savez if raw else np.savez_compressed)(stream, **arrays)
            _replace(temp, file)
        except OSError:
            _discard_temp(temp)
            raise
        episode = {"index": index, "seed": seed, "frames": len(states), "duration_s": round(len(states)*DT, 6),
                   "success": bool(metrics.get("success", False)) and not interrupted, "metrics": _plain(metrics),
                   "interrupted": interrupted, "reach_evidence": reach_evidence,
                   "reach_limit": reach_limit,
                   "npz_url": f"/api/v1/transport/datasets/{row['id']}/episodes/{index}"}
        row["episodes"].append(episode)
        row["dataset"]["episodes"].append({"index": index, "npz_url": episode["npz_url"]})
        row["summary"] = {"episodes": len(row["episodes"]), "successes": sum(e["success"] for e in row["episodes"]),
                          "frames": sum(e["frames"] for e in row["episodes"])}
        self._publish(row, durable=True)
        self._manifest(row)

    def _run(self, row):
        from .transport_policy import validate_camera_alignment
        env = controller = None
        states, actions, requested = [], [], []
        frames = {"front": [], "wrist": []}
        seed = row["spec"]["seed"]
        metrics = {}
        try:
            if self.env_factory is None:
                from .transport import MobileTransportEnv
                factory = MobileTransportEnv
            else:
                factory = self.env_factory
            if self._is_cancelled(row["id"]):
                row["status"] = "cancelled"
                return
            if any(step["status"] not in ("reused", "builtin", "succeeded") for step in row.get("asset_steps", [])):
                row["status"] = "generating"
                self._publish(row, durable=True)
                ready = prepare_assets(row, self.generation,
                    lambda: self._publish(row, durable=True),
                    lambda: self._is_cancelled(row["id"]), self._cancel[row["id"]].wait,
                    lambda ref, role: self.catalog.record_pending(ref, row["id"], role, row["spec"].get(f"{role}_name", "")))
                if not ready:
                    row["status"] = "cancelled"
                    return
                row["status"] = "queued"
                self._publish(row, durable=True)
            # Also reconcile succeeded asset steps after a process restart.
            for step in row.get("asset_steps", []):
                if step["status"] == "succeeded" and step.get("asset_id"):
                    row["spec"]["object_asset" if step["role"] == "object" else "bin_asset"] = step["asset_id"]
            self._refresh_reviews(row)
            if any(item["status"] == "pending" for item in row.get("asset_reviews", [])):
                row.update(status="awaiting_asset_review", error=None)
                return
            row["assets"] = self._validate_assets(TransportSpec.model_validate(row["spec"]))
            registry = self._policies()
            registry.validate(row["spec"])
            controller = registry.create(row["spec"])
            row["policy_identity"] = _plain(controller.identity)
            if callable(getattr(controller, "bind_runtime", None)):
                def diagnostics_changed(diagnostics):
                    row["policy_diagnostics"] = _plain(diagnostics)
                    self._publish(row)
                controller.bind_runtime(self._cancel[row["id"]], diagnostics_changed)
            self._publish(row, durable=True)
            env = factory(self.configs.load(), self.lab, row["spec"])
            self._record_configuration(row, env)
            for episode in range(row["spec"]["episodes"]):
                if self._is_cancelled(row["id"]):
                    row["status"] = "cancelled"
                    break
                seed = row["spec"]["seed"] + episode
                env.reset(seed)
                controller.reset(seed, row["spec"]["instruction"])
                states, actions, requested = [], [], []
                constraints = []
                frames = {"front": [], "wrist": []}
                row.update(status="running", current_episode=episode)
                preview = _plain(env.snapshot(render=True))
                preview["episode_index"] = episode
                row["preview"] = preview
                self._publish(row, durable=True)
                deadline = time.monotonic()
                metrics = preview.get("metrics", {})
                for tick in range(MAX_TICKS if row["spec"].get("task_kind", "deliver") == "deliver" else MAX_TICKS*2):
                    if self._is_cancelled(row["id"]):
                        row["status"] = "cancelled"
                        break
                    # Capture observations before the configured policy action.
                    state = {k: np.array(v, copy=True) if isinstance(v, (np.ndarray, list, tuple)) else v for k, v in env.state().items()}
                    validate_camera_alignment(preview, state.get("time"))
                    image_frames = {name: _image(preview.get("images", {}).get(name)) for name in frames}
                    action = controller.action(env, state, preview)
                    if self._is_cancelled(row["id"]):
                        row["status"] = "cancelled"
                        break
                    if hasattr(controller, "diagnostics"):
                        row["policy_diagnostics"] = _plain(controller.diagnostics)
                    requested_action = _action_array(action)
                    step_result = env.step(action)
                    if callable(getattr(controller, "mark_applied", None)):
                        controller.mark_applied()
                        row["policy_diagnostics"] = _plain(controller.diagnostics)
                    next_preview = None
                    actual = (step_result or {}).get("applied_action") if isinstance(step_result, dict) else None
                    if actual is None:
                        try:
                            next_preview = _plain(env.snapshot(render=True))
                        except Exception as exc:
                            raise ValueError("仿真已推进，但该步实际动作尚不可得且预览失败；只保存此前完整配对的采集帧。") from exc
                        actual = next_preview.get("applied_action")
                    if actual is None:
                        # Every controller requires the physical env's actual clamp.
                        raise ValueError("仿真没有返回实际施加动作，停止采集以避免动作时序不明。")
                    states.append(state)
                    actions.append(np.asarray(actual, dtype=float))
                    requested.append(requested_action)
                    applied_constraint = (step_result.get("arm_constraint")
                                          if isinstance(step_result, dict) else None)
                    constraints.append(_plain(applied_constraint) if applied_constraint else None)
                    for name in frames:
                        frames[name].append(image_frames[name])
                    if isinstance(step_result, dict) and "metrics" in step_result:
                        metrics = step_result["metrics"]
                    # This complete S_t/RGB_t/A_t row survives a failing render
                    # of the next observation; it was already physically applied.
                    if next_preview is None:
                        next_preview = _plain(env.snapshot(render=True))
                    next_preview["episode_index"] = episode
                    preview = next_preview
                    metrics = preview.get("metrics", {})
                    row["preview"] = preview
                    self._publish(row, durable=tick % 20 == 0)
                    if metrics.get("success") or metrics.get("failed"):
                        break
                    if self.realtime:
                        deadline += DT
                        delay = deadline-time.monotonic()
                        if delay > 0:
                            self._cancel[row["id"]].wait(min(delay, DT))
                        elif delay < -.25:
                            deadline = time.monotonic()
                interrupted = self._is_cancelled(row["id"])
                try:
                    self._save_episode(row, states, actions, requested, frames, seed, metrics, interrupted=interrupted,
                                       constraints=constraints)
                finally:
                    # A failed metadata write may follow a successful NPZ
                    # commit. Do not save the same frames as another episode.
                    states = []
                if interrupted:
                    row.update(status="cancelled", error="已停止并保存已采集的完整帧。")
                    break
                if not metrics.get("success"):
                    row.update(status="failed", error="本次完整任务程序未成功，已保存实际失败数据并停止后续示范。")
                    break
            else:
                row.update(status="succeeded", error=None)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            storage = exc if isinstance(exc, OSError) and exc.errno is not None else None
            try:
                self._save_episode(row, states, actions, requested, frames, seed, metrics, interrupted=True,
                                   constraints=constraints)
            except OSError as save_error:
                storage = save_error
            except Exception:
                error += " 未能保存本回合的完整帧。"
            row.update(status="cancelled" if self._is_cancelled(row["id"]) else "failed",
                       error=_storage_message(exc) if storage is exc else error)
            if storage is not None:
                self._note_storage_error(row, storage)
        finally:
            if controller is not None and hasattr(controller, "diagnostics"):
                row["policy_diagnostics"] = _plain(controller.diagnostics)
            if controller is not None and callable(getattr(controller, "close", None)):
                try:
                    controller.close()
                except Exception:
                    pass
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
            if self._is_cancelled(row["id"]):
                row["status"] = "cancelled"
            self._finish(row)

    def close(self):
        with self._lock:
            active = [rid for rid, row in self._rows.items() if row["status"] not in TERMINAL | {"awaiting_asset_review"}]
        for rid in active:
            self.stop(rid)
        deadline = time.monotonic()+30.
        for worker in list(self._workers.values()):
            worker.join(timeout=max(0., deadline-time.monotonic()))


class TransportBody(BaseModel):
    spec: TransportSpec
    client_request_id: str = Field(min_length=1, max_length=100)


class SuggestObjectBody(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    description: str = Field(min_length=1, max_length=1500)
    current_parameters: dict[str, float | None] = Field(default_factory=dict)


class AssetReviewBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: Literal["save", "use_once", "reject"]
    client_request_id: str = Field(min_length=1, max_length=100)


def register_transport(app, configs, lab_root, generation=None):
    store = TransportStore(configs, lab_root, generation=generation) if generation is not None else TransportStore(configs, lab_root)

    def invoke(fn, *args):
        try:
            return fn(*args)
        except KeyError:
            raise HTTPException(404, "组合任务或资产不存在。")
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(409, str(exc))
        except OSError as exc:
            raise HTTPException(_storage_status(exc), _storage_message(exc)) from exc

    @app.get("/api/v1/transport/capabilities")
    def capabilities():
        objects, chassis, bins = [], [], []
        paths = sorted((Path(lab_root) / "assets/imported").glob("*/manifest.json"))
        records = store.catalog.snapshot()["records"]
        for path in paths:
            try:
                ref = "imported/"+path.parent.name
                record = records.get(ref)
                if record is not None and not record.get("saved"):
                    continue
                default_name = "塑料小鸭 · Aholo" if ref == DUCK else "四轮底盘 · Aholo" if ref == CHASSIS else path.parent.name
                item = asset_info(Path(lab_root), ref, (record or {}).get("name") or default_name)
                ext = np.diff(np.asarray(item["bounds_m"]), axis=0)[0]
                if item["source"] != "lux3d":
                    continue
                if (item["kind"] == "task_object" and item["role"] == "dynamic" and 0 < item["height_m"] <= .075 and 0 < min(ext) and max(ext[:2]) <= .085):
                    objects.append(item)
                elif item["kind"] == "robot_visual":
                    chassis.append(item)
                elif item["kind"] == "static_prop" and item["role"] in ("static", "visual_only"):
                    bins.append(item)
            except (ValueError, KeyError, TypeError, OSError):
                continue
        objects.sort(key=lambda item: item["id"] != DUCK)
        chassis.sort(key=lambda item: item["id"] != CHASSIS)
        planner = generation.get("planner") if generation else None
        configured = bool(generation and generation["objects"].provider.configured())
        suggestion = planner.status() if planner else {"configured": False, "state": "waiting_for_configuration"}
        return {"objects": objects, "chassis": chassis, "bins": bins, "defaults": TransportSpec().model_dump(),
                "notes": plan_transport(TransportSpec())["warnings"], "configured": configured, "suggestion": suggestion}

    @app.post("/api/v1/transport/suggest-object")
    def suggest(body: SuggestObjectBody):
        try:
            return suggest_parameters(body.description, body.current_parameters, generation)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        except Exception as exc:
            # Provider error bodies can contain credentials or URLs; return a
            # bounded public message, keeping the user's editable form usable.
            raise HTTPException(502, f"参数建议服务未完成（{type(exc).__name__}），请手动填写或稍后重试。")

    @app.post("/api/v1/transport/plan")
    def plan(body: TransportSpec):
        return plan_transport(body)

    @app.post("/api/v1/transport/runs")
    def create(body: TransportBody):
        return invoke(store.create, body.spec, body.client_request_id)

    @app.get("/api/v1/transport/runs/{rid}")
    def read(rid: str, compact: bool = False):
        return invoke(store.get, rid, compact)

    @app.post("/api/v1/transport/runs/{rid}/stop")
    def stop(rid: str):
        return invoke(store.stop, rid)

    @app.post("/api/v1/transport/runs/{rid}/retry")
    def retry(rid: str):
        return invoke(store.retry, rid)

    @app.post("/api/v1/transport/runs/{rid}/assets/{role}/review")
    def review(rid: str, role: Literal["object", "bin"], body: AssetReviewBody):
        return invoke(store.review_asset, rid, role, body.decision, body.client_request_id)

    @app.get("/api/v1/transport/runs/{rid}/assets/{role}/visual.obj")
    def object_preview(rid: str, role: Literal["object", "bin"]):
        item = invoke(store.review_item, rid, role)
        return FileResponse(invoke(store.catalog.file, item, "visual.obj"), media_type="text/plain")

    @app.get("/api/v1/transport/runs/{rid}/assets/{role}/original.glb")
    def glb_preview(rid: str, role: Literal["object", "bin"]):
        item = invoke(store.review_item, rid, role)
        return FileResponse(invoke(store.catalog.file, item, "original.glb"), media_type="model/gltf-binary")

    @app.get("/api/v1/transport/runs/{rid}/assets/{role}/package")
    def asset_package(rid: str, role: Literal["object", "bin"]):
        item = invoke(store.review_item, rid, role)
        path = invoke(store.catalog.package, item, store._directory(rid) / f"{role}-model.zip")
        return FileResponse(path, media_type="application/zip", filename=f"{rid}-{role}.zip")

    def file_path(rid, file):
        path = store._directory(rid) / file
        if not path.is_file():
            raise KeyError(rid)
        return path

    @app.get("/api/v1/transport/datasets/{rid}/manifest")
    def manifest(rid: str):
        return FileResponse(invoke(file_path, rid, "manifest.json"), media_type="application/json", filename=f"{rid}.json")

    @app.get("/api/v1/transport/datasets/{rid}/config")
    def configuration(rid: str):
        return FileResponse(invoke(file_path, rid, "config.json"), media_type="application/json", filename=f"{rid}_config.json")

    @app.get("/api/v1/transport/datasets/{rid}/episodes/{index}")
    def episode(rid: str, index: int):
        if not 0 <= index <= 4:
            raise HTTPException(404, "样本不存在。")
        return FileResponse(invoke(file_path, rid, f"episodes/ep{index:03d}.npz"), media_type="application/octet-stream", filename=f"{rid}_ep{index:03d}.npz")

    @app.on_event("shutdown")
    def close_transport():
        store.close()

    return store
