"""Persistent requirements -> Aholo assets -> real scene -> collection workflow.

The four text fields describe appearance and a bounded execution task. Work
parameters are parsed by explicit rules, never presented as an LLM response.
Paid submissions are bound to a durable run/role ID before any provider call.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import uuid
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import HTTPException
from pydantic import BaseModel, Field

from .duck_demo import CHASSIS, DUCK, asset_info, make_config
from .generation.api import SpecPatchBody, apply_user_patch
from .generation.schema import ChatMessage
from .generation.sessions import _now
from .schema import SceneProp
from .studio import _number


class ScenarioSpec(BaseModel):
    environment: str = Field(default="仓库分拣区，一组整齐货架和周转箱，工作区前方保持空旷。", min_length=2, max_length=1500)
    robot: str = Field(default="四轮抓取机器人，紧凑底盘，白色与深绿色外壳，黑色轮胎，停靠执行抓取。", min_length=2, max_length=1500)
    object: str = Field(default="一只黄色塑料小鸭，橙色短嘴、黑色眼睛，光滑圆润的单个完整刚体。", min_length=2, max_length=1500)
    operation: str = Field(default="传送带不停，随机姿态跟随抓取物体，放入机器人旁的收纳箱，采集3次。", min_length=2, max_length=1500)
    environment_mode: Literal["generate", "reuse", "none"] = "generate"
    robot_mode: Literal["generate", "reuse"] = "generate"
    object_mode: Literal["generate", "reuse"] = "generate"
    environment_asset: str | None = None
    chassis_asset: str = CHASSIS
    object_asset: str = DUCK
    robot_mount: Literal["four_wheel", "fixed_arm"] = "four_wheel"
    episodes: int = Field(default=3, ge=1, le=20)
    seed: int = Field(default=0, ge=0, le=999999)
    belt_speed_m_s: float = Field(default=.025, ge=.01, le=.04)
    continuous: bool = True
    random_pose: bool = True


def plan_spec(spec: ScenarioSpec):
    spec = spec.model_copy(deep=True)
    text = spec.operation.lower()
    issues, warnings = [], [
        "环境按描述生成背景道具并组装；碰撞使用显式包围盒，工作区保持空旷。",
        "机器人外观可生成；关节臂和抓取控制沿用现有机构，四轮底盘停靠作业。",
        "物体按高6 cm、35 g导入；形状、摩擦和质量未经实物标定，新造型不保证抓取成功。",
        "工作描述按已支持的关键词解析；以下参数决定实际执行，未识别的文字不增加动作能力。",
    ]
    for pattern, message in (
        (r"导航|行驶|开车|行走|移动底盘|底盘移动|追着传送带走|绕障|避障|走到|开到|驶[向入到]|navigate|driv", "当前机器人停靠抓放，不支持底盘导航、驾驶或避障。"),
        (r"倒水|液体|开门|拧|旋拧|插入|插孔|装配|堆叠|叠放|焊接|切割|推[动移]|pour|assembl|stack|push", "当前执行器支持传送带跟随抓取入箱，不支持该复合操作。"),
        (r"双臂|双手|两台机器人|多个机器人|两个机器人|多物体|多个物体|同时抓|同时操作|连续来料|连续多[只个]|不间断来料", "当前每条示范操作一个物体；多机器人和连续多物体流尚未接入。"),
    ):
        if re.search(pattern, text):
            issues.append(message)
    if not any(w in text for w in ("抓", "夹", "拾", "分拣", "pick", "grasp")):
        issues.append("请明确抓取物体并放入收纳箱；当前支持这类抓放任务。")
    if re.search(r"四足|人形|机器人狗|飞行|无人机|六足|quadruped|humanoid", spec.robot.lower()):
        issues.append("当前支持停靠四轮底盘加机械臂，或固定机械臂；该机器人尚无可执行机构。")
    if re.search(r"厘米|毫米|\bcm\b|\bmm\b|质量|重量|摩擦|\d+\s*(?:kg|克|千克)", spec.object.lower()):
        issues.append("物体尺寸和物理参数目前采用演示规格；请将物品栏用于外形描述，不设置额外物理数值。")
    count = re.search(r"(?:采集|收集|录制|记录)\s*([\d零一二三四五六七八九十两]+)\s*(?:条|次|组)", text)
    if count:
        value = _number(count[1])
        if not 1 <= value <= 20:
            issues.append("单次采集数量支持1～20条。")
        else:
            spec.episodes = value
    seed = re.search(r"(?:seed|种子)\s*[:：=]?\s*(\d+)", text)
    if seed:
        if int(seed[1]) > 999999:
            issues.append("随机种子需要在0～999999之间。")
        else:
            spec.seed = int(seed[1])
    speed = re.search(r"(?:带速|传送带速度|belt\s*speed)\s*[:：=为]?\s*(\d+(?:\.\d+)?)\s*(cm/s|m/s|厘米每秒|米每秒|厘米/秒|米/秒)", text)
    if speed:
        value = float(speed[1]) / (100 if speed[2] in ("cm/s", "厘米每秒", "厘米/秒") else 1)
        if not .01 <= value <= .04:
            issues.append("本版传送带速度支持1～4 cm/s；过快目标可能超出抓取范围。")
        else:
            spec.belt_speed_m_s = value
    elif re.search(r"带速|传送带速度|belt\s*speed", text):
        issues.append("请把带速写为例如“带速2 cm/s”，使用明确单位。")
    if any(w in text for w in ("不停", "持续输送", "连续输送", "continuous")):
        spec.continuous = True
    elif any(w in text for w in ("停带", "暂停传送带", "到位停止", "节拍")):
        spec.continuous = False
    if any(w in text for w in ("固定姿态", "不要随机", "不随机", "直立")):
        spec.random_pose = False
    elif any(w in text for w in ("随机姿态", "不同姿态")):
        spec.random_pose = True
    if spec.robot_mount == "fixed_arm":
        if spec.robot_mode == "generate":
            warnings.append("固定机械臂沿用现有机构；已取消底盘外观生成。")
        spec.robot_mode = "reuse"
    for role, mode in (("物体", spec.object_mode), ("机器人底盘", spec.robot_mode), ("环境道具", spec.environment_mode)):
        if mode == "reuse" and not (role == "机器人底盘" and spec.robot_mount == "fixed_arm"):
            warnings.append(f"{role}选择了复用，外形沿用所选资产；若要按新描述修改外形，请改为生成新模型。")
    if spec.environment_mode == "reuse" and not spec.environment_asset:
        issues.append("请选择可复用的环境道具，或选择生成/不添加环境。")
    for mode, ref, title in ((spec.object_mode, spec.object_asset, "物体"),
                              (spec.robot_mode, spec.chassis_asset, "底盘")):
        if mode == "reuse" and not ref.startswith("imported/"):
            issues.append(f"请选择已导入的Aholo{title}。")
    count = sum((spec.environment_mode == "generate", spec.robot_mode == "generate" and spec.robot_mount == "four_wheel", spec.object_mode == "generate"))
    mount = "停靠四轮机械臂" if spec.robot_mount == "four_wheel" else "固定机械臂"
    summary = f"{mount}以{spec.belt_speed_m_s*100:g} cm/s{'不停带跟随' if spec.continuous else '停带'}抓取入箱，{'随机' if spec.random_pose else '固定'}初始姿态，采集{spec.episodes}条。"
    return {"ready": not issues, "spec": spec.model_dump(), "summary": summary,
            "steps": ["生成或复用环境道具、机器人外观和操作物体", "校验尺寸与机构，组装有碰撞的场景", "真实MuJoCo抓取仿真与双相机采集"],
            "warnings": warnings, "issues": list(dict.fromkeys(issues)), "generation_count": int(count),
            "parser": "structured_rules_v1"}


class ScenarioStore:
    _locks: dict[str, threading.RLock] = {}

    def __init__(self, root: Path, configs, jobs, generation: dict, *, background=False):
        self.lab = Path(root)
        self.dir = self.lab / "datasets/scenarios"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.configs, self.jobs, self.gen = configs, jobs, generation
        self.background = background
        self._lock = self._locks.setdefault(str(self.dir.resolve()), threading.RLock())
        self._active = set()

    def _save(self, row):
        row["updated"] = _now()
        path = self.dir / f"{row['id']}.json"
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(row, ensure_ascii=False, indent=1), encoding="utf-8")
        temp.replace(path)

    def get(self, rid):
        from .scene_assets import valid_id
        valid_id(rid)
        # Writers publish by atomic replace; progress reads must remain available
        # while a worker compiles a large mesh or starts a physics process.
        path = self.dir / f"{rid}.json"
        if not path.is_file():
            raise KeyError(rid)
        return json.loads(path.read_text(encoding="utf-8"))

    def _cancelled(self, row):
        if (self.dir / f"{row['id']}.cancel").exists() or row["status"] == "cancelled":
            for job in self.jobs.list():
                if (job["id"] == row.get("job_id") or job.get("params", {}).get("provenance", {}).get("scenario_id") == row["id"]):
                    if job["status"] in ("running", "queued"):
                        self.jobs.stop(job["id"])
                    row["job_id"] = job["id"]
            row.update(status="cancelled", error="已停止后续组装和采集；已提交的在线生成任务仍保留。")
            self._save(row)
            return True
        return False

    def list(self):
        with self._lock:
            return sorted([json.loads(p.read_text(encoding="utf-8")) for p in self.dir.glob("scenario_*.json")], key=lambda r: r["created"], reverse=True)

    def create(self, spec: ScenarioSpec, client_request_id: str):
        if not client_request_id or len(client_request_id) > 100:
            raise ValueError("需要稳定的提交ID，长度1～100。")
        plan = plan_spec(spec)
        if not plan["ready"]:
            raise ValueError("；".join(plan["issues"]))
        requested_hash = hashlib.sha256(spec.model_dump_json().encode()).hexdigest()
        with self._lock:
            for existing in self.list():
                if existing["client_request_id"] == client_request_id:
                    if existing["request_hash"] != requested_hash:
                        raise ValueError("提交ID已绑定其他要求；请创建新请求。")
                    return existing
            spec = ScenarioSpec.model_validate(plan["spec"])
            if plan["generation_count"] and not self.gen["objects"].provider.configured():
                raise ValueError("后端尚未配置Aholo，无法提交新模型生成。")
            # Validate reused refs before spending credits on the other roles.
            self._validate_reused(spec)
            rid = "scenario_" + uuid.uuid4().hex[:12]
            steps = []
            for role, mode, ref in (("environment", spec.environment_mode, spec.environment_asset),
                                    ("robot", spec.robot_mode, spec.chassis_asset),
                                    ("object", spec.object_mode, spec.object_asset)):
                skip = mode == "none" or (role == "robot" and spec.robot_mount == "fixed_arm")
                steps.append({"role": role, "status": "skipped" if skip else "reused" if mode == "reuse" else "pending",
                              "asset_id": ref if mode == "reuse" and not skip else None})
            row = {"id": rid, "status": "generating", "spec": spec.model_dump(), "plan": plan,
                   "client_request_id": client_request_id, "request_hash": requested_hash,
                   "steps": steps, "project_id": "scene_" + rid[9:], "dataset_id": "scene_" + rid[9:],
                   "created": _now(), "error": None}
            self._save(row)
        self._schedule(rid, self._start_generations)
        return self.get(rid)

    def _validate_reused(self, spec):
        if spec.object_mode == "reuse":
            item = asset_info(self.lab, spec.object_asset, "物体")
            extent = np.diff(np.asarray(item["bounds_m"]), axis=0)[0]
            if (item["source"] != "lux3d" or item["kind"] != "task_object" or item["role"] != "dynamic"
                    or not 0 < item["height_m"] <= .075 or np.max(extent[:2]) > .085):
                raise ValueError("所选物体不符合动态小物体规格（高≤7.5 cm，水平尺寸≤8.5 cm）。")
        if spec.robot_mount == "four_wheel" and spec.robot_mode == "reuse":
            item = asset_info(self.lab, spec.chassis_asset, "底盘")
            if item["source"] != "lux3d" or item["kind"] != "robot_visual":
                raise ValueError("所选底盘需要Aholo机器人外观模型。")
        if spec.environment_mode == "reuse":
            item = asset_info(self.lab, spec.environment_asset, "环境道具")
            if item["kind"] != "static_prop" or item["role"] not in ("static", "visual_only") or item["source"] != "lux3d":
                raise ValueError("所选环境需要Aholo静态道具模型。")
            from .scene_builder import _prop_xml
            _prop_xml(SceneProp(name="preflight_environment", asset=spec.environment_asset, collision=True), self.lab)

    def _schedule(self, rid, callback):
        with self._lock:
            if rid in self._active:
                return
            self._active.add(rid)
        def work():
            try:
                with self._lock:
                    row = self.get(rid)
                    if self._cancelled(row):
                        return
                    if row["status"] not in ("succeeded", "failed", "blocked", "cancelled"):
                        callback(row)
            except Exception as exc:
                with self._lock:
                    row = self.get(rid)
                    if row["status"] != "cancelled":
                        row.update(status="failed", error=str(exc))
                        self._save(row)
            finally:
                with self._lock:
                    self._active.discard(rid)
        if self.background:
            threading.Thread(target=work, daemon=True).start()
        else:
            work()

    def _start_generations(self, row):
        spec = ScenarioSpec.model_validate(row["spec"])
        for step in row["steps"]:
            if self._cancelled(row):
                return
            if step["status"] != "pending":
                continue
            role = step["role"]
            if not step.get("session_id"):
                session = self.gen["sessions"].create()
                step["session_id"] = session.session_id
                self._save(row)
            session = self.gen["sessions"].get(step["session_id"])
            if not step.get("patched"):
                appearance = {"environment": spec.environment, "robot": spec.robot, "object": spec.object}[role]
                instruction = {
                    "environment": "A compact background prop assembly for a miniature robot workspace. Shelves and storage crates only, flat base, no room shell, no floor, no ceiling, no robot, no conveyor, no task object. Visual requirements: ",
                    "robot": "A compact four-wheel robot chassis only, flat upper mounting deck, four black wheels, no arm, no gripper. Appearance requirements: ",
                    "object": "A compact graspable single solid object, no surrounding scene. Appearance requirements: ",
                }[role]
                data = {"expected_revision": session.revision, "name": f"scene-{role}", "appearance": instruction + appearance,
                        "user_words": appearance, "kind": {"environment": "static_prop", "robot": "robot_visual", "object": "task_object"}[role],
                        "role": {"environment": "static", "robot": "visual_only", "object": "dynamic"}[role],
                        "height_m": {"environment": .30, "robot": .065, "object": .06}[role],
                        "width_m": {"environment": .50, "robot": .18, "object": .048}[role],
                        "depth_m": {"environment": .22, "robot": .20, "object": .062}[role],
                        "mass_kg": {"environment": 1., "robot": .9, "object": .035}[role]}
                if role != "environment":
                    data["radius_m"] = .024 if role == "object" else .09
                apply_user_patch(session, SpecPatchBody(**data))
                session.messages.append(ChatMessage(role="user", content=appearance, ts=_now()))
                self.gen["sessions"].save(session, session.revision)
                step["patched"] = True
                self._save(row)
            # Same request ID across refresh/restart, including a response lost after submit.
            if self._cancelled(row):
                return
            result = self.gen["orchestrator"].confirm(step["session_id"],
                client_request_id=f"{row['id']}:{role}", provider="lux3d", background=True)
            job = result["job"]
            step.update(job_id=job["id"], status=job["state"], error=job.get("error"), asset_id=job.get("asset_id"))
            self._save(row)

    def poll(self, rid):
        self._schedule(rid, self._advance)
        return self.get(rid)

    def _advance(self, row):
        if row["status"] == "simulating":
            job = self.jobs.get(row["job_id"])
            if not job:
                row.update(status="blocked", error="采集作业记录缺失，未自动重新采集。")
            elif job["status"] == "succeeded":
                row.update(status="succeeded", error=None)
            elif job["status"] in ("failed", "interrupted", "cancelled"):
                row.update(status="cancelled" if job["status"] == "cancelled" else "failed", error=job.get("error") or "采集已停止，请查看作业日志。")
            self._save(row)
            return
        self._start_generations(row)
        if self._cancelled(row):
            return
        waiting, blockers = False, []
        for step in row["steps"]:
            if not step.get("job_id"):
                continue
            job = self.gen["objects"].poll(step["job_id"], background=True)
            step.update(status=job["state"], asset_id=job.get("asset_id"), error=job.get("error"), node=job.get("node"))
            if job["state"] in ("failed", "retryable_error", "submit_unknown", "unconfigured"):
                blockers.append(step)
            elif job["state"] != "succeeded":
                waiting = True
        if blockers:
            row.update(status="blocked", error="模型生成尚未完成：" + "；".join(f"{s['role']}: {s.get('error') or s['status']}" for s in blockers))
        self._save(row)
        if blockers or waiting:
            return
        row.update(status="assembling", error=None)
        self._save(row)
        self._assemble(row)

    def _assemble(self, row):
        spec = ScenarioSpec.model_validate(row["spec"])
        refs = {s["role"]: s.get("asset_id") for s in row["steps"]}
        cfg = make_config(self.configs.load(), self.lab, spec.seed, refs["object"], refs["robot"] or CHASSIS)
        cfg.project_id = row["project_id"]
        if spec.robot_mount == "fixed_arm":
            cfg.robots[0].links = [l for l in cfg.robots[0].links if l.name != "mobile_chassis"]
            cfg.robots[0].joints = [j for j in cfg.robots[0].joints if j.name != "chassis_mount"]
        cfg.task.duck_conveyor.continuous = spec.continuous
        cfg.task.duck_conveyor.random_pose = spec.random_pose
        cfg.task.duck_conveyor.belt_speed_m_s = spec.belt_speed_m_s
        cfg.task.instruction = spec.operation
        if refs["environment"]:
            item = asset_info(self.lab, refs["environment"], "环境道具")
            if item["kind"] != "static_prop" or item["role"] not in ("static", "visual_only"):
                raise ValueError("环境资产不是静态道具。")
            bounds = np.asarray(item["bounds_m"], dtype=float)
            ext = bounds[1] - bounds[0]
            scale = min(1., .65 / ext[0], .28 / ext[1], .35 / ext[2])
            # Keep the complete estimated box behind the robot's working volume.
            pos = [-float(bounds[:, 0].mean())*scale, .28-float(bounds[0, 1])*scale, -.061-float(bounds[0, 2])*scale]
            cfg.scene.props = [SceneProp(name="scenario_environment", asset=refs["environment"], position=pos, uniform_scale=scale, collision=True)]
        cfg.provenance.update({"scenario.id": row["id"], "scenario.parser": "structured_rules_v1",
                               "scenario.robot_mount": spec.robot_mount,
                               "scenario.environment": refs["environment"] or "none"})
        from .engine import PlatformEnv
        probe = cfg.model_copy(deep=True)
        probe.sensors = []
        env = PlatformEnv(probe, self.lab)
        try:
            for _ in range(8):
                env.step([0, 0, 0, 1])
            if not np.all(np.isfinite(env.data.qpos)):
                raise ValueError("场景短步校验出现非有限状态。")
        finally:
            env.close()
        if self._cancelled(row):
            return
        # Reconcile durable job intent after restart before ever creating a new job.
        existing = next((j for j in self.jobs.list() if j.get("params", {}).get("provenance", {}).get("scenario_id") == row["id"]), None)
        if existing is None:
            saved = self.configs.save(cfg)
            self._save(row)
            if self._cancelled(row):
                return
            existing = self.jobs.create("collect", {"project_id": saved.project_id, "config_version": saved.version,
                "dataset_id": row["dataset_id"], "realtime": True,
                "provenance": {"scenario_id": row["id"], "scenario_spec": spec.model_dump(), "assets": refs, "parser": "structured_rules_v1"},
                "plan": [{"asset": cfg.task.asset, "seed": spec.seed+i} for i in range(spec.episodes)]})
        row.update(job_id=existing["id"], status="simulating", error=None)
        self._save(row)

    def stop(self, rid):
        current = self.get(rid)
        if current["status"] == "succeeded":
            return current
        # A durable cancellation flag is visible to the worker immediately,
        # even while it owns the mutation lock during expensive mesh validation.
        (self.dir / f"{rid}.cancel").write_text(_now(), encoding="ascii")
        with self._lock:
            row = self.get(rid)
            if row["status"] == "succeeded":
                return row
            # A process may have stopped after persisting the collect job but
            # before saving its ID on the scenario. Reconcile that window too.
            related = [j for j in self.jobs.list() if j["id"] == row.get("job_id")
                       or j.get("params", {}).get("provenance", {}).get("scenario_id") == rid]
            for job in related:
                if job["status"] in ("running", "queued"):
                    self.jobs.stop(job["id"])
                row["job_id"] = job["id"]
            row.update(status="cancelled", error="已停止后续组装和采集；已提交的在线生成任务不会撤回，资产仍保留。")
            self._save(row)
            return row

    def retry(self, rid):
        with self._lock:
            row = self.get(rid)
            if row["status"] not in ("failed", "blocked") or row.get("job_id"):
                raise ValueError("仅可恢复尚未进入仿真的资产处理失败。")
            retry = []
            for step in row["steps"]:
                if not step.get("job_id"):
                    continue
                job = self.gen["objects"].get(step["job_id"])
                if job["state"] in ("submit_unknown", "unconfigured") or not job.get("remote_id"):
                    raise ValueError("提交结果未知，不能重复收费提交；请在完整生成任务中恢复远端任务ID。")
                if job["state"] in ("retryable_error", "failed"):
                    if job["node"] not in ("remote", "download", "import", "validate"):
                        raise ValueError("无法恢复该生成步骤；未创建新任务。")
                    retry.append((job["id"], job["node"]))
            for jid, node in retry:
                self.gen["objects"].retry_node(jid, node, background=True)
            row.update(status="generating", error=None)
            self._save(row)
        return self.poll(rid)


class ScenarioBody(BaseModel):
    spec: ScenarioSpec
    client_request_id: str = Field(min_length=1, max_length=100)


def register_scenarios(app, configs, jobs, lab_root, generation):
    scenarios = ScenarioStore(lab_root, configs, jobs, generation, background=True)

    @app.get("/api/v1/scenarios/capabilities")
    def capabilities():
        groups = {"objects": [], "chassis": [], "environments": []}
        for path in sorted((Path(lab_root) / "assets/imported").glob("*/manifest.json")):
            try:
                ref = "imported/" + path.parent.name
                item = asset_info(Path(lab_root), ref, "塑料小鸭 · Aholo" if ref == DUCK else "四轮底盘 · Aholo" if ref == CHASSIS else path.parent.name)
                if item["source"] != "lux3d":
                    continue
                ext = np.diff(np.asarray(item["bounds_m"]), axis=0)[0]
                if item["kind"] == "task_object" and item["role"] == "dynamic" and 0 < item["height_m"] <= .075 and np.max(ext[:2]) <= .085:
                    groups["objects"].append(item)
                elif item["kind"] == "robot_visual":
                    groups["chassis"].append(item)
                elif item["kind"] == "static_prop" and item["role"] in ("static", "visual_only"):
                    manifest = json.loads(path.read_text(encoding="utf-8"))
                    if manifest.get("calibrated"):
                        groups["environments"].append(item)
            except (ValueError, KeyError, TypeError):
                continue
        groups["objects"].sort(key=lambda a: a["id"] != DUCK)
        groups["chassis"].sort(key=lambda a: a["id"] != CHASSIS)
        return {"configured": generation["objects"].provider.configured(), "defaults": ScenarioSpec().model_dump(),
                "notes": plan_spec(ScenarioSpec())["warnings"], **groups}

    @app.post("/api/v1/scenarios/plan")
    def plan(body: ScenarioSpec):
        return plan_spec(body)

    def invoke(fn, *args):
        try:
            return fn(*args)
        except KeyError:
            raise HTTPException(404, "场景或资产记录不存在。")
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(409, str(exc))

    @app.post("/api/v1/scenarios")
    def create(body: ScenarioBody):
        return invoke(scenarios.create, body.spec, body.client_request_id)

    @app.get("/api/v1/scenarios")
    def history():
        return {"scenarios": scenarios.list()}

    @app.get("/api/v1/scenarios/{rid}")
    def read(rid: str):
        return invoke(scenarios.get, rid)

    @app.post("/api/v1/scenarios/{rid}/poll")
    def poll(rid: str):
        return invoke(scenarios.poll, rid)

    @app.post("/api/v1/scenarios/{rid}/stop")
    def stop(rid: str):
        return invoke(scenarios.stop, rid)

    @app.post("/api/v1/scenarios/{rid}/retry")
    def retry(rid: str):
        return invoke(scenarios.retry, rid)

    return scenarios
