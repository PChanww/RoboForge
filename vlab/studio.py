"""Local task studio: bounded text parsing, editable plans, real collection.

This is a deterministic parser, not a language-model response. Every execution
uses a separate project snapshot; the default project is never overwritten.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re
import uuid
from typing import Literal

from fastapi import HTTPException
from fastapi.responses import Response
import numpy as np
from pydantic import BaseModel, Field

from .engine import PlatformEnv
from .schema import SceneCollider
from .scene_assets import valid_id

ASSETS = {"block": "方块", "cylinder": "圆柱", "standard": "标准杯", "tall": "高杯", "wide": "宽口杯"}
COLORS = {"red": [.83, .25, .18], "blue": [.18, .44, .83], "green": [.20, .63, .43],
          "orange": [.93, .52, .19], "violet": [.55, .36, .77]}
SCENES = {"workbench": "简洁工作台", "sorting": "分拣工作台", "lab": "实验工作台"}


class StudioSpec(BaseModel):
    prompt: str = Field(default="", max_length=2000)
    scene: Literal["workbench", "sorting", "lab"] = "workbench"
    task: Literal["pick_place", "push"] = "pick_place"
    asset: Literal["block", "cylinder", "standard", "tall", "wide"] = "block"
    color: Literal["red", "blue", "green", "orange", "violet"] = "red"
    goal: Literal["left", "center", "right"] = "right"
    episodes: int = Field(default=3, ge=1, le=20)
    seed: int = Field(default=0, ge=0, le=999999)
    randomize: bool = True
    unsupported: list[str] = Field(default_factory=list, max_length=20)


class PromptBody(BaseModel):
    prompt: str = Field(min_length=2, max_length=2000)


def _number(text):
    if text.isdigit():
        return int(text)
    digits = dict(zip("零一二三四五六七八九", range(10)))
    if text == "两":
        return 2
    if "十" in text:
        a, b = text.split("十", 1)
        return (digits.get(a, 1) * 10) + digits.get(b, 0)
    return digits.get(text, 0)


def parse_prompt(prompt):
    text = prompt.lower().strip()
    spec = StudioSpec(prompt=prompt)
    defaults, unsupported = [], []
    if any(w in text for w in ("方块", "积木", "立方", "cube", "block")):
        spec.asset = "block"
    elif any(w in text for w in ("圆柱", "柱体", "cylinder")):
        spec.asset = "cylinder"
    elif "高杯" in text:
        spec.asset = "tall"
    elif "宽口" in text:
        spec.asset = "wide"
    elif any(w in text for w in ("杯", "cup")):
        spec.asset = "standard"
    else:
        unsupported.append("请指定操作物体：方块、圆柱或杯子。其他物体尚未接入本地执行器。")
    if any(w in text for w in ("推", "push")):
        spec.task = "push"
    elif not any(w in text for w in ("抓", "搬", "放", "移动", "pick", "place", "move")):
        unsupported.append("请说明要搬运放置，还是在桌面推移物体。")
    for words, key in ((('红', 'red'), 'red'), (('蓝', 'blue'), 'blue'),
                       (('绿', 'green'), 'green'), (('橙', 'orange'), 'orange'), (('紫', 'purple'), 'violet')):
        if any(w in text for w in words):
            spec.color = key
            break
    else:
        defaults.append("未指定颜色，使用红色；可在方案中修改。")
    if "分拣" in text or "sorting" in text:
        spec.scene = "sorting"
    elif any(w in text for w in ("实验", "lab")):
        spec.scene = "lab"
    elif not any(w in text for w in ("桌", "台", "table", "workbench")):
        defaults.append("未指定场景，使用简洁工作台。")
    goal = re.search(r"(?:到|入|至|to)\s*[^，。；,\n]{0,10}?(左|右|中间|中央|中心|left|right|center)", text)
    if goal:
        spec.goal = {"左": "left", "右": "right", "left": "left", "right": "right"}.get(goal[1], "center")
    else:
        defaults.append("未指定目标位置，使用右侧目标区。")
    count = re.search(r"(?:采集|收集|录制|记录|生成)\s*([\d零一二三四五六七八九十两]+)\s*(?:条|次|组|个)", text)
    if not count:
        count = re.search(r"([\d一二三四五六七八九十两]+)\s*(?:条示范|次采集|episodes?)", text)
    if count:
        n = _number(count[1])
        if 1 <= n <= 20:
            spec.episodes = n
        else:
            unsupported.append("单次采集数量支持 1～20 条，请调整数量。")
    else:
        defaults.append("未指定数量，采集 3 条示范。")
    seed = re.search(r"(?:seed|种子)\s*[:：=]?\s*(\d+)", text)
    if seed:
        if int(seed[1]) <= 999999:
            spec.seed = int(seed[1])
        else:
            unsupported.append("随机种子需要在 0～999999 之间。")
    spec.randomize = not any(w in text for w in ("固定摆位", "固定位置", "不要随机", "不随机"))
    for pattern, reason in (
        (r"倒水|倒入|开门|拧|旋拧|插入|插孔|堆叠|叠放|焊接|切割|双臂|行走|装配|绕障|避障", "这版执行器支持搬运放置和直线推移，暂不支持该操作。"),
        (r"厨房|客厅|户外|仓库|街道|工厂|机器人狗|四足", "这版场景支持工作台、分拣台和实验台；完整环境生成需要另外接入。"),
        (r"同时|多个物体|两[个种]物体|两个方块|三个方块|先.+再.+(?:杯|方块|圆柱)", "当前一次任务操作一个物体，多物体和复合任务尚未接入。"),
        (r"障碍物|加.{0,3}障碍|任意尺寸|厘米|毫米|\bcm\b|\bmm\b|摩擦|质量|重量", "自定义障碍、尺寸或物理参数尚未开放，请先使用当前标准规格。"),
    ):
        if re.search(pattern, text):
            unsupported.append(reason)
    if spec.task == "push" and spec.asset != "block":
        unsupported.append("桌面推移目前仅支持方块；圆柱和杯子可以使用搬运放置。")
    spec.unsupported = list(dict.fromkeys(unsupported))
    return {"spec": spec.model_dump(), "parser": "local_rules_v1", "ready": not unsupported,
            "defaults": defaults, "summary": describe(spec)}


def describe(spec):
    action = "桌面推移" if spec.task == "push" else "抓取与放置"
    goal = {"left": "左侧", "center": "中央", "right": "右侧"}[spec.goal]
    return f"在{SCENES[spec.scene]}中，对{ASSETS[spec.asset]}执行{action}，目标为{goal}，采集 {spec.episodes} 条示范。"


def build_config(spec, store, project_id="studio_preview"):
    if spec.unsupported:
        raise ValueError("请先修改未支持的要求，再生成方案。")
    if spec.task == "push" and spec.asset != "block":
        raise ValueError("推移任务目前仅支持方块。")
    cfg = store.load("default").model_copy(deep=True)
    cfg.project_id, cfg.version = project_id, 1
    cfg.scene.environment = None
    cfg.scene.seed = spec.seed
    cfg.scene.sensor_renderer, cfg.scene.use_default_table = "mujoco", True
    cfg.scene.colliders = []
    cfg.scene.table_color_rgb = {"workbench": [.29, .37, .39], "sorting": [.36, .42, .46], "lab": [.58, .62, .61]}[spec.scene]
    if spec.scene != "workbench":
        for i, x in enumerate((-.245, .245)):
            cfg.scene.colliders.append(SceneCollider(name=f"studio_fixture_{i}",
                full_size_m=[.055, .07, .026 if spec.scene == "sorting" else .045],
                position=[x, -.15, .013 if spec.scene == "sorting" else .0225]))
    task = cfg.task
    task.asset = spec.asset
    task.id = task.plugin = "tabletop_push" if spec.task == "push" else "cup_pick_place"
    task.object_color_rgb = COLORS[spec.color]
    task.time_limit_s = 10
    task.instruction = describe(spec)
    goal_x = {"left": -.015, "center": .055, "right": .135}[spec.goal]
    task.goal_xy = [goal_x, -.045 if spec.task == "push" else .065]
    task.goal_surface_m = 0 if spec.task == "push" else .009
    task.spawn_min, task.spawn_max = [-.145, -.065], [-.10, -.035]
    if spec.task == "push":
        task.spawn_min, task.spawn_max = [-.14, -.05], [-.12, -.04]
    if not spec.randomize:
        task.spawn_min = task.spawn_max = [-.13, -.045]
    for sensor in cfg.sensors:
        if sensor.id == "front":
            scale = 384 / sensor.resolution[1]
            sensor.resolution = [512, 384]
            sensor.intrinsics.fx *= scale
            sensor.intrinsics.fy *= scale
            sensor.intrinsics.cx, sensor.intrinsics.cy = 256, 192
    return cfg


@lru_cache(maxsize=2)
def _episode_arrays(path, mtime_ns):
    with np.load(path, allow_pickle=True) as arrays:
        return {key: arrays[key] for key in ("frame_time", "eef_pos", "gripper", "action", "jpeg_front", "jpeg_wrist")}


def register_studio(app, store, jobs, lab_root):
    @app.post("/api/v1/studio/plan")
    def plan(body: PromptBody):
        return parse_prompt(body.prompt)

    @app.post("/api/v1/studio/preview")
    def preview(spec: StudioSpec):
        try:
            cfg = build_config(spec, store)
            env = PlatformEnv(cfg, lab_root)
            try:
                obs = env.observation()
                if any(f is None for f in obs["images"].values()):
                    raise ValueError("相机首帧不可用")
                return {"images": {k: v["jpeg_b64"] for k, v in obs["images"].items()}, "state": obs["state"]}
            finally:
                env.close()
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @app.post("/api/v1/studio/execute")
    def execute(spec: StudioSpec):
        try:
            identity = uuid.uuid4().hex[:12]
            cfg = build_config(spec, store, "studio_" + identity)
            store.save(cfg)
            return jobs.create("collect", {"project_id": cfg.project_id, "dataset_id": "studio_" + identity,
                "realtime": True, "studio_spec": spec.model_dump(),
                "provenance": {"studio_spec": spec.model_dump(), "parser": "local_rules_v1"},
                "plan": [{"asset": spec.asset, "seed": spec.seed + i if spec.randomize else spec.seed}
                         for i in range(spec.episodes)]})
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    def episode_data(dataset, index):
        valid_id(dataset)
        if not dataset.startswith("collection__"):
            raise HTTPException(400, "only collection episodes support replay")
        root = Path(lab_root) / "datasets/collections" / valid_id(dataset[len("collection__"):])
        path = root / "episodes" / f"ep{index:03d}.npz"
        if index < 0 or not (root / "manifest.json").is_file() or not path.is_file():
            raise HTTPException(404, "completed episode not found")
        return _episode_arrays(str(path), path.stat().st_mtime_ns)

    @app.get("/api/v1/studio/datasets/{dataset}/episodes/{index}/trace")
    def trace(dataset: str, index: int):
        data = episode_data(dataset, index)
        return {key: data[key].tolist() for key in ("frame_time", "eef_pos", "gripper", "action")}

    @app.get("/api/v1/studio/datasets/{dataset}/episodes/{index}/frames/{frame}/{camera}")
    def frame(dataset: str, index: int, frame: int, camera: str):
        if camera not in ("front", "wrist"):
            raise HTTPException(400, "unknown camera")
        data = episode_data(dataset, index)
        if not 0 <= frame < len(data["frame_time"]):
            raise HTTPException(404, "frame not found")
        return Response(bytes(data["jpeg_" + camera][frame]), media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})
