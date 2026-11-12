"""Capability registry: the single source of truth for what the platform supports.

Injected into the planner prompt; the LLM cannot flip a capability to available.
Provider availability is read from backend environment only (never the browser).
"""
from __future__ import annotations

import os

from .schema import AssetKind, SceneMode

REGISTRY_VERSION = "1.0"

# kind -> (online provider, local import, notes)
ASSET_KIND_SUPPORT = {
    AssetKind.task_object: {"online": "lux3d", "local_import": True,
                            "note": "单体可操作物体；生成不保证精确尺寸/拓扑，需标定后才能绑定物理"},
    AssetKind.static_prop: {"online": "lux3d", "local_import": True,
                            "note": "静态道具；碰撞用 box/cylinder/plane 代理，不做凸分解"},
    AssetKind.robot_visual: {"online": "lux3d", "local_import": True,
                             "note": "仅机器人外观模型；不生成可动物理机器人/URDF/关节"},
    AssetKind.environment_splat: {"online": "aholo_world", "local_import": True,
                                  "note": "高斯背景；仅 hybrid 固定前视渲染，全景预览≠三维编辑"},
    AssetKind.environment_mesh: {"online": None, "local_import": True,
                                 "note": "预留：在线网格环境生成未支持；仅本地导入标定网格"},
}

PROVIDERS = {
    "lux3d": {
        "label": "Lux3D 单体生成（Aholo 中国）",
        "env_keys": ["AHOLO_API_KEY", "LUX3D_CN_API_KEY", "LUX3D_API_KEY"],
        "supported_params": ["prompt", "version", "style", "face_count", "output_format", "enable_pbr"],
        "unsupported_params": ["seed", "negative_prompt", "layout", "colliders", "urdf",
                               "mass", "friction", "joint_limits"],
        "outputs": ["glb"],
        "note": "negative_prompt 不支持，合并进文本；质量/摩擦/关节不传给 3D API",
    },
    "aholo_world": {
        "label": "Aholo World 场景生成/重建",
        "env_keys": ["AHOLO_API_KEY"],
        "supported_params": ["prompt", "ref_image", "scene", "task_quality"],
        "unsupported_params": ["seed", "layout", "colliders", "urdf", "mesh_physics"],
        "outputs": ["gaussian_ply", "panorama"],
        "note": "输出高斯/全景，非物理网格；需 hybrid + 显式碰撞代理",
    },
    "local_import": {
        "label": "本地导入（GLB/PLY）",
        "env_keys": [],
        "supported_params": ["file", "up_axis", "unit_scale", "calibrated"],
        "unsupported_params": [],
        "outputs": ["obj", "gaussian_ply"],
        "note": "始终可用；未标定资产仅预览，不绑定物理",
    },
}

PHYSICS_LIMITS = {
    "collision_proxies": ["box", "cylinder", "plane"],
    "convex_decomposition": False,
    "urdf_generation": False,
    "joint_generation": False,
    "part_split": "future_research",
    "auto_rig": "future_research",
    "sim2real": False,
    "inertia": "proxy-based estimate, labeled as such; never measured",
    "solid_cylinder_for_insertion": False,
}

SCENE_MODES = {
    SceneMode.background_scene: {
        "requires": ["world gaussian asset", "sensor_renderer=hybrid", "fixed front camera",
                     "default table or explicit static colliders"],
        "note": "World 外观 + 显式代理；固定前视 hybrid 限制保留",
    },
    SceneMode.composed_scene: {
        "requires": ["parameterized floor/workbench", "static props", "manipulable objects"],
        "note": "无需 World/标定即可组装；物体分别装配",
    },
}

RENDER_LIMITS = [
    "hybrid 仅 static_background_overlay 固定前视；无腕部/移动相机",
    "高斯全景是预览图，不是可编辑三维场景",
    "CPU 背景渲染约 10s+；前景为真实 MuJoCo geom",
]


def _configured(env_keys: list[str]) -> bool:
    return any(bool(os.environ.get(k)) for k in env_keys)


def capability_registry(aux_status: dict | None = None) -> dict:
    providers = {}
    for pid, p in PROVIDERS.items():
        providers[pid] = {**p, "configured": _configured(p["env_keys"]) if p["env_keys"] else True}
    return {
        "registry_version": REGISTRY_VERSION,
        "providers": providers,
        "asset_kinds": {k.value: v for k, v in ASSET_KIND_SUPPORT.items()},
        "physics_limits": PHYSICS_LIMITS,
        "scene_modes": {m.value: v for m, v in SCENE_MODES.items()},
        "render_limits": RENDER_LIMITS,
        "aux_model": aux_status or {"state": "waiting_for_configuration"},
        "rules": [
            "LLM 只理解与建议；程序校验、编排、落地",
            "模型估计的质量/摩擦/尺度/关节轴/可抓取性不是已测量事实",
            "未知尺寸→未标定视觉资产，处理前不绑定物理",
            "生成不承诺满足精确尺寸/拓扑/关节",
            "远程成功≠已导入≠可训练",
        ],
    }