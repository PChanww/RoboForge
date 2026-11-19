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
                            "note": "单体可操作物体；支持多动态物体共存（task_asset + asset_nodes role=dynamic）；需标定尺寸"},
    AssetKind.static_prop: {"online": "lux3d", "local_import": True,
                            "note": "静态道具；支持 box/cylinder/plane 及凹面体素复合盒分解（compound）保留容器内腔壁接触"},
    AssetKind.robot_visual: {"online": "lux3d", "local_import": True,
                             "note": "机械臂外观或部件模型；可用 part_split 拆件装配至运动学连杆并导出标准 URDF"},
    AssetKind.environment_splat: {"online": "aholo_world", "local_import": True,
                                  "note": "高斯背景；仅 hybrid 固定前视渲染，全景预览≠三维编辑"},
    AssetKind.environment_mesh: {"online": "lux3d", "local_import": True,
                                 "note": "三角网格工作台/环境；支持在线生成或本地导入，并自动生成复合凹面碰撞盒"},
}

PROVIDERS = {
    "lux3d": {
        "label": "Lux3D 单体生成（Aholo 中国）",
        "env_keys": ["AHOLO_API_KEY", "LUX3D_CN_API_KEY", "LUX3D_API_KEY"],
        "supported_params": ["prompt", "img", "generation_mode", "version", "style", "face_count", "output_format", "enable_pbr"],
        "unsupported_params": ["seed", "negative_prompt", "layout", "colliders", "urdf",
                               "mass", "friction", "joint_limits"],
        "outputs": ["glb"],
        "note": "支持文本/参考图生成 (text_to_3d / img_to_3d)；negative_prompt 不支持，质量/摩擦/关节不传给 3D API",
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
    "collision_proxies": ["box", "cylinder", "plane", "compound_boxes"],
    "convex_decomposition": "voxel_box_compound",
    "urdf_generation": "template_and_part_assembly",
    "urdf_export": True,
    "joint_generation": "part_assembly_with_kinematics",
    "part_split": "supported_local_and_lux3d_api",
    "auto_rig": "humanoid_only_not_for_robotics_urdf",
    "multi_dynamic_asset_nodes": True,
    "support_plane_z_parameterized": True,
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


def capability_registry(aux_status: dict | None = None, configured: dict[str, bool] | None = None) -> dict:
    providers = {}
    for pid, p in PROVIDERS.items():
        active = _configured(p["env_keys"]) if p["env_keys"] else True
        providers[pid] = {**p, "configured": (configured or {}).get(pid, active)}
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
