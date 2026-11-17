"""Environment MJCF builder: default tabletop OR imported scene visual + static colliders.

Separated from task objects (tray/cup) so a generated background never replaces the
dynamic task object. Static colliders have no freejoint (no mass dynamics).
"""
from __future__ import annotations

import json
from pathlib import Path

from .schema import ProjectConfig
from .transforms import quat_xyzw_to_wxyz, transform_matrix, mat_to_pos_quat_xyzw
from .scene_assets import scene_asset_dir, load_scene, contained_file, valid_id

LIGHTS = """
        <light pos="-.4 -.3 .85" dir=".25 .25 -1" diffuse=".69 .65 .59" castshadow="true"/>
        <light pos=".5 .4 .6" dir="-.6 -.3 -1" diffuse=".44 .48 .56" castshadow="false"/>"""
FLOOR = '<geom type="plane" size="2 2 .01" pos="0 0 -.061" rgba=".19 .23 .25 1"/>'
DEFAULT_TABLE = '<geom name="table" type="box" pos="0 0 -.028" size=".32 .215 .028" rgba=".29 .37 .39 1"/>'


def parse_asset_ref(ref: str):
    """Parse 'builtin/<name>' | 'imported/<slug>' | 'scenes/<asset_id>/<rev>'."""
    if not isinstance(ref, str) or "/" not in ref:
        raise ValueError(f"asset ref must be builtin/<name>, imported/<slug> or scenes/<id>/<rev>: {ref!r}")
    head, _, tail = ref.partition("/")
    if head == "builtin":
        return ("builtin", valid_id(tail))
    if head == "imported":
        return ("imported", valid_id(tail))
    if head == "scenes":
        parts = tail.split("/")
        if len(parts) != 2:
            raise ValueError("scene ref must be scenes/<asset_id>/<revision>")
        return ("scene", valid_id(parts[0]), int(parts[1]))
    raise ValueError(f"unknown asset ref namespace: {head}")


def resolve_asset_node(lab_root: Path, asset_ref: str) -> dict:
    """Resolve an asset_node reference to its visual mesh, metadata, and optional collision proxy."""
    lab_root = Path(lab_root)
    parsed = parse_asset_ref(asset_ref)
    kind = parsed[0]
    if kind == "imported":
        slug = parsed[1]
        d = (lab_root / "assets" / "imported" / slug).resolve()
        imp_root = (lab_root / "assets" / "imported").resolve()
        if not d.is_relative_to(imp_root):
            raise ValueError(f"asset ref {asset_ref!r} escapes imported asset root")
        mf_path = d / "manifest.json"
        if not mf_path.is_file():
            raise ValueError(f"imported asset {asset_ref!r} does not exist")
        mf = json.loads(mf_path.read_text(encoding="utf-8"))
        obj_rel = mf.get("obj", "visual.obj")
        obj_path = contained_file(d, obj_rel)
        if not obj_path.is_file():
            raise ValueError(f"imported asset {asset_ref!r} visual mesh missing: {obj_rel}")
        # Legacy cup manifests (import_asset.py) lack 'calibrated' flag but have height_m & radius_m.
        calibrated = bool(mf["calibrated"]) if "calibrated" in mf else bool(mf.get("height_m") and mf.get("radius_m"))
        proxy = mf.get("collision_proxy")
        if isinstance(proxy, str) and mf.get("radius_m") and mf.get("height_m"):
            r, h = float(mf["radius_m"]), float(mf["height_m"])
            proxy = {"shape": "cylinder", "size_m": [2 * r, 2 * r, h], "position": [0.0, 0.0, h / 2.0]}
        offset_z = 0.0
        return {
            "ref": asset_ref,
            "kind": mf.get("kind", "task_object"),
            "calibrated": calibrated,
            "mesh_bytes": obj_path.read_bytes(),
            "color_rgb": mf.get("color_rgb"),
            "collision_proxy": proxy if isinstance(proxy, dict) else None,
            "compound_proxy": mf.get("compound_proxy"),
            "concave_detected": bool(mf.get("concave_detected")),
            "bounds_m": mf.get("bounds_m"),
            "mass_kg": mf.get("mass_kg"),
            "friction": mf.get("friction"),
            "offset_z": offset_z,
            "manifest": mf,
        }
    if kind == "builtin":
        name = parsed[1]
        import sim_env
        if name not in sim_env.BUILTINS:
            raise ValueError(f"unknown builtin asset {asset_ref!r}")
        r, h, mass, color_str, mesh_bytes, _, _ = sim_env._load_asset(name)
        shifted_lines = []
        for line in mesh_bytes.decode("utf-8").splitlines():
            if line.startswith("v "):
                _, xs, ys, zs = line.split()
                shifted_lines.append(f"v {xs} {ys} {float(zs) + h / 2.0:.9g}")
            else:
                shifted_lines.append(line)
        shifted_bytes = ("\n".join(shifted_lines) + "\n").encode("utf-8")
        return {
            "ref": asset_ref,
            "kind": "task_object",
            "calibrated": True,
            "mesh_bytes": shifted_bytes,
            "color_rgb": [float(x) for x in color_str.split()],
            "collision_proxy": {"shape": "cylinder", "size_m": [2 * r, 2 * r, h], "position": [0.0, 0.0, h / 2.0]},
            "bounds_m": [[-r, -r, 0.0], [r, r, h]],
            "offset_z": 0.0,
            "manifest": None,
        }
    _, aid, rev = parsed
    d, mf = load_scene(lab_root, aid, rev, physics=False)
    if mf.get("ply_kind") == "gaussian" or not mf.get("visual_asset"):
        raise ValueError(f"scene {asset_ref!r} is gaussian/preview-only and cannot be placed as an asset_node; use background_ref")
    d, mf = load_scene(lab_root, aid, rev, physics=True)
    obj_path = contained_file(d, mf["visual_asset"])
    return {
        "ref": asset_ref,
        "kind": "scene_mesh",
        "calibrated": bool(mf.get("calibrated")),
        "mesh_bytes": obj_path.read_bytes(),
        "color_rgb": None,
        "collision_proxy": None,
        "bounds_m": mf.get("bbox"),
        "offset_z": 0.0,
        "manifest": mf,
    }


def _fmt(v):
    return " ".join(f"{x:.6g}" for x in v)


def _collider_xml(c, *, visual_present=False):
    q = _fmt(quat_xyzw_to_wxyz(c.quaternion_xyzw))
    rgba = ' rgba=".5 .6 .65 0"' if visual_present else ''
    common = (f'name="{c.name}" pos="{_fmt(c.position)}" quat="{q}"{rgba} '
              f'friction="{_fmt(c.friction)}" contype="{c.contype}" conaffinity="{c.conaffinity}"')
    fx, fy, fz = c.full_size_m
    if c.shape == "box":
        return f'<geom type="box" size="{fx/2:.6g} {fy/2:.6g} {fz/2:.6g}" {common}/>'
    if c.shape == "cylinder":
        return f'<geom type="cylinder" size="{fx/2:.6g} {fz/2:.6g}" {common}/>'
    return f'<geom type="plane" size="{fx/2:.6g} {fy/2:.6g} .01" {common}/>'


def _default_table_xml(work_surface_z: float = 0.0) -> str:
    cz = float(work_surface_z) - 0.028
    return f'<geom name="table" type="box" pos="0 0 {cz:.6g}" size=".32 .215 .028" rgba=".29 .37 .39 1"/>'


def _dynamic_node_geoms_xml(node, resolved: dict) -> str:
    s = float(node.uniform_scale)
    mf = resolved.get("manifest") or {}
    mass = float(node.mass_kg if node.mass_kg is not None else (resolved.get("mass_kg") or mf.get("mass_kg") or 0.12))
    fric = node.friction if node.friction is not None else (resolved.get("friction") or mf.get("friction") or [1.2, 0.005, 0.0001])
    fstr = _fmt(fric)
    mode = getattr(node, "collision_mode", "auto")
    comp = resolved.get("compound_proxy") or mf.get("compound_proxy")
    use_compound = isinstance(comp, dict) and comp.get("boxes") and (
        mode == "compound" or (mode == "auto" and resolved.get("concave_detected"))
    )
    if use_compound:
        boxes = comp["boxes"]
        pmass = max(1e-4, mass / len(boxes))
        out = []
        for i, b in enumerate(boxes):
            bp = [float(v) * s for v in b["position"]]
            hs = [max(5e-4, float(v) * s / 2.0) for v in b["size_m"]]
            q = _fmt(quat_xyzw_to_wxyz(b.get("quaternion_xyzw", [0.0, 0.0, 0.0, 1.0])))
            out.append(
                f'<geom name="node_{node.node_id}_col_{i}" type="box" pos="{_fmt(bp)}" quat="{q}" '
                f'size="{_fmt(hs)}" mass="{pmass:.6g}" friction="{fstr}" group="3" rgba=".2 .5 .6 0"/>'
            )
        return "".join(out)
    proxy = resolved.get("collision_proxy")
    if isinstance(proxy, dict):
        shape = proxy.get("shape", "box")
        sz = [float(v) * s for v in proxy["size_m"]]
        bp = [float(v) * s for v in proxy.get("position", [0.0, 0.0, 0.0])]
        q = _fmt(quat_xyzw_to_wxyz(proxy.get("quaternion_xyzw", [0.0, 0.0, 0.0, 1.0])))
        if shape == "cylinder":
            return (
                f'<geom name="node_{node.node_id}_col" type="cylinder" pos="{_fmt(bp)}" quat="{q}" '
                f'size="{sz[0]/2:.6g} {sz[2]/2:.6g}" mass="{mass:.6g}" friction="{fstr}" group="3" rgba=".2 .5 .6 0"/>'
            )
        return (
            f'<geom name="node_{node.node_id}_col" type="box" pos="{_fmt(bp)}" quat="{q}" '
            f'size="{sz[0]/2:.6g} {sz[1]/2:.6g} {sz[2]/2:.6g}" mass="{mass:.6g}" friction="{fstr}" group="3" rgba=".2 .5 .6 0"/>'
        )
    raise ValueError(f"dynamic asset node {node.node_id}: missing calibrated collision proxy")


def environment_xml(config: ProjectConfig, lab_root: Path):
    sc = config.scene
    parts = [LIGHTS]
    assets = {}
    mesh_scales = {}
    node_infos = []
    node_collider_refs = {n.collider_ref for n in sc.asset_nodes if n.collider_ref}
    info = {"environment": None, "colliders": [c.name for c in sc.colliders if c.enabled],
            "default_table": False, "asset_nodes": node_infos, "mesh_scales": mesh_scales}
    table_geom = _default_table_xml(sc.work_surface_z)
    if sc.environment:
        d, mf = load_scene(lab_root, sc.environment.asset_id, sc.environment.revision, physics=False)
        gaussian = mf.get("ply_kind") == "gaussian"
        if gaussian:
            if sc.sensor_renderer != "hybrid":
                raise ValueError("Gaussian scene requires sensor_renderer=hybrid (not a MJCF mesh)")
            import hashlib
            for name, expected in mf.get("files", {}).items():
                f = contained_file(d, name)
                if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest() != expected:
                    raise ValueError(f"Scene asset missing or checksum mismatch: {name}")
            if not (sc.use_default_table or any(c.enabled for c in sc.colliders)):
                raise ValueError("Gaussian scene needs a default table or static colliders for physics")
            info["environment"] = {"asset_id": sc.environment.asset_id, "revision": sc.environment.revision,
                                   "calibrated": mf.get("calibrated", False), "renderer": "hybrid-gaussian"}
            parts.append(FLOOR)
            if sc.use_default_table:
                parts.append(table_geom)
                info["default_table"] = True
        else:
            d, mf = load_scene(lab_root, sc.environment.asset_id, sc.environment.revision, physics=True)
            visual = contained_file(d, mf["visual_asset"])
            assets["scene_env.obj"] = visual.read_bytes()
            T = transform_matrix(sc.environment.position, sc.environment.quaternion_xyzw,
                                 sc.environment.uniform_scale)
            pos, q, s = mat_to_pos_quat_xyzw(T)
            parts.append(f'<geom class="visual" type="mesh" mesh="scene_env" pos="{_fmt(pos)}" '
                         f'quat="{_fmt(quat_xyzw_to_wxyz(q))}" rgba=".62 .66 .70 1"/>')
            info["environment"] = {"asset_id": sc.environment.asset_id, "revision": sc.environment.revision,
                                   "calibrated": mf.get("calibrated", False)}
            parts.append(FLOOR)
            if sc.use_default_table:
                parts.append(table_geom)
                info["default_table"] = True
    else:
        parts.append(FLOOR)
        if sc.use_default_table:
            parts.append(table_geom)
            info["default_table"] = True

    for node in sc.asset_nodes:
        resolved = resolve_asset_node(lab_root, node.asset_ref)
        if node.role in ("static", "dynamic") and not resolved["calibrated"]:
            raise ValueError(
                f"asset node {node.node_id}: {node.asset_ref} is uncalibrated; only role='visual_only' is allowed"
            )
        mesh_file = f"node_{node.node_id}.obj"
        assets[mesh_file] = resolved["mesh_bytes"]
        mesh_scales[mesh_file] = float(node.uniform_scale)
        rgb = resolved.get("color_rgb")
        rgba = f"{_fmt(rgb)} 1" if isinstance(rgb, list) and len(rgb) == 3 else ".68 .72 .76 1"
        oz = float(resolved.get("offset_z") or 0.0) * float(node.uniform_scale)
        if node.role == "dynamic":
            dyn_geoms = _dynamic_node_geoms_xml(node, resolved)
            parts.append(
                f'<body name="node_{node.node_id}" pos="{_fmt(node.position)}" '
                f'quat="{_fmt(quat_xyzw_to_wxyz(node.quaternion_xyzw))}">'
                f'<freejoint name="node_{node.node_id}_free"/>'
                f'{dyn_geoms}'
                f'<geom name="node_{node.node_id}_visual" class="visual" type="mesh" '
                f'mesh="node_{node.node_id}" pos="0 0 {oz:.6g}" rgba="{rgba}"/></body>'
            )
        else:
            parts.append(
                f'<body name="node_{node.node_id}" pos="{_fmt(node.position)}" '
                f'quat="{_fmt(quat_xyzw_to_wxyz(node.quaternion_xyzw))}">'
                f'<geom name="node_{node.node_id}_visual" class="visual" type="mesh" '
                f'mesh="node_{node.node_id}" pos="0 0 {oz:.6g}" rgba="{rgba}"/></body>'
            )
        node_infos.append({
            "node_id": node.node_id,
            "asset_ref": node.asset_ref,
            "role": node.role,
            "position": list(node.position),
            "quaternion_xyzw": list(node.quaternion_xyzw),
            "uniform_scale": float(node.uniform_scale),
            "collider_ref": node.collider_ref,
        })

    node_collider_prefixes = tuple(
        f"{ref[:-7]}_part_" if ref.endswith("_part_0") else f"{ref}_part_"
        for ref in node_collider_refs
    )
    for c in sc.colliders:
        if c.enabled:
            has_vis = (
                (sc.environment is not None)
                or (c.name in node_collider_refs)
                or c.name.startswith(node_collider_prefixes)
            )
            parts.append(_collider_xml(c, visual_present=has_vis))
    return "".join(parts), assets, info
