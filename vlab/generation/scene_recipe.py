"""Trainable scene recipes: layout + dependency graph + real physics validation + atomic apply.

A recipe is assembled separately from generation: parameterized floor/workbench and
static props become explicit collision proxies, the manipulable object becomes the
task asset, and a World background (optional) becomes the scene environment. Two modes:

- composed_scene: explicit proxies + parameterized table; works with NO World background
  and NO calibration (the table/colliders provide contact, the cup is a calibrated builtin
  or imported asset).
- background_scene: World appearance + explicit proxies. A gaussian background forces
  sensor_renderer=hybrid, which is restricted to the world-fixed front camera
  (static_background_overlay); it is not editable 3D Gaussian geometry.

Validation builds a real MuJoCo model and short-steps it; remote/mock success is never
treated as physics truth. Apply reuses the existing atomic scene commit (expected_project_version
-> 409 on conflict), so a failed validation leaves the current project untouched.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import threading
from pathlib import Path

from ..schema import ProjectConfig, SceneAssetNode, SceneCollider, SceneInstance
from ..scene_assets import load_scene, valid_id
from ..scene_builder import parse_asset_ref, resolve_asset_node
from ..transforms import compose, mat_to_pos_quat_xyzw, transform_matrix
from .schema import AssetKind, AssetNode, AssetRole, SceneMode, SceneRecipe
from .sessions import _now, new_id


_parse_ref = parse_asset_ref


def task_asset_id(ref: str) -> str:
    """Map a recipe task ref to the id understood by tasks.asset_params."""
    kind, *rest = _parse_ref(ref)
    if kind == "builtin":
        return rest[0]
    if kind == "imported":
        return "imported/" + rest[0]
    raise ValueError("task_asset_ref must be builtin/<name> or imported/<slug> (a scene is not a manipulable object)")


def recipe_hash(recipe: SceneRecipe) -> str:
    data = recipe.model_dump()
    data.pop("manifest_hash", None)
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class RecipeStore:
    def __init__(self, root: Path):
        self.dir = Path(root)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "scene_recipes.json"
        self._lock = threading.RLock()
        self._recipes = ({r["recipe_id"]: r for r in json.loads(self.path.read_text(encoding="utf-8"))}
                         if self.path.is_file() else {})

    def _save(self):
        with self._lock:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(list(self._recipes.values()), ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(self.path)

    def put(self, recipe: SceneRecipe) -> SceneRecipe:
        recipe.manifest_hash = recipe_hash(recipe)
        with self._lock:
            self._recipes[recipe.recipe_id] = recipe.model_dump()
            self._save()
        return recipe

    def get(self, rid: str) -> SceneRecipe | None:
        with self._lock:
            raw = self._recipes.get(rid)
        return SceneRecipe.model_validate(raw) if raw else None

    def list(self):
        with self._lock:
            return copy.deepcopy(sorted(self._recipes.values(), key=lambda r: r.get("recipe_id", "")))


def build_recipe_from_session(session, lab_root: Path, *, overrides: dict | None = None) -> SceneRecipe:
    """Derive a recipe from a generation session. The session's imported asset becomes the
    manipulable task object only if it is a calibrated dynamic task_object; a generated
    static_prop / robot_visual becomes an asset_node; otherwise the caller supplies refs."""
    ov = dict(overrides or {})
    spec = session.asset_spec
    task_ref = ov.get("task_asset_ref")
    latest_asset = None
    for jid in reversed(session.job_ids):
        aid = _session_asset_id(session, jid, lab_root)
        if aid:
            latest_asset = "imported/" + aid.split("/", 1)[1] if aid.startswith("imported/") else aid
            break
    if task_ref is None and spec.kind == AssetKind.task_object and latest_asset:
        task_ref = latest_asset
    asset_nodes = ov.get("asset_nodes")
    support_z = float(ov.get("support_plane_z", 0.0))
    use_as_goal_tray = bool(ov.get("use_as_goal_tray"))
    goal_tray_node_id = ov.get("goal_tray_node_id")
    goal_xy = ov.get("goal_xy")
    goal_surface_m = ov.get("goal_surface_m")
    goal_half_extents = ov.get("goal_half_extents")
    if asset_nodes is None:
        if spec.kind in (AssetKind.static_prop, AssetKind.robot_visual) and latest_asset:
            import re
            raw_id = re.sub(r"[^A-Za-z0-9_-]", "-", (spec.name or spec.kind.value)).strip("-") or "prop"
            node_id = valid_id(raw_id[:48])
            node_role = spec.role if spec.role != AssetRole.dynamic else AssetRole.static
            pos = [0.0, 0.0, support_z]
            if use_as_goal_tray:
                if spec.kind != AssetKind.static_prop:
                    raise ValueError("只有 static_prop 资产才能设为抓放目标托盘")
                res = resolve_asset_node(lab_root, latest_asset)
                if not res.get("calibrated"):
                    raise ValueError(f"{latest_asset} 未标定，不能设为抓放目标托盘（请先填写尺寸完成标定）")
                node_role = AssetRole.static
                gx, gy = goal_xy or [0.133, 0.050]
                pos = [float(gx), float(gy), support_z]
                goal_xy = [float(gx), float(gy)]
                goal_tray_node_id = goal_tray_node_id or node_id
                comp = res.get("compound_proxy")
                if res.get("concave_detected") and isinstance(comp, dict) and comp.get("boxes"):
                    fbox = comp["boxes"][int(comp.get("floor_box_index") or 0)]
                    fz = fbox["size_m"][2]
                    if goal_surface_m is None:
                        goal_surface_m = support_z + float(fbox["position"][2]) + float(fz) / 2.0
                    if goal_half_extents is None:
                        goal_half_extents = [float(fbox["size_m"][0]) / 2.0, float(fbox["size_m"][1]) / 2.0]
                else:
                    proxy = res.get("collision_proxy") or {}
                    sz = proxy.get("size_m") or []
                    if len(sz) == 3:
                        if goal_half_extents is None:
                            hx = float(sz[0]) / 2.0
                            hy = float(sz[0]) / 2.0 if proxy.get("shape") == "cylinder" else float(sz[1]) / 2.0
                            goal_half_extents = [hx, hy]
                        if goal_surface_m is None:
                            pz = float((proxy.get("position") or [0.0, 0.0, float(sz[2]) / 2.0])[2])
                            goal_surface_m = support_z + pz + float(sz[2]) / 2.0
            asset_nodes = [AssetNode(node_id=node_id, asset_ref=latest_asset, role=node_role, position=pos).model_dump()]
        else:
            if use_as_goal_tray and not goal_tray_node_id:
                raise ValueError("当前会话没有可用于放置目标托盘的 static_prop 资产")
            asset_nodes = []
    mode = SceneMode(ov.get("scene_mode", "composed_scene"))
    background_ref = ov.get("background_ref")
    if background_ref is None and spec.kind == AssetKind.environment_mesh and latest_asset and latest_asset.startswith("scenes/"):
        background_ref = latest_asset
    renderer = ov.get("sensor_renderer", "mujoco")
    if mode == SceneMode.background_scene and background_ref:
        # Mesh environments use mujoco renderer; gaussian scenes use hybrid renderer.
        if spec.kind == AssetKind.environment_mesh:
            renderer = ov.get("sensor_renderer", "mujoco")
        else:
            renderer = ov.get("sensor_renderer", "hybrid")
    recipe = SceneRecipe(
        recipe_id=ov.get("recipe_id") or new_id("recipe"),
        task_goal=ov.get("task_goal") or spec.description_zh or spec.name or "cup pick-and-place",
        scene_mode=mode,
        robot_ref=ov.get("robot_ref"),
        task_asset_ref=task_ref,
        support_plane_z=support_z,
        background_ref=background_ref,
        asset_nodes=asset_nodes,
        colliders=ov.get("colliders", []),
        use_default_table=ov.get("use_default_table", True),
        sensor_renderer=renderer,
        layout_constraints=ov.get("layout_constraints", []),
        calibration=ov.get("calibration", {}),
        goal_tray_node_id=goal_tray_node_id,
        goal_xy=goal_xy,
        goal_surface_m=goal_surface_m,
        goal_half_extents=goal_half_extents,
        versions={"generation_schema": session.schema_version, "session_id": session.session_id,
                  "spec_revision": session.revision},
    )
    return recipe


def _session_asset_id(session, jid, lab_root):
    """Find a succeeded object job's asset id from the persisted object store, if present."""
    path = Path(lab_root) / "datasets" / "generation_jobs" / "object_jobs.json"
    if not path.is_file():
        return None
    try:
        jobs = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    for j in jobs:
        if j.get("id") == jid and j.get("state") == "succeeded" and j.get("asset_id"):
            return j["asset_id"]
    return None


def _auto_node_colliders(node: AssetNode, resolved: dict, existing_names: set[str]) -> list[dict]:
    mf = resolved.get("manifest") or {}
    friction = mf.get("friction") or [0.95, 0.005, 0.0001]
    scale = float(node.uniform_scale)
    t_world_node = transform_matrix(node.position, node.quaternion_xyzw, scale)

    comp = resolved.get("compound_proxy") or mf.get("compound_proxy")
    mode = getattr(node, "collision_mode", "auto")
    use_compound = isinstance(comp, dict) and comp.get("boxes") and (
        mode == "compound" or (mode == "auto" and resolved.get("concave_detected"))
    )
    if use_compound:
        boxes = list(comp["boxes"])
        fidx = int(comp.get("floor_box_index") or 0)
        if 0 <= fidx < len(boxes):
            boxes = [boxes[fidx]] + [b for i, b in enumerate(boxes) if i != fidx]
        base = node.node_id if f"{node.node_id}_part_0" not in existing_names else f"{node.node_id}_proxy"
        while f"{base}_part_0" in existing_names:
            base = f"{base}_x"
        out = []
        for idx, b in enumerate(boxes):
            t_node_proxy = transform_matrix(
                b.get("position", [0.0, 0.0, 0.0]),
                b.get("quaternion_xyzw", [0.0, 0.0, 0.0, 1.0]),
                1.0,
            )
            pos_w, quat_w, _ = mat_to_pos_quat_xyzw(compose(t_world_node, t_node_proxy))
            cname = valid_id(f"{base}_part_{idx}")
            existing_names.add(cname)
            out.append(SceneCollider(
                name=cname,
                shape="box",
                full_size_m=[max(1e-3, float(v) * scale) for v in b["size_m"]],
                position=pos_w,
                quaternion_xyzw=quat_w,
                friction=friction,
                tag="asset_node",
            ).model_dump())
        return out

    proxy = resolved.get("collision_proxy")
    if not isinstance(proxy, dict):
        return []
    shape = proxy.get("shape", "box")
    if shape not in {"box", "cylinder", "plane"}:
        return []
    raw_size = proxy.get("size_m")
    if not isinstance(raw_size, list) or len(raw_size) != 3:
        return []
    t_node_proxy = transform_matrix(
        proxy.get("position", [0.0, 0.0, 0.0]),
        proxy.get("quaternion_xyzw", [0.0, 0.0, 0.0, 1.0]),
        1.0,
    )
    pos_w, quat_w, _ = mat_to_pos_quat_xyzw(compose(t_world_node, t_node_proxy))
    cname = node.node_id if node.node_id not in existing_names else f"{node.node_id}_proxy"
    idx = 2
    while cname in existing_names:
        cname = f"{node.node_id}_proxy_{idx}"
        idx += 1
    existing_names.add(cname)
    return [SceneCollider(
        name=valid_id(cname),
        shape=shape,
        full_size_m=[float(v) * scale for v in raw_size],
        position=pos_w,
        quaternion_xyzw=quat_w,
        friction=friction,
        tag="asset_node",
    ).model_dump()]


def materialize_recipe_nodes(recipe: SceneRecipe, lab_root: Path | None = None) -> tuple[list[dict], list[dict]]:
    """Validate and resolve asset_nodes, materializing static primitive/compound collision
    proxies or configuring dynamic rigid-body manifests."""
    colliders_out = copy.deepcopy(recipe.colliders)
    existing_names = {c.get("name") for c in colliders_out if c.get("name")}
    if lab_root is not None and recipe.background_ref and not colliders_out:
        kind, *rest = _parse_ref(recipe.background_ref)
        if kind == "scene":
            _, mf = load_scene(lab_root, rest[0], rest[1], physics=False)
            if mf.get("ply_kind") != "gaussian" and isinstance(mf.get("colliders"), list):
                cal = recipe.calibration or {}
                t_bg = transform_matrix(
                    cal.get("position", [0.0, 0.0, 0.0]),
                    cal.get("quaternion_xyzw", [0.0, 0.0, 0.0, 1.0]),
                    float(cal.get("uniform_scale", 1.0)),
                )
                scale = float(cal.get("uniform_scale", 1.0))
                for col in mf["colliders"]:
                    cname = valid_id(col["name"])
                    if cname in existing_names:
                        continue
                    t_local = transform_matrix(col["position"], col.get("quaternion_xyzw", [0, 0, 0, 1]), 1.0)
                    pw, qw, _ = mat_to_pos_quat_xyzw(compose(t_bg, t_local))
                    existing_names.add(cname)
                    colliders_out.append(SceneCollider(
                        name=cname,
                        shape=col.get("shape", "box"),
                        full_size_m=[float(v) * scale for v in col["full_size_m"]],
                        position=pw,
                        quaternion_xyzw=qw,
                        tag="environment_mesh",
                    ).model_dump())

    nodes_out: list[dict] = []
    for n in recipe.asset_nodes:
        if n.collider_ref and n.collider_ref not in existing_names:
            raise ValueError(f"node {n.node_id}: collider_ref {n.collider_ref} not present in recipe.colliders")
        nd = SceneAssetNode(
            node_id=n.node_id,
            asset_ref=n.asset_ref,
            role=n.role.value,
            position=list(n.position),
            quaternion_xyzw=list(n.quaternion_xyzw),
            uniform_scale=float(n.uniform_scale),
            collider_ref=n.collider_ref,
            depends_on=list(n.depends_on),
            mass_kg=n.mass_kg,
            friction=list(n.friction) if n.friction is not None else None,
            collision_mode=n.collision_mode,
        ).model_dump()
        if lab_root is not None:
            resolved = resolve_asset_node(lab_root, n.asset_ref)
            if n.role in (AssetRole.static, AssetRole.dynamic) and not resolved["calibrated"]:
                raise ValueError(
                    f"node {n.node_id}: {n.asset_ref} is uncalibrated; only visual_only role is allowed"
                )
            if n.role == AssetRole.static and n.collider_ref is None:
                auto_cols = _auto_node_colliders(n, resolved, existing_names)
                if auto_cols:
                    colliders_out.extend(auto_cols)
                    nd["collider_ref"] = auto_cols[0]["name"]
        nodes_out.append(nd)
    return colliders_out, nodes_out


def _apply_recipe_to_data(recipe: SceneRecipe, data: dict, lab_root: Path | None = None) -> dict:
    sc = data["scene"]
    sc["sensor_renderer"] = recipe.sensor_renderer
    sc["use_default_table"] = recipe.use_default_table
    sc["work_surface_z"] = recipe.support_plane_z
    colliders_out, nodes_out = materialize_recipe_nodes(recipe, lab_root=lab_root)
    sc["colliders"] = colliders_out
    sc["asset_nodes"] = nodes_out
    if recipe.background_ref:
        kind, *rest = _parse_ref(recipe.background_ref)
        if kind != "scene":
            raise ValueError("background_ref must be scenes/<asset_id>/<rev>")
        cal = recipe.calibration or {}
        sc["environment"] = SceneInstance(
            asset_id=rest[0], revision=rest[1],
            position=cal.get("position", [0, 0, 0]),
            quaternion_xyzw=cal.get("quaternion_xyzw", [0, 0, 0, 1]),
            uniform_scale=cal.get("uniform_scale", 1.0)).model_dump()
    else:
        sc["environment"] = None
    t = data["task"]
    if recipe.task_asset_ref:
        t["asset"] = task_asset_id(recipe.task_asset_ref)
    if recipe.goal_tray_node_id is not None:
        tray_node = next((nd for nd in nodes_out if nd["node_id"] == recipe.goal_tray_node_id), None)
        if tray_node is None:
            raise ValueError(f"goal_tray_node_id {recipe.goal_tray_node_id!r} not in asset_nodes")
        if tray_node.get("role") != "static":
            raise ValueError(f"goal_tray_node_id {recipe.goal_tray_node_id!r} must have role='static'")
        cref = tray_node.get("collider_ref")
        tc = next((c for c in colliders_out if c.get("name") == cref and c.get("enabled", True)), None)
        if tc is None:
            raise ValueError(
                f"goal tray node {recipe.goal_tray_node_id!r} requires an enabled static collider (collider_ref={cref!r})"
            )
        fx, fy, fz = [float(v) for v in tc["full_size_m"]]
        gx, gy = (list(recipe.goal_xy) if recipe.goal_xy is not None
                  else [float(tc["position"][0]), float(tc["position"][1])])
        hx, hy = (list(recipe.goal_half_extents) if recipe.goal_half_extents is not None
                  else ([fx / 2.0, fx / 2.0] if tc.get("shape") == "cylinder" else [fx / 2.0, fy / 2.0]))
        gs = (float(recipe.goal_surface_m) if recipe.goal_surface_m is not None
              else float(tc["position"][2]) + fz / 2.0)
        t["goal_xy"] = [float(gx), float(gy)]
        t["goal_half_extents"] = [float(hx), float(hy)]
        t["goal_surface_m"] = float(gs)
        t["goal_tray_collider_ref"] = cref
        t["show_builtin_tray"] = False
    else:
        if t.get("goal_tray_collider_ref") or t.get("show_builtin_tray") is False:
            t["goal_tray_collider_ref"] = None
            t["show_builtin_tray"] = True
            t["goal_surface_m"] = float(recipe.support_plane_z) + 0.009
            t["goal_half_extents"] = [0.074, 0.069]
        if recipe.goal_xy is not None:
            t["goal_xy"] = list(recipe.goal_xy)
        if recipe.goal_surface_m is not None:
            t["goal_surface_m"] = float(recipe.goal_surface_m)
        elif t.get("goal_surface_m", 0.009) <= recipe.support_plane_z + 1e-4:
            t["goal_surface_m"] = float(recipe.support_plane_z) + 0.009
        if recipe.goal_half_extents is not None:
            t["goal_half_extents"] = list(recipe.goal_half_extents)
    return data


def recipe_to_config_data(recipe: SceneRecipe, base: ProjectConfig, project_id: str,
                          lab_root: Path | None = None) -> dict:
    """Mapping recipe -> project config dict. Resolves asset_nodes when lab_root is provided."""
    data = base.model_dump()
    data["project_id"] = project_id
    return _apply_recipe_to_data(recipe, data, lab_root=lab_root)


def _dependency_issues(recipe: SceneRecipe):
    issues, warnings = [], []
    nodes = {n.node_id: n for n in recipe.asset_nodes}
    for n in recipe.asset_nodes:
        for dep in n.depends_on:
            if dep not in nodes:
                issues.append(f"node {n.node_id}: depends_on unknown node {dep}")
    # cycle detection (DFS)
    WHITE, GREY, BLACK = 0, 1, 2
    color = {nid: WHITE for nid in nodes}

    def visit(nid, stack):
        color[nid] = GREY
        for dep in nodes[nid].depends_on:
            if dep not in nodes:
                continue
            if color[dep] == GREY:
                issues.append(f"dependency cycle: {' -> '.join(stack + [dep])}")
            elif color[dep] == WHITE:
                visit(dep, stack + [dep])
        color[nid] = BLACK

    for nid in nodes:
        if color[nid] == WHITE:
            visit(nid, [nid])
    return issues, warnings


def _collider_aabb(c: dict):
    shape = c.get("shape", "box")
    fx, fy, fz = c.get("full_size_m", [0, 0, 0])
    px, py, pz = c.get("position", [0, 0, 0])
    if shape == "cylinder":
        hx = hy = fx / 2
    else:
        hx, hy = fx / 2, fy / 2
    hz = fz / 2 if shape != "plane" else 0.01
    return (px - hx, py - hy, pz - hz, px + hx, py + hy, pz + hz)


def _overlap(a, b):
    return (min(a[3], b[3]) - max(a[0], b[0]),
            min(a[4], b[4]) - max(a[1], b[1]),
            min(a[5], b[5]) - max(a[2], b[2]))


def _layout_issues(recipe: SceneRecipe, base: ProjectConfig, *,
                   effective_colliders: list[dict] | None = None,
                   resolved_nodes: dict[str, dict] | None = None):
    issues, warnings = [], []
    cols = recipe.colliders if effective_colliders is None else effective_colliders
    resolved_nodes = resolved_nodes or {}
    for n in recipe.asset_nodes:
        if not all(math.isfinite(v) for v in n.position) or not math.isfinite(n.uniform_scale):
            issues.append(f"node {n.node_id}: non-finite pose/scale")
        if n.role == AssetRole.static and not n.collider_ref:
            res = resolved_nodes.get(n.node_id)
            if res is not None and res.get("collision_proxy") is None:
                warnings.append(f"static node {n.node_id} has no collider_ref or manifest proxy; it will be visual-only (no contact)")
        if n.collider_ref and not any(c.get("name") == n.collider_ref for c in recipe.colliders):
            issues.append(f"node {n.node_id}: collider_ref {n.collider_ref} not present in recipe.colliders")
    boxes = [(c.get("name", f"c{i}"), _collider_aabb(c)) for i, c in enumerate(cols)]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            ox, oy, oz = _overlap(boxes[i][1], boxes[j][1])
            if ox > 1e-3 and oy > 1e-3 and oz > 1e-3:
                warnings.append(f"colliders {boxes[i][0]} and {boxes[j][0]} interpenetrate "
                                f"({ox:.3f}x{oy:.3f}x{oz:.3f} m); check the layout")
    if recipe.use_default_table:
        sz = float(recipe.support_plane_z)
        table_bb = (-0.32, -0.215, sz - 0.056, 0.32, 0.215, sz)
        for name, bb in boxes:
            ox, oy, oz = _overlap(table_bb, bb)
            if ox > 1e-3 and oy > 1e-3 and oz > 1e-3:
                warnings.append(f"collider {name} overlaps the default table; disable use_default_table "
                                f"or move the proxy to avoid redundant/penetrating support")
    # static props should rest on the support plane unless explicitly supported
    sz = float(recipe.support_plane_z)
    for c in cols:
        bb = _collider_aabb(c)
        if c.get("shape") != "plane" and bb[2] - sz > 0.02:
            warnings.append(f"collider {c.get('name')} floats {bb[2] - sz:.3f} m above the support plane; "
                            f"add a support or set depends_on")
    return issues, warnings


def _background_issues(recipe: SceneRecipe, lab_root: Path):
    issues, warnings, info = [], [], {}
    if not recipe.background_ref:
        if recipe.sensor_renderer == "hybrid":
            issues.append("sensor_renderer=hybrid requires a gaussian background_ref")
        return issues, warnings, info
    kind, *rest = _parse_ref(recipe.background_ref)
    if kind != "scene":
        issues.append("background_ref must be scenes/<asset_id>/<rev>")
        return issues, warnings, info
    try:
        _, mf = load_scene(lab_root, rest[0], rest[1], physics=False)
    except Exception as exc:
        issues.append(f"background unavailable: {exc}")
        return issues, warnings, info
    gaussian = mf.get("ply_kind") == "gaussian"
    info["background"] = {"asset_id": rest[0], "revision": rest[1], "gaussian": gaussian,
                          "calibrated": mf.get("calibrated", False)}
    if gaussian:
        if recipe.sensor_renderer != "hybrid":
            issues.append("gaussian background requires sensor_renderer=hybrid (it is not a MJCF mesh)")
        if not (recipe.use_default_table or any(c.get("enabled", True) for c in recipe.colliders)):
            issues.append("gaussian background needs a default table or explicit static colliders for physics")
        warnings.append("hybrid background is static_background_overlay: world-fixed front camera only; "
                        "the panorama is a preview, not editable 3D Gaussian geometry")
    else:
        if recipe.sensor_renderer == "hybrid":
            issues.append("mesh background uses sensor_renderer=mujoco, not hybrid")
        if not mf.get("calibrated"):
            issues.append("mesh background must be calibrated before physics use")
    return issues, warnings, info


def validate_recipe(recipe: SceneRecipe, lab_root: Path, config_store, project_id: str) -> dict:
    """Full validation: refs, dependency graph, layout, background/renderer, real MuJoCo
    build + reachability + short-step stability. Returns a structured report."""
    from ..tasks import asset_params
    from ..store import validate_config

    lab_root = Path(lab_root)
    issues, warnings, checks = [], [], {}
    base = config_store.load(project_id) if config_store.exists(project_id) else config_store.load("default")

    dep_i, dep_w = _dependency_issues(recipe)
    issues += dep_i
    warnings += dep_w
    checks["dependency_graph"] = {"nodes": len(recipe.asset_nodes), "acyclic": not dep_i}

    node_issues: list[str] = []
    resolved_nodes: dict[str, dict] = {}
    node_summaries: list[dict] = []
    for n in recipe.asset_nodes:
        try:
            res = resolve_asset_node(lab_root, n.asset_ref)
            resolved_nodes[n.node_id] = res
            if n.role in (AssetRole.static, AssetRole.dynamic) and not res["calibrated"]:
                node_issues.append(
                    f"node {n.node_id}: {n.asset_ref} is uncalibrated; only visual_only role is allowed"
                )
            node_summaries.append({
                "node_id": n.node_id,
                "asset_ref": n.asset_ref,
                "role": n.role.value,
                "calibrated": res["calibrated"],
                "has_proxy": bool(res.get("collision_proxy") or res.get("compound_proxy")),
                "concave": bool(res.get("concave_detected")),
            })
        except Exception as exc:
            node_issues.append(f"node {n.node_id} ({n.asset_ref}): {exc}")
    issues += node_issues
    checks["asset_nodes"] = {"count": len(recipe.asset_nodes), "resolved": node_summaries, "issues": node_issues}

    effective_colliders = None
    if not node_issues:
        try:
            effective_colliders, _ = materialize_recipe_nodes(recipe, lab_root=lab_root)
        except Exception as exc:
            issues.append(str(exc))

    lay_i, lay_w = _layout_issues(
        recipe, base, effective_colliders=effective_colliders, resolved_nodes=resolved_nodes
    )
    issues += lay_i
    warnings += lay_w
    checks["layout"] = {
        "colliders": len(effective_colliders) if effective_colliders is not None else len(recipe.colliders),
        "issues": lay_i,
        "warnings": lay_w,
    }

    bg_i, bg_w, bg_info = _background_issues(recipe, lab_root)
    issues += bg_i
    warnings += bg_w
    checks["background"] = bg_info.get("background", {})

    if recipe.robot_ref and recipe.robot_ref not in {r.id for r in base.robots}:
        issues.append(f"robot_ref {recipe.robot_ref} not in project robots")

    # resolve the manipulable task object (must be a calibrated desktop asset)
    task_info = {}
    if recipe.task_asset_ref:
        try:
            aid = task_asset_id(recipe.task_asset_ref)
            r, h, mass, *_ = asset_params(aid, lab_root)
            task_info = {"asset": aid, "radius_m": r, "height_m": h, "mass_kg": mass}
        except Exception as exc:
            issues.append(f"task asset unresolved/uncalibrated: {exc}")
    checks["task_object"] = task_info

    # real MuJoCo build + reachability via the existing config validator
    candidate = None
    if not issues:
        try:
            data = recipe_to_config_data(recipe, base, project_id, lab_root=lab_root)
            candidate = ProjectConfig.model_validate(data)
        except Exception as exc:
            issues.append(f"recipe does not map to a valid config: {exc}")
    physics = {"built": False, "stable": False}
    if candidate is not None:
        checks["goal_tray"] = {
            "custom_node_id": recipe.goal_tray_node_id,
            "show_builtin_tray": candidate.task.show_builtin_tray,
            "goal_tray_collider_ref": candidate.task.goal_tray_collider_ref,
            "goal_xy": list(candidate.task.goal_xy),
            "goal_half_extents": list(candidate.task.goal_half_extents),
            "goal_surface_m": candidate.task.goal_surface_m,
        }
        build_issues = validate_config(candidate, lab_root)
        issues += build_issues
        physics["built"] = not build_issues
        checks["config_build"] = {"issues": build_issues}
        if not build_issues:
            physics.update(_short_step_stability(candidate, lab_root))
    checks["physics"] = physics
    if physics.get("built") and not physics.get("stable"):
        issues.append("short-step simulation unstable: " + "; ".join(physics.get("instability", [])))

    return {"recipe_id": recipe.recipe_id, "valid": not issues, "issues": issues,
            "warnings": warnings, "checks": checks, "manifest_hash": recipe.manifest_hash or recipe_hash(recipe)}


def _short_step_stability(candidate: ProjectConfig, lab_root: Path) -> dict:
    """Build the env and short-step it; cup and dynamic nodes must settle above table, no NaN."""
    import numpy as np
    import mujoco
    from ..engine import PlatformEnv
    out = {"stable": False, "instability": [], "steps": 0}
    env = None
    try:
        env = PlatformEnv(candidate, lab_root)
        cup = env.model.body("cup").id
        mujoco.mj_step(env.model, env.data, nstep=250)
        out["steps"] = 250
        if not (np.isfinite(env.data.qpos).all() and np.isfinite(env.data.qvel).all()):
            out["instability"].append("non-finite state after stepping")
        ws_z = float(candidate.scene.work_surface_z)
        z = float(env.data.xpos[cup][2])
        upright = float(env.data.xmat[cup][8])
        out["cup_z_m"] = round(z, 4)
        out["cup_upright_dot"] = round(upright, 3)
        if z <= ws_z - 0.015 or z > ws_z + 0.5:
            out["instability"].append(f"cup left the support region (z={z:.3f} m)")
        if upright < 0.9:
            out["instability"].append(f"cup tipped over (upright dot={upright:.2f})")
        dyn_info = {}
        for node in candidate.scene.asset_nodes:
            if node.role == "dynamic":
                nbid = env.model.body(f"node_{node.node_id}").id
                nz = float(env.data.xpos[nbid][2])
                dyn_info[node.node_id] = {"z_m": round(nz, 4)}
                if nz <= ws_z - 0.015 or nz > ws_z + 0.5:
                    out["instability"].append(f"dynamic node {node.node_id} left support region (z={nz:.3f} m)")
        if dyn_info:
            out["dynamic_nodes"] = dyn_info
        out["stable"] = not out["instability"]
    except Exception as exc:
        out["instability"].append(f"build/step failed: {exc}")
    finally:
        if env is not None:
            env.close()
    return out


def apply_mutate(recipe: SceneRecipe, lab_root: Path | None = None):
    """Return a mutate(data) closure for the existing atomic _commit_scene."""
    def mutate(data):
        _apply_recipe_to_data(recipe, data, lab_root=lab_root)
    return mutate