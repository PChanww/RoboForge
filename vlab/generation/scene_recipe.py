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

from ..schema import ProjectConfig, SceneCollider, SceneInstance
from ..scene_assets import load_scene, valid_id
from .schema import AssetKind, AssetRole, SceneMode, SceneRecipe
from .sessions import _now, new_id


def _parse_ref(ref: str):
    """'builtin/<name>' | 'imported/<slug>' | 'scenes/<asset_id>/<rev>'."""
    if not isinstance(ref, str) or "/" not in ref:
        raise ValueError(f"asset ref must be builtin/<name>, imported/<slug> or scenes/<id>/<rev>: {ref!r}")
    head, _, tail = ref.partition("/")
    if head == "builtin":
        return ("builtin", tail)
    if head == "imported":
        return ("imported", tail)
    if head == "scenes":
        parts = tail.split("/")
        if len(parts) != 2:
            raise ValueError("scene ref must be scenes/<asset_id>/<revision>")
        return ("scene", parts[0], int(parts[1]))
    raise ValueError(f"unknown asset ref namespace: {head}")


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
    manipulable task object only if it is a calibrated dynamic task_object; otherwise the
    caller must supply task_asset_ref (we never invent a calibrated object)."""
    ov = dict(overrides or {})
    spec = session.asset_spec
    task_ref = ov.get("task_asset_ref")
    if task_ref is None and spec.kind == AssetKind.task_object:
        for jid in reversed(session.job_ids):
            asset_id = _session_asset_id(session, jid, lab_root)
            if asset_id:
                task_ref = "imported/" + asset_id.split("/", 1)[1] if asset_id.startswith("imported/") else asset_id
                break
    mode = SceneMode(ov.get("scene_mode", "composed_scene"))
    background_ref = ov.get("background_ref")
    renderer = ov.get("sensor_renderer", "mujoco")
    if mode == SceneMode.background_scene and background_ref:
        renderer = ov.get("sensor_renderer", "hybrid")
    recipe = SceneRecipe(
        recipe_id=ov.get("recipe_id") or new_id("recipe"),
        task_goal=ov.get("task_goal") or spec.description_zh or spec.name or "cup pick-and-place",
        scene_mode=mode,
        robot_ref=ov.get("robot_ref"),
        task_asset_ref=task_ref,
        support_plane_z=0.0,
        background_ref=background_ref,
        asset_nodes=ov.get("asset_nodes", []),
        colliders=ov.get("colliders", []),
        use_default_table=ov.get("use_default_table", True),
        sensor_renderer=renderer,
        layout_constraints=ov.get("layout_constraints", []),
        calibration=ov.get("calibration", {}),
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


def recipe_to_config_data(recipe: SceneRecipe, base: ProjectConfig, project_id: str) -> dict:
    """Pure mapping recipe -> project config dict (no IO). The caller validates/commits it."""
    data = base.model_dump()
    data["project_id"] = project_id
    sc = data["scene"]
    sc["sensor_renderer"] = recipe.sensor_renderer
    sc["use_default_table"] = recipe.use_default_table
    sc["work_surface_z"] = recipe.support_plane_z
    sc["colliders"] = copy.deepcopy(recipe.colliders)
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
    if recipe.task_asset_ref:
        data["task"]["asset"] = task_asset_id(recipe.task_asset_ref)
    return data


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


def _layout_issues(recipe: SceneRecipe, base: ProjectConfig):
    issues, warnings = [], []
    for n in recipe.asset_nodes:
        if not all(math.isfinite(v) for v in n.position) or not math.isfinite(n.uniform_scale):
            issues.append(f"node {n.node_id}: non-finite pose/scale")
        if n.role == AssetRole.static and not n.collider_ref:
            warnings.append(f"static node {n.node_id} has no collider_ref; it will be visual-only (no contact)")
        if n.collider_ref and not any(c.get("name") == n.collider_ref for c in recipe.colliders):
            issues.append(f"node {n.node_id}: collider_ref {n.collider_ref} not present in recipe.colliders")
    boxes = [(c.get("name", f"c{i}"), _collider_aabb(c)) for i, c in enumerate(recipe.colliders)]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            ox, oy, oz = _overlap(boxes[i][1], boxes[j][1])
            if ox > 1e-3 and oy > 1e-3 and oz > 1e-3:
                warnings.append(f"colliders {boxes[i][0]} and {boxes[j][0]} interpenetrate "
                                f"({ox:.3f}x{oy:.3f}x{oz:.3f} m); check the layout")
    if recipe.use_default_table:
        table = ("default_table", (-0.32, -0.215, -0.056, 0.32, 0.215, 0.0))
        for name, bb in boxes:
            ox, oy, oz = _overlap(table, bb)
            if ox > 1e-3 and oy > 1e-3 and oz > 1e-3:
                warnings.append(f"collider {name} overlaps the default table; disable use_default_table "
                                f"or move the proxy to avoid redundant/penetrating support")
    # static props should rest on the support plane (z=0) unless explicitly supported
    for c in recipe.colliders:
        bb = _collider_aabb(c)
        if c.get("shape") != "plane" and bb[2] > 0.02:
            warnings.append(f"collider {c.get('name')} floats {bb[2]:.3f} m above the support plane; "
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

    lay_i, lay_w = _layout_issues(recipe, base)
    issues += lay_i
    warnings += lay_w
    checks["layout"] = {"colliders": len(recipe.colliders), "issues": lay_i, "warnings": lay_w}

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
    try:
        data = recipe_to_config_data(recipe, base, project_id)
        candidate = ProjectConfig.model_validate(data)
    except Exception as exc:
        issues.append(f"recipe does not map to a valid config: {exc}")
    physics = {"built": False, "stable": False}
    if candidate is not None:
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
    """Build the env and short-step it; the cup must settle upright above the table, no NaN."""
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
        z = float(env.data.xpos[cup][2])
        upright = float(env.data.xmat[cup][8])
        out["cup_z_m"] = round(z, 4)
        out["cup_upright_dot"] = round(upright, 3)
        if z <= 0.0 or z > 0.5:
            out["instability"].append(f"cup left the support region (z={z:.3f} m)")
        if upright < 0.9:
            out["instability"].append(f"cup tipped over (upright dot={upright:.2f})")
        out["stable"] = not out["instability"]
    except Exception as exc:
        out["instability"].append(f"build/step failed: {exc}")
    finally:
        if env is not None:
            env.close()
    return out


def apply_mutate(recipe: SceneRecipe):
    """Return a mutate(data) closure for the existing atomic _commit_scene."""
    def mutate(data):
        sc = data["scene"]
        sc["sensor_renderer"] = recipe.sensor_renderer
        sc["use_default_table"] = recipe.use_default_table
        sc["work_surface_z"] = recipe.support_plane_z
        sc["colliders"] = copy.deepcopy(recipe.colliders)
        if recipe.background_ref:
            kind, *rest = _parse_ref(recipe.background_ref)
            cal = recipe.calibration or {}
            sc["environment"] = SceneInstance(
                asset_id=rest[0], revision=rest[1],
                position=cal.get("position", [0, 0, 0]),
                quaternion_xyzw=cal.get("quaternion_xyzw", [0, 0, 0, 1]),
                uniform_scale=cal.get("uniform_scale", 1.0)).model_dump()
        else:
            sc["environment"] = None
        if recipe.task_asset_ref:
            data["task"]["asset"] = task_asset_id(recipe.task_asset_ref)
    return mutate