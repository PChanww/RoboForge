"""FastAPI backend: versioned v1 API + legacy /api/* compatibility + WebSocket."""
from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import parse_qs

import mujoco
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import gymnasium

from . import sessions as sess_mod
from .engine import SCHEMA_VERSION, UnsupportedMode
from .schema import ProjectConfig
from .store import ConfigStore, validate_config
from .scene_assets import valid_id, scene_asset_dir, load_scene, contained_file

LAB_ROOT = Path(os.environ.get("VLAB_ROOT", Path(__file__).resolve().parents[1]))


class ResetBody(BaseModel):
    seed: int | None = None


class CreateSessionBody(BaseModel):
    project: str = "default"
    robot_id: str | None = None
    controller_id: str | None = None


class AcquireBody(BaseModel):
    owner: str


class TokenBody(BaseModel):
    token: str
    epoch: int | None = None


class ExpertStepBody(TokenBody):
    steps: int = 1


class PolicyStepBody(TokenBody):
    model_id: str = "hackathon_vla_bc"
    steps: int = 1
    instruction: str | None = None  # language input for VLA policies; ignored by fixed-action models


class ActionBody(BaseModel):
    token: str
    action: list[float]
    tick: int | None = None
    epoch: int | None = None


class ActionBlockBody(BaseModel):
    token: str
    actions: list[list[float]]
    epoch: int | None = None


class JobBody(BaseModel):
    kind: str
    params: dict = {}


class ImportBody(BaseModel):
    filename: str
    data_b64: str
    slug: str
    height_m: float = 0.09
    mass_kg: float = 0.12
    radius_m: float | None = None
    up: str = "y"


class SceneImportBody(BaseModel):
    filename: str
    data_b64: str
    asset_id: str
    up_axis: str = "Y"
    unit_scale: float = 1.0
    calibrated: bool = False


class BindSceneBody(BaseModel):
    asset_id: str
    revision: int = 1
    use_default_table: bool = False
    position: list[float] = [0, 0, 0]
    quaternion_xyzw: list[float] = [0, 0, 0, 1]
    uniform_scale: float = 1.0


class GenerationBody(BaseModel):
    kind: str = "generation"
    params: dict = {}


class Lux3dBody(BaseModel):
    name: str
    prompt: str = ""
    height_m: float = 0.09
    mass_kg: float = 0.12
    radius_m: float | None = None
    kind: str | None = None
    role: str = "dynamic"
    width_m: float | None = None
    depth_m: float | None = None


class SceneInstanceBody(BaseModel):
    position: list[float] = [0, 0, 0]
    quaternion_xyzw: list[float] = [0, 0, 0, 1]
    uniform_scale: float = 1.0
    use_default_table: bool | None = None
    expected_version: int | None = None


class CollidersBody(BaseModel):
    colliders: list[dict] = []
    expected_version: int | None = None


class SceneCommitBody(BaseModel):
    environment: dict | None = None
    colliders: list[dict] = []
    asset_nodes: list[dict] | None = None
    use_default_table: bool | None = None
    work_surface_z: float | None = None
    expected_version: int


class AssemblePartsBody(BaseModel):
    project_id: str = "default"
    robot_id: str = "assembled_5r"
    parts_slug: str
    mapping: dict[str, str]
    replace_primary: bool = True


class DemoSetupBody(BaseModel):
    project_id: str = "hackathon_demo"
    with_world_background: bool = False
    support_plane_z: float = 0.0


class So100SetupBody(BaseModel):
    project_id: str = "so100_real_loop"
    live_generate: bool = False
    with_world_background: bool = False


def create_app(lab_root: Path = LAB_ROOT) -> FastAPI:
    app = FastAPI(title="VLA Lab Platform", version=SCHEMA_VERSION)
    store = ConfigStore(lab_root / "projects")
    manager = sess_mod.SessionManager()
    scene_lock = threading.Lock()

    def _validate_candidate(candidate):
        env = candidate.scene.environment
        if env is not None:
            load_scene(lab_root, env.asset_id, env.revision,
                       physics=candidate.scene.sensor_renderer != "hybrid")
        issues = validate_config(candidate, lab_root)
        if issues:
            raise HTTPException(400, {"issues": issues})

    def _commit_scene(pid, mutate, expected_version=None):
        """Validate the whole config, then save once. Any failure leaves current untouched."""
        valid_id(pid)
        with scene_lock:
            cfg = store.load(pid) if store.exists(pid) else store.load("default")
            if expected_version is not None and cfg.version != expected_version:
                raise HTTPException(409, {"error": f"scene version conflict: expected v{expected_version}, current v{cfg.version}",
                                          "current_version": cfg.version, "expected_version": expected_version})
            data = cfg.model_dump()
            data["project_id"] = pid
            data["validation_status"] = "candidate"
            mutate(data)
            from pydantic import ValidationError
            try:
                candidate = ProjectConfig.model_validate(data)
            except ValidationError as exc:
                raise HTTPException(400, {"issues": [str(e) for e in exc.errors()]})
            _validate_candidate(candidate)
            return store.save(candidate.model_copy(update={"project_id": pid, "validation_status": "validated"}))
    if not store.exists("default"):
        from .make_default_config import build
        store.save(build())

    # Legacy single-threaded lab, created lazily on first legacy request so startup
    # does not open an OpenGL context (a second GLFW context in a background process
    # can segfault; the first session renderer becomes the primary context).
    import server
    legacy_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="legacy")
    legacy_lab = None
    legacy_lock = threading.Lock()

    def _get_legacy():
        nonlocal legacy_lab
        with legacy_lock:
            if legacy_lab is None:
                legacy_lab = legacy_worker.submit(server.Lab).result()
        return legacy_lab

    robot_mesh_dir = lab_root / "assets" / "robot"
    if robot_mesh_dir.is_dir():
        app.mount("/assets/robot", StaticFiles(directory=robot_mesh_dir), name="robot_meshes")

    @app.get("/assets/robots/{rid}/{fname}")
    def custom_robot_mesh(rid: str, fname: str):
        valid_id(rid)
        if not fname.endswith(".obj") or "/" in fname or "\\" in fname or ".." in fname:
            raise HTTPException(400, "invalid robot mesh filename")
        p = contained_file(lab_root / "assets" / "robots" / rid, fname)
        if not p.is_file():
            raise HTTPException(404, "robot mesh not found")
        return FileResponse(p, media_type="text/plain")

    @app.get("/api/v1/cup.obj")
    def cup_mesh(asset: str = "standard"):
        from . import tasks
        try:
            *_, mesh, offset, handle = tasks.asset_params(asset, lab_root)
        except Exception as exc:
            raise HTTPException(400, f"unknown asset {asset}: {exc}")
        return Response(mesh, media_type="text/plain")

    @app.get("/api/v1/asset-mesh")
    def asset_mesh(ref: str):
        from .scene_builder import resolve_asset_node
        try:
            resolved = resolve_asset_node(lab_root, ref)
        except Exception as exc:
            raise HTTPException(404, f"unknown asset mesh {ref}: {exc}")
        return Response(resolved["mesh_bytes"], media_type="text/plain")

    @app.exception_handler(sess_mod.ControlConflict)
    async def _conflict(_req, exc):
        return JSONResponse({"error": str(exc)}, status_code=409)

    @app.exception_handler(UnsupportedMode)
    async def _unsupported(_req, exc):
        return JSONResponse({"error": str(exc)}, status_code=400)

    @app.exception_handler(ValueError)
    async def _bad_value(_req, exc):
        return JSONResponse({"error": str(exc)}, status_code=400)

    @app.exception_handler(FileNotFoundError)
    async def _missing_file(_req, exc):
        return JSONResponse({"error": "Requested project or asset file does not exist"}, status_code=404)

    def _session(sid):
        try:
            return manager.get(sid)
        except KeyError:
            raise HTTPException(404, f"unknown session {sid}")

    # -- capabilities & config ---------------------------------------------
    @app.get("/api/v1/capabilities")
    def capabilities():
        cfg = store.load("default")
        from .engine import PlatformEnv
        robots = []
        for r in cfg.robots:
            has5r = {"waist", "shoulder", "elbow", "wrist_pitch", "wrist_yaw"} <= {j.name for j in r.joints}
            ctrl = next((c.id for c in cfg.controllers
                         if (c.mode == "cartesian_delta_ik") == has5r), cfg.controllers[0].id)
            env = PlatformEnv(cfg, lab_root, robot_id=r.id, controller_id=ctrl)
            robots.append({"id": r.id, "joints": [j.name for j in r.joints if j.type != "fixed"],
                           "end_effector": r.end_effector, "supported_modes": env.supported_modes()})
            env.close()
        return {"schema_version": SCHEMA_VERSION, "physics_backend": "mujoco",
                "mujoco_version": mujoco.__version__, "gymnasium_version": gymnasium.__version__,
                "robots": robots, "controllers": [c.model_dump() for c in cfg.controllers],
                "sensors": [s.model_dump() for s in cfg.sensors], "task": cfg.task.model_dump()}

    @app.get("/api/v1/config")
    def get_config(project: str = "default"):
        return store.load(project).model_dump()

    @app.post("/api/v1/config/validate")
    def validate(config: dict):
        from pydantic import ValidationError
        try:
            parsed = ProjectConfig.model_validate(config)
        except ValidationError as exc:
            return {"valid": False, "issues": [str(e) for e in exc.errors()]}
        issues = validate_config(parsed, lab_root)
        return {"valid": not issues, "issues": issues}

    @app.put("/api/v1/config")
    def put_config(config: dict, project: str = "default"):
        from pydantic import ValidationError
        try:
            parsed = ProjectConfig.model_validate(config)
        except ValidationError as exc:
            raise HTTPException(400, {"issues": [str(e) for e in exc.errors()]})
        issues = validate_config(parsed, lab_root)
        if issues:
            raise HTTPException(400, {"issues": issues})
        parsed = parsed.model_copy(update={"project_id": project, "validation_status": "validated"})
        return store.save(parsed).model_dump()

    @app.get("/api/v1/config/versions")
    def config_versions(project: str = "default"):
        return {"versions": store.versions(project)}

    @app.get("/api/v1/config/versions/{version}")
    def config_version(version: int, project: str = "default"):
        return store.load_version(project, version).model_dump()

    # -- sessions -----------------------------------------------------------
    @app.post("/api/v1/sessions")
    def create_session(body: CreateSessionBody):
        cfg = store.load(body.project)
        s = manager.create(cfg, lab_root, body.robot_id, body.controller_id)
        return {"id": s.id, "robot": s.env.robot.id, "controller": s.env.controller.id,
                "config_version": cfg.version, "epoch": s.epoch,
                "observation": s.submit(s.env.observation)}

    @app.get("/api/v1/sessions")
    def list_sessions():
        return {"sessions": manager.list()}

    @app.post("/api/v1/sessions/{sid}/reset")
    def reset_session(sid: str, body: ResetBody):
        return _session(sid).reset(body.seed)

    @app.get("/api/v1/sessions/{sid}/observation")
    def observation(sid: str):
        s = _session(sid)
        obs = s.submit(s.env.observation)
        return {**obs, "epoch": s.epoch}

    @app.post("/api/v1/sessions/{sid}/control/acquire")
    def acquire(sid: str, body: AcquireBody):
        return {"token": _session(sid).acquire(body.owner), "owner": body.owner}

    @app.post("/api/v1/sessions/{sid}/control/release")
    def release(sid: str, body: TokenBody):
        _session(sid).release(body.token)
        return {"released": True}

    @app.post("/api/v1/sessions/{sid}/action")
    def action(sid: str, body: ActionBody):
        s = _session(sid)
        try:
            return s.execute(body.token, body.action, tick=body.tick, epoch=body.epoch)
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @app.post("/api/v1/sessions/{sid}/action_block")
    def action_block(sid: str, body: ActionBlockBody):
        s = _session(sid)
        try:
            return s.execute_block(body.token, body.actions, epoch=body.epoch)
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @app.post("/api/v1/sessions/{sid}/expert_step")
    def expert_step(sid: str, body: ExpertStepBody):
        s = _session(sid)
        try:
            return s.expert(body.token, epoch=body.epoch, steps=body.steps)
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @app.post("/api/v1/sessions/{sid}/policy_step")
    def policy_step(sid: str, body: PolicyStepBody):
        s = _session(sid)
        try:
            return s.policy(body.token, model_id=body.model_id, epoch=body.epoch, steps=body.steps,
                            instruction=body.instruction)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc))
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @app.get("/api/v1/sessions/{sid}/result")
    def result(sid: str):
        s = _session(sid)
        return {"observation": s.submit(s.env.observation), "metrics": s.submit(s.env.metrics)}

    @app.get("/api/v1/sessions/{sid}/frame.jpg")
    def frame(sid: str, camera: str = "front", width: int = 800, height: int = 540):
        from .engine import _jpeg_b64
        import base64
        s = _session(sid)
        arr = s.submit(s.env.render, camera, width, height)
        return Response(base64.b64decode(_jpeg_b64(arr)), media_type="image/jpeg")

    @app.delete("/api/v1/sessions/{sid}")
    def delete_session(sid: str):
        manager.close(sid)
        return {"closed": sid}

    # -- WebSocket observation/action loop ---------------------------------
    @app.websocket("/api/v1/ws/{sid}")
    async def ws(ws: WebSocket, sid: str):
        await ws.accept()
        try:
            s = _session(sid)
        except HTTPException:
            await ws.close(code=4004)
            return
        try:
            await ws.send_json({"type": "observation", "data": s.submit(s.env.observation)})
            while True:
                msg = await ws.receive_json()
                kind = msg.get("type")
                if kind == "observation":
                    await ws.send_json({"type": "observation", "data": s.submit(s.env.observation)})
                elif kind == "action":
                    s.check_control(msg.get("token", ""))
                    obs = s.submit(s.env.step, msg["action"])
                    await ws.send_json({"type": "observation", "data": obs})
                elif kind == "acquire":
                    await ws.send_json({"type": "token", "token": s.acquire(msg["owner"])})
                else:
                    await ws.send_json({"type": "error", "error": f"unknown type {kind}"})
        except WebSocketDisconnect:
            pass
        except sess_mod.ControlConflict as exc:
            await ws.send_json({"type": "error", "error": str(exc), "code": 409})
        except ValueError as exc:
            await ws.send_json({"type": "error", "error": str(exc), "code": 400})

    # -- legacy compatibility (unchanged behavior) -------------------------
    def _legacy(method, path, query, body):
        try:
            lab = _get_legacy()
            result = legacy_worker.submit(lab.run, method, path, query, body).result()
        except server.APIError as exc:
            raise HTTPException(exc.status, str(exc))
        if isinstance(result, bytes):
            return Response(result, media_type="image/jpeg")
        return result

    @app.get("/api/assets")
    def legacy_assets():
        return _legacy("GET", "/api/assets", {}, {})

    @app.post("/api/reset")
    def legacy_reset(body: dict):
        return _legacy("POST", "/api/reset", {}, body)

    @app.get("/api/observation")
    def legacy_observation(images: str = "0"):
        return _legacy("GET", "/api/observation", {"images": [images]}, {})

    @app.post("/api/step")
    def legacy_step(body: dict):
        return _legacy("POST", "/api/step", {}, body)

    @app.post("/api/baseline-step")
    def legacy_baseline(body: dict):
        return _legacy("POST", "/api/baseline-step", {}, body)

    @app.get("/api/result")
    def legacy_result():
        return _legacy("GET", "/api/result", {}, {})

    @app.get("/api/frame.jpg")
    def legacy_frame():
        return _legacy("GET", "/api/frame.jpg", {}, {})

    # -- background jobs / library / datasets / deployments ------------------
    from .jobs import JobManager
    from .aux_model import AuxModel
    jobs = JobManager(lab_root)

    @app.post("/api/v1/jobs")
    def create_job(body: JobBody):
        try:
            return jobs.create(body.kind, body.params)
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @app.get("/api/v1/jobs")
    def list_jobs():
        return {"jobs": jobs.list()}

    @app.get("/api/v1/jobs/{jid}")
    def get_job(jid: str):
        j = jobs.get(jid)
        if not j:
            raise HTTPException(404, "unknown job")
        return j

    @app.get("/api/v1/jobs/{jid}/logs")
    def job_logs(jid: str, tail: int = 200):
        return {"lines": jobs.logs(jid, tail)}

    @app.post("/api/v1/jobs/{jid}/stop")
    def stop_job(jid: str):
        try:
            return jobs.stop(jid)
        except KeyError:
            raise HTTPException(404, "unknown job")

    @app.get("/api/v1/projects")
    def list_projects():
        out = []
        for d in sorted(store.root.iterdir()):
            cur = d / "current.json"
            if cur.is_file():
                cfg = json.loads(cur.read_text(encoding="utf-8"))
                out.append({"id": d.name, "version": cfg.get("version"),
                            "robots": [r["id"] for r in cfg.get("robots", [])],
                            "task": cfg.get("task", {}).get("id")})
        return {"projects": out}

    @app.get("/api/v1/datasets")
    def list_datasets():
        out = []
        staging = lab_root / "datasets" / "staging" / "manifest.json"
        if staging.is_file():
            m = json.loads(staging.read_text(encoding="utf-8"))
            out.append({"id": "staging", "kind": "staging", "episodes": m["summary"]["episodes"],
                        "frames": m["summary"]["frames"], "successes": m["summary"]["successes"],
                        "control_source": m.get("control_source")})
        collections = lab_root / "datasets" / "collections"
        if collections.is_dir():
            for d in sorted(collections.iterdir()):
                mf = d / "manifest.json"
                if mf.is_file():
                    m = json.loads(mf.read_text(encoding="utf-8"))
                    out.append({"id": "collection__" + d.name, "collection_id": d.name,
                                "kind": "staging", **m["summary"]})
        ler = lab_root / "datasets" / "lerobot"
        if ler.is_dir():
            for d in sorted(ler.iterdir()):
                if (d / "meta" / "info.json").is_file():
                    info = json.loads((d / "meta" / "info.json").read_text(encoding="utf-8"))
                    out.append({"id": d.name, "kind": "lerobot", "episodes": info.get("total_episodes"),
                                "frames": info.get("total_frames"), "fps": info.get("fps")})
        return {"datasets": out}

    @app.get("/api/v1/datasets/{did}")
    def get_dataset(did: str):
        valid_id(did)
        if did == "staging":
            p = lab_root / "datasets" / "staging" / "manifest.json"
        elif did.startswith("collection__"):
            p = lab_root / "datasets" / "collections" / valid_id(did[len("collection__"):]) / "manifest.json"
        else:
            p = lab_root / "datasets" / "lerobot" / did / "meta" / "info.json"
        if not p.is_file():
            raise HTTPException(404, "unknown dataset")
        return json.loads(p.read_text(encoding="utf-8"))

    @app.get("/api/v1/datasets/{did}/preview-frames")
    def dataset_preview_frames(did: str, episode: int = 0):
        import base64 as _b64_mod
        import numpy as np
        valid_id(did)
        if did == "staging":
            col_dir = lab_root / "datasets" / "staging"
        else:
            raw_id = did[len("collection__"):] if did.startswith("collection__") else did
            col_dir = lab_root / "datasets" / "collections" / valid_id(raw_id)
        npz_path = col_dir / "episodes" / f"ep{int(episode):03d}.npz"
        mf_path = col_dir / "manifest.json"
        if not npz_path.is_file():
            return {"dataset": did, "cameras": [], "frames": []}
        cams = ["front", "wrist"]
        if mf_path.is_file():
            mf = json.loads(mf_path.read_text(encoding="utf-8"))
            if isinstance(mf.get("cameras"), list) and mf["cameras"]:
                cams = list(mf["cameras"])
        with np.load(npz_path, allow_pickle=True) as z:
            n = int(len(z["frame_time"]))
            if n == 0:
                return {"dataset": did, "cameras": cams, "frames": []}
            picks = [(0, "起始"), (n // 2, "中段"), (n - 1, "结束")]
            seen = set()
            frames = []
            for idx, label in picks:
                if idx in seen:
                    continue
                seen.add(idx)
                item = {
                    "index": int(idx),
                    "label": label,
                    "sim_time": round(float(z["frame_time"][idx]), 3),
                    "action": [round(float(v), 4) for v in z["action"][idx].tolist()],
                    "images": {},
                }
                for c in cams:
                    k = f"jpeg_{c}"
                    if k in z:
                        raw = bytes(z[k][idx])
                        item["images"][c] = _b64_mod.b64encode(raw).decode("ascii")
                frames.append(item)
        return {"dataset": did, "cameras": cams, "frames": frames}

    @app.get("/api/v1/models")
    def list_models():
        models_dir = lab_root / "datasets" / "models"
        reports_dir = lab_root / "datasets" / "reports"
        entries: dict[str, dict] = {}
        if models_dir.is_dir():
            for p in sorted(models_dir.iterdir()):
                if p.is_dir() and (p / "config.json").is_file() and (p / "so100_stats.json").is_file():
                    mid = p.name
                    try:
                        st = json.loads((p / "so100_stats.json").read_text(encoding="utf-8"))
                    except Exception:
                        st = {}
                    from .smolvla_infer import is_smolvla_loaded
                    rec = entries.setdefault(mid, {
                        "id": mid, "has_pt": True, "has_npz": False,
                        "camera_keys": list(st.get("camera_keys") or ["front", "wrist"]),
                        "runnable": True, "arch": str(st.get("arch") or "smolvla"),
                        "language": True,
                        "loaded": is_smolvla_loaded(lab_root, mid),
                        "default_instruction": st.get("task_instruction") or "",
                        "instructions": list(st.get("instructions") or []),
                    })
                elif p.suffix in {".pt", ".npz"}:
                    mid = p.stem
                    rec = entries.setdefault(mid, {
                        "id": mid, "has_pt": False, "has_npz": False,
                        "camera_keys": ["front"], "runnable": False,
                        "language": False,
                    })
                    if p.suffix == ".pt":
                        rec["has_pt"] = True
                    if p.suffix == ".npz":
                        rec["has_npz"] = True
                        rec["runnable"] = True
                        try:
                            import numpy as np
                            with np.load(p, allow_pickle=False) as d:
                                if "camera_keys" in d:
                                    rec["camera_keys"] = json.loads(str(d["camera_keys"].item()))
                        except Exception:
                            pass
                else:
                    continue
                rp = reports_dir / f"train_{mid}.json"
                if rp.is_file():
                    try:
                        rep = json.loads(rp.read_text(encoding="utf-8"))
                        for fld in ("type", "camera_keys", "episodes", "frames", "val_mse", "gen_mse", "dataset"):
                            if fld in rep:
                                rec[fld] = rep[fld]
                    except Exception:
                        pass
        return {"models": sorted(entries.values(), key=lambda x: (not x["runnable"], x["id"]))}

    @app.get("/api/v1/asset-library")
    def asset_library():
        items = [{"id": k, "name": v, "category": "object", "source": "builtin"} for k, v in
                 {"standard": "标准杯", "tall": "高杯", "wide": "宽口杯"}.items()]
        imp = lab_root / "assets" / "imported"
        if imp.is_dir():
            for d in sorted(imp.iterdir()):
                mf = d / "manifest.json"
                if mf.is_file():
                    m = json.loads(mf.read_text(encoding="utf-8"))
                    items.append({"id": f"imported/{d.name}", "name": d.name, "category": "object",
                                  "kind": m.get("kind", "task_object"), "role": m.get("role", "dynamic"),
                                  "calibrated": m.get("calibrated", True),
                                  "source": m.get("source"), "generation_id": m.get("generation_id"),
                                  "height_m": m.get("height_m"), "width_m": m.get("width_m"),
                                  "depth_m": m.get("depth_m"), "collision_proxy": m.get("collision_proxy"),
                                  "concave_detected": bool(m.get("concave_detected")),
                                  "compound_boxes": (m.get("compound_proxy") or {}).get("box_count")})
        rob = lab_root / "assets" / "robot"
        if rob.is_dir():
            items.append({"id": "robot/lux3d", "name": "lux3d 机械臂外观", "category": "robot",
                          "source": "lux3d", "generation_id": "5025807"})
        return {"assets": items}

    def _pkg_manifest(name: str):
        mf = lab_root / "deploy" / name / "manifest.json"
        if not mf.is_file():
            raise HTTPException(404, "unknown deployment package")
        return json.loads(mf.read_text(encoding="utf-8"))

    @app.get("/api/v1/deployments")
    def list_deployments():
        out = []
        dep = lab_root / "deploy"
        if dep.is_dir():
            for d in sorted(dep.iterdir()):
                mf = d / "manifest.json"
                if mf.is_file():
                    m = json.loads(mf.read_text(encoding="utf-8"))
                    kind = m.get("kind") or ("visual" if m.get("architecture") else "baseline")
                    out.append({"id": d.name, "package_id": m.get("package_id") or d.name, "kind": kind,
                                "robot": m.get("robot"), "created": m.get("created") or m.get("created_utc")})
        return {"deployments": out}

    @app.get("/api/v1/deployments/{name}")
    def deployment_detail(name: str):
        return _pkg_manifest(name)

    @app.get("/api/v1/deployments/{name}/compat")
    def deployment_compat(name: str, robot_id: str | None = None, project: str = "default"):
        from .compat import check_contract
        mf = _pkg_manifest(name)
        contract = mf.get("contract")
        if not contract:
            return {"compatible": None, "mismatches": ["baseline package has no visual model contract"],
                    "robot": robot_id}
        if contract.get("joint_names"):
            mapped = contract
        else:  # legacy simplified contract
            mapped = {"joint_names": contract["state_names"][:5],
                      "action_mode": contract.get("action_mode"), "action_dim": len(contract["action_names"]),
                      "action_names": contract["action_names"], "camera_keys": [contract["image_key"]],
                      "fps": contract.get("fps"), "privileged": contract.get("privileged", False)}
        cfg = store.load(project)
        return check_contract(mapped, cfg, robot_id)

    # -- scene assets --------------------------------------------------------
    import base64 as _b64
    import tempfile as _tmp
    from . import scene_import
    from .schema import SceneInstance

    @app.post("/api/v1/scene-assets/import")
    def scene_import_ep(body: SceneImportBody):
        tmp = None
        try:
            valid_id(body.asset_id)
            if len(body.data_b64) > (scene_import.MAX_SCENE * 4 // 3 + 8):
                raise ValueError("Scene exceeds 256 MiB")
            data = _b64.b64decode(body.data_b64, validate=True)
            with _tmp.NamedTemporaryFile(suffix=Path(body.filename).suffix, delete=False) as f:
                f.write(data)
                tmp = Path(f.name)
            mf = scene_import.import_scene(tmp, lab_root, body.asset_id, 1,
                                           up_axis=body.up_axis, unit_scale=body.unit_scale,
                                           calibrated=body.calibrated)
            return json.loads(mf.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        finally:
            if tmp is not None:
                tmp.unlink(missing_ok=True)

    @app.get("/api/v1/scene-assets")
    def scene_list():
        out = []
        root = lab_root / "assets" / "scenes"
        if root.is_dir():
            for d in sorted(root.iterdir()):
                for rev in sorted(d.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 0):
                    mf = rev / "manifest.json"
                    if mf.is_file():
                        out.append(json.loads(mf.read_text(encoding="utf-8")))
        return {"scenes": out}

    @app.get("/api/v1/scene-assets/{asset_id}")
    def scene_detail(asset_id: str, revision: int | None = None):
        valid_id(asset_id)
        if revision is not None:
            return load_scene(lab_root, asset_id, revision)[1]
        d = lab_root / "assets" / "scenes" / asset_id
        revs = sorted([p for p in d.iterdir() if p.name.isdigit()], key=lambda p: int(p.name)) if d.is_dir() else []
        if not revs:
            raise HTTPException(404, "unknown scene asset")
        return json.loads((revs[-1] / "manifest.json").read_text(encoding="utf-8"))

    @app.post("/api/v1/scene-assets/{asset_id}/validate")
    def scene_validate(asset_id: str, revision: int | None = None):
        mf = scene_detail(asset_id, revision)
        try:
            load_scene(lab_root, asset_id, mf["revision"], physics=True)
            return {"valid": True, "issues": [], "status": mf.get("status")}
        except ValueError as exc:
            return {"valid": False, "issues": [str(exc)], "status": mf.get("status")}

    @app.get("/api/v1/scene-assets/{asset_id}/revisions/{revision}/visual.obj")
    def scene_visual(asset_id: str, revision: int):
        d, mf = load_scene(lab_root, asset_id, revision, physics=True)
        return FileResponse(contained_file(d, mf["visual_asset"]), media_type="text/plain")

    @app.get("/api/v1/scene-assets/{asset_id}/revisions/{revision}/preview")
    def scene_preview(asset_id: str, revision: int):
        d, mf = load_scene(lab_root, asset_id, revision)
        name = mf.get("preview_asset")
        if not name or name not in mf.get("files", {}):
            raise HTTPException(404, "Scene has no preview image")
        return FileResponse(contained_file(d, name), media_type="image/jpeg")

    @app.get("/api/v1/scene-assets/{asset_id}/revisions/{revision}/point-cloud")
    def scene_point_cloud(asset_id: str, revision: int, max_points: int = 24000):
        import base64
        import numpy as np
        from .gaussian import load_gaussians
        try:
            d, mf = load_scene(lab_root, asset_id, revision, physics=False)
        except ValueError as exc:
            raise HTTPException(404, str(exc))
        if mf.get("ply_kind") != "gaussian":
            raise HTTPException(400, "Scene is not a gaussian point cloud")
        ply_rel = next((k for k in mf.get("files", {}) if k.lower().endswith(".ply")), None)
        if not ply_rel:
            raise HTTPException(404, "Gaussian PLY file not found")
        g = load_gaussians(contained_file(d, ply_rel))
        means = g["means"]
        op = g["opacity"]
        rgb = g["rgb"]
        sc = g["scales"].max(axis=1)
        mask = (op >= 0.25) & (sc <= 0.08)
        if int(mask.sum()) < 100:
            mask = op >= 0.10
        idx = np.flatnonzero(mask)
        limit = max(500, min(int(max_points), 60000))
        if len(idx) > limit:
            order = np.argsort(-op[idx])[:limit]
            idx = idx[order]
        pts = np.ascontiguousarray(means[idx], dtype=np.float32)
        cols = np.ascontiguousarray(np.clip(rgb[idx] * 255.0, 0, 255), dtype=np.uint8)
        floor_p5 = float(np.percentile(pts[:, 2], 5)) if len(pts) else 0.0
        return {
            "asset_id": asset_id,
            "revision": revision,
            "count": int(len(pts)),
            "auto_floor_offset_z": round(-floor_p5 - 0.06, 4),
            "bbox_min": pts.min(axis=0).tolist() if len(pts) else [0, 0, 0],
            "bbox_max": pts.max(axis=0).tolist() if len(pts) else [0, 0, 0],
            "positions_f32_b64": base64.b64encode(pts.tobytes()).decode("ascii"),
            "colors_u8_b64": base64.b64encode(cols.tobytes()).decode("ascii"),
        }

    @app.put("/api/v1/projects/{pid}/scene")
    def commit_scene(pid: str, body: SceneCommitBody):
        def mutate(data):
            data["scene"]["environment"] = body.environment
            data["scene"]["colliders"] = body.colliders
            if body.asset_nodes is not None:
                data["scene"]["asset_nodes"] = body.asset_nodes
            if body.use_default_table is not None:
                data["scene"]["use_default_table"] = body.use_default_table
            if body.work_surface_z is not None:
                data["scene"]["work_surface_z"] = body.work_surface_z
        saved = _commit_scene(pid, mutate, body.expected_version)
        return {"version": saved.version,
                "environment": saved.scene.environment.model_dump() if saved.scene.environment else None,
                "colliders": [c.model_dump() for c in saved.scene.colliders],
                "asset_nodes": [n.model_dump() for n in saved.scene.asset_nodes],
                "use_default_table": saved.scene.use_default_table,
                "work_surface_z": saved.scene.work_surface_z}

    @app.put("/api/v1/projects/{pid}/scene/instance")
    def set_scene_instance(pid: str, body: SceneInstanceBody):
        def mutate(data):
            env = data["scene"].get("environment")
            if env is None:
                raise HTTPException(400, "no environment bound; bind a scene first")
            env = dict(env)
            env.update({"position": body.position, "quaternion_xyzw": body.quaternion_xyzw,
                        "uniform_scale": body.uniform_scale})
            data["scene"]["environment"] = env
            if body.use_default_table is not None:
                data["scene"]["use_default_table"] = body.use_default_table
        saved = _commit_scene(pid, mutate, body.expected_version)
        return {"version": saved.version, "environment": saved.scene.environment.model_dump()}

    @app.get("/api/v1/projects/{pid}/scene")
    def get_scene(pid: str):
        cfg = store.load(pid) if store.exists(pid) else store.load("default")
        return {"environment": cfg.scene.environment.model_dump() if cfg.scene.environment else None,
                "colliders": [c.model_dump() for c in cfg.scene.colliders],
                "asset_nodes": [n.model_dump() for n in cfg.scene.asset_nodes],
                "use_default_table": cfg.scene.use_default_table,
                "work_surface_z": cfg.scene.work_surface_z, "version": cfg.version}

    @app.put("/api/v1/projects/{pid}/scene/colliders")
    def set_colliders(pid: str, body: CollidersBody):
        def mutate(data):
            data["scene"]["colliders"] = body.colliders
        saved = _commit_scene(pid, mutate, body.expected_version)
        return {"version": saved.version, "colliders": [c.model_dump() for c in saved.scene.colliders]}

    @app.post("/api/v1/projects/{pid}/bind-scene")
    def bind_scene(pid: str, body: BindSceneBody):
        valid_id(pid)
        load_scene(lab_root, body.asset_id, body.revision, physics=True)
        cfg = store.load(pid) if store.exists(pid) else store.load("default")
        cfg = cfg.model_copy(update={"project_id": pid})
        cfg.scene.environment = SceneInstance(asset_id=body.asset_id, revision=body.revision,
                                              position=body.position, quaternion_xyzw=body.quaternion_xyzw,
                                              uniform_scale=body.uniform_scale)
        cfg.scene.use_default_table = body.use_default_table
        issues = validate_config(cfg, lab_root)
        if issues:
            raise HTTPException(400, {"issues": issues})
        return store.save(cfg).model_dump()

    @app.get("/api/v1/compat")
    def compat_check(robot_id: str | None = None, model: str = "state", project: str = "default"):
        from .compat import check_contract
        cfg = store.load(project)
        base = {"joint_names": ["waist", "shoulder", "elbow", "wrist_pitch", "wrist_yaw"],
                "action_mode": "cartesian_delta_ik", "action_dim": 4,
                "action_names": ["dx", "dy", "dz", "grip"], "fps": 20.0,
                "camera_keys": ["front", "wrist"], "privileged": False}
        if model == "priv":
            base["privileged"] = True
        elif model == "visual":
            base["camera_keys"] = ["front"]
        elif model != "state":
            raise HTTPException(400, "model must be state|priv|visual")
        return check_contract(base, cfg, robot_id)

    @app.post("/api/v1/compat")
    def compat_contract(body: dict):
        from .compat import check_contract
        cfg = (ProjectConfig.model_validate(body["config"]) if "config" in body
               else store.load(body.get("project", "default")))
        return check_contract(body["contract"], cfg, body.get("robot_id"))

    from .world_provider import GenerationJobStore
    gen_store = GenerationJobStore(lab_root / "datasets" / "generation_jobs", lab_root=lab_root)

    from .generation.api import register_generation_api
    generation_ctx = register_generation_api(app, lab_root, store=store, world_store=gen_store,
                                             commit_scene=_commit_scene)

    @app.get("/api/v1/lux3d/status")
    def lux3d_status():
        key = os.environ.get("AHOLO_API_KEY") or os.environ.get("LUX3D_CN_API_KEY") or os.environ.get("LUX3D_API_KEY")
        return {"provider": "lux3d", "configured": bool(key),
                "note": "set AHOLO_API_KEY in the backend environment to enable single-asset generation"}

    @app.post("/api/v1/lux3d/generate")
    def lux3d_generate(body: Lux3dBody):
        key = os.environ.get("AHOLO_API_KEY") or os.environ.get("LUX3D_CN_API_KEY") or os.environ.get("LUX3D_API_KEY")
        if not key:
            raise HTTPException(409, "AHOLO_API_KEY not configured; object generation is unavailable (local import still works)")
        slug = body.name.strip().lower().replace(" ", "-")
        try:
            return jobs.create("lux3d_generate", {
                "slug": slug, "prompt": body.prompt, "height_m": body.height_m,
                "mass_kg": body.mass_kg, "radius_m": body.radius_m,
                "asset_kind": body.kind, "role": body.role, "width_m": body.width_m, "depth_m": body.depth_m,
                "provider": "lux3d", "kind": "object_generation"})
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @app.get("/api/v1/generation-jobs")
    def generation_status():
        return {"provider": "aholo-world", "aux": AuxModel().status(),
                "configured": gen_store.provider.configured(),
                "note": "set AHOLO_API_KEY / AUX_MODEL_* in the backend environment to enable live generation",
                "jobs": gen_store.list()}

    @app.post("/api/v1/generation-jobs")
    def generation_submit(body: GenerationBody):
        return gen_store.submit(body.kind, body.params, background=True)

    @app.get("/api/v1/generation-jobs/{jid}")
    def generation_get(jid: str):
        j = gen_store.get(jid)
        if not j:
            raise HTTPException(404, "unknown generation job")
        return j

    @app.post("/api/v1/generation-jobs/{jid}/poll")
    def generation_poll(jid: str):
        j = gen_store.poll(jid, background=True)
        if not j:
            raise HTTPException(404, "unknown generation job")
        return j

    @app.get("/api/v1/projects/{pid}/robot.urdf")
    def export_robot_urdf(pid: str, robot_id: str | None = None):
        from .robot_builder import robot_to_urdf
        try:
            valid_id(pid)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        cfg = store.load(pid) if store.exists(pid) else store.load("default")
        r = cfg.robots[0] if robot_id is None else next((x for x in cfg.robots if x.id == robot_id), None)
        if r is None:
            raise HTTPException(404, f"unknown robot {robot_id!r}")
        urdf = robot_to_urdf(r, mesh_uri_prefix=r.mesh_dir)
        return Response(urdf, media_type="application/xml",
                        headers={"Content-Disposition": f'attachment; filename="{pid}-{r.id}.urdf"'})

    @app.post("/api/v1/robots/assemble-parts")
    def assemble_robot_parts_ep(body: AssemblePartsBody):
        from .robot_builder import assemble_robot_from_parts
        try:
            valid_id(body.project_id)
            slug = valid_id(body.parts_slug)
            parts_root = (lab_root / "assets" / "imported" / slug / "parts").resolve()
            if not parts_root.is_dir():
                raise HTTPException(404, f"parts directory missing for {slug!r}; run part-split first")
            resolved_parts = {slot: contained_file(parts_root, rel) for slot, rel in body.mapping.items()}
            cfg = store.load(body.project_id) if store.exists(body.project_id) else store.load("default")
            base_robot = cfg.robots[0]
            new_robot = assemble_robot_from_parts(base_robot, resolved_parts, lab_root, body.robot_id)
            updated_robots = ([new_robot] if body.replace_primary
                              else [*[r for r in cfg.robots if r.id != new_robot.id], new_robot])
            candidate = cfg.model_copy(update={"project_id": body.project_id, "robots": updated_robots, "validation_status": "candidate"})
            issues = validate_config(candidate, lab_root)
            if issues:
                raise HTTPException(400, {"issues": issues})
            saved = store.save(candidate.model_copy(update={"validation_status": "validated"}))
            return {"project_id": saved.project_id, "version": saved.version, "robot": new_robot.model_dump()}
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(400, str(exc))

    @app.post("/api/v1/hackathon-demo/setup")
    def hackathon_demo_setup(body: DemoSetupBody):
        from .hackathon_demo import setup_hackathon_demo
        try:
            valid_id(body.project_id)
            return setup_hackathon_demo(
                lab_root, store,
                project_id=body.project_id,
                with_world_background=body.with_world_background,
                support_plane_z=body.support_plane_z,
            )
        except Exception as exc:
            raise HTTPException(400, str(exc))

    @app.post("/api/v1/so100-prototype/setup")
    def so100_prototype_setup(body: So100SetupBody):
        from .so100_prototype import setup_so100_project
        try:
            valid_id(body.project_id)
            return setup_so100_project(
                lab_root, store,
                project_id=body.project_id,
                live_generate=body.live_generate,
                with_world_background=body.with_world_background,
            )
        except Exception as exc:
            raise HTTPException(400, str(exc))

    @app.on_event("shutdown")
    def _shutdown():
        manager.close_all()
        if legacy_lab is not None:
            legacy_worker.submit(legacy_lab.env.close).result()
        legacy_worker.shutdown(wait=True)

    return app


app = create_app()
