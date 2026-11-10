"""FastAPI backend: versioned v1 API + legacy /api/* compatibility + WebSocket."""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import parse_qs

import mujoco
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import gymnasium

from . import sessions as sess_mod
from .engine import SCHEMA_VERSION, UnsupportedMode
from .schema import ProjectConfig
from .store import ConfigStore, validate_config

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


def create_app(lab_root: Path = LAB_ROOT) -> FastAPI:
    app = FastAPI(title="VLA Lab Platform", version=SCHEMA_VERSION)
    store = ConfigStore(lab_root / "projects")
    manager = sess_mod.SessionManager()
    if not store.exists("default"):
        from .make_default_config import build
        store.save(build())

    # Legacy single-threaded lab, created lazily on first legacy request so startup
    # does not open an OpenGL context (a second GLFW context in a background process
    # can segfault; the first session renderer becomes the primary context).
    import server
    import threading
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

    @app.get("/api/v1/cup.obj")
    def cup_mesh(asset: str = "standard"):
        from . import tasks
        try:
            *_, mesh, offset, handle = tasks.asset_params(asset, lab_root)
        except Exception as exc:
            raise HTTPException(400, f"unknown asset {asset}: {exc}")
        return Response(mesh, media_type="text/plain")

    @app.exception_handler(sess_mod.ControlConflict)
    async def _conflict(_req, exc):
        return JSONResponse({"error": str(exc)}, status_code=409)

    @app.exception_handler(UnsupportedMode)
    async def _unsupported(_req, exc):
        return JSONResponse({"error": str(exc)}, status_code=400)

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
            env = PlatformEnv(cfg, lab_root, robot_id=r.id)
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
    def expert_step(sid: str, body: TokenBody):
        s = _session(sid)
        try:
            return s.expert(body.token, epoch=body.epoch)
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
        if did == "staging":
            p = lab_root / "datasets" / "staging" / "manifest.json"
        else:
            p = lab_root / "datasets" / "lerobot" / did / "meta" / "info.json"
        if not p.is_file():
            raise HTTPException(404, "unknown dataset")
        return json.loads(p.read_text(encoding="utf-8"))

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
                                  "source": m.get("source"), "generation_id": m.get("generation_id"),
                                  "height_m": m.get("height_m")})
        rob = lab_root / "assets" / "robot"
        if rob.is_dir():
            items.append({"id": "robot/lux3d", "name": "lux3d 机械臂外观", "category": "robot",
                          "source": "lux3d", "generation_id": "5025807"})
        return {"assets": items}

    @app.get("/api/v1/deployments")
    def list_deployments():
        out = []
        dep = lab_root / "deploy"
        if dep.is_dir():
            for d in sorted(dep.iterdir()):
                mf = d / "manifest.json"
                if mf.is_file():
                    m = json.loads(mf.read_text(encoding="utf-8"))
                    out.append({"id": d.name, "package_id": m.get("package_id"),
                                "robot": m.get("robot"), "created": m.get("created_utc")})
        return {"deployments": out}

    @app.get("/api/v1/compat")
    def compat_check(robot_id: str | None = None, model: str = "state"):
        from .compat import check_contract
        cfg = store.load("default")
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

    @app.get("/api/v1/generation-jobs")
    def generation_status():
        return {"provider": "aholo-lux3d", "aux": AuxModel().status(),
                "note": "set AHOLO_API_KEY / AUX_MODEL_* in the backend environment to enable live generation"}

    @app.on_event("shutdown")
    def _shutdown():
        manager.close_all()
        if legacy_lab is not None:
            legacy_worker.submit(legacy_lab.env.close).result()
        legacy_worker.shutdown(wait=True)

    return app


app = create_app()