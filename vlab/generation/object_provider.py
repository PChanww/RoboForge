"""Lux3D single-object provider + node-granular persistent job store.

Mirrors the World GenerationJobStore semantics (intent-before-submit, unknown
submit outcomes never blindly resubmitted, known remote_id reused on retry) but
for single assets, with explicit nodes: submit -> remote -> download -> import ->
validate. A local receipt cannot guarantee remote exactly-once billing across
crashes; that limit is declared, not hidden. MockLux3DProvider is a test double,
NOT an online capability.
"""
from __future__ import annotations

import copy
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from ..scene_assets import valid_id

NODES = ["submit", "remote", "download", "import", "validate"]
TERMINAL = {"succeeded", "failed"}


def _now():
    return datetime.now(timezone.utc).isoformat()


class Lux3DProvider:
    def __init__(self, key_provider=None, *, timeout_ms=60000):
        self.key = (os.environ.get("AHOLO_API_KEY") or os.environ.get("LUX3D_CN_API_KEY")
                    or os.environ.get("LUX3D_API_KEY"))
        self._client = None
        self._client_key = None
        self._key_provider = key_provider
        self._timeout_ms = timeout_ms
        self._client_lock = threading.Lock()

    def _key(self):
        return self._key_provider() if self._key_provider is not None else self.key

    def configured(self):
        return bool(self._key())

    def _sdk(self):
        with self._client_lock:
            key = self._key()
            if not key:
                raise NotImplementedError("Set AHOLO_API_KEY to enable Lux3D generation")
            if self._client is None or key != self._client_key:
                try:
                    from manycore.aholo_sdk_core import AholoClientConfig
                    from manycore.aholo_sdk_lux3d import create_lux3d_client
                except ImportError as exc:
                    raise NotImplementedError("install manycore-aholo-sdk-lux3d==1.9.0") from exc
                self._client = create_lux3d_client(AholoClientConfig(
                    api_key=key, region="cn", timeout_ms=self._timeout_ms))
                self._client_key = key
            return self._client

    def submit(self, prompt, payload):
        c = self._sdk()
        payload = payload or {}
        mode = payload.get("generation_mode") or ("img_to_3d" if payload.get("img") and not payload.get("use_text_with_img") else "text_to_3d")
        if mode == "img_to_3d":
            img = payload.get("img")
            imgs = payload.get("imgs")
            kwargs = {k: payload[k] for k in ("version", "face_count", "output_format", "enable_pbr", "ai_predict_size")
                      if k in payload}
            return str(c.img_to_3d.create(img=img, imgs=imgs, **kwargs))
        kwargs = {k: payload[k] for k in ("version", "style", "img", "face_count", "output_format", "enable_pbr", "ai_predict_size")
                  if k in payload}
        return str(c.text_to_3d.create(prompt=prompt, **kwargs))

    def submit_part_split(self, glb_url: str) -> str:
        c = self._sdk()
        return str(c.part_split.create(glb_url=glb_url))

    def get_status(self, task_id):
        c = self._sdk()
        r = c.tasks.retrieve(task_id)
        status = r.get("status")
        if status == 3:
            state = "succeeded"
        elif status in (4, 6):
            state = "failed"
        else:
            state = "remote_processing"
        return {"state": state, "provider_status": status, "remote_id": task_id,
                "outputs": r.get("outputs") or [],
                "error": f"Lux3D status {status}" if state == "failed" else None}

    def list_remote_tasks(self, *, page: int = 1, page_size: int = 20, status: int | None = None) -> dict:
        c = self._sdk()
        res = c.tasks.list(page=page, page_size=page_size, status=status)
        items = []
        for it in (res.get("items") or []):
            st_code = it.get("status")
            st = "succeeded" if st_code == 3 else ("failed" if st_code in (4, 6) else "remote_processing")
            items.append({
                "remote_id": str(it.get("taskId")),
                "provider": "lux3d",
                "task_type": it.get("taskType"),
                "version": it.get("version"),
                "state": st,
                "provider_status": st_code,
                "created_ms": it.get("created"),
                "updated_ms": it.get("lastModified"),
                "consumed_credits": it.get("consumedCredits"),
            })
        return {
            "provider": "lux3d",
            "total": int(res.get("total") or len(items)),
            "page": int(res.get("page") or page),
            "page_size": int(res.get("pageSize") or page_size),
            "items": items,
        }

    def download(self, outputs, directory):
        from asset_pipeline.import_asset import download_mesh
        return download_mesh(outputs, directory)


class MockLux3DProvider:
    """Controlled test double (NOT an online capability)."""

    def __init__(self, behavior="success"):
        self.behavior = behavior
        self.submits = 0
        self.calls = []

    def configured(self):
        return True

    def submit(self, prompt, payload):
        self.submits += 1
        self.calls.append(("submit", prompt, copy.deepcopy(payload)))
        if self.behavior == "submit_timeout":
            raise TimeoutError("simulated submit timeout")
        return f"mock-{self.submits}"

    def submit_part_split(self, glb_url: str) -> str:
        self.submits += 1
        self.calls.append(("submit_part_split", glb_url))
        if self.behavior == "submit_timeout":
            raise TimeoutError("simulated part_split timeout")
        return f"mock-split-{self.submits}"

    def get_status(self, task_id):
        if self.behavior == "failed":
            return {"state": "failed", "remote_id": task_id, "outputs": [], "error": "mock failure"}
        return {"state": "succeeded", "remote_id": task_id, "outputs": [], "error": None}

    def list_remote_tasks(self, *, page: int = 1, page_size: int = 20, status: int | None = None) -> dict:
        return {
            "provider": "lux3d",
            "total": 1,
            "page": page,
            "page_size": page_size,
            "items": [{
                "remote_id": "mock-1",
                "provider": "lux3d",
                "task_type": "text-to-3d",
                "version": "g1-turbo",
                "state": "failed" if self.behavior == "failed" else "succeeded",
                "provider_status": 4 if self.behavior == "failed" else 3,
                "created_ms": None,
                "updated_ms": None,
                "consumed_credits": 0,
            }],
        }

    def download(self, outputs, directory):
        import struct
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "result.glb"
        # Minimal valid glTF2 binary: a non-flat tetrahedron with an embedded buffer.
        verts = struct.pack("<12f", 0, 0, 0, 0.05, 0, 0, 0, 0.05, 0, 0, 0, 0.05)
        idx = struct.pack("<12H", 0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3)
        buf = verts + idx
        gltf = {"asset": {"version": "2.0"},
                "buffers": [{"byteLength": len(buf)}],
                "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(verts)},
                                {"buffer": 0, "byteOffset": len(verts), "byteLength": len(idx)}],
                "accessors": [{"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3",
                               "min": [0, 0, 0], "max": [0.05, 0.05, 0.05]},
                              {"bufferView": 1, "componentType": 5123, "count": 12, "type": "SCALAR"}],
                "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
                "nodes": [{"mesh": 0}], "scenes": [{"nodes": [0]}], "scene": 0}
        js = json.dumps(gltf).encode()
        js += b" " * ((-len(js)) % 4)
        buf += b"\x00" * ((-len(buf)) % 4)
        total = 12 + 8 + len(js) + 8 + len(buf)
        header = struct.pack("<4sII", b"glTF", 2, total)
        data = header + struct.pack("<I4s", len(js), b"JSON") + js + struct.pack("<I4s", len(buf), b"BIN\x00") + buf
        partial = path.with_suffix(".glb.part")
        try:
            partial.write_bytes(data)
            partial.replace(path)
        finally:
            partial.unlink(missing_ok=True)
        return path


class ObjectJobStore:
    def __init__(self, root: Path, provider=None, *, lab_root=None):
        self.dir = Path(root)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "object_jobs.json"
        self.provider = provider or Lux3DProvider()
        self.lab_root = Path(lab_root) if lab_root else self.dir / "registered"
        self._lock = threading.RLock()
        self._active = set()
        self._jobs = {j["id"]: j for j in json.loads(self.path.read_text(encoding="utf-8"))} if self.path.is_file() else {}
        for j in self._jobs.values():
            if j["state"] == "submitting":
                j.update(state="submit_unknown", node="submit",
                         error="Submit interrupted; remote id unknown. Check provider history before resubmitting.")
        self._save()

    def _save(self):
        with self._lock:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(list(self._jobs.values()), ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(self.path)

    def _commit(self, job):
        with self._lock:
            self._jobs[job["id"]] = copy.deepcopy(job)
            self._save()

    def list(self):
        with self._lock:
            return copy.deepcopy(sorted(self._jobs.values(), key=lambda j: j["created"], reverse=True))

    def get(self, jid):
        with self._lock:
            return copy.deepcopy(self._jobs.get(jid))

    def find_by_request(self, client_request_id):
        with self._lock:
            return copy.deepcopy(self._find_by_request_locked(client_request_id))

    def _find_by_request_locked(self, client_request_id):
        if not client_request_id:
            return None
        for j in self._jobs.values():
            if j.get("client_request_id") == client_request_id:
                return j
        return None

    @staticmethod
    def _binding_signature(params):
        keys = (
            "prompt", "payload", "kind", "role",
            "height_m", "width_m", "depth_m", "radius_m",
            "mass_kg", "friction", "collision_mode",
            "spec_id", "spec_revision", "spec_hash", "request_hash",
            "provider", "project_id", "config_version",
        )
        return {k: params.get(k) for k in keys if params.get(k) is not None}

    def _allocate_unique_slug_locked(self, requested_slug: str, jid: str) -> str:
        used = {j.get("slug") for j in self._jobs.values()}
        imported_root = self.lab_root / "assets" / "imported"
        if requested_slug not in used and not (imported_root / requested_slug).exists():
            return requested_slug
        base = requested_slug[:48].rstrip("-") or "asset"
        candidate = valid_id(f"{base}-{jid[:6]}")
        idx = 2
        while candidate in used or (imported_root / candidate).exists():
            candidate = valid_id(f"{base[:42]}-{jid[:6]}-{idx}")
            idx += 1
        return candidate

    def submit(self, params, *, background=False):
        """params: prompt, payload, slug, kind, role, dims, mass, friction, spec binding,
        client_request_id, provider, project_id, config_version. Persists intent first."""
        if not isinstance(params.get("prompt"), str) or not params["prompt"].strip():
            raise ValueError("Object generation requires a prompt")
        requested_slug = valid_id(params["slug"])
        crid = params.get("client_request_id")
        with self._lock:
            existing = self._find_by_request_locked(crid)
            if existing is not None:
                if self._binding_signature(params) != self._binding_signature(existing.get("params", {})):
                    raise ValueError(f"client_request_id {crid!r} already bound to a different payload")
                return copy.deepcopy(existing)  # atomic idempotent reuse
            jid = uuid.uuid4().hex[:12]
            slug = self._allocate_unique_slug_locked(requested_slug, jid)
            stored_params = copy.deepcopy(params)
            stored_params["requested_slug"] = requested_slug
            stored_params["slug"] = slug
            job = {"id": jid, "kind": "object_generation", "provider": "lux3d",
                   "params": stored_params, "slug": slug, "requested_slug": requested_slug,
                   "client_request_id": crid,
                   "state": "submitting", "node": "submit", "remote_id": None,
                   "spec_id": params.get("spec_id"), "spec_revision": params.get("spec_revision"),
                   "spec_hash": params.get("spec_hash"), "request_hash": params.get("request_hash"),
                   "project_id": params.get("project_id"),
                   "config_version": params.get("config_version"),
                   "created": _now(), "error": None, "retries": 0,
                   "nodes": {n: "pending" for n in NODES}}
            if not params.get("remote_id") and not self.provider.configured():
                job.update(state="unconfigured", error="Set AHOLO_API_KEY to enable Lux3D generation")
            self._jobs[jid] = job
            self._save()  # intent BEFORE remote submit
        if job["state"] == "unconfigured":
            return self.get(jid)
        if background:
            threading.Thread(target=self._submit, args=(jid,), daemon=True).start()
        else:
            self._submit(jid)
        return self.get(jid)

    def _submit(self, jid):
        job = self.get(jid)
        p = job["params"]
        try:
            if p.get("remote_id"):
                rid = str(p["remote_id"])
            else:
                rid = self.provider.submit(p["prompt"], p.get("payload", {}))
            job.update(remote_id=rid, state="remote_processing", node="remote")
            job["nodes"]["submit"] = "done"
        except TimeoutError as exc:
            job.update(state="submit_unknown", node="submit", error=str(exc))
        except NotImplementedError as exc:
            job.update(state="unconfigured", node="submit", error=str(exc))
        except Exception as exc:
            job.update(state="failed", node="submit", error=str(exc))
        self._commit(job)

    def attach_remote(self, jid: str, remote_id: str, *, provider: str | None = None) -> dict:
        """Bind a recovered remote_id onto an existing submit_unknown/unconfigured job without resubmitting."""
        rid = valid_id(str(remote_id or "").strip())
        with self._lock:
            job = self._jobs.get(jid)
            if job is None:
                raise KeyError(jid)
            if provider is not None and provider != job.get("provider", "lux3d"):
                raise ValueError(f"provider mismatch: job uses {job.get('provider')}, got {provider}")
            if jid in self._active:
                raise ValueError("cannot attach remote_id while job is actively running")
            if job.get("remote_id"):
                if job["remote_id"] == rid:
                    return copy.deepcopy(job)
                raise ValueError(f"job {jid} already bound to remote_id {job['remote_id']}")
            if job["state"] not in {"submit_unknown", "unconfigured", "retryable_error"}:
                raise ValueError(f"cannot attach remote_id in state {job['state']!r}")
            for other in self._jobs.values():
                if other["id"] != jid and other.get("remote_id") == rid and other.get("spec_hash") != job.get("spec_hash"):
                    raise ValueError(f"remote_id {rid!r} already belongs to a different job ({other['id']})")
            should_verify = self.provider.configured()

        if should_verify:
            try:
                st = self.provider.get_status(rid)
            except Exception as exc:
                raise ValueError(f"remote task {rid!r} verification failed: {exc}") from exc
            if not isinstance(st, dict) or st.get("remote_id") not in (None, rid):
                raise ValueError(f"remote task {rid!r} response mismatch")

        with self._lock:
            job = self._jobs.get(jid)
            if job is None:
                raise KeyError(jid)
            if jid in self._active:
                raise ValueError("cannot attach remote_id while job is actively running")
            if job.get("remote_id"):
                if job["remote_id"] == rid:
                    return copy.deepcopy(job)
                raise ValueError(f"job {jid} already bound to remote_id {job['remote_id']}")
            if job["state"] not in {"submit_unknown", "unconfigured", "retryable_error"}:
                raise ValueError(f"cannot attach remote_id in state {job['state']!r}")
            for other in self._jobs.values():
                if other["id"] != jid and other.get("remote_id") == rid and other.get("spec_hash") != job.get("spec_hash"):
                    raise ValueError(f"remote_id {rid!r} already belongs to a different job ({other['id']})")
            prev_state = job["state"]
            job.setdefault("audit", []).append({
                "event": "attach_remote", "remote_id": rid, "prev_state": prev_state, "ts": _now(),
            })
            job["remote_id"] = rid
            job.setdefault("params", {})["remote_id"] = rid
            job["state"] = "remote_processing"
            job["node"] = "remote"
            job["nodes"]["submit"] = "done"
            job["error"] = None
            self._commit(job)
            return copy.deepcopy(job)

    def poll(self, jid, *, background=False):
        with self._lock:
            job = self._jobs.get(jid)
            if (not job or not job.get("remote_id") or job["state"] in TERMINAL
                    or job["state"] == "submit_unknown" or jid in self._active):
                return self.get(jid)
            self._active.add(jid)
        if background:
            threading.Thread(target=self._run_nodes, args=(jid, "remote"), daemon=True).start()
        else:
            self._run_nodes(jid, "remote")
        return self.get(jid)

    def retry_node(self, jid, node, *, background=False):
        if node not in {"download", "import", "validate", "remote"}:
            raise ValueError("retry node must be remote/download/import/validate")
        with self._lock:
            job = self._jobs.get(jid)
            if not job or not job.get("remote_id"):
                raise ValueError("cannot retry without a known remote_id; recover the task first")
            if jid in self._active:
                return self.get(jid)
            job["retries"] = job.get("retries", 0) + 1
            if job["retries"] > 5:
                job.update(state="failed", error="retry limit reached (5); manual recovery required")
                self._commit(job)
                return self.get(jid)
            self._active.add(jid)
        force_redownload = (node == "download")
        if background:
            threading.Thread(
                target=self._run_nodes, args=(jid, node),
                kwargs={"force_redownload": force_redownload}, daemon=True,
            ).start()
        else:
            self._run_nodes(jid, node, force_redownload=force_redownload)
        return self.get(jid)

    @staticmethod
    def _verified_download_path(job, outdir: Path) -> Path | None:
        import hashlib
        from asset_pipeline.import_asset import validate_mesh_file
        if job.get("nodes", {}).get("download") != "done":
            return None
        expected_sha = job.get("download_sha256")
        raw_path = job.get("download_path")
        if not expected_sha or not raw_path:
            return None
        path = Path(raw_path)
        if not path.is_file():
            return None
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
            return None
        try:
            validate_mesh_file(path)
        except Exception:
            return None
        return path

    @staticmethod
    def _clear_download_artifacts(outdir: Path) -> None:
        if not outdir.is_dir():
            return
        for p in outdir.iterdir():
            if p.is_file() and (p.name.startswith("result.") or p.name.endswith(".part")):
                p.unlink(missing_ok=True)

    def _run_nodes(self, jid, start, *, force_redownload=False):
        import hashlib
        from asset_pipeline.import_asset import read_glb, validate_mesh_file
        job = self.get(jid)
        try:
            if start == "remote":
                st = self.provider.get_status(job["remote_id"])
                job.update(error=st.get("error"), provider_status=st.get("provider_status"))
                if st["state"] == "failed":
                    job.update(state="failed", node="remote")
                    self._commit(job)
                    return
                if st["state"] != "succeeded":
                    job.update(state="remote_processing", node="remote")
                    self._commit(job)
                    return
                job["nodes"]["remote"] = "done"
                job["outputs"] = st["outputs"]
                start = "download"
            if start == "download":
                outdir = self.dir / jid / "outputs"
                if force_redownload:
                    self._clear_download_artifacts(outdir)
                    job.pop("download_sha256", None)
                    job.pop("download_path", None)
                    job.pop("asset_id", None)
                    job.pop("validation", None)
                    job.pop("asset_status", None)
                    job["nodes"]["download"] = "pending"
                    job["nodes"]["import"] = "pending"
                    job["nodes"]["validate"] = "pending"
                job.update(state="downloading", node="download", error=None)
                self._commit(job)
                verified = None if force_redownload else self._verified_download_path(job, outdir)
                if verified is not None:
                    path = verified
                else:
                    self._clear_download_artifacts(outdir)
                    downloaded = Path(self.provider.download(job.get("outputs", []), outdir))
                    validate_mesh_file(downloaded)
                    path = downloaded
                job["download_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                job["download_path"] = str(path)
                job["nodes"]["download"] = "done"
                start = "import"
            if start == "import":
                job.update(state="importing", node="import")
                self._commit(job)
                from asset_pipeline.import_object import register_environment_mesh, register_object
                p = job["params"]
                dl_path = Path(job.get("download_path") or "")
                if not dl_path.is_file() or hashlib.sha256(dl_path.read_bytes()).hexdigest() != job.get("download_sha256"):
                    raise ValueError("downloaded mesh missing or checksum mismatch; retry download first")
                expected_glb_sha = hashlib.sha256(read_glb(dl_path)).hexdigest()
                if p.get("kind") == "environment_mesh":
                    scene_mf_path = self.lab_root / "assets" / "scenes" / job["slug"] / "1" / "manifest.json"
                    if not scene_mf_path.exists():
                        register_environment_mesh(
                            dl_path, self.lab_root, job["slug"], revision=1,
                            height=p.get("height_m"), width=p.get("width_m"), depth=p.get("depth_m"),
                            source="lux3d", task_id=job["remote_id"], prompt=p["prompt"],
                            spec_id=job.get("spec_id"), spec_revision=job.get("spec_revision"),
                            spec_hash=job.get("spec_hash"), job_id=job["id"],
                        )
                    job["asset_id"] = f"scenes/{job['slug']}/1"
                    mf_check_path = scene_mf_path
                else:
                    out = self.lab_root / "assets" / "imported" / job["slug"]
                    mf_path = out / "manifest.json"
                    if mf_path.exists():
                        mf = json.loads(mf_path.read_text(encoding="utf-8"))
                        same_job = (
                            mf.get("generation_id") == job.get("remote_id")
                            and mf.get("spec_hash") == job.get("spec_hash")
                            and mf.get("sha256_glb") == expected_glb_sha
                            and mf.get("job_id") in (None, job["id"])
                        )
                        if not same_job:
                            if mf.get("job_id") == job["id"]:
                                with self._lock:
                                    job["slug"] = self._allocate_unique_slug_locked(
                                        job.get("requested_slug") or job["slug"], job["id"]
                                    )
                                out = self.lab_root / "assets" / "imported" / job["slug"]
                                mf_path = out / "manifest.json"
                            else:
                                raise ValueError(
                                    f"Refusing to reuse existing asset directory {out}: manifest provenance "
                                    f"(job_id={mf.get('job_id')}, generation_id={mf.get('generation_id')}, "
                                    f"spec_hash={mf.get('spec_hash')}) does not match job {job['id']}"
                                )
                    if not mf_path.exists():
                        register_object(
                            dl_path, out, kind=p.get("kind", "task_object"),
                            role=p.get("role", "dynamic"), height=p.get("height_m"), width=p.get("width_m"),
                            depth=p.get("depth_m"), radius=p.get("radius_m"), mass=p.get("mass_kg"),
                            friction=p.get("friction"), source="lux3d",
                            source_detail="downloaded from authenticated Aholo API task",
                            task_id=job["remote_id"], prompt=p["prompt"], spec_id=job.get("spec_id"),
                            spec_revision=job.get("spec_revision"), spec_hash=job.get("spec_hash"),
                            compiler_version=p.get("compiler_version"), job_id=job["id"],
                            scale_mode=p.get("scale_mode", "auto"),
                            collision_mode=p.get("collision_mode", "auto"))
                    job["asset_id"] = "imported/" + job["slug"]
                    mf_check_path = out / "manifest.json"
                job["nodes"]["import"] = "done"
                start = "validate"
            if start == "validate":
                job.update(state="validating", node="validate")
                self._commit(job)
                if str(job.get("asset_id") or "").startswith("scenes/"):
                    mf_file = self.lab_root / "assets" / "scenes" / job["slug"] / "1" / "manifest.json"
                else:
                    mf_file = self.lab_root / "assets" / "imported" / job["slug"] / "manifest.json"
                mf = json.loads(mf_file.read_text(encoding="utf-8"))
                issues = []
                if not mf.get("triangles"):
                    issues.append("no triangles")
                if mf.get("unsatisfied_dimensions"):
                    issues.append(f"unsatisfied dimension constraints: {mf['unsatisfied_dimensions']}")
                if mf.get("role") == "dynamic" and not mf.get("calibrated"):
                    issues.append("dynamic asset is uncalibrated; physics binding blocked")
                job["validation"] = {"issues": issues, "physics_status": mf.get("physics_status"),
                                     "calibrated": mf.get("calibrated"), "role": mf.get("role"),
                                     "unsatisfied_dimensions": mf.get("unsatisfied_dimensions", {}),
                                     "concave_detected": mf.get("concave_detected"),
                                     "compound_boxes": (mf.get("compound_proxy") or {}).get("box_count")}
                job["nodes"]["validate"] = "done"
                job.update(state="succeeded", asset_status=mf.get("physics_status"), finished=_now())
        except Exception as exc:
            job.update(state="retryable_error", error=str(exc))
        finally:
            with self._lock:
                self._active.discard(jid)
                self._commit(job)
