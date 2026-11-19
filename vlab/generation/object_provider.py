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
import logging
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
    def __init__(self):
        self.key = (os.environ.get("AHOLO_API_KEY") or os.environ.get("LUX3D_CN_API_KEY")
                    or os.environ.get("LUX3D_API_KEY"))
        self._client = None

    def configured(self):
        return bool(self.key)

    def _sdk(self):
        if self._client is None:
            if not self.key:
                raise NotImplementedError("Set AHOLO_API_KEY to enable Lux3D generation")
            try:
                from manycore.aholo_sdk_core import AholoClientConfig
                from manycore.aholo_sdk_lux3d import create_lux3d_client
            except ImportError as exc:
                raise NotImplementedError("install manycore-aholo-sdk-lux3d==1.9.0") from exc
            self._client = create_lux3d_client(AholoClientConfig(api_key=self.key, region="cn"))
        return self._client

    def submit(self, prompt, payload):
        c = self._sdk()
        kwargs = {k: payload[k] for k in ("version", "style", "face_count", "output_format", "enable_pbr")
                  if k in payload}
        return str(c.text_to_3d.create(prompt=prompt, **kwargs))

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

    def download(self, outputs, directory):
        from asset_pipeline.import_asset import download_mesh
        return download_mesh(outputs, directory)


class MockLux3DProvider:
    """Controlled test double (NOT an online capability)."""

    def __init__(self, behavior="success"):
        self.behavior = behavior
        self.submits = 0

    def configured(self):
        return True

    def submit(self, prompt, payload):
        self.submits += 1
        if self.behavior == "submit_timeout":
            raise TimeoutError("simulated submit timeout")
        return f"mock-{self.submits}"

    def get_status(self, task_id):
        if self.behavior == "failed":
            return {"state": "failed", "remote_id": task_id, "outputs": [], "error": "mock failure"}
        return {"state": "succeeded", "remote_id": task_id, "outputs": [], "error": None}

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
        path.write_bytes(data)
        return path


class ObjectJobStore:
    def __init__(self, root: Path, provider=None, *, lab_root=None):
        self.dir = Path(root)
        self.path = self.dir / "object_jobs.json"
        self.provider = provider or Lux3DProvider()
        self.lab_root = Path(lab_root) if lab_root else self.dir / "registered"
        self._lock = threading.RLock()
        self._active = set()
        self._jobs = {j["id"]: j for j in json.loads(self.path.read_text(encoding="utf-8"))} if self.path.is_file() else {}
        interrupted = [j for j in self._jobs.values() if j["state"] == "submitting"]
        if interrupted:
            for j in interrupted:
                j.update(state="submit_unknown", node="submit",
                         error="Submit interrupted; remote id unknown. Check provider history before resubmitting.")
            try:
                self._save()
            except OSError as exc:
                logging.getLogger(__name__).warning("Generation state recovery not saved (%s): %s", self.path, exc)

    def _save(self):
        with self._lock:
            self.dir.mkdir(parents=True, exist_ok=True)
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
            for j in self._jobs.values():
                if j.get("client_request_id") == client_request_id:
                    return copy.deepcopy(j)
        return None

    def submit(self, params, *, background=False):
        """params: prompt, payload, slug, kind, role, dims, mass, friction, spec binding,
        client_request_id, provider, project_id, config_version. Persists intent first."""
        if not isinstance(params.get("prompt"), str) or not params["prompt"].strip():
            raise ValueError("Object generation requires a prompt")
        slug = valid_id(params["slug"])
        crid = params.get("client_request_id")
        if crid:
            existing = self.find_by_request(crid)
            if existing:
                return existing  # idempotent: duplicate confirm -> same local job
        jid = uuid.uuid4().hex[:12]
        job = {"id": jid, "kind": "object_generation", "provider": "lux3d",
               "params": copy.deepcopy(params), "slug": slug, "client_request_id": crid,
               "state": "submitting", "node": "submit", "remote_id": None,
               "spec_id": params.get("spec_id"), "spec_revision": params.get("spec_revision"),
               "spec_hash": params.get("spec_hash"), "project_id": params.get("project_id"),
               "config_version": params.get("config_version"),
               "created": _now(), "error": None, "retries": 0,
               "nodes": {n: "pending" for n in NODES}}
        with self._lock:
            self._jobs[jid] = job
            self._save()  # intent BEFORE remote submit
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
        if background:
            threading.Thread(target=self._run_nodes, args=(jid, node), daemon=True).start()
        else:
            self._run_nodes(jid, node)
        return self.get(jid)

    def _run_nodes(self, jid, start):
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
                job.update(state="downloading", node="download")
                self._commit(job)
                outdir = self.dir / jid / "outputs"
                existing = sorted(outdir.glob("result.*")) if outdir.is_dir() else []
                if existing:
                    path = existing[0]  # reuse downloaded file on retry
                else:
                    path = self.provider.download(job.get("outputs", []), outdir)
                import hashlib
                job["download_sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
                job["download_path"] = str(path)
                job["nodes"]["download"] = "done"
                start = "import"
            if start == "import":
                job.update(state="importing", node="import")
                self._commit(job)
                from asset_pipeline.import_object import register_object
                p = job["params"]
                out = self.lab_root / "assets" / "imported" / job["slug"]
                if not (out / "manifest.json").exists():
                    register_object(
                        Path(job["download_path"]), out, kind=p.get("kind", "task_object"),
                        role=p.get("role", "dynamic"), height=p.get("height_m"), width=p.get("width_m"),
                        depth=p.get("depth_m"), radius=p.get("radius_m"), mass=p.get("mass_kg"),
                        friction=p.get("friction"), source="lux3d",
                        source_detail="downloaded from authenticated Aholo API task",
                        task_id=job["remote_id"], prompt=p["prompt"], spec_id=job.get("spec_id"),
                        spec_revision=job.get("spec_revision"), spec_hash=job.get("spec_hash"),
                        compiler_version=p.get("compiler_version"))
                job["asset_id"] = "imported/" + job["slug"]
                job["nodes"]["import"] = "done"
                start = "validate"
            if start == "validate":
                job.update(state="validating", node="validate")
                self._commit(job)
                mf = json.loads((self.lab_root / "assets" / "imported" / job["slug"] / "manifest.json")
                                .read_text(encoding="utf-8"))
                issues = []
                if not mf.get("triangles"):
                    issues.append("no triangles")
                if mf.get("role") == "dynamic" and not mf.get("calibrated"):
                    issues.append("dynamic asset is uncalibrated; physics binding blocked")
                job["validation"] = {"issues": issues, "physics_status": mf.get("physics_status"),
                                     "calibrated": mf.get("calibrated"), "role": mf.get("role")}
                job["nodes"]["validate"] = "done"
                job.update(state="succeeded", asset_status=mf.get("physics_status"), finished=_now())
        except Exception as exc:
            job.update(state="retryable_error", error=str(exc))
        finally:
            with self._lock:
                self._active.discard(jid)
                self._commit(job)