"""Aholo World provider adapter (generation / reconstruction) + persistent job store.

Real calls require the World SDK and AHOLO_API_KEY. Unknown submit outcomes are
never automatically resubmitted; a local receipt cannot guarantee remote exactly-once
creation across crashes. Poll/download retries reuse the known remote ID.
"""
from __future__ import annotations

import json
import copy
import hashlib
import shutil
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen

from .scene_assets import valid_id, scene_asset_dir


def _now():
    return datetime.now(timezone.utc).isoformat()


class WorldProvider:
    """Thin adapter over the official World SDK; methods mirror its documented API."""

    def __init__(self):
        self.key = os.environ.get("AHOLO_API_KEY")
        self._client = None

    def configured(self):
        return bool(self.key)

    def _sdk(self):
        if self._client is None:
            if not self.key:
                raise NotImplementedError("Set AHOLO_API_KEY to enable World generation")
            try:
                from manycore.aholo_sdk_world import create_world_client
                from manycore.aholo_sdk_core import AholoClientConfig
            except ImportError as exc:
                raise NotImplementedError("install manycore-aholo-sdk-world to enable online World") from exc
            self._client = create_world_client(AholoClientConfig(api_key=self.key, region="cn"))
        return self._client

    def submit_generation(self, prompt, ref_image=None, **params):
        c = self._sdk()
        resources = [{"url": ref_image, "type": "image"}] if ref_image else None
        op = c.generations.create(name=params.get("name", "RoboForge scene"), prompt=prompt, resources=resources)
        return str(op["worldId"])

    def submit_reconstruction(self, inputs, **params):
        c = self._sdk()
        op = c.reconstructions.create(name=params.get("name", "RoboForge reconstruction"),
                resources=inputs, scene="space", task_quality=params.get("task_quality", "normal"))
        return str(op["worldId"])

    def get_status(self, task_id):
        c = self._sdk()
        r = c.retrieve(task_id)
        status = r.get("status")
        if status in {"PENDING", "PREPROCESSING", "RUNNING"}:
            state = "remote_processing"
        elif status == "SUCCEEDED":
            state = "succeeded"
        elif status in {"FAILED", "CANCELED", "TIMEOUT", "REJECTED"}:
            state = "failed"
        else:
            raise ValueError(f"Unknown World status: {status}")
        assets = r.get("assets") or {}
        urls = (assets.get("splats") or {}).get("urls") or {}
        outputs = []
        for field, filename in [("plyPath", "scene.ply"), ("spzPath", "scene.spz")]:
            if urls.get(field):
                outputs.append({"url": urls[field], "filename": filename, "role": "scene"})
                break
        pano = (assets.get("imagery") or {}).get("panoUrl")
        if pano:
            outputs.append({"url": pano, "filename": "panorama.jpg", "role": "preview"})
        return {"state": state, "provider_status": status, "remote_id": task_id, "outputs": outputs,
                "up_axis": (assets.get("semanticsMetadata") or {}).get("upAxis"),
                "error": str(r.get("error") or status) if state == "failed" else None}

    def list_remote_tasks(self, *, page_num: int = 0, page_size: int = 20) -> dict:
        c = self._sdk()
        res = c.list(page_num=page_num, page_size=page_size)
        items = []
        for it in (res.get("result") or []):
            st_raw = it.get("status")
            if st_raw in {"PENDING", "PREPROCESSING", "RUNNING"}:
                st = "remote_processing"
            elif st_raw == "SUCCEEDED":
                st = "succeeded"
            else:
                st = "failed"
            items.append({
                "remote_id": str(it.get("worldId")),
                "provider": "aholo_world",
                "name": it.get("name"),
                "scene": it.get("scene"),
                "state": st,
                "provider_status": st_raw,
                "progress": it.get("progress"),
                "created_ms": it.get("createTime"),
                "updated_ms": it.get("updateTime"),
                "cover": it.get("cover"),
            })
        return {
            "provider": "aholo_world",
            "total": int(res.get("totalCount") or len(items)),
            "page": int(res.get("pageNum") or page_num),
            "page_size": int(res.get("pageSize") or page_size),
            "items": items,
        }

    def download_assets(self, task_id, directory, *, status=None):
        st = status or self.get_status(task_id)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        files = []
        for o in st["outputs"]:
            if urlsplit(o["url"]).scheme != "https" or o["filename"] not in {"scene.ply", "scene.spz", "panorama.jpg"}:
                raise ValueError("Invalid World output URL or filename")
            dest = directory / o["filename"]
            if dest.exists() and self._valid_world_file(dest):
                files.append({"path": str(dest), "role": o["role"], "sha256": hashlib.sha256(dest.read_bytes()).hexdigest()})
                continue
            dest.unlink(missing_ok=True)
            partial = dest.with_suffix(dest.suffix + ".part")
            try:
                with urlopen(o["url"], timeout=90) as resp, partial.open("wb") as f:
                    expected_raw = resp.headers.get("Content-Length") if getattr(resp, "headers", None) else None
                    expected_len = int(expected_raw) if expected_raw and expected_raw.isdigit() else None
                    while chunk := resp.read(1024 * 1024):
                        if f.tell() + len(chunk) > 256 * 1024**2:
                            raise ValueError("World output exceeds 256 MiB")
                        f.write(chunk)
                    if expected_len is not None and f.tell() != expected_len:
                        raise ValueError(f"Truncated World download: expected {expected_len}, got {f.tell()}")
                if not self._valid_world_file(partial, suffix=dest.suffix):
                    raise ValueError(f"Invalid World output file signature for {o['filename']}")
                partial.replace(dest)
            finally:
                partial.unlink(missing_ok=True)
            files.append({"path": str(dest), "role": o["role"], "sha256": hashlib.sha256(dest.read_bytes()).hexdigest()})
        return files

    @staticmethod
    def _valid_world_file(path: Path, suffix: str | None = None) -> bool:
        if not path.is_file() or path.stat().st_size == 0:
            return False
        ext = (suffix or path.suffix).lower()
        head = path.read_bytes()[:4]
        if ext == ".ply":
            return head.startswith(b"ply")
        if ext == ".jpg":
            return head.startswith(b"\xff\xd8")
        return True


class MockWorldProvider:
    """Controlled test double (NOT an online capability)."""

    def __init__(self, behavior="success"):
        self.behavior = behavior
        self.submits = 0

    def configured(self):
        return True

    def submit_generation(self, prompt, ref_image=None, **p):
        self.submits += 1
        if self.behavior == "submit_timeout":
            raise TimeoutError("simulated submit timeout")
        return f"mock-{self.submits}"

    def submit_reconstruction(self, inputs, **p):
        return self.submit_generation("recon")

    def get_status(self, task_id):
        if self.behavior == "failed":
            return {"state": "failed", "remote_id": task_id, "outputs": [], "error": "mock failure"}
        return {"state": "succeeded", "remote_id": task_id, "outputs": [], "error": None}

    def list_remote_tasks(self, *, page_num: int = 0, page_size: int = 20) -> dict:
        return {
            "provider": "aholo_world",
            "total": 1,
            "page": page_num,
            "page_size": page_size,
            "items": [{
                "remote_id": "mock-1",
                "provider": "aholo_world",
                "name": "Mock World Scene",
                "scene": "indoor",
                "state": "failed" if self.behavior == "failed" else "succeeded",
                "provider_status": "FAILED" if self.behavior == "failed" else "SUCCEEDED",
                "progress": 0.0 if self.behavior == "failed" else 1.0,
                "created_ms": None,
                "updated_ms": None,
                "cover": None,
            }],
        }

    def download_assets(self, task_id, directory, *, status=None):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        src = directory / "scene.ply"
        src.write_text("ply\nformat ascii 1.0\nelement vertex 1\nproperty float f_dc_0\nend_header\n0\n")
        return [{"path": str(src), "role": "scene", "sha256": hashlib.sha256(src.read_bytes()).hexdigest()}]


class GenerationJobStore:
    def __init__(self, root: Path, provider=None, *, lab_root=None):
        self.dir = Path(root)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "generation_jobs.json"
        self.provider = provider or WorldProvider()
        self.lab_root = Path(lab_root) if lab_root else self.dir / "registered"
        self._lock = threading.RLock()
        self._active = set()
        self._jobs = {j["id"]: j for j in json.loads(self.path.read_text(encoding="utf-8"))} if self.path.is_file() else {}
        for j in self._jobs.values():
            if j["state"] == "submitting":
                j.update(state="interrupted", error="Submit interrupted; check remote history before creating another task")
        self._save()

    def _save(self):
        with self._lock:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(list(self._jobs.values()), ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(self.path)

    def list(self):
        with self._lock:
            return copy.deepcopy(sorted(self._jobs.values(), key=lambda j: j["created"], reverse=True))

    def _commit(self, job):
        with self._lock:
            self._jobs[job["id"]] = copy.deepcopy(job)
            self._save()

    def get(self, jid):
        with self._lock:
            return copy.deepcopy(self._jobs.get(jid))

    def find_by_request(self, client_request_id: str | None):
        with self._lock:
            return copy.deepcopy(self._find_by_request_locked(client_request_id))

    def _find_by_request_locked(self, client_request_id: str | None):
        if not client_request_id:
            return None
        for j in self._jobs.values():
            if j.get("params", {}).get("client_request_id") == client_request_id:
                return j
        return None

    @staticmethod
    def _binding_signature(kind: str, params: dict):
        keys = ("prompt", "ref_image", "inputs", "spec_id", "spec_revision",
                "spec_hash", "request_hash", "project_id", "config_version")
        return {"kind": kind, **{k: params.get(k) for k in keys if params.get(k) is not None}}

    def submit(self, kind, params, *, background=False):
        if kind not in {"generation", "reconstruction"}:
            raise ValueError("kind must be generation or reconstruction")
        if kind == "generation" and (not isinstance(params.get("prompt"), str) or not params["prompt"].strip()):
            raise ValueError("Generation requires a prompt")
        if params.get("remote_id"):
            valid_id(str(params["remote_id"]))
        if params.get("ref_image") and urlsplit(str(params["ref_image"])).scheme != "https":
            raise ValueError("Reference image must be an HTTPS URL")
        if kind == "reconstruction":
            inputs = params.get("inputs", [])
            if not isinstance(inputs, list) or not inputs:
                raise ValueError("Reconstruction requires input resources")
            if any(not isinstance(r, dict) or r.get("type") not in {"image", "video", "insv"}
                   or urlsplit(str(r.get("url", ""))).scheme != "https" for r in inputs):
                raise ValueError("Resources require an HTTPS URL and image/video/insv type")
            if all(r["type"] == "image" for r in inputs) and len(inputs) < 20:
                raise ValueError("Image reconstruction requires at least 20 images")
        crid = params.get("client_request_id")
        with self._lock:
            existing = self._find_by_request_locked(crid)
            if existing is not None:
                if self._binding_signature(kind, params) != self._binding_signature(
                    existing["kind"], existing.get("params", {})
                ):
                    raise ValueError(f"client_request_id {crid!r} already bound to a different payload")
                return copy.deepcopy(existing)
            jid = uuid.uuid4().hex[:12]
            job = {"id": jid, "kind": kind, "params": copy.deepcopy(params), "state": "submitting",
                   "remote_id": None, "created": _now(), "error": None}
            self._jobs[jid] = job
            self._save()  # persist intent BEFORE remote submit (no duplicate on retry)
        if background:
            threading.Thread(target=self._submit, args=(jid,), daemon=True).start()
        else:
            self._submit(jid)
        return self.get(jid)

    def _submit(self, jid):
        job = self.get(jid)
        kind, params = job["kind"], job["params"]
        try:
            if params.get("remote_id"):
                rid = params["remote_id"]
            elif kind == "generation":
                rid = self.provider.submit_generation(params["prompt"], params.get("ref_image"), name=params.get("name", "RoboForge scene"))
            else:
                rid = self.provider.submit_reconstruction(params.get("inputs", []), name=params.get("name", "RoboForge reconstruction"), task_quality=params.get("task_quality", "normal"))
            job.update(remote_id=str(rid), state="remote_processing")
        except TimeoutError as exc:
            job.update(state="interrupted", error=str(exc))  # remote id unknown; resume manually
        except NotImplementedError as exc:
            job.update(state="unconfigured", error=str(exc))
        except Exception as exc:
            job.update(state="failed", error=str(exc))
        self._commit(job)

    def attach_remote(self, jid: str, remote_id: str, *, provider: str | None = None) -> dict:
        """Attach a recovered remote_id onto an interrupted/unconfigured World job without resubmitting."""
        rid = valid_id(str(remote_id or "").strip())
        with self._lock:
            job = self._jobs.get(jid)
            if job is None:
                raise KeyError(jid)
            if provider is not None and provider != "aholo_world":
                raise ValueError(f"provider mismatch: world job expects aholo_world, got {provider}")
            if jid in self._active:
                raise ValueError("cannot attach remote_id while job is actively running")
            if job.get("remote_id"):
                if job["remote_id"] == rid:
                    return copy.deepcopy(job)
                raise ValueError(f"job {jid} already bound to remote_id {job['remote_id']}")
            if job["state"] not in {"interrupted", "submit_unknown", "unconfigured", "retryable_error"}:
                raise ValueError(f"cannot attach remote_id in state {job['state']!r}")
            for other in self._jobs.values():
                if other["id"] != jid and other.get("remote_id") == rid:
                    raise ValueError(f"remote_id {rid!r} already belongs to job {other['id']}")
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
            if job["state"] not in {"interrupted", "submit_unknown", "unconfigured", "retryable_error"}:
                raise ValueError(f"cannot attach remote_id in state {job['state']!r}")
            for other in self._jobs.values():
                if other["id"] != jid and other.get("remote_id") == rid:
                    raise ValueError(f"remote_id {rid!r} already belongs to job {other['id']}")
            prev_state = job["state"]
            job.setdefault("audit", []).append({
                "event": "attach_remote", "remote_id": rid, "prev_state": prev_state, "ts": _now(),
            })
            job["remote_id"] = rid
            job.setdefault("params", {})["remote_id"] = rid
            job["state"] = "remote_processing"
            job["error"] = None
            self._commit(job)
            return copy.deepcopy(job)

    def poll(self, jid, *, background=False):
        with self._lock:
            job = self._jobs.get(jid)
            if not job or not job.get("remote_id") or job["state"] == "failed" or (job["state"] == "succeeded" and job.get("asset_id")) or jid in self._active:
                return self.get(jid)
            self._active.add(jid)
        if background:
            threading.Thread(target=self._poll, args=(jid,), daemon=True).start()
        else:
            self._poll(jid)
        return self.get(jid)

    def _poll(self, jid):
        job = self.get(jid)
        try:
            st = self.provider.get_status(job["remote_id"])
            job.update(error=st.get("error"), provider_status=st.get("provider_status"))
            if st["state"] == "succeeded":
                job["state"] = "downloading"
                self._commit(job)
                files = self.provider.download_assets(job["remote_id"], self.dir / jid / "outputs", status=st)
                job.update(state="processing", files=files)
                self._commit(job)
                from .scene_import import import_scene
                src = next((Path(f["path"]) for f in files if f["role"] == "scene"), None)
                if src is None or src.suffix != ".ply":
                    raise ValueError("No importable PLY; downloaded files retained (SPZ import not supported)")
                aid = valid_id("world_" + jid)
                manifest_path = scene_asset_dir(self.lab_root, aid, 1) / "manifest.json"
                if not manifest_path.exists():
                    manifest_path = import_scene(src, self.lab_root, aid, source_kind=job["kind"], up_axis=st.get("up_axis") or "Z")
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest.update(provider="aholo_world", remote_task_id=job["remote_id"], source_detail=job["params"].get("prompt", ""))
                preview = next((Path(f["path"]) for f in files if f["role"] == "preview"), None)
                if preview is not None:
                    dest = manifest_path.parent / "preview" / "panorama.jpg"
                    dest.parent.mkdir(exist_ok=True)
                    shutil.copyfile(preview, dest)
                    manifest["preview_asset"] = "preview/panorama.jpg"
                    manifest["files"]["preview/panorama.jpg"] = hashlib.sha256(dest.read_bytes()).hexdigest()
                tmp = manifest_path.with_suffix(".tmp")
                tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
                tmp.replace(manifest_path)
                job.update(state="succeeded", asset_id=aid, revision=1, asset_status=manifest["status"], finished=_now())
            else:
                job["state"] = st["state"]
        except Exception as exc:
            job.update(state="retryable_error", error=str(exc))
        finally:
            with self._lock:
                self._active.discard(jid)
                self._commit(job)
