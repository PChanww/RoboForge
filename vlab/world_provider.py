"""Aholo World provider adapter (generation / reconstruction) + persistent job store.

Real online calls require the optional World SDK and AHOLO_API_KEY; without them the
adapter reports unconfigured and only controlled/mock tests run. Remote task IDs are
persisted BEFORE submit completes so a retry/restart never creates a duplicate
billable task. Local stop != remote cancel (the provider exposes no cancel here).
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path


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
            try:
                from manycore.aholo_sdk_world import create_world_client
                from manycore.aholo_sdk_core import AholoClientConfig
            except ImportError as exc:
                raise NotImplementedError("install manycore-aholo-sdk-world to enable online World") from exc
            self._client = create_world_client(AholoClientConfig(api_key=self.key, region="cn"))
        return self._client

    def submit_generation(self, prompt, ref_image=None, **params):
        c = self._sdk()
        kw = {"prompt": prompt}
        if ref_image:
            kw["img"] = ref_image
        return str(c.text_to_world.create(**kw, **params))

    def submit_reconstruction(self, inputs, **params):
        c = self._sdk()
        return str(c.reconstruction.create(inputs=inputs, **params))

    def get_status(self, task_id):
        c = self._sdk()
        r = c.tasks.get(task_id)
        return {"state": r.get("status"), "remote_id": task_id, "outputs": r.get("outputs", []),
                "error": r.get("error")}

    def download_assets(self, task_id, directory):
        st = self.get_status(task_id)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        files = []
        for i, o in enumerate(st["outputs"]):
            url = o.get("content")
            if not isinstance(url, str) or not url.startswith("https"):
                continue
            import urllib.request
            dest = directory / f"asset_{i}.bin"
            with urllib.request.urlopen(url, timeout=90) as resp, dest.open("wb") as f:
                f.write(resp.read())
            files.append(str(dest))
        return files


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

    def download_assets(self, task_id, directory):
        return []


class GenerationJobStore:
    def __init__(self, root: Path, provider=None):
        self.dir = Path(root)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "generation_jobs.json"
        self.provider = provider or WorldProvider()
        self._lock = threading.Lock()
        self._jobs = {j["id"]: j for j in json.loads(self.path.read_text())} if self.path.is_file() else {}

    def _save(self):
        self.path.write_text(json.dumps(list(self._jobs.values()), ensure_ascii=False, indent=1))

    def list(self):
        return sorted(self._jobs.values(), key=lambda j: j["created"], reverse=True)

    def get(self, jid):
        return self._jobs.get(jid)

    def submit(self, kind, params):
        jid = uuid.uuid4().hex[:12]
        job = {"id": jid, "kind": kind, "params": params, "state": "submitting",
               "remote_id": None, "created": _now(), "error": None}
        with self._lock:
            self._jobs[jid] = job
            self._save()  # persist intent BEFORE remote submit (no duplicate on retry)
        try:
            if kind == "generation":
                rid = self.provider.submit_generation(params.get("prompt", ""), params.get("ref_image"))
            else:
                rid = self.provider.submit_reconstruction(params.get("inputs", []))
            job.update(remote_id=rid, state="remote_processing")
        except TimeoutError as exc:
            job.update(state="interrupted", error=str(exc))  # remote id unknown; resume manually
        except NotImplementedError as exc:
            job.update(state="unconfigured", error=str(exc))
        except Exception as exc:
            job.update(state="failed", error=str(exc))
        with self._lock:
            self._save()
        return job

    def poll(self, jid):
        job = self._jobs.get(jid)
        if not job or not job.get("remote_id"):
            return job
        try:
            st = self.provider.get_status(job["remote_id"])
            job.update(state=st["state"], error=st.get("error"))
        except Exception as exc:
            job.update(error=str(exc))
        with self._lock:
            self._save()
        return job