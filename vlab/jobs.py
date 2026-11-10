"""Persisted background jobs for collect / train / eval / export.

Commands are built from a whitelist of kinds using subprocess argument arrays
(no shell string interpolation of user input). Logs and results are written under
datasets/jobs/<id>/. On backend restart, previously-running jobs are marked
interrupted so the UI can offer recovery.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

KINDS = {"collect", "train", "eval", "export", "convert", "checks"}


def _now():
    return datetime.now(timezone.utc).isoformat()


class JobManager:
    def __init__(self, lab_root: Path):
        self.lab = Path(lab_root)
        self.dir = self.lab / "datasets" / "jobs"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir / "jobs.json"
        self._lock = threading.Lock()
        self._procs = {}
        self._jobs = self._load()
        for j in self._jobs.values():
            if j["status"] == "running":
                j["status"] = "interrupted"
                j["finished"] = _now()
        self._save()

    def _load(self):
        if self.index_path.is_file():
            return {j["id"]: j for j in json.loads(self.index_path.read_text(encoding="utf-8"))}
        return {}

    def _save(self):
        self.index_path.write_text(json.dumps(list(self._jobs.values()), ensure_ascii=False, indent=1),
                                   encoding="utf-8")

    def list(self):
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j["created"], reverse=True)

    def get(self, jid):
        with self._lock:
            return self._jobs.get(jid)

    def logs(self, jid, tail=200):
        p = self.dir / jid / "log.txt"
        if not p.is_file():
            return []
        return p.read_text(encoding="utf-8", errors="replace").splitlines()[-tail:]

    def _command(self, kind, params):
        lab = str(self.lab)
        if kind == "collect":
            out = params.get("out", str(self.lab / "datasets" / "staging"))
            return [sys.executable, "scripts/collect_expert.py", "--out", out], lab
        if kind == "eval":
            args = [sys.executable, "scripts/eval_policy.py", "--group", params.get("group", "all")]
            if params.get("priv"):
                args.append("--priv")
            return args, lab
        if kind == "export":
            return [sys.executable, "scripts/export_deploy.py"], lab
        if kind in ("convert", "checks"):
            distro = os.environ.get("VLAB_WSL_DISTRO", "Ubuntu-22.04")
            linux_lab = os.environ.get("VLAB_LINUX_LAB", "/mnt/c/Users/31295/Documents/Codex/2026-11-09/"
                                                          "https-www-aholo3d-cn-hackathon-utm/outputs/vla-lab")
            deps = os.environ.get("VLAB_LINUX_DEPS", "/mnt/c/Users/31295/Documents/Codex/2026-11-09/"
                                                    "https-www-aholo3d-cn-hackathon-utm/work/policy/deps")
            py = os.environ.get("VLAB_LINUX_PYTHON", "~/miniconda3/envs/voxcpm/bin/python")
            if kind == "convert":
                inner = (f"cd {linux_lab} && PYTHONPATH={deps} HF_HUB_OFFLINE=1 {py} scripts/lerobot_writer.py "
                         f"--staging datasets/staging --out datasets/lerobot/expert21")
            else:
                inner = (f"cd {linux_lab} && PYTHONPATH={deps} HF_HUB_OFFLINE=1 {py} scripts/data_checks.py "
                         f"--lab {linux_lab}")
            return ["wsl", "-d", distro, "--", "bash", "-lc", inner], lab
        if kind == "train":
            distro = os.environ.get("VLAB_WSL_DISTRO", "Ubuntu-22.04")
            linux_lab = os.environ.get("VLAB_LINUX_LAB", "/mnt/c/Users/31295/Documents/Codex/2026-11-09/"
                                                          "https-www-aholo3d-cn-hackathon-utm/outputs/vla-lab")
            deps = os.environ.get("VLAB_LINUX_DEPS", "/mnt/c/Users/31295/Documents/Codex/2026-11-09/"
                                                    "https-www-aholo3d-cn-hackathon-utm/work/policy/deps")
            py = os.environ.get("VLAB_LINUX_PYTHON", "~/miniconda3/envs/voxcpm/bin/python")
            script = "train_priv.py" if params.get("priv") else "train_bc.py"
            epochs = int(params.get("epochs", 60))
            inner = (f"cd {linux_lab} && PYTHONPATH={deps}:. HF_HUB_OFFLINE=1 PYTHONWARNINGS=ignore "
                     f"{py} scripts/{script} --lab {linux_lab} --epochs {epochs}")
            return ["wsl", "-d", distro, "--", "bash", "-lc", inner], lab
        raise ValueError(f"unknown job kind {kind}")

    def create(self, kind, params):
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {sorted(KINDS)}")
        jid = uuid.uuid4().hex[:12]
        jdir = self.dir / jid
        jdir.mkdir(parents=True, exist_ok=True)
        job = {"id": jid, "kind": kind, "params": params, "status": "queued",
               "created": _now(), "started": None, "finished": None,
               "log": str(jdir / "log.txt"), "returncode": None}
        with self._lock:
            self._jobs[jid] = job
            self._save()
        threading.Thread(target=self._run, args=(jid,), daemon=True).start()
        return job

    def _run(self, jid):
        job = self.get(jid)
        try:
            cmd, cwd = self._command(job["kind"], job["params"])
        except ValueError as exc:
            job.update(status="failed", finished=_now(), error=str(exc))
            with self._lock:
                self._save()
            return
        logp = self.dir / jid / "log.txt"
        with open(logp, "w", encoding="utf-8") as log:
            log.write(f"$ {' '.join(cmd)}\n")
            log.flush()
            proc = subprocess.Popen(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                                    env={**os.environ, "PYTHONIOENCODING": "utf-8"})
            with self._lock:
                self._procs[jid] = proc
                job.update(status="running", started=_now())
                self._save()
            rc = proc.wait()
        with self._lock:
            self._procs.pop(jid, None)
            job.update(status="succeeded" if rc == 0 else "failed", finished=_now(), returncode=rc)
            self._save()

    def stop(self, jid):
        with self._lock:
            proc = self._procs.get(jid)
            job = self._jobs.get(jid)
        if not job:
            raise KeyError(jid)
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            job.update(status="cancelled", finished=_now())
        else:
            job.update(status="cancelled", finished=_now())
        with self._lock:
            self._save()
        return job