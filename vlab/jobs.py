"""Persisted background jobs for collect / train / eval / export.

Commands are built from a whitelist of kinds using subprocess argument arrays
(no shell string interpolation of user input). Logs and results are written under
datasets/jobs/<id>/. On backend restart, previously-running jobs are marked
interrupted so the UI can offer recovery.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from .scene_assets import valid_id
from .store import ConfigStore

KINDS = {"collect", "train", "train_visual", "eval", "eval_visual", "export", "export_visual",
         "convert", "checks", "lux3d_generate"}


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
        tmp = self.index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(list(self._jobs.values()), ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.index_path)

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
            if "config_snapshot" not in params or "out" not in params:
                raise ValueError("Collection requires a saved job config snapshot and output directory")
            args = [sys.executable, "scripts/collect_expert.py", "--out", str(params["out"]),
                    "--config", str(params["config_snapshot"])]
            if params.get("plan_path"):
                args += ["--plan", str(params["plan_path"])]
            if params.get("provenance"):
                args += ["--provenance", json.dumps(params["provenance"], ensure_ascii=False)]
            return args, lab
        if kind == "eval":
            args = [sys.executable, "scripts/eval_policy.py", "--group", params.get("group", "all")]
            if params.get("priv"):
                args.append("--priv")
            return args, lab
        if kind == "export":
            return [sys.executable, "scripts/export_deploy.py"], lab
        if kind == "lux3d_generate":
            slug = valid_id(params["slug"])
            args = [sys.executable, "asset_pipeline/generate_lux3d.py", "--slug", slug,
                    "--height-m", str(float(params.get("height_m", 0.09))),
                    "--mass-kg", str(float(params.get("mass_kg", 0.12)))]
            if params.get("prompt"):
                args += ["--prompt", str(params["prompt"])]
            if params.get("radius_m") is not None:
                args += ["--radius-m", str(float(params["radius_m"]))]
            if params.get("asset_kind"):
                args += ["--kind", str(params["asset_kind"]), "--role", str(params.get("role", "dynamic"))]
            if params.get("width_m") is not None:
                args += ["--width-m", str(float(params["width_m"]))]
            if params.get("depth_m") is not None:
                args += ["--depth-m", str(float(params["depth_m"]))]
            return args, lab
        if kind in {"train_visual", "eval_visual", "convert", "checks", "export_visual"}:
            distro = os.environ.get("VLAB_WSL_DISTRO", "Ubuntu-22.04")
            linux_lab = os.environ.get("VLAB_LINUX_LAB", "/mnt/c/Users/31295/Documents/Codex/2026-11-09/"
                                                          "https-www-aholo3d-cn-hackathon-utm/outputs/vla-lab")
            deps = os.environ.get("VLAB_LINUX_DEPS", "/mnt/c/Users/31295/Documents/Codex/2026-11-09/"
                                                    "https-www-aholo3d-cn-hackathon-utm/work/policy/deps")
            py = os.environ.get("VLAB_LINUX_PYTHON", "~/miniconda3/envs/voxcpm/bin/python")
            q = shlex.quote
            python = '"$HOME"/' + q(py[2:]) if py.startswith("~/") else q(py)
            dataset = valid_id(params.get("dataset_id", params.get("dataset", "expert21")))
            if kind == "train_visual":
                epochs = int(params.get("epochs", 40))
                if not 1 <= epochs <= 10000:
                    raise ValueError("epochs must be between 1 and 10000")
                args = ["scripts/train_image_bc.py", "--lab", linux_lab, "--dataset", dataset,
                        "--model-out", valid_id(params.get("model_id", params.get("model_out", "visual_bc"))),
                        "--epochs", str(epochs)]
            elif kind == "eval_visual":
                snapshot = Path(params["config_snapshot"]).relative_to(self.lab).as_posix()
                args = ["scripts/eval_image_policy.py", "--lab", linux_lab,
                        "--model", valid_id(params.get("model_id", params.get("model", "visual_bc"))),
                        "--base-project", valid_id(params.get("project_id", params.get("base_project", "default"))),
                        "--config", linux_lab + "/" + snapshot,
                        "--api-base", os.environ.get("VLAB_API_BASE", "http://127.0.0.1:8767"),
                        "--out", linux_lab + "/" + str(params["report_relative"])]
                if params.get("allow_legacy_contract"):
                    args.append("--allow-legacy-contract")
            elif kind == "convert":
                collection = valid_id(params.get("collection_id", "staging"))
                staging = "datasets/staging" if collection == "staging" else "datasets/collections/" + collection
                args = ["scripts/lerobot_writer.py", "--staging", staging, "--out", "datasets/lerobot/" + dataset]
            elif kind == "export_visual":
                args = ["scripts/export_visual_package.py", "--lab", linux_lab,
                        "--model", valid_id(params.get("model_id", "visual_hybrid")),
                        "--project", valid_id(params.get("project_id", "hybrid_engine")),
                        "--out", str(params["out"])]
            else:
                if dataset != "expert21":
                    raise ValueError("Legacy checks only support expert21; selected datasets use conversion read-back verification")
                args = ["scripts/data_checks.py", "--lab", linux_lab]
            inner = (f"cd {q(linux_lab)} && no_proxy=127.0.0.1,localhost NO_PROXY=127.0.0.1,localhost "
                     f"PYTHONPATH={q(deps + ':.')} HF_HUB_OFFLINE=1 {python} " + " ".join(q(a) for a in args))
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
        params = dict(params)
        jdir = self.dir / jid
        jdir.mkdir(parents=True, exist_ok=True)
        if kind in {"collect", "eval_visual"}:
            project = valid_id(params.get("project_id", params.get("base_project", "default")))
            store = ConfigStore(self.lab / "projects")
            cfg = (store.load_version(project, int(params["config_version"])) if params.get("config_version")
                   else store.load(project))
            snapshot = jdir / "config.json"
            snapshot.write_text(cfg.model_dump_json(indent=2), encoding="utf-8")
            params.update(project_id=project, config_version=cfg.version, config_snapshot=str(snapshot))
            if kind == "collect":
                dataset = valid_id(params.get("dataset_id", "collect_" + jid))
                out = self.lab / "datasets" / "collections" / dataset
                if out.exists():
                    raise ValueError(f"Collection {dataset} already exists; choose a new dataset_id")
                out.mkdir(parents=True, exist_ok=False)
                params.update(dataset_id=dataset, out=str(out))
                if "plan" in params:
                    plan = params["plan"]
                    if not isinstance(plan, list) or not plan or any(not isinstance(p, dict) or "asset" not in p or not isinstance(p.get("seed"), int) for p in plan):
                        raise ValueError("plan must contain asset/seed entries")
                    plan_path = jdir / "plan.json"
                    plan_path.write_text(json.dumps(plan), encoding="utf-8")
                    params["plan_path"] = str(plan_path)
            else:
                params["report_relative"] = (jdir / "evaluation.json").relative_to(self.lab).as_posix()
        if kind == "export_visual":
            params.setdefault("model_id", valid_id(params.get("model", "visual_hybrid")))
            params.setdefault("project_id", valid_id(params.get("project", "hybrid_engine")))
            params.setdefault("out", f"deploy/{params['model_id']}_package_{jid[:6]}")
        self._command(kind, params)  # reject invalid parameters before starting a process
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
            with self._lock:
                if job["status"] == "cancelled":
                    return
                try:
                    proc = subprocess.Popen(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                                            env={**os.environ, "PYTHONIOENCODING": "utf-8"})
                except OSError as exc:
                    job.update(status="failed", error=str(exc), finished=_now())
                    self._save()
                    return
                self._procs[jid] = proc
                job.update(status="running", started=_now())
                self._save()
            rc = proc.wait()
        with self._lock:
            self._procs.pop(jid, None)
            job.update(status="cancelled" if job["status"] == "cancelled" else ("succeeded" if rc == 0 else "failed"), finished=_now(), returncode=rc)
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
