"""Explicit local asset-library acceptance, separate from generation caches.

Schema 1 snapshots existing imported assets as a historical reusable whitelist
once. Transport-managed cache entries are hidden until an explicit ``save``
decision; untracked imports retain the original modules' library behavior.
Transactions use a process lock and OS file lock, reread disk under the lock,
and atomically replace JSON, so multiple store instances do not lose decisions.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import threading
import uuid
import zipfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .scene_assets import contained_file, valid_id

_LOCKS = {}
_LOCKS_GUARD = threading.Lock()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _ref(value):
    if not isinstance(value, str) or not value.startswith("imported/"):
        raise ValueError("模型必须来自本地Aholo导入缓存。")
    valid_id(value.split("/", 1)[1])
    return value


class AssetCatalog:
    def __init__(self, lab_root):
        self.lab = Path(lab_root)
        self.root = self.lab / "datasets/transport_asset_catalog"
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "catalog.json"
        with _LOCKS_GUARD:
            self._lock = _LOCKS.setdefault(str(self.path.resolve()), threading.RLock())
        with self._transaction() as data:
            if not data:
                data.update(schema_version="1.0", baseline_at=_now(),
                            policy="historical and untracked imports retain existing behavior; managed caches require explicit save",
                            records={}, decisions={}, review_requests={})
                for path in (self.lab / "assets/imported").glob("*/manifest.json"):
                    try:
                        ref = _ref("imported/"+path.parent.name)
                    except ValueError:
                        continue
                    data["records"][ref] = {"asset_id": ref, "state": "legacy", "saved": True,
                                             "storage": "library", "saved_at": None}

    @contextmanager
    def _transaction(self, *, write=True):
        with self._lock, (self.root / "catalog.lock").open("a+b") as lockfile:
            lockfile.seek(0, 2)
            if lockfile.tell() == 0:
                lockfile.write(b"0"); lockfile.flush()
            lockfile.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lockfile.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(lockfile, fcntl.LOCK_EX)
            try:
                data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
                before = copy.deepcopy(data)
                yield data
                if write and data != before:
                    temp = self.path.with_name("catalog-"+uuid.uuid4().hex+".tmp")
                    try:
                        temp.write_text(json.dumps(data, indent=1, ensure_ascii=False, allow_nan=False), encoding="utf-8")
                        temp.replace(self.path)
                    finally:
                        temp.unlink(missing_ok=True)
            finally:
                lockfile.seek(0)
                if os.name == "nt":
                    msvcrt.locking(lockfile.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(lockfile, fcntl.LOCK_UN)

    def snapshot(self):
        with self._transaction(write=False) as data:
            return copy.deepcopy(data)

    def is_reusable(self, asset_ref):
        record = self.snapshot()["records"].get(_ref(asset_ref))
        return record is None or bool(record.get("saved"))

    def record_pending(self, asset_ref, run_id, role, name=""):
        with self._transaction() as data:
            record = data["records"].setdefault(_ref(asset_ref), {
                "asset_id": asset_ref, "state": "pending", "saved": False, "storage": "cache",
                "created": _now(), "run_id": run_id, "role": role})
            if name and record.get("run_id") == run_id:
                record.setdefault("name", name)

    def decision(self, run_id, role):
        return self.snapshot()["decisions"].get(f"{run_id}:{role}")

    def review(self, asset_ref, run_id, role, decision, request_id):
        if decision not in ("save", "use_once", "reject"):
            raise ValueError("请选择保存入库、仅本次使用或放弃。")
        if not isinstance(request_id, str) or not 1 <= len(request_id) <= 100:
            raise ValueError("审阅需要长度1～100的稳定提交ID。")
        with self._transaction() as data:
            key = f"{run_id}:{role}"
            requests = data.setdefault("review_requests", {})
            if request_id in requests and requests[request_id] != key:
                raise ValueError("该审阅提交ID已绑定其他模型。")
            # Preserve bindings written by schema-1 catalogs predating the
            # explicit request-alias index as well.
            for other_key, item in data["decisions"].items():
                if item["client_request_id"] == request_id and other_key != key:
                    raise ValueError("该审阅提交ID已绑定其他模型。")
            prior = data["decisions"].get(key)
            if prior:
                if prior["decision"] != decision or prior["asset_id"] != asset_ref:
                    raise ValueError("该模型已有审阅结论，不能在运行中改变决定。")
                requests[request_id] = key
                return copy.deepcopy(prior)
            directory = self.asset_directory(asset_ref)
            manifest = contained_file(directory, "manifest.json")
            if not manifest.is_file():
                raise ValueError("待审阅模型缓存不存在。")
            record = data["records"].setdefault(_ref(asset_ref), {"asset_id": asset_ref})
            accepted = {"asset_id": asset_ref, "run_id": run_id, "role": role, "decision": decision,
                        "client_request_id": request_id, "reviewed_at": _now()}
            data["decisions"][key] = accepted
            requests[request_id] = key
            if decision == "save":
                record.update(state="saved", saved=True, storage="library", saved_at=accepted["reviewed_at"],
                              manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
            elif not record.get("saved"):
                record.update(state="temporary" if decision == "use_once" else "rejected",
                              saved=False, storage="cache")
            return copy.deepcopy(accepted)

    def asset_directory(self, asset_ref):
        asset_ref = _ref(asset_ref)
        root = (self.lab / "assets/imported").resolve()
        directory = (self.lab / "assets" / asset_ref).resolve()
        if not directory.is_relative_to(root):
            raise ValueError("模型目录越出导入缓存。")
        return directory

    def review_item(self, run, step):
        role, ref = step["role"], step["asset_id"]
        self.record_pending(ref, run["id"], role, run["spec"].get(f"{role}_name", ""))
        record = self.snapshot()["records"][ref]
        decision = self.decision(run["id"], role)
        base = f"/api/v1/transport/runs/{run['id']}/assets/{role}"
        directory = self.asset_directory(ref)
        glb = base+"/original.glb" if contained_file(directory, "original.glb").is_file() else None
        obj = base+"/visual.obj"
        return {"role": role, "asset_id": ref, "name": record.get("name", ""), "session_id": step.get("session_id"),
                "status": "pending" if decision is None else "rejected" if decision["decision"] == "reject" else "accepted",
                "decision": decision["decision"] if decision else None, "saved": bool(record.get("saved")),
                "storage": record.get("storage", "cache"), "saved_at": record.get("saved_at"),
                "reviewed_at": decision.get("reviewed_at") if decision else None,
                "description": run["spec"]["object_description" if role == "object" else "bin_description"],
                "parameters": run["spec"].get("object_parameters", {}) if role == "object" else {
                    "size_m": run["spec"]["bin_size_m"], "color": run["spec"]["bin_color"]},
                "parameter_provenance": run.get("parameter_provenance", {}) if role == "object" else {},
                "preview": {"obj_url": obj, "glb_url": glb}, "preview_url": glb or obj,
                "preview_format": "glb" if glb else "obj", "package_url": base+"/package",
                "download_url": base+"/package", "source": "lux3d"}

    def file(self, item, filename):
        if filename not in ("visual.obj", "original.glb", "manifest.json"):
            raise ValueError("不支持的模型文件。")
        path = contained_file(self.asset_directory(item["asset_id"]), filename)
        if not path.is_file():
            raise KeyError(filename)
        return path

    def package(self, item, destination):
        """Export original GLB, imported OBJ, manifest and task-specific estimates."""
        destination = Path(destination)
        temp = destination.with_name(destination.stem+"-"+uuid.uuid4().hex+".tmp")
        try:
            with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name in ("original.glb", "visual.obj", "manifest.json"):
                    archive.write(self.file(item, name), name)
                metadata = {key: item[key] for key in ("asset_id", "role", "source", "description", "parameters",
                    "parameter_provenance", "status", "decision", "saved", "saved_at", "storage")}
                metadata["name"] = item.get("name", "")
                metadata["note"] = "Original GLB appearance, imported calibrated OBJ, estimated physical parameters; not real-world measurements."
                archive.writestr("transport_context.json", json.dumps(metadata, indent=2, ensure_ascii=False))
            temp.replace(destination)
            return destination
        finally:
            temp.unlink(missing_ok=True)
