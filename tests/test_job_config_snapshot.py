"""T14: a submitted job keeps the config version it started from (no drift).

The collect job writes a config snapshot; later edits to the project's current config
must not change that job's parameters or snapshot.
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
from vlab.jobs import JobManager  # noqa: E402
from vlab.schema import ProjectConfig  # noqa: E402
from vlab.store import ConfigStore  # noqa: E402


def test_job_uses_config_snapshot_after_edit():
    root = Path(tempfile.mkdtemp(prefix="vlab_snap_"))
    try:
        cfg = ProjectConfig.model_validate_json((LAB / "configs" / "default.json").read_text(encoding="utf-8"))
        cfg = cfg.model_copy(update={"project_id": "snap_proj", "version": 1})
        store = ConfigStore(root / "projects")
        v1 = store.save(cfg)
        assert v1.version == 1 and v1.task.time_limit_s == cfg.task.time_limit_s

        mgr = JobManager(root)
        mgr._run = lambda jid: None  # validate/plan only; never launch the subprocess
        job = mgr.create("collect", {"project_id": "snap_proj", "dataset_id": "snapshot_set",
                                     "plan": [{"asset": "cup", "seed": 0}]})
        snap_path = Path(job["params"]["config_snapshot"])
        assert snap_path.is_file() and job["params"]["config_version"] == 1
        assert Path(job["params"]["out"]).is_dir()

        v2 = store.save(cfg.model_copy(update={"version": 1,
                                               "task": cfg.task.model_copy(update={"time_limit_s": cfg.task.time_limit_s + 5})}))
        assert v2.version == 2 and v2.task.time_limit_s != cfg.task.time_limit_s
        snap_cfg = ProjectConfig.model_validate_json(snap_path.read_text(encoding="utf-8"))
        assert snap_cfg.version == 1, "job snapshot was overwritten by the new current config"
        assert snap_cfg.task.time_limit_s == cfg.task.time_limit_s
        assert mgr.get(job["id"])["params"]["config_version"] == 1

        with_snapshot = json.loads((root / "datasets" / "jobs" / job["id"] / "config.json").read_text(encoding="utf-8"))
        assert with_snapshot["version"] == 1 and with_snapshot["task"]["time_limit_s"] == cfg.task.time_limit_s

        bad = 0
        for plan in ([], [{"asset": "cup"}], [{"seed": 1}], [{"asset": "cup", "seed": "x"}]):
            try:
                mgr.create("collect", {"project_id": "snap_proj", "dataset_id": f"bad_{bad}",
                                       "plan": plan})
            except ValueError:
                bad += 1
        assert bad == 4, "invalid plans must be rejected before any job is queued"
        print({"snapshot_version": snap_cfg.version, "current_version": store.load("snap_proj").version,
               "invalid_plans_rejected": bad})
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main():
    test_job_uses_config_snapshot_after_edit()


if __name__ == "__main__":
    main()
