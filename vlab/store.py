"""Config persistence: editable current, immutable version snapshots, run snapshots."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .schema import ProjectConfig
from .scene_assets import valid_id


class ConfigStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _dir(self, project_id):
        valid_id(project_id)
        d = self.root / project_id
        (d / "versions").mkdir(parents=True, exist_ok=True)
        return d

    def exists(self, project_id):
        return (self._dir(project_id) / "current.json").is_file()

    def load(self, project_id="default") -> ProjectConfig:
        return ProjectConfig.model_validate_json((self._dir(project_id) / "current.json").read_text(encoding="utf-8"))

    def save(self, config: ProjectConfig) -> ProjectConfig:
        d = self._dir(config.project_id)
        if (d / "current.json").is_file():
            version = self.load(config.project_id).version + 1
        else:
            version = config.version
        config = config.model_copy(update={"version": version})
        text = config.model_dump_json(indent=2)
        (d / "versions" / f"v{config.version}.json").write_text(text, encoding="utf-8")
        (d / "current.json").write_text(text, encoding="utf-8")
        return config

    def load_version(self, project_id, version) -> ProjectConfig:
        return ProjectConfig.model_validate_json(
            (self._dir(project_id) / "versions" / f"v{version}.json").read_text(encoding="utf-8"))

    def versions(self, project_id):
        return sorted(int(p.stem[1:]) for p in (self._dir(project_id) / "versions").glob("v*.json"))

    def snapshot_run(self, config: ProjectConfig, run_id: str) -> Path:
        d = self.root / "_runs" / run_id
        d.mkdir(parents=True, exist_ok=True)
        p = d / "snapshot.json"
        p.write_text(config.model_dump_json(indent=2), encoding="utf-8")
        return p


def validate_config(config: ProjectConfig, lab_root: Path):
    """Static + physics-build validation. Returns a list of issue strings (empty = ok)."""
    from .compat import select_controller
    from .engine import PlatformEnv
    issues = []
    has_runnable_robot = False
    for robot in config.robots:
        ctrl = select_controller(config, robot)
        if ctrl is None:
            continue
        has_runnable_robot = True
        try:
            env = PlatformEnv(config, lab_root, robot_id=robot.id, controller_id=ctrl.id)
            env.close()
        except Exception as exc:
            issues.append(f"robot {robot.id}: scene build failed: {exc}")
            continue
        if any(c.mode == "cartesian_delta_ik" for c in config.controllers) and robot.id == "lux3d_5r":
            import arm
            ranges = None
            unreachable = 0
            for x in np.linspace(-.16, .14, 5):
                for y in np.linspace(-.09, .06, 4):
                    for z in np.linspace(.03, .17, 4):
                        q = np.asarray(arm.ik([x, y, z]))
                        if np.hypot(np.hypot(x, y - arm.BASE_Y), z + arm.TOOL - arm.SHOULDER_Z) > arm.L1 + arm.L2:
                            unreachable += 1
            if unreachable:
                issues.append(f"robot {robot.id}: {unreachable} task-region IK targets unreachable")
    if not has_runnable_robot:
        issues.append("project requires at least one compatible robot/controller pair")
    return issues
