"""Python controller plugin protocol.

A plugin module exposes `create_controller(config)` returning an object with
`reset(observation)` and `act(observation) -> list[float]`. Loaded by explicit
path; the source SHA-256 and errors are recorded. Arbitrary/untrusted remote code
must run in a subprocess with resource limits (not enabled here) — this loader is
for locally audited plugins only.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path


class ControllerPlugin:
    def reset(self, observation):
        pass

    def act(self, observation) -> list[float]:
        raise NotImplementedError


def load_plugin(path: Path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"plugin not found: {path}")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location(f"vlab_plugin_{sha[:8]}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "create_controller"):
        raise ValueError(f"plugin {path} missing create_controller(config)")
    return module.create_controller, {"path": str(path), "sha256": sha}


def run_plugin_episode(env, create_controller, max_ticks=200):
    """Observation-action loop with a plugin; validates actions and records errors."""
    plugin = create_controller(env.config)
    obs = env.observation()
    plugin.reset(obs)
    for _ in range(max_ticks):
        action = plugin.act(obs)
        try:
            obs = env.step(list(action))
        except ValueError as exc:
            return {"success": False, "error": f"invalid plugin action: {exc}", "tick": env.scheduler.tick}
        if env.metrics()["success"]:
            break
    return {"success": env.metrics()["success"], "tick": env.scheduler.tick}