"""Verify auxiliary model adapter (no key -> waiting) + structured import + plugin loop."""
import json
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
from vlab.aux_model import AuxModel, validate_suggestion
from vlab.engine import PlatformEnv
from vlab.plugins import load_plugin, run_plugin_episode
from vlab.schema import ProjectConfig


def main():
    cfg = ProjectConfig.model_validate_json((LAB / "configs" / "default.json").read_text(encoding="utf-8"))
    aux = AuxModel()
    status = aux.status()
    waiting = aux.suggest("design a 4-dof arm")
    suggestion = {"robots": cfg.model_dump()["robots"], "notes": "structured import test"}
    check = validate_suggestion(suggestion, cfg, LAB)
    create, info = load_plugin(LAB / "plugins" / "example_openloop.py")
    env = PlatformEnv(cfg, LAB)
    try:
        loop = run_plugin_episode(env, create, max_ticks=40)
    finally:
        env.close()
    report = {"aux_model_status": status, "aux_waiting_response": waiting.get("status"),
              "structured_import": check, "plugin": {**info, "loop": loop}}
    (LAB / "datasets" / "reports").mkdir(parents=True, exist_ok=True)
    (LAB / "datasets" / "reports" / "aux_plugin.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"aux_state": status["state"], "structured_valid": check["valid"],
                      "plugin_sha": info["sha256"][:12], "plugin_ticks": loop["tick"],
                      "plugin_error": loop.get("error")}))


if __name__ == "__main__":
    main()