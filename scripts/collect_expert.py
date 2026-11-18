"""Collect expert episodes into datasets/staging. Run from vla-lab:
python scripts/collect_expert.py [--out datasets/staging]"""
import argparse
import json
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))

from vlab.collect import collect
from vlab.schema import ProjectConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(LAB / "datasets" / "staging"))
    ap.add_argument("--config", default=str(LAB / "configs" / "default.json"))
    ap.add_argument("--expert", default="servo", choices=["servo", "openloop"])
    ap.add_argument("--plan", help="JSON asset/seed episode plan; defaults to project asset or 21-episode baseline")
    ap.add_argument("--episodes", type=int, default=0, help="Number of seeds on config task.asset (overrides default plan)")
    ap.add_argument("--provenance", help="Inline JSON or path to JSON: recipe/spec revision + generation source metadata")
    args = ap.parse_args()
    cfg = ProjectConfig.model_validate_json(Path(args.config).read_text(encoding="utf-8"))
    provenance = None
    if args.provenance:
        try:
            provenance = json.loads(args.provenance)
        except json.JSONDecodeError:
            provenance = json.loads(Path(args.provenance).read_text(encoding="utf-8"))
    if args.plan:
        plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    elif args.episodes and args.episodes > 0:
        plan = [{"asset": cfg.task.asset, "seed": s} for s in range(args.episodes)]
    elif cfg.project_id == "default" and cfg.task.asset == "standard":
        plan = ([{"asset": a, "seed": s} for a in ("standard", "tall", "wide") for s in range(5)]
                + [{"asset": "standard", "seed": 30}, {"asset": "standard", "seed": 31},
                   {"asset": "tall", "seed": 30}, {"asset": "wide", "seed": 30},
                   {"asset": "imported/lux3d-ivory", "seed": 0}, {"asset": "imported/lux3d-ivory", "seed": 1}])
    else:
        plan = [{"asset": cfg.task.asset, "seed": s} for s in range(6)]
    collect(cfg, LAB, plan, Path(args.out), expert=args.expert, provenance=provenance)


if __name__ == "__main__":
    main()
