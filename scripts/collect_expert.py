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
    ap.add_argument("--plan", help="JSON asset/seed episode plan; defaults to the existing 21-episode baseline")
    ap.add_argument("--preview-dir", help="Write live camera frames and collection progress to this directory")
    ap.add_argument("--realtime", action="store_true", help="Pace collection to simulation time for live viewing")
    ap.add_argument("--provenance", help="Inline JSON or path to JSON: recipe/spec revision + generation source metadata")
    args = ap.parse_args()
    cfg = ProjectConfig.model_validate_json(Path(args.config).read_text(encoding="utf-8"))
    provenance = None
    if args.provenance:
        try:
            provenance = json.loads(args.provenance)
        except json.JSONDecodeError:
            provenance = json.loads(Path(args.provenance).read_text(encoding="utf-8"))
    plan = ([{"asset": a, "seed": s} for a in ("standard", "tall", "wide") for s in range(5)]
            + [{"asset": "standard", "seed": 30}, {"asset": "standard", "seed": 31},
               {"asset": "tall", "seed": 30}, {"asset": "wide", "seed": 30},
               {"asset": "imported/lux3d-ivory", "seed": 0}, {"asset": "imported/lux3d-ivory", "seed": 1}])
    if args.plan:
        plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    collect(cfg, LAB, plan, Path(args.out), expert=args.expert, provenance=provenance,
            preview_dir=Path(args.preview_dir) if args.preview_dir else None, realtime=args.realtime)


if __name__ == "__main__":
    main()
