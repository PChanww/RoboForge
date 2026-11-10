"""Collect expert episodes into datasets/staging. Run from vla-lab:
python scripts/collect_expert.py [--out datasets/staging]"""
import argparse
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
    args = ap.parse_args()
    cfg = ProjectConfig.model_validate_json(Path(args.config).read_text(encoding="utf-8"))
    plan = ([{"asset": a, "seed": s} for a in ("standard", "tall", "wide") for s in range(5)]
            + [{"asset": "standard", "seed": 30}, {"asset": "standard", "seed": 31},
               {"asset": "tall", "seed": 30}, {"asset": "wide", "seed": 30},
               {"asset": "imported/lux3d-ivory", "seed": 0}, {"asset": "imported/lux3d-ivory", "seed": 1}])
    collect(cfg, LAB, plan, Path(args.out), expert=args.expert)


if __name__ == "__main__":
    main()