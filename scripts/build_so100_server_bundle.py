"""Package a standalone GPU/NPU training bundle for SO-ARM100 Action Chunking + 450M SmolVLA.

Produces `exports/so100_server_bundle/` containing:
- `scripts/train_smolvla.py`, `scripts/smolvla_worker.py` (real 450M SmolVLA Flow-Matching)
- `scripts/train_image_bc.py`, `scripts/visual_bc_model.py` (fast CNN Action Chunking baseline)
- minimal `vlab/` runtime modules (`schema.py`, `compat.py`, `scene_assets.py`, `visual_infer.py`, `smolvla_infer.py`)
- `datasets/lerobot/<dataset>/` (LeRobot v3 parquet + meta + dual-camera frames)
- `run_3090_smolvla.sh`, `run_ascend910_smolvla.sh`, `run_3090.sh`, `run_ascend910.sh`
- `requirements.txt` and `README_SERVER_TRAIN.md`
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def build_bundle(lab_root: Path, out_dir: Path, dataset_id: str = "so100_real_demos",
                 model_out: str = "so100_act_policy", chunk_size: int = 8, epochs: int = 160) -> Path:
    lab_root = Path(lab_root).resolve()
    out_dir = Path(out_dir).resolve()
    ds_src = lab_root / "datasets" / "lerobot" / dataset_id
    if not (ds_src / "meta" / "simulation.json").is_file():
        raise FileNotFoundError(f"LeRobot dataset not found: {ds_src}")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "scripts").mkdir(parents=True, exist_ok=True)
    (out_dir / "vlab").mkdir(parents=True, exist_ok=True)
    (out_dir / "datasets" / "lerobot").mkdir(parents=True, exist_ok=True)

    for rel in (
        "scripts/train_image_bc.py",
        "scripts/visual_bc_model.py",
        "scripts/train_smolvla.py",
        "scripts/smolvla_worker.py",
        "vlab/schema.py",
        "vlab/compat.py",
        "vlab/scene_assets.py",
        "vlab/visual_infer.py",
        "vlab/smolvla_infer.py",
    ):
        shutil.copy2(lab_root / rel, out_dir / rel)
    (out_dir / "vlab" / "__init__.py").write_text("", encoding="utf-8")

    mf_src = lab_root / "datasets" / "collections" / dataset_id / "manifest.json"
    if mf_src.is_file():
        mf_dst = out_dir / "datasets" / "collections" / dataset_id / "manifest.json"
        mf_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(mf_src, mf_dst)

    shutil.copytree(ds_src, out_dir / "datasets" / "lerobot" / dataset_id)

    (out_dir / "requirements.txt").write_text(
        "torch>=2.2\nnumpy>=1.26\npillow>=10.0\npydantic>=2.6\n"
        "transformers>=4.49\nsafetensors>=0.4\nlerobot>=0.4.0\n",
        encoding="utf-8",
    )

    run_3090_smolvla = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'cd "$(dirname "$0")"\n'
        "export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}\n"
        "export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}\n"
        f"python scripts/train_smolvla.py --lab . --dataset {dataset_id} "
        "--model-out so100_smolvla --cameras front,wrist --chunk-size 16 "
        "--steps 1000 --batch-size 16 --lr 1e-4 --device cuda \"$@\"\n"
        'echo "SmolVLA training finished. Copy datasets/models/so100_smolvla and datasets/reports/train_so100_smolvla.json back to RoboForge."\n'
    )
    (out_dir / "run_3090_smolvla.sh").write_text(run_3090_smolvla, encoding="utf-8", newline="\n")

    run_910_smolvla = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'cd "$(dirname "$0")"\n'
        "export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}\n"
        "export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0}\n"
        f"python scripts/train_smolvla.py --lab . --dataset {dataset_id} "
        "--model-out so100_smolvla --cameras front,wrist --chunk-size 16 "
        "--steps 1000 --batch-size 16 --lr 1e-4 --device npu \"$@\"\n"
        'echo "SmolVLA training finished on Ascend 910. Copy datasets/models/so100_smolvla and datasets/reports/train_so100_smolvla.json back to RoboForge."\n'
    )
    (out_dir / "run_ascend910_smolvla.sh").write_text(run_910_smolvla, encoding="utf-8", newline="\n")

    run_3090 = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'cd "$(dirname "$0")"\n'
        "export HF_HUB_OFFLINE=1\n"
        "export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}\n"
        f"python scripts/train_image_bc.py --lab . --dataset {dataset_id} "
        f"--model-out {model_out} --cameras front,wrist --chunk-size {chunk_size} "
        f"--epochs {epochs} --device cuda\n"
        'echo "Training finished. Copy datasets/models/* and datasets/reports/* back to RoboForge."\n'
    )
    (out_dir / "run_3090.sh").write_text(run_3090, encoding="utf-8", newline="\n")

    run_910 = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'cd "$(dirname "$0")"\n'
        "export HF_HUB_OFFLINE=1\n"
        "export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0}\n"
        f"python scripts/train_image_bc.py --lab . --dataset {dataset_id} "
        f"--model-out {model_out} --cameras front,wrist --chunk-size {chunk_size} "
        f"--epochs {epochs} --device npu\n"
        'echo "Training finished. Copy datasets/models/* and datasets/reports/* back to RoboForge."\n'
    )
    (out_dir / "run_ascend910.sh").write_text(run_910, encoding="utf-8", newline="\n")

    readme = f"""# SO-ARM100 Standalone Server Training Bundle (`{dataset_id}`)

## Hardware & VRAM Footprint (Well below 50 GB / 3×3090 budget)
1. **Real 450M `SmolVLA` (`SmolVLM2-500M` + Flow-Matching Action Expert, `so100_smolvla`)** — **Recommended**:
   - Inputs: Dual cameras (`front + wrist` 256×256) + 18D SO-ARM100 proprioception + natural language task prompt
   - Outputs: 16-step 4D continuous delta-IK + gripper Flow-Matching trajectory chunk
   - **VRAM usage (frozen VLM prefix cache + bf16 expert)**: **~3.5–5.5 GB** on a **single RTX 3090 (24 GB)** or **single Ascend 910 (32 GB)** (vs ~11–14 GB if unfreezing the full VLM)
   - **Wall-clock time**: **~3–6 minutes** on 1× RTX 3090 for 1,000 Flow-Matching steps (`batch_size=16`)
   - *Base weights*: place `work/smolvla-libero` and `work/smolvlm-processor` under `./base_weights/` (or pass `--base-weights` / `--processor`).
2. **Lightweight Fast CNN Baseline (`{model_out}`, `chunk_size={chunk_size}`)**:
   - **VRAM usage**: **< 1.5 GB** on 1× RTX 3090 / 1× Ascend 910, runs **< 1 minute** for `{epochs}` epochs.

## Quick Start on RTX 3090 Server (SmolVLA)
```bash
pip install -r requirements.txt
bash run_3090_smolvla.sh
```

## Quick Start on Huawei Ascend 910 Server (SmolVLA)
```bash
# Ensure torch + torch_npu are activated in your CANN conda env
bash run_ascend910_smolvla.sh
```

## Drop Weights Back Into RoboForge
After training finishes, copy the generated artifacts back into `outputs/vla-lab/`:
- `datasets/models/so100_smolvla/` (`config.json`, `model.safetensors`, `so100_stats.json`)
- `datasets/reports/train_so100_smolvla.json`

Once copied back, `so100_smolvla (front+wrist)` appears automatically in the RoboForge toolbar (`运行视觉策略`) for live closed-loop MuJoCo rollout.
"""
    (out_dir / "README_SERVER_TRAIN.md").write_text(readme, encoding="utf-8", newline="\n")
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--out", default=None)
    ap.add_argument("--dataset", default="so100_real_demos")
    ap.add_argument("--model-out", default="so100_act_policy")
    ap.add_argument("--chunk-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=160)
    args = ap.parse_args()
    lab = Path(args.lab)
    out = Path(args.out) if args.out else lab / "exports" / "so100_server_bundle"
    res = build_bundle(lab, out, dataset_id=args.dataset, model_out=args.model_out,
                       chunk_size=args.chunk_size, epochs=args.epochs)
    print(f"Bundle ready at {res}")


if __name__ == "__main__":
    main()
