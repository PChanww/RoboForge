#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
python scripts/train_smolvla.py --lab . --dataset so100_real_demos --model-out so100_smolvla --cameras front,wrist --chunk-size 16 --steps 1000 --batch-size 16 --lr 1e-4 --device cuda "$@"
echo "SmolVLA training finished. Copy datasets/models/so100_smolvla and datasets/reports/train_so100_smolvla.json back to RoboForge."
