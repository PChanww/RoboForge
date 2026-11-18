#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0}
python scripts/train_image_bc.py --lab . --dataset so100_real_demos --model-out so100_act_policy --cameras front,wrist --chunk-size 8 --epochs 160 --device npu
echo "Training finished. Copy datasets/models/* and datasets/reports/* back to RoboForge."
