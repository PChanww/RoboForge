#!/usr/bin/env bash
# Reuse the policy runtime and assets prepared in this Windows workspace.
set -eu
cd "$(dirname "$0")/../../.."
assets="$PWD/work"
runtime="${SMOLVLA_PYTHON:-$HOME/miniconda3/envs/voxcpm/bin/python}"
if [[ ! -x "$runtime" || ! -d "$assets/policy/deps" || ! -f "$assets/smolvla-libero/model.safetensors" ]]; then
    echo "This workspace's policy runtime is missing. For a portable install, see integrations/README.md." >&2
    exit 2
fi
export PYTHONPATH="$assets/policy/deps${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
exec "$runtime" -u outputs/vla-lab/integrations/smolvla.py \
    --weights "$assets/smolvla-libero" --processor "$assets/smolvlm-processor" "$@"
