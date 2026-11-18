# SO-ARM100 Standalone Server Training Bundle (`so100_real_demos`)

## Hardware & VRAM Footprint (Well below 50 GB / 3×3090 budget)
1. **Real 450M `SmolVLA` (`SmolVLM2-500M` + Flow-Matching Action Expert, `so100_smolvla`)** — **Recommended**:
   - Inputs: Dual cameras (`front + wrist` 256×256) + 18D SO-ARM100 proprioception + natural language task prompt
   - Outputs: 16-step 4D continuous delta-IK + gripper Flow-Matching trajectory chunk
   - **VRAM usage (frozen VLM prefix cache + bf16 expert)**: **~3.5–5.5 GB** on a **single RTX 3090 (24 GB)** or **single Ascend 910 (32 GB)** (vs ~11–14 GB if unfreezing the full VLM)
   - **Wall-clock time**: **~3–6 minutes** on 1× RTX 3090 for 1,000 Flow-Matching steps (`batch_size=16`)
   - *Base weights*: place `work/smolvla-libero` and `work/smolvlm-processor` under `./base_weights/` (or pass `--base-weights` / `--processor`).
2. **Lightweight Fast CNN Baseline (`so100_act_policy`, `chunk_size=8`)**:
   - **VRAM usage**: **< 1.5 GB** on 1× RTX 3090 / 1× Ascend 910, runs **< 1 minute** for `160` epochs.

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

## General (multi-task / instruction-conditioned) training

The trainer now conditions on **each sample's own task instruction** (per-frame `task` from the
LeRobot dataset) instead of one global prompt expanded to the whole batch. This is what turns the
model from "single-task" into a general, language-conditioned VLA — provided the dataset actually
contains **multiple distinct tasks/instructions**.

- Dataset requirement: a LeRobot dataset whose `task`/instruction varies across episodes (e.g.
  cup / canister / tray / static-prop pick-and-place). A single-task dataset still trains, but the
  model will only follow that task.
- Train command (single GPU 3090):
  ```bash
  bash run_3090_smolvla.sh --dataset <multi_task_dataset> --model-out so100_smolvla_multi --steps 3000
  ```
- `so100_stats.json` now also records `instructions: [..]` (unique list) and `task_by_episode`.
  The web UI ("运行视觉策略") shows a language instruction input for these models; fixed-action
  VisualBC/ACT models keep the same runner contract and simply ignore the instruction.

### Ready-to-train multi-task dataset (`vla_multi_task`)

A 10-episode, 5-instruction dataset has been collected in sim and converted to LeRobot:
`datasets/lerobot/vla_multi_task` (front+wrist, 18-D state, 4-D delta-IK, 745 frames, 5 distinct tasks:
standard / tall / wide / blue cup / ceramic tumbler, all 10/10 expert success).

Upload to the GPU server, then train with per-sample instruction conditioning:
```bash
# server-side layout
#   ./exports/so100_server_bundle/   (this bundle)
#   ./datasets/lerobot/vla_multi_task
#   ./base_weights/smolvla-libero, ./base_weights/smolvlm-processor   (or pass --base-weights/--processor)
bash run_3090_smolvla.sh --dataset vla_multi_task --model-out so100_smolvla_multi --steps 3000 --batch-size 16
```
Return `datasets/models/so100_smolvla_multi/` (config.json, model.safetensors, so100_stats.json)
and `datasets/reports/train_so100_smolvla_multi.json` into the RoboForge `vla-lab` checkout; it then
appears in the toolbar with a language instruction input. Note: generalization is bounded by task
diversity — add more objects/instructions (and larger steps) for broader coverage.
