# SO-ARM100 Standalone Server Training Bundle (`so100_real_demos`)

## Hardware & VRAM Footprint (Well below 50 GB / 3×3090 budget)
- **Default architecture (`so100_act_policy`)**: Dual-camera (`front + wrist`) CNN + Proprioception + Action Chunking (`chunk_size=8`, 32 output action dims over 8 horizon ticks)
  - **VRAM usage**: **< 1.5 GB** on single RTX 3090 (or single Ascend 910 NPU)
  - **Wall-clock time**: **< 1 minute** on 1× RTX 3090 for `160` epochs
- **Optional full VLM-based VLA (`SmolVLA` 450M Flow-Matching)**:
  - **VRAM usage**: **~11–14 GB** fp16/bf16 on a **single RTX 3090 (24 GB)**
  - Fits easily within 1 card (< 50 GB limit).

## Quick Start on RTX 3090 Server
```bash
pip install -r requirements.txt
bash run_3090.sh
```

## Quick Start on Huawei Ascend 910 Server
```bash
# Ensure torch + torch_npu are activated in your CANN conda env
bash run_ascend910.sh
```

## Drop Weights Back Into RoboForge
After training finishes, copy the generated artifacts back into `outputs/vla-lab/`:
- `datasets/models/so100_act_policy.npz`
- `datasets/models/so100_act_policy.pt`
- `datasets/reports/train_so100_act_policy.json`

*(Even if only `so100_act_policy.pt` is copied back, `vlab/visual_infer.py` automatically converts `.pt` to `.npz` on load.)*
