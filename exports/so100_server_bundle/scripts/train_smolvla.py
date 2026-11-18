"""Fine-tune the real 450M SmolVLA (SmolVLM2-500M + Flow-Matching Action Expert) on SO-ARM100 LeRobot v3 data.

Uses dual cameras (front + wrist), 18-dim proprioceptive state, and natural-language
task instructions to predict multi-step 4-dim delta-IK + gripper action chunks.
Works on NVIDIA RTX 3090 / 4070 CUDA GPUs as well as Huawei Ascend 910 NPUs.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vlab.compat import visual_contract  # noqa: E402
from vlab.scene_assets import valid_id  # noqa: E402
from vlab.schema import ProjectConfig  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks, pad_vector  # noqa: E402

JOINTS = ["waist", "shoulder", "elbow", "wrist_pitch", "wrist_yaw", "left", "right"]


def _select_device(requested: str) -> torch.device:
    req = (requested or "auto").strip().lower()
    if req != "auto":
        return torch.device(req)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu")
    return torch.device("cpu")


def _default_pretrained_paths(lab: Path) -> tuple[Path, Path]:
    for cand_root in (lab, lab.parent, lab.parent.parent):
        w1 = cand_root / "base_weights" / "smolvla-libero"
        p1 = cand_root / "base_weights" / "smolvlm-processor"
        if w1.is_dir() and p1.is_dir():
            return w1, p1
        w2 = cand_root / "work" / "smolvla-libero"
        p2 = cand_root / "work" / "smolvlm-processor"
        if w2.is_dir() and p2.is_dir():
            return w2, p2
    raise FileNotFoundError("Could not locate base weights (smolvla-libero & smolvlm-processor)")


def _resize_chw_256(img: torch.Tensor) -> torch.Tensor:
    if img.shape[0] != 3:
        img = img.permute(2, 0, 1)
    if img.shape[1] != 256 or img.shape[2] != 256:
        img = F.interpolate(img.unsqueeze(0), size=(256, 256), mode="bilinear", align_corners=False)[0]
    return img


def _load_dataset(ds: LeRobotDataset, camera_keys: list[str], chunk_size: int):
    cams = {c: [] for c in camera_keys}
    states, acts, eps, tasks = [], [], [], []
    for i in range(len(ds)):
        s = ds[i]
        for cam in camera_keys:
            k = f"observation.images.{cam}"
            if k not in s:
                raise ValueError(f"Dataset missing camera stream {k!r}")
            cams[cam].append(_resize_chw_256(s[k]))
        states.append(s["observation.state"])
        acts.append(s["action"])
        eps.append(int(s["episode_index"]))
        tasks.append(str(s.get("task") or "Pick up the object and place it inside the tray."))
    cam_tensors = {c: torch.stack(v) for c, v in cams.items()}
    state = torch.stack(states).float()
    act = torch.stack(acts).float()
    ep = np.asarray(eps)

    act_chunks = torch.zeros((len(act), chunk_size, act.shape[1]), dtype=torch.float32)
    for e in np.unique(ep):
        idx = np.where(ep == e)[0]
        end = len(idx) - 1
        for k in range(chunk_size):
            gather_pos = [idx[min(pos + k, end)] for pos in range(len(idx))]
            act_chunks[idx, k] = act[gather_pos]
    return cam_tensors, state, act_chunks, ep, tasks


@torch.no_grad()
def _embed_frozen_img_lang(policy: SmolVLAPolicy, images, img_masks, lang_tokens, lang_masks):
    embs = []
    pad_masks = []
    vlm_exp = policy.model.vlm_with_expert
    for img, img_mask in zip(images, img_masks, strict=False):
        if policy.model.add_image_special_tokens:
            st = vlm_exp.embed_language_tokens(
                policy.model.global_image_start_token.to(device=img.device)
            ).unsqueeze(0).expand(img.shape[0], -1, -1)
            embs.append(st)
            pad_masks.append(torch.ones_like(st[:, :, 0], dtype=torch.bool))
        img_emb = vlm_exp.embed_image(img)
        img_emb = img_emb * math.sqrt(img_emb.shape[-1])
        bsize, n_emb = img_emb.shape[:2]
        embs.append(img_emb)
        pad_masks.append(img_mask[:, None].expand(bsize, n_emb))
        if policy.model.add_image_special_tokens:
            et = vlm_exp.embed_language_tokens(
                policy.model.image_end_token.to(device=img.device)
            ).unsqueeze(0).expand(img.shape[0], -1, -1)
            embs.append(et)
            pad_masks.append(torch.ones_like(et[:, :, 0], dtype=torch.bool))
    lang_emb = vlm_exp.embed_language_tokens(lang_tokens)
    lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
    embs.append(lang_emb)
    pad_masks.append(lang_masks)
    return torch.cat(embs, dim=1), torch.cat(pad_masks, dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab", required=True)
    ap.add_argument("--dataset", default="so100_real_demos")
    ap.add_argument("--model-out", default="so100_smolvla")
    ap.add_argument("--base-weights", default=None)
    ap.add_argument("--processor", default=None)
    ap.add_argument("--cameras", default="front,wrist")
    ap.add_argument("--chunk-size", type=int, default=16)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=None, help="Optional epoch override to compute steps")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    valid_id(args.dataset)
    valid_id(args.model_out)
    lab = Path(args.lab).resolve()
    camera_keys = [valid_id(c.strip()) for c in args.cameras.split(",") if c.strip()]
    if not camera_keys:
        raise ValueError("Provide at least one camera key")
    dev = _select_device(args.device)

    default_w, default_p = _default_pretrained_paths(lab)
    base_weights = Path(args.base_weights).resolve() if args.base_weights else default_w
    processor_dir = Path(args.processor).resolve() if args.processor else default_p

    meta_path = lab / "datasets" / "lerobot" / args.dataset / "meta" / "simulation.json"
    if not meta_path.is_file():
        raise ValueError("Dataset lacks meta/simulation.json contract")
    source = json.loads(meta_path.read_text(encoding="utf-8"))
    visual_contract(ProjectConfig.model_validate(source["config_snapshot"]))

    ds = LeRobotDataset(repo_id=f"local/{args.dataset}", root=lab / "datasets" / "lerobot" / args.dataset)
    cam_tensors, state, act_chunks, ep, tasks = _load_dataset(ds, camera_keys, args.chunk_size)
    state_dim = int(state.shape[1])
    act_dim = int(act_chunks.shape[2])

    unique_eps = sorted(np.unique(ep).tolist())
    n_ep = len(unique_eps)
    n_tr = max(1, min(n_ep - 1, int(round(0.7 * n_ep))))
    tr_set = set(unique_eps[:n_tr])
    rest = unique_eps[n_tr:]
    va_set = set(rest[:max(1, len(rest) // 2)])
    gen_set = set(rest[max(1, len(rest) // 2):]) or va_set
    tr = np.isin(ep, list(tr_set))
    va = np.isin(ep, list(va_set))
    gen = np.isin(ep, list(gen_set))

    smean, sstd = state[tr].mean(0), state[tr].std(0) + 1e-6
    amean, astd = act_chunks[tr, 0].mean(0), act_chunks[tr, 0].std(0) + 1e-6
    sn = (state - smean) / sstd
    an = (act_chunks - amean) / astd

    cfg = PreTrainedConfig.from_pretrained(base_weights, local_files_only=True)
    cfg.vlm_model_name = str(processor_dir)
    cfg.load_vlm_weights = False
    cfg.device = "cpu"
    cfg.freeze_vision_encoder = True
    cfg.train_expert_only = True
    cfg.train_state_proj = True
    cfg.chunk_size = int(args.chunk_size)
    cfg.n_action_steps = int(args.chunk_size)
    cfg.num_steps = 10
    cfg.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
        **{f"observation.images.{c}": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)) for c in camera_keys},
    }
    cfg.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(act_dim,)),
    }

    policy = SmolVLAPolicy.from_pretrained(base_weights, config=cfg, strict=True)
    policy.to(dev)
    policy.model.vlm_with_expert.set_requires_grad()
    policy.model.set_requires_grad()
    for mod in (
        policy.model.state_proj,
        policy.model.action_in_proj,
        policy.model.action_out_proj,
        policy.model.action_time_mlp_in,
        policy.model.action_time_mlp_out,
    ):
        for p in mod.parameters():
            p.requires_grad = True
    for name, p in policy.model.vlm_with_expert.lm_expert.named_parameters():
        if "lm_head" not in name:
            p.requires_grad = True

    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)

    tok = policy.model.vlm_with_expert.processor.tokenizer
    tok.padding_side = "right"
    unique_task = tasks[0].rstrip() + "\n"
    lang_tok = tok(
        [unique_task], padding="max_length", max_length=cfg.tokenizer_max_length, truncation=True, return_tensors="pt"
    )
    single_ids = lang_tok.input_ids.to(dev)
    single_mask = lang_tok.attention_mask.bool().to(dev)

    # Precompute frozen vision+language prefix tokens once (exact since vlm is frozen)
    use_bf16 = dev.type in ("cuda", "npu")
    n_total = len(state)
    cached_embs = []
    cached_masks = []
    cache_bs = 16
    policy.eval()
    t_cache0 = time.perf_counter()
    for start in range(0, n_total, cache_bs):
        end = min(n_total, start + cache_bs)
        mini = {f"observation.images.{c}": cam_tensors[c][start:end].to(dev) for c in camera_keys}
        imgs, img_masks = policy.prepare_images(mini)
        b = end - start
        lt = single_ids.expand(b, -1)
        lm = single_mask.expand(b, -1)
        with torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_bf16):
            emb_c, mask_c = _embed_frozen_img_lang(policy, imgs, img_masks, lt, lm)
        cached_embs.append(emb_c.cpu())
        cached_masks.append(mask_c.cpu())
    all_il_embs = torch.cat(cached_embs, dim=0)
    all_il_masks = torch.cat(cached_masks, dim=0)
    l_il = int(all_il_embs.shape[1])
    print(f"Cached frozen VLM prefixes ({all_il_embs.shape}) in {time.perf_counter() - t_cache0:.2f}s", flush=True)

    train_idx_all = torch.from_numpy(np.where(tr)[0]).long()
    total_steps = int(args.steps)
    if args.epochs is not None:
        total_steps = max(1, int(args.epochs) * max(1, len(train_idx_all) // max(1, args.batch_size)))

    opt_params = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(opt_params, lr=args.lr, betas=(0.9, 0.95), weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=max(1e-6, args.lr * 0.05))

    torch.manual_seed(0)
    policy.train()
    base_il_att = torch.zeros(l_il + 1, dtype=torch.bool, device=dev)
    base_il_att[-1] = True  # image+language = 0, state = 1
    peak_vram_mb = 0.0

    for step in range(1, total_steps + 1):
        perm = train_idx_all[torch.randint(0, len(train_idx_all), (args.batch_size,))]
        il_emb = all_il_embs[perm].to(dev, dtype=torch.float32)
        il_mask = all_il_masks[perm].to(dev)
        st_b = pad_vector(sn[perm].to(dev), cfg.max_state_dim)
        act_b = pad_vector(an[perm].to(dev), cfg.max_action_dim)

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_bf16):
            state_emb = policy.model.state_proj(st_b)
            if state_emb.ndim == 2:
                state_emb = state_emb[:, None, :]
            prefix_embs = torch.cat([il_emb.to(state_emb.dtype), state_emb], dim=1)
            prefix_pad = torch.cat([il_mask, torch.ones((len(perm), 1), dtype=torch.bool, device=dev)], dim=1)
            prefix_att = base_il_att[None, :].expand(len(perm), -1)

            noise = policy.model.sample_noise(act_b.shape, dev)
            flow_t = policy.model.sample_time(len(perm), dev)
            t_exp = flow_t[:, None, None]
            x_t = t_exp * noise + (1.0 - t_exp) * act_b
            u_t = noise - act_b
            suffix_embs, suffix_pad, suffix_att = policy.model.embed_suffix(x_t, flow_t)
            pad_masks = torch.cat([prefix_pad, suffix_pad], dim=1)
            att_masks = torch.cat([prefix_att, suffix_att], dim=1)
            att_2d = make_att_2d_masks(pad_masks, att_masks)
            pos_ids = torch.cumsum(pad_masks, dim=1) - 1
            (_, suffix_out), _ = policy.model.vlm_with_expert.forward(
                attention_mask=att_2d,
                position_ids=pos_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                fill_kv_cache=False,
            )
            suffix_out = suffix_out[:, -cfg.chunk_size:].float()
            v_t = policy.model.action_out_proj(suffix_out)
            flow_loss = F.mse_loss(v_t[:, :, :act_dim], u_t[:, :, :act_dim]) + 0.02 * F.mse_loss(v_t[:, :, act_dim:], u_t[:, :, act_dim:])
            pred_a0 = x_t - t_exp * v_t
            aux_mse = F.mse_loss(pred_a0[:, :, :act_dim], act_b[:, :, :act_dim])
            loss = flow_loss + 0.5 * aux_mse

        loss.backward()
        torch.nn.utils.clip_grad_norm_(opt_params, 10.0)
        opt.step()
        sched.step()
        if dev.type == "cuda":
            peak_vram_mb = max(peak_vram_mb, torch.cuda.max_memory_allocated() / 1e6)
        if step % 50 == 0 or step == total_steps:
            print({
                "step": step,
                "loss": round(float(loss), 5),
                "flow_loss": round(float(flow_loss), 5),
                "aux_mse": round(float(aux_mse), 5),
                "lr": round(float(sched.get_last_lr()[0]), 7),
            }, flush=True)

    # Full end-to-end 10-step Flow-Matching ODE evaluation
    policy.eval()

    @torch.no_grad()
    def _eval_split(mask_np: np.ndarray, max_samples: int = 48) -> float:
        idxs = np.where(mask_np)[0]
        if len(idxs) > max_samples:
            pick = np.linspace(0, len(idxs) - 1, max_samples, dtype=int)
            idxs = idxs[pick]
        sq_errs = []
        amean_t = amean.to(dev)
        astd_t = astd.to(dev)
        for start in range(0, len(idxs), 8):
            sub = idxs[start:start + 8]
            b = len(sub)
            batch = {
                "observation.state": sn[sub].to(dev),
                "observation.language.tokens": single_ids.expand(b, -1),
                "observation.language.attention_mask": single_mask.expand(b, -1),
                **{f"observation.images.{c}": cam_tensors[c][sub].to(dev) for c in camera_keys},
            }
            zero_noise = torch.zeros((b, cfg.chunk_size, cfg.max_action_dim), device=dev, dtype=torch.float32)
            with torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_bf16):
                pred_norm = policy.predict_action_chunk(batch, noise=zero_noise)
            pred_unnorm = pred_norm.float() * astd_t + amean_t
            target_unnorm = act_chunks[sub].to(dev)
            sq_errs.append(((pred_unnorm - target_unnorm) ** 2).mean().item())
        return float(np.mean(sq_errs))

    val_mse = _eval_split(va)
    gen_mse = _eval_split(gen)

    out_dir = lab / "datasets" / "models" / args.model_out
    out_dir.mkdir(parents=True, exist_ok=True)
    policy.cpu()
    policy.save_pretrained(out_dir)

    stats_payload = {
        "model_id": args.model_out,
        "arch": "smolvla",
        "camera_keys": camera_keys,
        "state_dim": state_dim,
        "action_dim": act_dim,
        "chunk_size": int(args.chunk_size),
        "replan_steps": 4,
        "state_joints": source.get("joints") or JOINTS,
        "task_instruction": tasks[0],
        "state_mean": smean.numpy().tolist(),
        "state_std": sstd.numpy().tolist(),
        "action_mean": amean.numpy().tolist(),
        "action_std": astd.numpy().tolist(),
    }
    (out_dir / "so100_stats.json").write_text(json.dumps(stats_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    render_meta = {}
    mf_path = lab / "datasets" / "collections" / args.dataset / "manifest.json"
    if mf_path.is_file():
        mf = json.loads(mf_path.read_text(encoding="utf-8"))
        render_meta = {k: mf.get(k) for k in ("render_mode", "sensor_renderer", "renderer_version") if mf.get(k) is not None}

    summary = {
        "type": "SmolVLA (450M VLM + Flow-Matching Action Expert, front+wrist+language+state)",
        "arch": "smolvla",
        "camera_keys": camera_keys,
        "chunk_size": int(args.chunk_size),
        "steps": total_steps,
        "device": str(dev),
        "total_params_m": round(total_params / 1e6, 2),
        "trainable_params_m": round(trainable_params / 1e6, 2),
        "vram_mb": round(peak_vram_mb, 1),
        "episodes": len(unique_eps),
        "frames": int(len(state)),
        "train_frames": int(tr.sum()),
        "val_mse": round(val_mse, 5),
        "gen_mse": round(gen_mse, 5),
        "dataset": args.dataset,
        "render_meta": render_meta,
        "checkpoint_dir": str(out_dir),
    }
    reports = lab / "datasets" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / f"train_{args.model_out}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
