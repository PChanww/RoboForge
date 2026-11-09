"""Run a real LIBERO-trained SmolVLA policy against the lab's HTTP API."""
import argparse
import base64
import hashlib
import io
import json
import math
import time
from pathlib import Path
from urllib.request import Request, build_opener, ProxyHandler
from importlib.metadata import version

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


class Policy:
    def __init__(self, weights, processor, device):
        weights = Path(weights)
        config = PreTrainedConfig.from_pretrained(weights, local_files_only=True)
        config.vlm_model_name = str(Path(processor).resolve())
        # The complete policy checkpoint already contains the VLM parameters.
        config.load_vlm_weights = False
        config.device = "cpu"
        self.model = SmolVLAPolicy.from_pretrained(weights, config=config, strict=True)
        self.device = device
        self.model.to(device).eval()
        self.tokenizer = self.model.model.vlm_with_expert.processor.tokenizer
        self.tokenizer.padding_side = "right"
        self.state_stats = load_file(str(weights / "policy_preprocessor_step_5_normalizer_processor.safetensors"))
        self.action_stats = load_file(str(weights / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"))

    def camera(self, encoded):
        data = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(data)) as image:
            rgb = np.array(image.convert("RGB"))
        return torch.from_numpy(rgb).permute(2, 0, 1)[None].to(self.device, torch.float32) / 255

    @torch.inference_mode()
    def predict(self, packet):
        obs = packet["observation"]
        x, y, z = obs["eef"]
        jaw = float(obs["gripper"]) * .04
        # Approximate LIBERO's tabletop reference height and a down-facing TCP.
        state = torch.tensor([x, y, z + .60, math.pi, 0., 0., jaw, -jaw])
        state = (state - self.state_stats["observation.state.mean"]) / (self.state_stats["observation.state.std"] + 1e-8)
        tokens = self.tokenizer([obs["instruction"].rstrip() + "\n"], padding="max_length",
                                max_length=48, truncation=True, return_tensors="pt")
        batch = {
            "observation.state": state.unsqueeze(0).to(self.device),
            "observation.images.camera1": self.camera(packet["rgb_jpeg"]),
            "observation.images.camera2": self.camera(packet["wrist_jpeg"]),
            "observation.language.tokens": tokens.input_ids.to(self.device),
            "observation.language.attention_mask": tokens.attention_mask.bool().to(self.device),
        }
        actions = self.model.predict_action_chunk(batch)[0].float().cpu()
        actions = actions * self.action_stats["action.std"] + self.action_stats["action.mean"]
        if actions.shape != (50, 7) or not torch.isfinite(actions).all():
            raise ValueError("Expected 50 finite, seven-dimensional LIBERO actions")
        return actions.numpy()


def project(action):
    # LIBERO gripper: -1 open, +1 close. Gantry gripper uses the opposite sign.
    # Rotational commands cannot be executed by this Cartesian gantry.
    return np.clip([*action[:3], -action[6]], -1., 1.).tolist()


def run(args):
    torch.manual_seed(args.seed)
    with (Path(args.weights) / "model.safetensors").open("rb") as weights:
        checksum = hashlib.file_digest(weights, "sha256").hexdigest()
    policy = Policy(args.weights, args.processor, args.device)
    # Local simulation traffic need not pass through OS/WSL HTTP proxies.
    opener = build_opener(ProxyHandler({}))

    def api(path, body=None):
        request = Request(args.url.rstrip("/") + path,
                          data=None if body is None else json.dumps(body).encode(),
                          headers={"Content-Type": "application/json"})
        with opener.open(request, timeout=20) as response:
            return json.load(response)

    first = api("/api/reset", {"asset": args.asset, "seed": args.seed})
    episode = first["episode"]
    tick = first["tick"]
    official_sha = "9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8"
    report = {"checkpoint": "lerobot/smolvla_libero" if checksum == official_sha else "custom",
              "weights_sha256": checksum, "device": args.device,
              "runtime": {"lerobot": version("lerobot"), "torch": torch.__version__,
                          "transformers": version("transformers")},
              "asset": args.asset, "seed": args.seed, "episode": episode,
              "replan_interval": args.replan, "start": first["observation"], "chunks": []}
    while tick < args.steps:
        packet = api("/api/observation?images=1")
        if packet["episode"] != episode or packet["tick"] != tick:
            raise RuntimeError("Another controller changed the scene")
        start = time.perf_counter()
        raw = policy.predict(packet)
        record = {"tick": tick, "inference_seconds": round(time.perf_counter() - start, 3),
                  "instruction": packet["observation"]["instruction"], "actions": []}
        for row in raw[:min(args.replan, args.steps - tick)]:
            action = project(row)
            applied = api("/api/step", {"episode": episode, "tick": tick, "action": action})
            tick = applied["tick"]
            record["actions"].append({"raw7": row.tolist(), "applied4": action})
        report["chunks"].append(record)
        report["result"] = api("/api/result")
        if report["result"]["episode"] != episode or report["result"]["tick"] != tick:
            raise RuntimeError("Another controller changed the scene")
        print(f"tick {tick}: {record['inference_seconds']:.3f}s / chunk", flush=True)
        if report["result"]["metrics"]["success"]:
            break
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["result"], ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--processor", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--asset", default="standard")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=80, choices=range(1, 501), metavar="1..500")
    parser.add_argument("--replan", type=int, default=5, choices=range(1, 51), metavar="1..50")
    parser.add_argument("--report")
    run(parser.parse_args())
