"""Local real SmolVLA inference, with an explicit LIBERO-to-arm adapter.

Run in the isolated learning environment. No simulator or expert controller is
imported by this process. The published checkpoint has no mobile-base action;
its three rotation outputs remain in raw evidence and never become wheel speeds.
"""
import argparse
import base64
from contextlib import nullcontext
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import re
import sys
import threading
import time

import numpy as np
from PIL import Image

LAB_ROOT = Path(__file__).resolve().parents[1]
if not __package__:
    sys.path.insert(0, str(LAB_ROOT))


class Runtime:
    def __init__(self, weights, processor, device, precision="auto"):
        self.weights, self.processor, self.device = Path(weights), Path(processor), device
        self.precision = precision
        self.lock = threading.Lock()
        self.ready, self.error, self.identity = False, None, {}

    def load(self):
        try:
            import torch
            self.cuda = torch.device(self.device).type == "cuda"
            if self.cuda and not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is unavailable. Install the CUDA PyTorch build; CPU fallback is disabled.")
            if self.precision == "auto":
                self.precision = "bfloat16" if self.cuda and torch.cuda.is_bf16_supported() else "float32"
            if self.precision == "bfloat16" and (not self.cuda or not torch.cuda.is_bf16_supported()):
                raise RuntimeError("bfloat16 inference requires a CUDA GPU with BF16 support.")
            self.amp_dtype = torch.bfloat16 if self.precision == "bfloat16" else None
            if self.cuda:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            from safetensors.torch import load_file
            from lerobot.configs.policies import PreTrainedConfig
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            torch.set_num_threads(4)
            config = PreTrainedConfig.from_pretrained(self.weights, local_files_only=True)
            config.vlm_model_name = str(self.processor.resolve())
            config.load_vlm_weights = False
            config.device = self.device
            self.model = SmolVLAPolicy.from_pretrained(self.weights, config=config, strict=True)
            self.model.to(self.device).eval()
            model_device = next(self.model.parameters()).device
            if model_device.type != torch.device(self.device).type:
                raise RuntimeError("Loaded model is not on the requested inference device.")
            self.tokenizer = self.model.model.vlm_with_expert.processor.tokenizer
            self.tokenizer.padding_side = "right"
            self.state_stats = load_file(str(self.weights / "policy_preprocessor_step_5_normalizer_processor.safetensors"))
            self.action_stats = load_file(str(self.weights / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"))
            with (self.weights / "model.safetensors").open("rb") as stream:
                checksum = hashlib.file_digest(stream, "sha256").hexdigest()
            self.identity = {"kind": "vla", "label": "SmolVLA · LIBERO 实验适配", "learned_policy": True,
                "checkpoint": "lerobot/smolvla_libero", "weights_sha256": checksum,
                "action_schema": "mobile7_v1", "action_dim": 7, "raw_action_schema": "libero_xyz_rpy_grip7",
                "adapter": "libero_delta_to_mobile7_arm_only_v1", "controls": ["arm_xyz", "gripper"],
                "mobile_base_policy": False, "privileged_observations": False, "device": str(model_device),
                "device_name": torch.cuda.get_device_name(model_device) if self.cuda else "CPU",
                "inference_precision": self.precision, "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda if self.cuda else None,
                "state_adapter": "base-local measured EEF, z+0.60 and down-facing TCP; measured jaw opening",
                "action_adapter": "clip raw XYZ to [-1,1], invert LIBERO gripper; rotation is not executable; wheel and hold actions zero"}
            self.ready = True
            print(json.dumps({"ready": True, "identity": self.identity}, ensure_ascii=False), flush=True)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            print(json.dumps({"ready": False, "error": self.error}), flush=True)

    def _camera(self, encoded):
        import torch
        raw = base64.b64decode(encoded.split(",", 1)[-1], validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            if image.width*image.height > 1920*1080:
                raise ValueError("Camera exceeds protocol resolution")
            rgb = np.asarray(image.convert("RGB")).copy()
        return torch.from_numpy(rgb).permute(2, 0, 1)[None].to(self.device, torch.float32) / 255.

    def predict(self, packet):
        import torch
        if not self.ready:
            raise ValueError("Model is not ready")
        if packet.get("action_schema") != "mobile7_v1":
            raise ValueError("Unsupported action schema")
        steps = int(packet["max_steps"])
        if not 1 <= steps <= 10:
            raise ValueError("max_steps outside 1..10")
        obs = packet["observation"]
        proprio = np.asarray(obs["state"], dtype=float)
        xyz = np.asarray(obs["eef_position_base"], dtype=float)
        grip = float(obs["gripper_open_fraction"])
        if proprio.shape != (28,) or xyz.shape != (3,) or not np.isfinite(proprio).all() or not np.isfinite(xyz).all() or not 0 <= grip <= 1:
            raise ValueError("Invalid measured robot observation")
        language = {"deliver": "Pick the object from the conveyor and place it in the box.",
                    "pick_return": "Pick up the object from the conveyor, then place it back on the conveyor.",
                    "bin_retrieve": "Put the conveyor object into the box, then pick it up again and hold it steadily above the box.",
                    "bin_return": "Put the conveyor object into the box, pick it up again and return it to the conveyor."}.get(packet.get("task_kind"), "")
        instruction = language + " " + str(packet["instruction"])
        with self.lock, torch.inference_mode():
            torch.manual_seed(int(packet["seed"]) + int(packet["tick"]))
            jaw = grip*.04
            state = torch.tensor([*xyz[:2], xyz[2]+.60, np.pi, 0., 0., jaw, -jaw], dtype=torch.float32)
            state = (state-self.state_stats["observation.state.mean"])/(self.state_stats["observation.state.std"]+1e-8)
            tokens = self.tokenizer([instruction.rstrip()+"\n"], padding="max_length", max_length=48,
                                    truncation=True, return_tensors="pt")
            batch = {"observation.state": state[None].to(self.device),
                     "observation.images.camera1": self._camera(obs["images"]["front"]),
                     "observation.images.camera2": self._camera(obs["images"]["wrist"]),
                     "observation.language.tokens": tokens.input_ids.to(self.device),
                     "observation.language.attention_mask": tokens.attention_mask.bool().to(self.device)}
            if self.cuda:
                torch.cuda.synchronize(self.device)
                torch.cuda.reset_peak_memory_stats(self.device)
            started = time.perf_counter()
            autocast = torch.autocast("cuda", dtype=self.amp_dtype) if self.amp_dtype is not None else nullcontext()
            with autocast:
                raw = self.model.predict_action_chunk(batch)[0].float().cpu()
            model_inference_s = time.perf_counter()-started
            peak_memory_mib = torch.cuda.max_memory_allocated(self.device)/(1024**2) if self.cuda else None
            raw = (raw*self.action_stats["action.std"]+self.action_stats["action.mean"]).numpy()
            if raw.shape != (50, 7) or not np.isfinite(raw).all():
                raise ValueError("SmolVLA checkpoint did not output finite 50x7 LIBERO actions")
            raw = raw[:steps]
            adapted = np.zeros((steps, 7))
            adapted[:, :3] = np.clip(raw[:, :3], -1., 1.)
            adapted[:, 3] = np.clip(-raw[:, 6], -1., 1.)
        return {"request_id": packet["request_id"], "session_id": packet["session_id"], "tick": packet["tick"],
                "action_schema": "mobile7_v1", "weights_sha256": self.identity["weights_sha256"],
                "raw_actions": raw.tolist(), "actions": adapted.tolist(), "inference_s": time.perf_counter()-started,
                "model_inference_s": model_inference_s, "device": self.identity["device"],
                "peak_memory_mib": peak_memory_mib}


def serve(runtime, host, port):
    from integrations.transport_mobile_models import MobilePolicies
    mobile = MobilePolicies(LAB_ROOT, runtime)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send_json(self, status, value):
            payload = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            match = re.fullmatch(r"/mobile/([0-9a-f]{12})/health", self.path)
            if match:
                try:
                    return self.send_json(200, mobile.health(match[1]))
                except (OSError, ValueError, RuntimeError, KeyError) as exc:
                    return self.send_json(200, {"ready": False, "state": "error", "error": str(exc)})
            if self.path != "/health":
                return self.send_json(404, {"error": "Not found"})
            self.send_json(200, {"ready": runtime.ready, "state": "ready" if runtime.ready else "error" if runtime.error else "loading",
                                 "identity": runtime.identity, "error": runtime.error,
                                 "requested_device": runtime.device})

        def do_POST(self):
            match = re.fullmatch(r"/mobile/([0-9a-f]{12})/infer", self.path)
            if not match and self.path != "/infer":
                return self.send_json(404, {"error": "Not found"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 8*1024*1024:
                    raise ValueError("Invalid payload length")
                packet = json.loads(self.rfile.read(length))
                result = mobile.predict(match[1], packet) if match else runtime.predict(packet)
                self.send_json(200, result)
            except Exception as exc:
                self.send_json(422, {"error": f"{type(exc).__name__}: {exc}"})

    server = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=runtime.load, daemon=True).start()
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--processor", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("auto", "float32", "bfloat16"), default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8772)
    args = parser.parse_args()
    serve(Runtime(args.weights, args.processor, args.device, args.precision), args.host, args.port)
