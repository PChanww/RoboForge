"""Local HTTP interface to a single MuJoCo episode. Run with python server.py."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import math
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

from PIL import Image

from sim_env import TabletopEnv

ROOT = Path(__file__).resolve().parent
BUILTINS = {"standard": "标准杯", "wide": "宽口杯", "tall": "高杯"}
MAX_STEPS = 500


class APIError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def assets():
    result = {key: (label, key) for key, label in BUILTINS.items()}
    for p in sorted((ROOT / "assets" / "imported").glob("*/manifest.json")):
        result["imported/" + p.parent.name] = (p.parent.name, str(p))
    return result


def jpeg(frame):
    buffer = BytesIO()
    Image.fromarray(frame).save(buffer, format="JPEG", quality=86)
    return buffer.getvalue()


class Lab:
    def __init__(self):
        self.env = None
        self.reset({"asset": "standard", "seed": 0})

    def reset(self, body):
        name, seed = body.get("asset", "standard"), body.get("seed", 0)
        available = assets()
        if not isinstance(name, str) or name not in available:
            raise APIError(400, "Unknown asset. See /api/assets.")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise APIError(400, "seed must be an integer from 0 to 4294967295")
        env = TabletopEnv(asset=available[name][1], seed=seed)
        if self.env is not None:
            self.env.close()
        self.env = env
        self.name, self.seed, self.episode, self.tick = name, seed, uuid4().hex, 0

    def snapshot(self, images=False):
        result = {"episode": self.episode, "tick": self.tick, "asset": self.name,
                  "seed": self.seed, "observation": self.env.observation()}
        if images:
            result["rgb_jpeg"] = base64.b64encode(jpeg(self.env.render(256, 256))).decode("ascii")
            result["wrist_jpeg"] = base64.b64encode(jpeg(self.env.render(256, 256, camera="wrist"))).decode("ascii")
        return result

    def check_version(self, body):
        if body.get("episode") != self.episode or type(body.get("tick")) is not int or body["tick"] != self.tick:
            raise APIError(409, "Episode or tick changed. Request a fresh observation.")

    def run(self, method, path, query, body):
        if method == "GET":
            if path == "/api/assets":
                return {"assets": [{"id": k, "name": v[0]} for k, v in assets().items()]}
            if path == "/api/observation":
                return self.snapshot(query.get("images") == ["1"])
            if path == "/api/result":
                return {**self.snapshot(), "metrics": self.env.metrics(), "limit": MAX_STEPS}
            if path == "/api/frame.jpg":
                return jpeg(self.env.render(800, 540))
        elif method == "POST":
            if path == "/api/reset":
                self.reset(body)
                return self.snapshot()
            if path in ("/api/step", "/api/baseline-step"):
                self.check_version(body)
                if self.tick >= MAX_STEPS:
                    raise APIError(409, "Step limit reached. Reset the episode.")
                if path.endswith("baseline-step"):
                    action = self.env.expert_action()
                else:
                    action = body.get("action")
                    if not isinstance(action, list) or len(action) != 4 or not all(
                        type(x) in (int, float) and math.isfinite(x) and -1 <= x <= 1 for x in action
                    ):
                        raise APIError(400, "action requires four finite numbers in [-1,1]")
                self.env.step(action)
                self.tick += 1
                return self.snapshot()
        raise APIError(404, "Unknown endpoint")


class Handler(BaseHTTPRequestHandler):
    def send_bytes(self, status, data, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def handle_api(self, method):
        parts = urlsplit(self.path)
        try:
            if method == "GET" and parts.path == "/":
                return self.send_bytes(200, (ROOT / "index.html").read_bytes(), "text/html; charset=utf-8")
            body = {}
            if method == "POST":
                origin = self.headers.get("Origin")
                if origin and urlsplit(origin).netloc != self.headers.get("Host"):
                    raise APIError(403, "Cross-origin control is disabled.")
                if self.headers.get_content_type() != "application/json":
                    raise APIError(415, "Use application/json")
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 16384:
                    raise APIError(400, "Invalid JSON body size")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise APIError(400, "Expected a JSON object")
            result = self.server.worker.submit(self.server.lab.run, method, parts.path,
                                               parse_qs(parts.query), body).result()
            if isinstance(result, bytes):
                return self.send_bytes(200, result, "image/jpeg")
            self.send_bytes(200, json.dumps(result, allow_nan=False).encode(), "application/json")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception as exc:
            if isinstance(exc, APIError):
                status = exc.status
            elif isinstance(exc, (ValueError, UnicodeError)):
                status = 400
            else:
                status = 500
                self.log_error("%r", exc)
            self.send_bytes(status, json.dumps({"error": str(exc)}).encode(), "application/json")

    def do_GET(self):
        self.handle_api("GET")

    def do_POST(self):
        self.handle_api("POST")

    def log_request(self, code="-", size="-"):
        if isinstance(code, int) and code >= 400:
            super().log_request(code, size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=1) as worker, ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
        server.worker = worker
        server.lab = worker.submit(Lab).result()
        print(f"VLA tabletop lab: http://127.0.0.1:{args.port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            worker.submit(server.lab.env.close).result()


if __name__ == "__main__":
    main()
