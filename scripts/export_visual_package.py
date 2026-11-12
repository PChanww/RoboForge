"""Export a self-contained visual-policy package (weights + full contract + pure
inference adapter + HTTP runner) and verify loading through the package manifest."""
import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "scripts"))
from visual_policy_adapter import VisualBCAdapter  # noqa: E402

RUN_HTTP = '''"""Run the packaged visual policy over the platform HTTP API."""
import argparse, json, sys, urllib.request
from pathlib import Path

PKG = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG / "policy_code"))
from visual_policy_adapter import VisualBCAdapter
from visual_bc_model import check_target_config


def req(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(API + path, data=data, method=method,
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=180) as resp:
        return json.loads(resp.read())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", default=str(PKG))
    ap.add_argument("--project", default="__PROJECT__")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ticks", type=int, default=200)
    ap.add_argument("--api", default="http://127.0.0.1:8767")
    a = ap.parse_args()
    API = a.api.rstrip("/")
    ad = VisualBCAdapter.from_package(a.package)
    cfg = req("GET", f"/api/v1/config?project={a.project}")
    mismatches = check_target_config(ad.contract(), cfg)
    if mismatches:
        print(json.dumps({"error": "incompatible target", "mismatches": mismatches}))
        sys.exit(2)
    sid = tok = None
    m = {"success": False, "tick": 0}
    try:
        s = req("POST", "/api/v1/sessions", {"project": a.project})
        sid = s["id"]
        tok = req("POST", f"/api/v1/sessions/{sid}/control/acquire", {"owner": "visual_pkg"})["token"]
        ep = req("POST", f"/api/v1/sessions/{sid}/reset", {"seed": a.seed})["epoch"]
        for _ in range(a.ticks):
            obs = req("GET", f"/api/v1/sessions/{sid}/observation")
            req("POST", f"/api/v1/sessions/{sid}/action",
                {"token": tok, "action": ad.act(obs), "epoch": ep})
            m = req("GET", f"/api/v1/sessions/{sid}/result")["metrics"]
            if m["success"]:
                break
        print(json.dumps({"success": m["success"], "tick": m["tick"]}))
    finally:
        if sid:
            if tok:
                try:
                    req("POST", f"/api/v1/sessions/{sid}/control/release", {"token": tok})
                except Exception:
                    pass
            try:
                req("DELETE", f"/api/v1/sessions/{sid}")
            except Exception:
                pass
'''


def fixed_input(adapter):
    img = np.linspace(0, 1, adapter.size * adapter.size * 3,
                      dtype=np.float32).reshape(3, adapter.size, adapter.size)
    st = torch.zeros(1, 18)
    pn = torch.zeros(1, 4)
    with torch.no_grad():
        return adapter.model(torch.from_numpy(img)[None], st, pn)[0].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab", required=True)
    ap.add_argument("--model", default="visual_hybrid")
    ap.add_argument("--project", default="hybrid_engine")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    lab = Path(args.lab)
    out = Path(args.out) if args.out else lab / "deploy" / f"{args.model}_package"
    if out.exists():
        raise ValueError("package exists; choose a new out dir")
    (out / "policy_code").mkdir(parents=True)
    src_model = lab / "datasets" / "models" / f"{args.model}.pt"
    shutil.copy(src_model, out / "model.pt")
    for name in ("visual_policy_adapter.py", "visual_bc_model.py"):
        shutil.copy(LAB / "scripts" / name, out / "policy_code" / name)
    (out / "run_http.py").write_text(RUN_HTTP.replace("__PROJECT__", args.project), encoding="utf-8")
    (out / "requirements.txt").write_text("torch>=2.6\nnumpy\npillow\n", encoding="utf-8")
    sha = hashlib.sha256((out / "model.pt").read_bytes()).hexdigest()
    ck = torch.load(out / "model.pt", weights_only=False)
    adapter = VisualBCAdapter(checkpoint=ck)
    contract = adapter.contract()
    a1 = fixed_input(adapter)
    manifest = {
        "package_version": 2, "package_id": out.name,
        "created": datetime.now(timezone.utc).isoformat(), "kind": "visual",
        "robot": "lux3d_5r", "model_id": args.model, "architecture": "visual_bc_cnn_v1",
        "weights": "model.pt", "weights_sha256": sha,
        "contract": contract,
        "entry": {"package": "VisualBCAdapter.from_package(<dir>)", "http": "run_http.py"},
        "requirements": "torch>=2.6, pillow, numpy (Python 3.11); no vlab/lerobot/pydantic at inference",
        "self_contained": True,
        "limitations": "image BC (no language); trained/evaluated on hybrid 5R front camera only",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    packaged = VisualBCAdapter.from_package(out)
    a2 = fixed_input(packaged)
    diff = float(np.abs(a1 - a2).max())
    manifest["reload_check"] = {"max_abs_diff_package_vs_source": diff,
                                "fixed_action": [round(float(x), 6) for x in a1],
                                "loaded_from": "package manifest weights + sha256"}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "README.md").write_text(
        "# 视觉策略运行包\n\n自包含：权重 `model.pt`、完整训练契约与纯推理代码。\n\n"
        "```bash\npython run_http.py --package . --project hybrid_engine --seed 0 "
        "--api http://127.0.0.1:8767\n```\n"
        "运行前会用 `manifest.contract` 对照目标项目配置（关节顺序、控制器、相机、FPS）；不兼容会拒绝。\n",
        encoding="utf-8")
    print(json.dumps({"package": str(out), "reload_max_abs_diff": diff,
                      "action": [round(float(x), 4) for x in a1]}))
    if diff > 1e-6:
        raise SystemExit(f"package reload mismatch {diff}")


if __name__ == "__main__":
    main()
