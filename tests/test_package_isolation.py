"""P0-6: exported package loads its own weights, survives isolation, rejects tampering.

Runs with torch (WSL learning env). The HTTP runner checks target compatibility and
releases/deletes its session even on errors.
"""
import json
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - Windows venv has no torch
    torch = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

API = "http://127.0.0.1:8767"


def fixed_action(ad):
    img = np.linspace(0, 1, ad.size * ad.size * 3, dtype=np.float32).reshape(3, ad.size, ad.size)
    with torch.no_grad():
        return ad.model(torch.from_numpy(img)[None], torch.zeros(1, 18), torch.zeros(1, 4))[0].numpy()


def http(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(API + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def test_package_isolation():
    if torch is None:
        import pytest
        pytest.skip("torch required (run in WSL)")
    from visual_policy_adapter import VisualBCAdapter
    from visual_bc_model import check_target_config

    tmp = Path(tempfile.mkdtemp(prefix="vlab_pkg_"))
    try:
        lab = tmp / "lab"
        (lab / "datasets" / "models").mkdir(parents=True)
        shutil.copy(ROOT / "datasets" / "models" / "visual_hybrid_v3.pt",
                    lab / "datasets" / "models" / "visual.pt")
        pkg = tmp / "pkg"
        run = subprocess.run([sys.executable, str(ROOT / "scripts" / "export_visual_package.py"),
                              "--lab", str(lab), "--model", "visual", "--project", "hybrid",
                              "--out", str(pkg)], capture_output=True, text=True)
        assert run.returncode == 0, run.stderr + run.stdout
        source = VisualBCAdapter.from_checkpoint(lab / "datasets" / "models" / "visual.pt")
        a1 = fixed_action(source)
        contract = source.contract()
        assert {"camera_specs", "controller_spec", "state_joint_names"} <= set(contract)

        # mutate the original project weights: the package must not change
        ck = torch.load(lab / "datasets" / "models" / "visual.pt", weights_only=False)
        for v in ck["state_dict"].values():
            v.zero_()
        torch.save(ck, lab / "datasets" / "models" / "visual.pt")
        packaged = VisualBCAdapter.from_package(pkg)
        a2 = fixed_action(packaged)
        assert np.abs(a1 - a2).max() == 0, "package silently used the original project weights"

        # remove the whole lab: package is self-contained
        shutil.rmtree(lab)
        a3 = fixed_action(VisualBCAdapter.from_package(pkg))
        assert np.abs(a1 - a3).max() == 0

        # tampered weights must be rejected by the manifest hash
        bad = tmp / "bad"
        shutil.copytree(pkg, bad)
        data = (bad / "model.pt").read_bytes()
        (bad / "model.pt").write_bytes(data[:-16] + b"0" * 16)
        try:
            VisualBCAdapter.from_package(bad)
            raise AssertionError("tampered weights loaded")
        except ValueError as exc:
            assert "sha256" in str(exc)

        # full-contract target check
        cfg = json.loads((ROOT / "configs" / "default.json").read_text(encoding="utf-8"))
        assert check_target_config(packaged.contract(), cfg) == []
        changed = json.loads(json.dumps(cfg))
        changed["sensors"][0]["intrinsics"]["fx"] += 10
        changed["controllers"][0]["max_delta_m"] = 0.02
        bad_mismatches = check_target_config(packaged.contract(), changed)
        assert any("camera front" in m for m in bad_mismatches), bad_mismatches
        assert any("max_delta_m" in m for m in bad_mismatches), bad_mismatches

        # runner: incompatible target is rejected before creating a session
        try:
            before = http("GET", "/api/v1/sessions")
            n0 = len(before.get("sessions", before) if isinstance(before, dict) else before)
        except Exception:
            print({"package": "ok", "runner": "skipped (backend not reachable)"})
            return {"package": "ok", "tamper": "rejected", "runner": "skipped"}
        cfg2 = json.loads((ROOT / "configs" / "default.json").read_text(encoding="utf-8"))
        cfg2["sensors"][0]["intrinsics"]["fx"] = 999.0
        http("PUT", "/api/v1/config?project=qa_pkg_incompat", cfg2)
        r = subprocess.run([sys.executable, str(pkg / "run_http.py"), "--package", str(pkg),
                            "--project", "qa_pkg_incompat", "--ticks", "1", "--api", API],
                           capture_output=True, text=True, cwd=str(tmp))
        assert r.returncode == 2, (r.returncode, r.stdout, r.stderr)
        assert "incompatible" in r.stdout
        after = http("GET", "/api/v1/sessions")
        n1 = len(after.get("sessions", after) if isinstance(after, dict) else after)
        assert n1 == n0, (n0, n1)

        # compatible run releases and deletes its session
        r = subprocess.run([sys.executable, str(pkg / "run_http.py"), "--package", str(pkg),
                            "--project", "hybrid_engine", "--ticks", "1", "--api", API],
                           capture_output=True, text=True, cwd=str(tmp))
        after2 = http("GET", "/api/v1/sessions")
        n2 = len(after2.get("sessions", after2) if isinstance(after2, dict) else after2)
        assert n2 == n0, (n0, n2, r.returncode, r.stdout, r.stderr)
        out = {"package": "ok", "isolation": "ok", "tamper": "rejected",
               "contract_full": True, "runner_returncode": r.returncode, "sessions": n2}
        print(out)
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_package_isolation()
