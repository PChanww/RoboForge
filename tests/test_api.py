"""Phase B acceptance: config versioning, session isolation, control ownership,
action validation, legacy compatibility, Gymnasium wrapper. Run: python tests/test_api.py"""
import sys
import tempfile
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))

from fastapi.testclient import TestClient
from vlab.api import create_app
from vlab.make_default_config import build
from vlab.store import ConfigStore


def test_store():
    with tempfile.TemporaryDirectory() as tmp:
        store = ConfigStore(Path(tmp))
        cfg = build()
        v1 = store.save(cfg)
        assert v1.version == 1
        v2 = store.save(v1.model_copy(update={"task": v1.task.model_copy(update={"instruction": "changed"})}))
        assert v2.version == 2
        assert store.load("default").task.instruction == "changed"
        assert store.load_version("default", 1).task.instruction == "Place the cup in the tray."
        assert store.versions("default") == [1, 2]
        snap = store.snapshot_run(v2, "run-x")
        assert snap.is_file()


def test_api():
    app = create_app(LAB)
    with TestClient(app) as client:
        cap = client.get("/api/v1/capabilities").json()
        assert cap["schema_version"] == "1.0"
        assert cap["robots"][0]["id"] == "lux3d_5r"
        assert "cartesian_delta_ik" in cap["robots"][0]["supported_modes"]

        cfg = build().model_dump()
        proj = "test_api"
        r = client.put(f"/api/v1/config?project={proj}", json=cfg)
        assert r.status_code == 200, r.text
        # invalid config rejected
        bad = build().model_dump()
        bad["robots"][0]["joints"][0]["axis"] = [0, 0, 0]
        r = client.post("/api/v1/config/validate", json=bad)
        assert r.json()["valid"] is False

        # two isolated sessions
        s1 = client.post("/api/v1/sessions", json={"project": proj}).json()["id"]
        s2 = client.post("/api/v1/sessions", json={"project": proj}).json()["id"]
        assert s1 != s2

        # action without control owner -> 409
        r = client.post(f"/api/v1/sessions/{s1}/action", json={"token": "x", "action": [0, 0, 0, 0]})
        assert r.status_code == 409, r.text
        tok1 = client.post(f"/api/v1/sessions/{s1}/control/acquire", json={"owner": "external"}).json()["token"]
        # conflicting owner -> 409
        r = client.post(f"/api/v1/sessions/{s1}/control/acquire", json={"owner": "manual"})
        assert r.status_code == 409
        # wrong dim -> 400
        r = client.post(f"/api/v1/sessions/{s1}/action", json={"token": tok1, "action": [0, 0, 0]})
        assert r.status_code == 400
        # valid step advances tick
        o1 = client.post(f"/api/v1/sessions/{s1}/action", json={"token": tok1, "action": [0, 0, 0, 1]}).json()
        assert o1["tick"] == 1
        # stale tick -> 409
        r = client.post(f"/api/v1/sessions/{s1}/action", json={"token": tok1, "action": [0, 0, 0, 1], "tick": 0})
        assert r.status_code == 409
        # session 2 unaffected by session 1 stepping
        o2 = client.get(f"/api/v1/sessions/{s2}/observation").json()
        assert o2["tick"] == 0
        # timing fields present
        assert "image_age_s" in o1["timing"] and "actual_capture_intervals_s" in o1["timing"]

        # legacy compatibility
        lr = client.post("/api/reset", json={"asset": "standard", "seed": 0})
        assert lr.status_code == 200
        ep = lr.json()["episode"]
        ls = client.post("/api/step", json={"episode": ep, "tick": 0, "action": [0, 0, 0, 1]})
        assert ls.status_code == 200 and ls.json()["tick"] == 1
        # legacy stale episode -> 409
        r = client.post("/api/step", json={"episode": "wrong", "tick": 1, "action": [0, 0, 0, 1]})
        assert r.status_code == 409

        client.delete(f"/api/v1/sessions/{s1}")
        client.delete(f"/api/v1/sessions/{s2}")


def test_gym():
    import numpy as np
    from vlab.gym_env import VLabGymEnv
    cfg = build()
    env = VLabGymEnv(cfg, LAB)
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert obs["state"].shape[0] == len(env.env._joints) * 2 + 4
    act = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(act)
    assert env.observation_space.contains(obs)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert "success" in info
    env.close()


if __name__ == "__main__":
    test_store()
    print("store OK")
    test_api()
    print("api OK")
    test_gym()
    print("gym OK")