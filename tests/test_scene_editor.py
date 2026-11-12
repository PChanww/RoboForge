"""Scene calibration & collider editing API: persist/reload, validation, physics build."""
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
from fastapi.testclient import TestClient
from vlab.api import create_app


def main():
    app = create_app(LAB)
    with TestClient(app) as c:
        pid = "test_scene_edit"
        # bind demo scene (rev1 exists) then edit instance + colliders
        r = c.post(f"/api/v1/projects/{pid}/bind-scene",
                   json={"asset_id": "demo_workspace", "revision": 1, "use_default_table": True})
        assert r.status_code == 200, r.text
        r = c.put(f"/api/v1/projects/{pid}/scene/instance",
                  json={"position": [0, 0, 0.05], "quaternion_xyzw": [0, 0, 0, 1],
                        "uniform_scale": 1.2, "use_default_table": False})
        assert r.status_code == 200, r.text
        assert abs(r.json()["environment"]["uniform_scale"] - 1.2) < 1e-9
        colliders = [
            {"name": "desk", "shape": "box", "full_size_m": [0.8, 0.6, 0.05], "position": [0, 0, -0.025]},
            {"name": "post", "shape": "cylinder", "full_size_m": [0.1, 0.1, 0.4], "position": [0.3, 0, 0.2]},
        ]
        r = c.put(f"/api/v1/projects/{pid}/scene/colliders", json={"colliders": colliders})
        assert r.status_code == 200, r.text
        assert [x["name"] for x in r.json()["colliders"]] == ["desk", "post"]
        # invalid elliptical cylinder rejected
        bad = [{"name": "bad", "shape": "cylinder", "full_size_m": [0.1, 0.2, 0.4]}]
        r = c.put(f"/api/v1/projects/{pid}/scene/colliders", json={"colliders": bad})
        assert r.status_code == 400, r.text
        # persisted: reload scene
        s = c.get(f"/api/v1/projects/{pid}/scene").json()
        assert s["environment"]["uniform_scale"] == 1.2 and len(s["colliders"]) == 2
    # physics build uses edited collider (desk top z=0) and cup rests on it
    from vlab.schema import ProjectConfig
    from vlab.engine import PlatformEnv
    cfg = ProjectConfig.model_validate_json((LAB / "projects" / pid / "current.json").read_text(encoding="utf-8"))
    env = PlatformEnv(cfg, LAB)
    try:
        m = env.metrics()
        assert abs(m["object_xyz"][2] - env._asset_height / 2) < 0.01
    finally:
        env.close()
    print({"scene_edit_persist": True, "cylinder_rejected": True, "physics_build_ok": True})


if __name__ == "__main__":
    main()