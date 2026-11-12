"""P0-4: atomic scene commit with expected_version and no persistence of invalid data."""
import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
from vlab.api import create_app  # noqa: E402

PID = "qa_scene_commit"
ENV = {"asset_id": "demo_workspace", "revision": 1, "position": [0, 0, 0],
       "quaternion_xyzw": [0, 0, 0, 1], "uniform_scale": 1.0}
DESK = {"name": "desk", "shape": "box", "full_size_m": [0.8, 0.6, 0.05], "position": [0, 0, -0.025],
        "quaternion_xyzw": [0, 0, 0, 1]}


def setup(c):
    base = json.loads((LAB / "configs" / "default.json").read_text(encoding="utf-8"))
    r = c.put(f"/api/v1/config?project={PID}", json=base)
    assert r.status_code == 200, r.text
    return r.json()["version"]


def scene(c):
    r = c.get(f"/api/v1/projects/{PID}/scene")
    assert r.status_code == 200
    return r.json()


def commit(c, **kw):
    body = {"environment": ENV, "colliders": [DESK], "use_default_table": False, **kw}
    return c.put(f"/api/v1/projects/{PID}/scene", json=body)


def main():
    app = create_app(LAB)
    with TestClient(app) as c:
        return _run(c)


def _run(c):
    setup(c)
    before = scene(c)
    v = before["version"]

    r = commit(c, expected_version=v)
    assert r.status_code == 200, r.text
    ok = r.json()
    assert ok["version"] == v + 1
    now = scene(c)
    assert now["version"] == v + 1 and now["environment"]["asset_id"] == "demo_workspace"
    assert [x["name"] for x in now["colliders"]] == ["desk"]

    def unchanged():
        s = scene(c)
        assert s["version"] == now["version"], s
        assert s["environment"] == now["environment"] and s["colliders"] == now["colliders"]

    cas = [
        ("negative scale", {"environment": dict(ENV, uniform_scale=-1.0)}),
        ("zero quaternion", {"environment": dict(ENV, quaternion_xyzw=[0, 0, 0, 0])}),
        ("short position", {"environment": dict(ENV, position=[0, 0])}),
        ("duplicate colliders", {"colliders": [DESK, dict(DESK)]}),
        ("bad friction", {"colliders": [dict(DESK, friction=[-1, 0, 0])]}),
        ("bad cylinder", {"colliders": [{"name": "cyl", "shape": "cylinder", "full_size_m": [0.2, 0.3, 0.2]}]}),
        ("unknown asset", {"environment": dict(ENV, asset_id="no_such_scene")}),
        ("partial failure keeps old env", {"colliders": [dict(DESK, full_size_m=[0, 0.2, 0.1])]}),
    ]
    for name, patch in cas:
        body = {"environment": ENV, "colliders": [DESK], "use_default_table": False,
                "expected_version": now["version"], **patch}
        r = c.put(f"/api/v1/projects/{PID}/scene", json=body)
        assert r.status_code == 400, (name, r.status_code, r.text)
        unchanged()

    # conflict: stale writer
    r = commit(c, expected_version=now["version"] - 1)
    assert r.status_code == 409, r.text
    detail = r.json()["detail"]
    assert detail["current_version"] == now["version"]
    unchanged()

    # legacy split endpoints still validate and never persist invalid values
    r = c.put(f"/api/v1/projects/{PID}/scene/instance",
              json={"position": [0, 0, 0], "quaternion_xyzw": [0, 0, 0, 1], "uniform_scale": -1})
    assert r.status_code == 400, r.text
    unchanged()
    r = c.put(f"/api/v1/projects/{PID}/scene/colliders",
              json={"colliders": [dict(DESK, name="desk", friction=[-1, 0, 0])]})
    assert r.status_code == 400, r.text
    unchanged()
    r = c.put(f"/api/v1/projects/{PID}/scene/colliders", json={"colliders": [DESK]})
    assert r.status_code == 200 and r.json()["version"] == now["version"] + 1
    # now stale instance request is rejected instead of overwriting newer edits
    r = c.put(f"/api/v1/projects/{PID}/scene/instance",
              json={"position": [1, 2, 3], "quaternion_xyzw": [0, 0, 0, 1], "uniform_scale": 1,
                    "expected_version": now["version"]})
    assert r.status_code == 409, r.text
    out = {"version_after_valid": ok["version"], "invalid_cases": len(cas), "conflict": 409,
           "legacy_split_does_not_persist_invalid": True}
    print(out)
    return out


if __name__ == "__main__":
    main()
