"""P0-3 regression: control ownership, epoch/tick staleness, action-block prevalidation."""
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
from fastapi.testclient import TestClient
from vlab.api import create_app


def main():
    app = create_app(LAB)
    with TestClient(app) as c:
        s = c.post("/api/v1/sessions", json={"project": "default"}).json()
        sid, epoch = s["id"], s["epoch"]
        tok = c.post(f"/api/v1/sessions/{sid}/control/acquire", json={"owner": "manual"}).json()["token"]
        out = {}
        # valid action with epoch
        r = c.post(f"/api/v1/sessions/{sid}/action", json={"token": tok, "action": [0, 0, 0, 1], "epoch": epoch})
        out["action_ok"] = r.status_code == 200
        # mutual exclusion: another owner cannot acquire
        r = c.post(f"/api/v1/sessions/{sid}/control/acquire", json={"owner": "external"})
        out["owner_conflict_409"] = r.status_code == 409
        # reset bumps epoch; old-epoch action rejected
        r = c.post(f"/api/v1/sessions/{sid}/reset", json={"seed": 0}).json()
        new_epoch = r["epoch"]
        out["epoch_incremented"] = new_epoch == epoch + 1
        r = c.post(f"/api/v1/sessions/{sid}/action", json={"token": tok, "action": [0, 0, 0, 1], "epoch": epoch})
        out["stale_epoch_409"] = r.status_code == 409
        # stale tick rejected
        r = c.post(f"/api/v1/sessions/{sid}/action", json={"token": tok, "action": [0, 0, 0, 1], "epoch": new_epoch, "tick": 999})
        out["stale_tick_409"] = r.status_code == 409
        # action block prevalidation: invalid member rejects whole block, no partial exec
        tick_before = c.get(f"/api/v1/sessions/{sid}/observation").json()["tick"]
        r = c.post(f"/api/v1/sessions/{sid}/action_block",
                   json={"token": tok, "epoch": new_epoch, "actions": [[0, 0, 0, 1], [0, 0, 0, 5]]})
        tick_after = c.get(f"/api/v1/sessions/{sid}/observation").json()["tick"]
        out["block_prevalidate_400"] = r.status_code == 400
        out["block_no_partial_exec"] = tick_before == tick_after
        # valid block executes all
        r = c.post(f"/api/v1/sessions/{sid}/action_block",
                   json={"token": tok, "epoch": new_epoch, "actions": [[0, 0, 0, 1], [0, 0, 0, 1]]})
        out["block_executed_2"] = r.status_code == 200 and r.json()["executed"] == 2
        # wrong token rejected
        r = c.post(f"/api/v1/sessions/{sid}/action", json={"token": "bad", "action": [0, 0, 0, 1], "epoch": new_epoch})
        out["bad_token_409"] = r.status_code == 409
        c.delete(f"/api/v1/sessions/{sid}")
    assert all(out.values()), out
    print(out)


if __name__ == "__main__":
    main()