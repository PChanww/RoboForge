"""Controlled World-provider tests (mock, NOT online). Covers success/failure/timeout,
restart recovery via persisted remote ID, no duplicate submit, unconfigured state."""
import sys
import tempfile
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
from vlab.world_provider import GenerationJobStore, MockWorldProvider, WorldProvider


def main():
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        s = GenerationJobStore(Path(tmp), MockWorldProvider("success"))
        j = s.submit("generation", {"prompt": "a room"})
        out["success_remote_id"] = j["remote_id"] is not None and j["state"] == "remote_processing"
        j = s.poll(j["id"])
        out["success_poll"] = j["state"] == "succeeded"
        # restart recovery: new store instance reads persisted remote id
        s2 = GenerationJobStore(Path(tmp), MockWorldProvider("success"))
        r = s2.get(j["id"])
        out["restart_recovery"] = r is not None and r["remote_id"] == j["remote_id"]
        before = s2.provider.submits
        s2.poll(j["id"])  # polling must not resubmit
        out["no_duplicate_submit"] = s2.provider.submits == before
    with tempfile.TemporaryDirectory() as tmp:
        s = GenerationJobStore(Path(tmp), MockWorldProvider("failed"))
        j = s.submit("generation", {"prompt": "x"})
        j = s.poll(j["id"])
        out["failure_visible"] = j["state"] == "failed" and j["error"]
    with tempfile.TemporaryDirectory() as tmp:
        s = GenerationJobStore(Path(tmp), MockWorldProvider("submit_timeout"))
        j = s.submit("generation", {"prompt": "x"})
        out["timeout_interrupted"] = j["state"] == "interrupted" and j["remote_id"] is None
    with tempfile.TemporaryDirectory() as tmp:
        s = GenerationJobStore(Path(tmp), WorldProvider())  # no key
        j = s.submit("generation", {"prompt": "x"})
        out["unconfigured"] = j["state"] == "unconfigured"
    assert all(out.values()), out
    print(out)


if __name__ == "__main__":
    main()