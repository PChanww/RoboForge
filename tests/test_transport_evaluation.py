"""Evidence reports must preserve raw data and distinguish expert from VLA."""
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vlab.transport_evaluation import ACTION_FIELDS, TransportEvaluation, register_transport_evaluation
from vlab.transport_evaluation import TASK_SEQUENCES


RID = "transport_evaluation_test"


@pytest.fixture(autouse=True)
def deterministic_policy_status(monkeypatch):
    from vlab.transport_policy import PolicyRegistry
    monkeypatch.setattr(PolicyRegistry, "_health", lambda *_args, **_kwargs: {"ready": False, "state": "offline"})


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf8")


def make_run(lab, *, success=True, status=None, legacy=False, n=8):
    directory = lab / "datasets/transport" / RID
    (directory / "episodes").mkdir(parents=True, exist_ok=True)
    fields = ACTION_FIELDS if not legacy else [v for v in ACTION_FIELDS if v != "base_angular_rad_s"]
    config = {"controllers": [{"frequency_hz": 20., "max_delta_m": .012, "frame": "world"}],
              "sensors": [{"id": "front", "resolution": [8, 6]}, {"id": "wrist", "resolution": [4, 3]}]}
    write_json(directory / "config.json", config)
    metrics = {"success": success, "grasped": True, "lifted": True, "controlled_release": success,
               "delivery_failed": not success, "tray_contact": success, "contacts": 0, "stable_ticks": 6 if success else 0}
    row = {"id": RID, "spec": {"instruction": "Conveyor pick, mobile transport, place in bin."},
           "status": status or ("succeeded" if success else "failed"), "error": None,
           "episodes": [{"index": 0, "seed": 3, "frames": n, "duration_s": n*.05,
                         "success": success, "interrupted": False, "metrics": metrics}],
           "configuration": {"sha256": hashlib.sha256((directory / "config.json").read_bytes()).hexdigest()}}
    if not legacy:
        row["spec"]["speed_scale"] = 1.6
    arrays = {"time": np.arange(n)*.05, "action": np.zeros((n,len(fields))), "requested_action": np.zeros((n,len(fields))),
              "arm_qpos": np.zeros((n,7)), "base_pose": np.tile([0,0,-.033,1,0,0,0],(n,1)),
              "base_velocity": np.zeros((n,6)), "wheel_qpos": np.zeros((n,4)), "wheel_qvel": np.zeros((n,4)),
              "object_pose": np.tile([0,0,.03,1,0,0,0],(n,1)), "object_velocity": np.zeros((n,6)),
              "phase": np.array([0,4,5,6,7,8,9,12] if success else [0,4,5,6,7,8,9,9])[:n],
              "contacts": np.array([0,2,2,2,2,2,1,0])[:n],
              "image_front": np.zeros((n,6,8,3),np.uint8), "image_wrist": np.zeros((n,3,4,3),np.uint8)}
    write_json(directory / "run.json",row)
    write_json(directory / "manifest.json",{"action_fields":fields,"frequency_hz":20})
    np.savez_compressed(directory / "episodes/ep000.npz",**arrays)
    return directory,row,arrays


def source_hashes(directory):
    return {p.relative_to(directory).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.rglob("*") if p.is_file()}


@pytest.mark.parametrize("endpoint", ["evaluation/download", "policy-contract"])
def test_json_can_be_fetched_without_attachment_for_browser_save(tmp_path, endpoint):
    make_run(tmp_path)
    app = FastAPI()
    register_transport_evaluation(app, tmp_path)
    with TestClient(app) as client:
        path = f"/api/v1/transport/runs/{RID}/{endpoint}"
        attachment = client.get(path)
        inline = client.get(path, params={"inline": "true"})
    assert attachment.status_code == inline.status_code == 200
    assert attachment.headers["content-disposition"].startswith("attachment;")
    assert "content-disposition" not in inline.headers
    assert inline.headers["content-type"].startswith("application/json")
    assert attachment.json() == inline.json()


def test_success_is_expert_evidence_with_explicit_7d_contract_and_header_only_images(tmp_path,monkeypatch):
    directory,row,arrays=make_run(tmp_path)
    original=np.lib.npyio.NpzFile.__getitem__
    def numeric_arrays_only(self,key):
        assert not key.startswith("image_"), "Reports must inspect image headers without loading RGB arrays"
        return original(self,key)
    monkeypatch.setattr(np.lib.npyio.NpzFile,"__getitem__",numeric_arrays_only)
    before=source_hashes(directory)
    evaluator=TransportEvaluation(tmp_path)
    report=evaluator.report(RID)
    contract=evaluator.contract(RID)
    assert report["verdict"]["code"]=="passed"
    assert report["controller"]["kind"]=="privileged_expert"
    assert not report["policy"]["vla_connected"] and not report["controller"]["learned_policy"]
    assert report["summary"]=={"episodes":1,"successes":1,"frames":8,"success_rate":1.}
    assert report["data_quality"]["valid"]
    assert contract["action"]["dim"]==7 and contract["action"]["names"]==ACTION_FIELDS
    assert not contract["vla_evaluated"]
    assert contract["observation"]["state"]["dim"]==28
    assert contract["observation"]["images"]["front"]["shape"]==[6,8,3]
    assert contract["compatibility"]["current_mobile_7d"]
    assert contract["config_sha256"]==row["configuration"]["sha256"]
    assert source_hashes(directory)==before


def test_failed_contact_history_reports_observed_placement_failure_without_inventing_cause(tmp_path):
    make_run(tmp_path,success=False)
    report=TransportEvaluation(tmp_path).report(RID)
    assert report["verdict"]["code"]=="failed" and report["data_quality"]["valid"]
    failure=report["episodes"][0]["failure"]
    assert failure["phase"]==9 and failure["stage"]=="越过目标箱沿"
    assert failure["last_bilateral_contact_s"]==.25
    assert "失去双侧夹持" in failure["reason"]
    assert "IK" not in failure["reason"] and "角度" not in failure["reason"]
    assert report["summary"]["success_rate"]==0


@pytest.mark.parametrize("field,edit",[
    ("time",lambda a:a.__setitem__(4,.201)),
    ("time",lambda a:a.__setitem__(0,-.05)),
    ("action",lambda a:a.__setitem__((3,0),1.2)),
    ("action",lambda a:a.__setitem__((3,4),.11)),
    ("action",lambda a:a.__setitem__((3,5),.81)),
    ("action",lambda a:a.__setitem__((3,6),.3)),
    ("requested_action",lambda a:a.__setitem__((3,2),np.nan)),
    ("phase",lambda a:a.__setitem__(3,15)),
    ("contacts",lambda a:a.__setitem__(3,3)),
])
def test_numeric_contract_and_sampling_anomalies_invalidate_data(tmp_path,field,edit):
    directory,row,arrays=make_run(tmp_path)
    edit(arrays[field])
    np.savez_compressed(directory / "episodes/ep000.npz",**arrays)
    report=TransportEvaluation(tmp_path).report(RID)
    assert report["verdict"]["code"]=="invalid_data"
    assert not report["data_quality"]["valid"]
    assert report["data_quality"]["issues"]


@pytest.mark.parametrize("field",["action","image_front"])
def test_missing_arrays_and_wrong_action_width_are_detected(tmp_path,field):
    directory,_,arrays=make_run(tmp_path)
    if field=="action":
        arrays[field]=arrays[field][:,:6]
    else:
        del arrays[field]
    np.savez_compressed(directory / "episodes/ep000.npz",**arrays)
    report=TransportEvaluation(tmp_path).report(RID)
    assert report["verdict"]["code"]=="invalid_data"


def test_image_dtype_and_shape_must_match_effective_camera_config(tmp_path):
    directory,_,arrays=make_run(tmp_path)
    arrays["image_front"]=np.zeros((8,5,8,3),np.uint8)
    arrays["image_wrist"]=arrays["image_wrist"].astype(np.float32)
    np.savez_compressed(directory / "episodes/ep000.npz",**arrays)
    issues=TransportEvaluation(tmp_path).report(RID)["data_quality"]["issues"]
    assert any("image_front" in issue for issue in issues)
    assert any("image_wrist" in issue for issue in issues)


def test_config_hash_and_manifest_contract_corruption_are_detected(tmp_path):
    directory,_,_=make_run(tmp_path)
    write_json(directory / "manifest.json",{"action_fields":ACTION_FIELDS[:-1],"frequency_hz":10})
    config=json.loads((directory / "config.json").read_text(encoding="utf8"))
    config["controllers"][0]["frequency_hz"]=10
    write_json(directory / "config.json",config)
    issues=TransportEvaluation(tmp_path).report(RID)["data_quality"]["issues"]
    assert any("SHA-256" in issue for issue in issues)
    assert any("清单" in issue for issue in issues)
    assert any("控制配置" in issue for issue in issues)


@pytest.mark.parametrize("corruption",["success_label","physical_success","terminal_status","episode_count"])
def test_inconsistent_success_labels_and_physical_evidence_are_detected(tmp_path,corruption):
    directory,row,_=make_run(tmp_path)
    if corruption=="success_label":
        row["episodes"][0]["metrics"]["success"]=False
    elif corruption=="physical_success":
        row["episodes"][0]["metrics"].update(tray_contact=False,stable_ticks=0,contacts=2)
    elif corruption=="terminal_status":
        row["episodes"][0]["success"]=False
        row["episodes"][0]["metrics"]["success"]=False
    else:
        row["spec"]["episodes"]=2
    write_json(directory / "run.json",row)
    assert TransportEvaluation(tmp_path).report(RID)["verdict"]["code"]=="invalid_data"


def test_interrupted_episode_does_not_claim_success_even_if_last_physical_tick_passed(tmp_path):
    directory,row,_=make_run(tmp_path,status="cancelled")
    row["episodes"][0].update(interrupted=True,success=False)
    write_json(directory / "run.json",row)
    report=TransportEvaluation(tmp_path).report(RID)
    assert report["verdict"]["code"]=="incomplete"
    assert report["data_quality"]["valid"]
    assert report["summary"]["successes"]==0
    assert "中断" in report["episodes"][0]["failure"]["reason"]


def test_archive_preserves_all_original_file_bytes_hashes_and_embeds_report_and_contract(tmp_path):
    directory,_,_=make_run(tmp_path,success=False)
    # Preserve an orphaned original episode too, for forensic completeness.
    (directory / "episodes/ep001.npz").write_bytes((directory / "episodes/ep000.npz").read_bytes())
    before=source_hashes(directory)
    evaluator=TransportEvaluation(tmp_path)
    path=evaluator.archive(RID)
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        assert set(z.namelist())==set(before)|{"evaluation.json","policy-contract.json"}
        report=json.loads(z.read("evaluation.json"))
        for entry in report["artifacts"]:
            assert hashlib.sha256(z.read(entry["file"])).hexdigest()==entry["sha256"]==before[entry["file"]]
        assert json.loads(z.read("policy-contract.json"))["action"]["dim"]==7
        assert report["verdict"]["code"]=="failed"
    assert source_hashes(directory)==before
    assert evaluator.archive(RID)==path


@pytest.mark.parametrize("filename",["config.json","manifest.json","episodes/ep000.npz"])
def test_missing_required_file_is_reported_and_prevents_incomplete_archive(tmp_path,filename):
    directory,_,_=make_run(tmp_path)
    (directory / filename).unlink()
    evaluator=TransportEvaluation(tmp_path)
    assert not evaluator.report(RID)["data_quality"]["valid"]
    with pytest.raises(ValueError,match="缺失"):
        evaluator.archive(RID)


@pytest.mark.parametrize("case",["running","awaiting_asset_review","no_episodes","zero_frames"])
def test_unfinished_or_frameless_runs_cannot_export_complete_experiment(tmp_path,case):
    directory,row,_=make_run(tmp_path,n=0 if case=="zero_frames" else 8)
    if case=="no_episodes":
        row["episodes"]=[]
    elif case not in ("zero_frames",):
        row["status"]=case
    write_json(directory / "run.json",row)
    evaluator=TransportEvaluation(tmp_path)
    with pytest.raises(ValueError,match="实验结束"):
        evaluator.archive(RID)
    if case in ("no_episodes","zero_frames"):
        assert evaluator.report(RID)["verdict"]["code"]=="invalid_data"


def test_historical_six_dimensional_recording_keeps_its_own_contract_and_refuses_7d_compatibility(tmp_path):
    make_run(tmp_path,legacy=True)
    evaluator=TransportEvaluation(tmp_path)
    report=evaluator.report(RID)
    contract=evaluator.contract(RID)
    assert report["data_quality"]["valid"]
    assert report["action_contract"]["dim"]==6
    assert not report["action_contract"]["current_mobile_7d_supported"]
    assert contract["action"]["dim"]==6 and "base_angular_rad_s" not in contract["action"]["names"]
    assert contract["compatibility"]["legacy_6d_recording"]
    assert not contract["compatibility"]["current_mobile_7d"]
    assert "unsupported" in contract["compatibility"]["reason"]
    with zipfile.ZipFile(evaluator.archive(RID)) as z:
        assert json.loads(z.read("policy-contract.json"))["action"]["dim"]==6


@pytest.mark.parametrize("rid",["../transport_evaluation_test","transport_../secret","wrong_prefix","transport_x/../secret"])
def test_run_id_path_traversal_is_rejected(tmp_path,rid):
    make_run(tmp_path)
    with pytest.raises(ValueError):
        TransportEvaluation(tmp_path).report(rid)


def test_readonly_http_report_contract_downloads_and_missing_run(tmp_path):
    make_run(tmp_path)
    app=FastAPI()
    register_transport_evaluation(app,tmp_path)
    with TestClient(app) as client:
        base=f"/api/v1/transport/runs/{RID}"
        assert client.get(base+"/evaluation").json()["verdict"]["code"]=="passed"
        response=client.get(base+"/policy-contract")
        assert response.status_code==200 and "attachment" in response.headers["content-disposition"]
        assert client.get(base+"/archive").headers["content-type"]=="application/zip"
        assert client.get("/api/v1/transport/runs/transport_missing/evaluation").status_code==404
        assert not client.get("/api/v1/transport/policy-capabilities").json()["vla_connected"]


def make_return_run(lab, kind):
    directory, row, arrays = make_run(lab)
    row["spec"].update(task_kind=kind, task_sequence=TASK_SEQUENCES[kind])
    metrics = row["episodes"][0]["metrics"]
    # Returning succeeds on actual roller support; final tray_contact is false.
    metrics.update(task_kind=kind, program_complete=True, returned_to_belt=True,
                   delivered_to_bin=kind == "bin_return", regrasped_from_bin=kind == "bin_return",
                   tray_contact=False, stable_ticks=0, return_stable_ticks=6)
    if kind == "bin_return":
        events = [
            {"stage": "pick_from_belt", "tick": 1, "contacts": 2},
            {"stage": "transport_to_bin", "tick": 2},
            {"stage": "place_in_bin", "tick": 3, "tray_contact": True, "controlled_release": True, "stable_ticks": 6},
            {"stage": "regrasp_from_bin", "tick": 4, "contacts": 2},
            {"stage": "lifted_from_bin", "tick": 5, "contacts": 2, "object_low_z": .15},
            {"stage": "return_along_route", "tick": 6},
        ]
        arrays["phase"] = np.array([0, 4, 5, 9, 14, 18, 20, 25])
    else:
        events = [{"stage": "pick_from_belt", "tick": 1, "contacts": 2}]
        arrays["phase"] = np.array([0, 4, 5, 22, 23, 24, 25, 25])
    events.append({"stage": "return_to_belt", "tick": 8, "roller_contact": True, "controlled_release": True,
                   "contacts": 0, "stable_ticks": 6})
    for event in events:
        event["time_s"] = event["tick"]*.05
    metrics["task_events"] = events
    write_json(directory / "run.json", row)
    np.savez_compressed(directory / "episodes/ep000.npz", **arrays)
    return directory, row, arrays


@pytest.mark.parametrize("kind", ["pick_return", "bin_return"])
def test_return_program_is_judged_by_final_belt_support_and_ordered_subtasks(tmp_path, kind):
    directory, row, _ = make_return_run(tmp_path, kind)
    before = source_hashes(directory)
    evaluator = TransportEvaluation(tmp_path)
    report, contract = evaluator.report(RID), evaluator.contract(RID)
    assert report["verdict"]["code"] == "passed", report["data_quality"]["issues"]
    assert report["task_kind"] == contract["task_kind"] == kind
    assert report["task_sequence"] == contract["task_sequence"] == TASK_SEQUENCES[kind]
    episode = report["episodes"][0]
    assert episode["program_complete"]
    assert all(stage["completed"] for stage in episode["subtasks"])
    assert not episode["metrics"]["tray_contact"]
    assert source_hashes(directory) == before


@pytest.mark.parametrize("corruption", ["incomplete", "missing_return", "bad_support", "early_success", "out_of_order", "future_event", "missing_second_lift"])
def test_return_success_cannot_hide_incomplete_program_or_missing_physical_evidence(tmp_path, corruption):
    directory, row, _ = make_return_run(tmp_path, "bin_return")
    metrics = row["episodes"][0]["metrics"]
    events = metrics["task_events"]
    if corruption == "incomplete":
        metrics["program_complete"] = False
    elif corruption == "missing_return":
        metrics["returned_to_belt"] = False
    elif corruption == "bad_support":
        events[-1]["roller_contact"] = False
    elif corruption == "early_success":
        metrics["task_events"] = events[:3]
    elif corruption == "out_of_order":
        events[2]["stage"], events[3]["stage"] = events[3]["stage"], events[2]["stage"]
    elif corruption == "future_event":
        events[-1].update(time_s=.5, tick=10)
    else:
        metrics["task_events"] = [event for event in events if event["stage"] != "lifted_from_bin"]
    write_json(directory / "run.json", row)
    assert TransportEvaluation(tmp_path).report(RID)["verdict"]["code"] == "invalid_data"


def test_initial_bin_success_is_reported_as_incomplete_return_with_exact_last_stage(tmp_path):
    directory, row, _ = make_return_run(tmp_path, "bin_return")
    row["status"] = "failed"
    row["episodes"][0]["success"] = False
    row["episodes"][0]["metrics"].update(success=False, program_complete=False,
        regrasped_from_bin=False, returned_to_belt=False,
        task_events=row["episodes"][0]["metrics"]["task_events"][:3])
    write_json(directory / "run.json", row)
    report = TransportEvaluation(tmp_path).report(RID)
    assert report["verdict"]["code"] == "failed"
    assert report["data_quality"]["valid"]
    assert "已完成稳定入箱" in report["verdict"]["reason"]
    assert [sub["completed"] for sub in report["episodes"][0]["subtasks"]] == [True, True, True, False, False, False]


@pytest.mark.parametrize("corruption", [None, "low_clearance", "unstable", "one_sided", "missing_event"])
def test_bin_retrieval_requires_stable_bilateral_hold_above_rim(tmp_path, corruption):
    directory, row, arrays = make_return_run(tmp_path, "bin_return")
    row["spec"].update(task_kind="bin_retrieve", task_sequence=TASK_SEQUENCES["bin_retrieve"])
    metrics = row["episodes"][0]["metrics"]
    metrics.update(task_kind="bin_retrieve", returned_to_belt=False, retrieved_from_bin=True, contacts=2)
    metrics["task_events"] = metrics["task_events"][:5]
    lifted = metrics["task_events"][-1]
    lifted.update(tick=8, time_s=.4, stable_ticks=5, object_low_z=.17, rim_z_m=.12)
    if corruption == "low_clearance":
        lifted["object_low_z"] = .13
    elif corruption == "unstable":
        lifted["stable_ticks"] = 1
    elif corruption == "one_sided":
        lifted["contacts"] = 1
    elif corruption == "missing_event":
        metrics["task_events"].pop()
    arrays["phase"][-1] = 26
    arrays["contacts"][-1] = 2
    np.savez_compressed(directory / "episodes/ep000.npz", **arrays)
    write_json(directory / "run.json", row)
    report = TransportEvaluation(tmp_path).report(RID)
    assert report["verdict"]["code"] == ("invalid_data" if corruption else "passed")
    assert report["task_sequence"][-1] == "lifted_from_bin"


def test_learned_controller_metadata_is_preserved_without_using_global_availability_as_identity(tmp_path):
    directory, row, _ = make_return_run(tmp_path, "pick_return")
    identity = {"id": "test-vla", "kind": "vla", "label": "Actual learned policy", "learned_policy": True,
                "weights_sha256": "b"*64, "adapter": "test_arm_only", "controls": ["arm_xyz", "gripper"],
                "mobile_base_policy": False, "privileged_observations": False}
    row["spec"].update(controller="vla", policy_id="test-vla")
    row["policy_identity"] = identity
    row["policy_diagnostics"] = {"inference_count": 2, "executed_model_actions": 8, "inference_s": .4}
    write_json(directory / "run.json", row)
    evaluator = TransportEvaluation(tmp_path)
    report, contract = evaluator.report(RID), evaluator.contract(RID)
    assert report["verdict"]["code"] == "passed"
    assert report["controller"] == contract["controller_identity"] == identity
    assert report["vla_evaluated"] and contract["vla_evaluated"]
    assert report["policy_diagnostics"] == contract["policy_diagnostics"] == row["policy_diagnostics"]
    assert "VLA" in report["verdict"]["label"] and "专家" not in report["verdict"]["label"]
    assert not report["policy"]["vla_connected"]


def test_failed_pre_execution_vla_request_is_not_reported_as_expert_or_evaluated_vla(tmp_path):
    directory, row, _ = make_run(tmp_path, success=False)
    row["spec"].update(controller="vla", policy_id="offline-vla")
    row["episodes"] = []
    write_json(directory / "run.json", row)
    evaluator = TransportEvaluation(tmp_path)
    report, contract = evaluator.report(RID), evaluator.contract(RID)
    assert report["controller"]["kind"] == contract["controller_source"] == "unverified_vla"
    assert not report["vla_evaluated"] and not contract["vla_evaluated"]
