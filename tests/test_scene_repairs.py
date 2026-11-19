"""Regression tests for the reviewed failures. No paid requests or existing data writes."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import trimesh

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
from vlab.transforms import transform_matrix, mat_to_pos_quat_xyzw, quat_xyzw_to_mat
from vlab.scene_import import import_scene
from vlab.scene_assets import load_scene, scene_asset_dir
from vlab.schema import SceneInstance
from vlab.make_default_config import build
from vlab.store import ConfigStore
from vlab.jobs import JobManager
from vlab.world_provider import WorldProvider, MockWorldProvider, GenerationJobStore
from vlab.compat import visual_contract, check_contract


class Repairs(unittest.TestCase):
    def test_random_rotation_roundtrips(self):
        rng = np.random.default_rng(3)
        for q in [np.array([0, 0, np.sin(np.deg2rad(75)), np.cos(np.deg2rad(75))]), *rng.normal(size=(200, 4))]:
            original = transform_matrix([1, 2, 3], q, 2.5)
            p, result, scale = mat_to_pos_quat_xyzw(original)
            np.testing.assert_allclose(transform_matrix(p, result, scale), original, atol=1e-10)

    def test_units_and_identifiers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "box.ply"
            trimesh.creation.box(extents=[1000]*3).export(src)
            manifest = import_scene(src, root, "metric", up_axis="Z", unit_scale=.001, calibrated=True)
            data = json.loads(manifest.read_text())
            np.testing.assert_allclose(np.ptp(data["bbox"], axis=0), [1]*3)
            load_scene(root, "metric", 1, physics=True)
            for bad in ["../escape", "..\\escape", str(root / "outside"), "a/b", "a:b"]:
                with self.assertRaises(ValueError):
                    import_scene(src, root, bad)
            self.assertFalse((root / "outside").exists())
            with self.assertRaises(ValueError):
                import_scene(src, root, "invalid_scale", unit_scale=float("nan"))
            (manifest.parent / "processed/visual.obj").write_text("tampered")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_scene(root, "metric", 1, physics=True)

    def test_scene_pose_validation(self):
        for fields in [dict(asset_id="../x"), dict(asset_id="ok", revision=0),
                       dict(asset_id="ok", quaternion_xyzw=[0]*4),
                       dict(asset_id="ok", position=[float("nan"), 0, 0])]:
            with self.assertRaises(ValueError):
                SceneInstance(**fields)

    def test_official_sdk_shape(self):
        create = Mock(return_value={"worldId": "real-shape-id"})
        reconstruct = Mock(return_value={"worldId": "recon-id"})
        retrieve = Mock(return_value={"status": "SUCCEEDED", "assets": {
            "splats": {"urls": {"plyPath": "https://example.com/scene.ply"}},
            "semanticsMetadata": {"upAxis": "Z"}}})
        provider = WorldProvider()
        sdk = SimpleNamespace(generations=SimpleNamespace(create=create),
                              reconstructions=SimpleNamespace(create=reconstruct), retrieve=retrieve)
        self.enterContext(patch.object(provider, "_sdk", return_value=sdk))
        self.assertEqual(provider.submit_generation("room", "https://example.com/ref.png"), "real-shape-id")
        self.assertEqual(create.call_args.kwargs["resources"], [{"url": "https://example.com/ref.png", "type": "image"}])
        self.assertEqual(provider.submit_reconstruction([{"url": "https://example.com/a.mp4", "type": "video"}]), "recon-id")
        self.assertEqual(reconstruct.call_args.kwargs["scene"], "space")
        result = provider.get_status("real-shape-id")
        self.assertEqual(result["state"], "succeeded")
        self.assertEqual(result["outputs"][0]["filename"], "scene.ply")
        self.assertEqual(result["up_axis"], "Z")

    def test_generation_register_retry_and_restart(self):
        class RetryProvider(MockWorldProvider):
            downloads = 0
            def download_assets(self, *args, **kwargs):
                self.downloads += 1
                if self.downloads == 1:
                    raise TimeoutError("download interrupted")
                return super().download_assets(*args, **kwargs)
        with tempfile.TemporaryDirectory() as tmp:
            provider = RetryProvider()
            store = GenerationJobStore(Path(tmp), provider)
            job = store.submit("generation", {"prompt": "room"})
            failed = store.poll(job["id"])
            self.assertEqual(failed["state"], "retryable_error")
            finished = store.poll(job["id"])
            self.assertEqual(finished["state"], "succeeded")
            self.assertEqual(finished["asset_status"], "preview_only")
            _, mf = load_scene(store.lab_root, finished["asset_id"], 1)
            self.assertEqual(mf["provider"], "aholo_world")
            self.assertEqual(mf["remote_task_id"], job["remote_id"])
            self.assertEqual(provider.submits, 1)
            restored = GenerationJobStore(Path(tmp), provider)
            restored.poll(job["id"])
            self.assertEqual(provider.downloads, 2)
            with self.assertRaises(ValueError):
                load_scene(store.lab_root, finished["asset_id"], 1, physics=True)

    def test_unknown_submit_is_not_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            provider = MockWorldProvider("submit_timeout")
            store = GenerationJobStore(Path(tmp), provider)
            job = store.submit("generation", {"prompt": "room"})
            GenerationJobStore(Path(tmp), provider).poll(job["id"])
            self.assertEqual(provider.submits, 1)
            self.assertEqual(job["state"], "interrupted")

    def test_missing_key_even_with_sdk_installed(self):
        with patch.dict(os.environ, {"AHOLO_API_KEY": ""}):
            with self.assertRaises(NotImplementedError):
                WorldProvider().submit_generation("room")

    def test_binding_rejects_bad_assets_without_saving(self):
        from fastapi.testclient import TestClient
        from vlab.api import create_app
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with TestClient(create_app(root)) as client:
                before = (root / "projects/default/current.json").read_bytes()
                for aid in ["missing", "../escape"]:
                    r = client.post("/api/v1/projects/default/bind-scene", json={"asset_id": aid})
                    self.assertEqual(r.status_code, 400, r.text)
                source = root / "gaussian.ply"
                source.write_text("ply\nformat ascii 1.0\nelement vertex 1\nproperty float f_dc_0\nend_header\n0\n")
                import_scene(source, root, "gaussian", up_axis="Z", calibrated=True)
                r = client.post("/api/v1/projects/default/bind-scene", json={"asset_id": "gaussian"})
                self.assertEqual(r.status_code, 400, r.text)
                self.assertIn("preview-only", r.text)
                self.assertEqual(before, (root / "projects/default/current.json").read_bytes())
                valid = client.post("/api/v1/scene-assets/gaussian/validate?revision=1").json()
                self.assertFalse(valid["valid"])

    def test_job_inputs_and_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            configs = ConfigStore(root / "projects")
            cfg = build(); cfg.project_id = "scene_project"
            configs.save(cfg)
            jobs = JobManager(root)
            with patch.object(jobs, "_run"):
                job = jobs.create("collect", {"project_id": "scene_project", "dataset_id": "capture_a"})
                cfg.task.instruction = "modified later"
                configs.save(cfg)
                snapshot = json.loads(Path(job["params"]["config_snapshot"]).read_text())
                self.assertNotEqual(snapshot["task"]["instruction"], cfg.task.instruction)
                args, _ = jobs._command("collect", job["params"])
                self.assertIn("--config", args)
                self.assertIn("capture_a", str(args))
                with self.assertRaises(ValueError):
                    jobs.create("collect", {"project_id": "scene_project", "dataset_id": "capture_a"})
                cmd, _ = jobs._command("train_visual", {"dataset_id": "scene_data", "model_id": "scene_model", "epochs": 2})
                self.assertIn("--dataset scene_data", cmd[-1])
                self.assertIn("--model-out scene_model", cmd[-1])
                evaluation = jobs.create("eval_visual", {"project_id": "scene_project", "model_id": "scene_model"})
                cmd, _ = jobs._command("eval_visual", evaluation["params"])
                self.assertIn("--model scene_model", cmd[-1])
                self.assertIn("--config", cmd[-1])
                with self.assertRaises(ValueError):
                    jobs._command("train_visual", {"dataset_id": "x;touch pwned"})

    def test_actual_project_camera_contract(self):
        cfg = build()
        contract = visual_contract(cfg)
        cfg.sensors[0].intrinsics.fx += 10
        self.assertFalse(check_contract(contract, cfg)["compatible"])
        cfg.controllers[0].frequency_hz += 1
        self.assertIn("controller", " ".join(check_contract(contract, cfg)["mismatches"]))

    def test_table_contact_is_required(self):
        from vlab.engine import PlatformEnv, UnsupportedMode
        cfg = build()
        cfg.scene.environment = SceneInstance(asset_id="demo_workspace", revision=2)
        cfg.scene.use_default_table = False
        with self.assertRaisesRegex(UnsupportedMode, "explicit static colliders"):
            PlatformEnv(cfg, LAB)
        with self.assertRaisesRegex(ValueError, "table collision support"):
            visual_contract(cfg)

    def test_delayed_camera_collection_has_real_frames(self):
        from vlab.collect import collect
        cfg = build()
        for camera in cfg.sensors:
            camera.latency_s = .05
        with tempfile.TemporaryDirectory() as tmp:
            collect(cfg, LAB, [{"asset": "standard", "seed": 0}], Path(tmp), max_ticks=2)
            with np.load(Path(tmp) / "episodes/ep000.npz", allow_pickle=True) as data:
                self.assertTrue(all(len(frame) > 0 for frame in data["jpeg_front"]))
                self.assertTrue(np.all(data["capture_time_front"] <= data["frame_time"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
