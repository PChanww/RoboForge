from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Thread
from urllib.error import HTTPError

import pytest
from fastapi.testclient import TestClient

from vlab.api import create_app
from vlab.generation.object_provider import Lux3DProvider
from vlab.world_provider import WorldProvider


SETTINGS = "/api/v1/settings/providers"
GENERATION = "/api/v1/generation"


@pytest.fixture(autouse=True)
def clear_provider_environment(monkeypatch):
    for name in ("AHOLO_API_KEY", "LUX3D_CN_API_KEY", "LUX3D_API_KEY",
                 "AUX_MODEL_API_KEY", "AUX_MODEL_BASE_URL", "AUX_MODEL_MODEL"):
        monkeypatch.delenv(name, raising=False)


def local_client(app):
    return TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000),
                      headers={"Origin": "http://localhost:5173"})


@contextmanager
def fake_chat_service():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, self.headers["Authorization"], body))
            content = ("OK" if body["messages"] == [{"role": "user", "content": "Reply with OK."}]
                       else json.dumps({"reply_zh": "fixture planner ready", "complete": True}))
            response = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/v1", requests
        finally:
            server.shutdown()
            thread.join(timeout=3)


def test_aux_preview_save_and_existing_generation_session_use_new_settings(tmp_path):
    with fake_chat_service() as (base_url, requests), local_client(create_app(tmp_path)) as client:
        old_session = client.post(GENERATION + "/sessions", json={}).json()
        assert old_session["aux_state"] == "waiting_for_configuration"
        update = {"aux_base_url": base_url + "/chat/completions", "aux_api_key": "fixture-aux-key",
                  "aux_model": "first-model"}

        preview = client.post(SETTINGS + "/test/aux", json=update)
        assert preview.status_code == 200 and preview.json()["results"][0]["ok"]
        assert not client.get(SETTINGS).json()["aux"]["key_configured"]
        assert not (tmp_path / ".env.providers.json").exists()
        assert requests[-1][:2] == ("/v1/chat/completions", "Bearer fixture-aux-key")
        assert "fixture-aux-key" not in preview.text

        saved = client.put(SETTINGS, json=update)
        assert saved.status_code == 200 and saved.headers["cache-control"] == "no-store"
        assert saved.json()["aux"] == {"base_url": base_url, "model": "first-model", "key_configured": True}
        assert "fixture-aux-key" not in saved.text
        assert "fixture-aux-key" not in client.get(SETTINGS).text
        assert client.get(GENERATION + "/capabilities").json()["aux_model"]["model"] == "first-model"

        assert client.put(SETTINGS, json={"aux_model": "updated-model"}).status_code == 200
        message = client.post(f"{GENERATION}/sessions/{old_session['session_id']}/messages", json={
            "content": "A blue ceramic cup", "expected_revision": old_session["revision"],
        })
        assert message.status_code == 200 and "fixture planner ready" in message.text
        assert requests[-1][2]["model"] == "updated-model"
        assert requests[-1][1] == "Bearer fixture-aux-key"

    with local_client(create_app(tmp_path)) as reopened:
        assert reopened.get(SETTINGS).json()["aux"]["model"] == "updated-model"


def test_aholo_test_is_read_only_and_key_rotation_updates_generation(tmp_path, monkeypatch):
    seen = []

    def lux_list(self, **kwargs):
        seen.append(("lux3d", self._key()))
        if self._key() == "preview-failed-key":
            raise HTTPError("https://example.invalid", 401, self._key(), {}, None)
        return {"items": [], "total": 0}

    def world_list(self, **kwargs):
        seen.append(("world", self._key()))
        return {"items": [], "total": 0}

    monkeypatch.setattr(Lux3DProvider, "list_remote_tasks", lux_list)
    monkeypatch.setattr(WorldProvider, "list_remote_tasks", world_list)
    with local_client(create_app(tmp_path)) as client:
        assert not client.get(GENERATION + "/capabilities").json()["providers"]["lux3d"]["configured"]
        preview = client.post(SETTINGS + "/test/aholo", json={"aholo_api_key": "preview-failed-key"})
        assert [item["ok"] for item in preview.json()["results"]] == [False, True]
        assert "401" in preview.text and "preview-failed-key" not in preview.text
        assert not (tmp_path / ".env.providers.json").exists()

        for key in ("fixture-aholo-first", "fixture-aholo-rotated"):
            saved = client.put(SETTINGS, json={"aholo_api_key": key})
            assert saved.status_code == 200 and key not in saved.text
            assert client.get("/api/v1/lux3d/status").json()["configured"]
            assert client.get("/api/v1/generation-jobs").json()["configured"]
            for provider in ("lux3d", "aholo_world"):
                listing = client.get(GENERATION + "/remote-tasks", params={"provider": provider})
                assert listing.json()["configured"] and key not in listing.text
            assert seen[-2:] == [("lux3d", key), ("world", key)]


def test_settings_reject_nonlocal_clients_and_validation_does_not_echo_keys(tmp_path):
    app = create_app(tmp_path)
    payload = {"aholo_api_key": "do-not-echo-this", "aux_base_url": "file:///secret"}
    with local_client(app) as client:
        invalid = client.put(SETTINGS, json=payload)
        assert invalid.status_code == 400 and "do-not-echo-this" not in invalid.text
        cross_site = client.put(SETTINGS, json=payload, headers={"Origin": "https://untrusted.example"})
        assert cross_site.status_code == 403
        spoofed_host = client.put(SETTINGS, json=payload, headers={"Host": "untrusted.example"})
        assert spoofed_host.status_code == 403
    with TestClient(app, base_url="http://localhost", client=("192.0.2.15", 50000)) as remote:
        assert remote.get(SETTINGS).status_code == 403
        assert remote.post(SETTINGS + "/test/aux", json=payload).status_code == 403
    assert not (tmp_path / ".env.providers.json").exists()
