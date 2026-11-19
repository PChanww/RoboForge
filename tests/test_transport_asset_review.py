"""Explicit generated-asset consent and cache/library boundaries, without fees."""
import io
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zipfile import ZipFile

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from test_transport_api import Plant, finished, eventually
from test_transport_generation import generated, awaiting
from vlab.transport_api import TransportSpec, TransportStore, register_transport
from vlab.transport_asset_review import AssetCatalog


def test_pending_models_require_every_role_and_http_exports_have_source_and_parameters(generated):
    app = FastAPI()
    store = register_transport(app, generated.store.configs, generated.store.lab, generated.generation)
    store.env_factory, store.realtime = Plant, False
    with TestClient(app) as client:
        row = client.post('/api/v1/transport/runs', json={
            'spec': TransportSpec(object_mode='generate', bin_mode='generate').model_dump(),
            'client_request_id': 'review-http'}).json()
        rid = row['id']
        row = awaiting(store, rid)
        assert not Plant.instances and generated.provider.submits == 2
        assert all(item['status'] == 'pending' and not item['saved'] for item in row['asset_reviews'])
        caps = client.get('/api/v1/transport/capabilities').json()
        assert row['spec']['object_asset'] not in [item['id'] for item in caps['objects']]
        assert row['spec']['bin_asset'] not in [item['id'] for item in caps['bins']]
        item = row['asset_reviews'][0]
        assert item['preview_format'] == 'glb'
        assert client.get(item['preview_url']).content[:4] == b'glTF'
        assert b'v ' in client.get(item['preview']['obj_url']).content
        exported = client.get(item['download_url'])
        assert exported.status_code == 200
        with ZipFile(io.BytesIO(exported.content)) as archive:
            assert set(archive.namelist()) == {'original.glb', 'visual.obj', 'manifest.json', 'transport_context.json'}
            context = json.loads(archive.read('transport_context.json'))
            assert context['decision'] is None and context['storage'] == 'cache'
            assert context['parameters']['height_m'] == .06
            assert json.loads(archive.read('manifest.json'))['source'] == 'lux3d'
        path = f'/api/v1/transport/runs/{rid}/assets/object/review'
        result = client.post(path, json={'decision': 'save', 'client_request_id': 'save-object'})
        assert result.status_code == 200 and result.json()['status'] == 'awaiting_asset_review'
        assert not Plant.instances
        assert client.post(path, json={'decision': 'save', 'client_request_id': 'save-object'}).status_code == 200
        saved = result.json()['asset_reviews'][0]
        assert saved['saved'] and saved['saved_at'] and saved['storage'] == 'library'
        caps = client.get('/api/v1/transport/capabilities').json()
        assert saved['asset_id'] in [item['id'] for item in caps['objects']]
        assert not caps['bins']
        assert client.post(f'/api/v1/transport/runs/{rid}/assets/bin/review',
            json={'decision': 'use_once', 'client_request_id': 'use-bin'}).status_code == 200
        final = finished(store, rid)
        assert final['status'] == 'succeeded' and len(Plant.instances) == 1
        assert generated.provider.submits == 2
        assert final['asset_reviews'][1]['decision'] == 'use_once' and not final['asset_reviews'][1]['saved']
        assert client.post(path, json={'decision': 'reject', 'client_request_id': 'change-mind'}).status_code == 409
        assert client.get(f'/api/v1/transport/runs/{rid}/assets/unknown/visual.obj').status_code == 422


def test_partial_reviews_survive_close_and_restart_without_accepting_unanswered_model(generated):
    store = generated.store
    row = awaiting(store, store.create(TransportSpec(object_mode='generate', bin_mode='generate'), 'restart-review')['id'])
    rid = row['id']
    store.review_asset(rid, 'object', 'save', 'keep-object')
    store.close()
    assert store.get(rid)['status'] == 'awaiting_asset_review'
    reopened = TransportStore(store.configs, store.lab, env_factory=Plant, realtime=False, generation=generated.generation)
    try:
        restored = reopened.get(rid)
        assert restored['status'] == 'awaiting_asset_review' and not Plant.instances
        assert restored['asset_reviews'][0]['decision'] == 'save'
        assert restored['asset_reviews'][1]['status'] == 'pending'
        reopened.review_asset(rid, 'bin', 'use_once', 'temporary-bin')
        final = finished(reopened, rid)
        assert final['status'] == 'succeeded' and generated.provider.submits == 2
        assert len(Plant.instances) == 1
    finally:
        reopened.close()


def test_reject_and_stop_preserve_caches_and_do_not_start_physics(generated):
    store = generated.store
    row = awaiting(store, store.create(TransportSpec(object_mode='generate'), 'reject-review')['id'])
    rid, ref = row['id'], row['spec']['object_asset']
    rejected = store.review_asset(rid, 'object', 'reject', 'reject-object')
    assert rejected['status'] == 'cancelled' and not Plant.instances
    assert (store.lab / 'assets' / ref / 'manifest.json').is_file()
    assert not store.catalog.is_reusable(ref)
    with pytest.raises(ValueError, match='尚未保存'):
        store.create(TransportSpec(object_asset=ref), 'reuse-rejected-cache')
    row = awaiting(store, store.create(TransportSpec(object_mode='generate'), 'stop-review')['id'])
    assert store.stop(row['id'])['status'] == 'cancelled'
    assert not Plant.instances and generated.provider.submits == 2


def test_catalog_transaction_history_and_concurrent_instances_do_not_lose_approvals(tmp_path):
    historical = tmp_path / 'assets/imported/historical'
    historical.mkdir(parents=True)
    (historical / 'manifest.json').write_text('{}', encoding='utf-8')
    first = AssetCatalog(tmp_path)
    assert first.is_reusable('imported/historical')
    refs = [f'imported/new-{i}' for i in range(12)]
    for ref in refs:
        directory = tmp_path / 'assets' / ref
        directory.mkdir(parents=True)
        (directory / 'manifest.json').write_text('{}', encoding='utf-8')
    assert first.is_reusable(refs[0]), 'Untracked imports retain other modules original behavior'
    for i, ref in enumerate(refs):
        first.record_pending(ref, f'run-{i}', 'object')
    second = AssetCatalog(tmp_path)
    assert not second.is_reusable(refs[0]), 'A second catalog instance must not absorb new caches as historical'
    def approve(index):
        catalog = first if index % 2 else second
        catalog.record_pending(refs[index], f'run-{index}', 'object')
        return catalog.review(refs[index], f'run-{index}', 'object', 'save', f'request-{index}')
    with ThreadPoolExecutor(max_workers=6) as pool:
        assert len(list(pool.map(approve, range(len(refs))))) == len(refs)
    snapshot = first.snapshot()
    assert len(snapshot['decisions']) == len(refs)
    assert all(second.is_reusable(ref) for ref in refs)
    # Different IDs can repeat one decision, but each ID stays bound even when
    # it was introduced as an alias of an already-accepted model.
    first.review(refs[0], 'run-0', 'object', 'save', 'alias-zero')
    with pytest.raises(ValueError, match='绑定其他模型'):
        second.review(refs[1], 'run-1', 'object', 'save', 'alias-zero')
    with pytest.raises(ValueError, match='绑定其他模型'):
        second.review(refs[0], 'run-0', 'object', 'save', 'request-1')
    with pytest.raises(ValueError):
        first.asset_directory('imported/../../outside')


def test_crash_after_durable_decision_resumes_approved_model_without_regeneration(generated):
    store = generated.store
    row = awaiting(store, store.create(TransportSpec(object_mode='generate'), 'decision-crash')['id'])
    rid = row['id']
    # Simulate a crash between catalog commit and updating the run record.
    store.catalog.review(row['spec']['object_asset'], rid, 'object', 'use_once', 'durable-choice')
    store.close()
    reopened = TransportStore(store.configs, store.lab, env_factory=Plant, realtime=False, generation=generated.generation)
    try:
        final = finished(reopened, rid)
        assert final['status'] == 'succeeded'
        assert len(Plant.instances) == 1 and generated.provider.submits == 1
    finally:
        reopened.close()


def test_last_approval_and_duplicates_wait_for_previous_worker_without_duplicate_physics(generated, monkeypatch):
    store = generated.store
    entered, release = threading.Event(), threading.Event()
    original = store._manifest
    def held_manifest(row):
        if row['status'] == 'awaiting_asset_review' and not entered.is_set():
            entered.set()
            release.wait(timeout=3)
        return original(row)
    monkeypatch.setattr(store, '_manifest', held_manifest)
    try:
        rid = store.create(TransportSpec(object_mode='generate'), 'approval-race')['id']
        assert entered.wait(timeout=3)
        assert store.get(rid)['status'] == 'awaiting_asset_review'
        old_worker = store._workers[rid]
        assert old_worker.is_alive()
        start = time.monotonic()
        first = store.review_asset(rid, 'object', 'use_once', 'first-choice')
        assert time.monotonic()-start < .2, 'Approval must not block while the prior worker is closing'
        assert first['status'] == 'queued' and not Plant.instances
        replacement = store._workers[rid]
        for index in range(4):
            store.review_asset(rid, 'object', 'use_once', f'repeated-choice-{index}')
        assert store._workers[rid] is replacement and replacement is not old_worker
        assert not Plant.instances
        release.set()
        final = finished(store, rid)
        assert final['status'] == 'succeeded'
        assert len(Plant.instances) == 1 and generated.provider.submits == 1
    finally:
        release.set()
