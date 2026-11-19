"""Driver ownership, network fail-safe and recorded state/action alignment."""
import json
import shutil
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import vlab.mobile_api as mobile_api
from vlab.duck_demo import CHASSIS
from vlab.make_default_config import build
from vlab.mobile_api import MobileConflict, MobileManager, register_mobile


LAB = Path(__file__).resolve().parents[1]


def eventually(predicate, timeout=3.):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(.01)
    raise AssertionError('Timed out waiting for driver state')


class ClockEnv:
    """Deterministic owner-checked plant, for precise timing/data assertions."""
    def __init__(self, config, lab, asset):
        self.owner = threading.get_ident()
        self.asset = {'id': asset, 'source': 'lux3d', 'mesh_sha256': 'test-hash'}
        self.reset()

    def check(self):
        assert threading.get_ident() == self.owner

    def reset(self):
        self.check()
        self.tick, self.v, self.w, self.x = 0, 0., 0., 0.

    def state(self):
        self.check()
        return {'time': self.tick * .05, 'base_pose': np.array([self.x, 0, .03, 1, 0, 0, 0]),
                'base_velocity': np.array([self.v, 0, 0, 0, 0, self.w]),
                'wheel_qpos': np.full(4, self.x / .028), 'wheel_qvel': np.full(4, self.v / .028),
                'arm_qpos': np.zeros(7)}

    def step(self, v, w, dt=.05):
        self.check()
        self.v += float(np.clip(v-self.v, -.0175, .0175))
        self.w += float(np.clip(w-self.w, -.15, .15))
        self.x += self.v * dt
        self.tick += 1
        return self.snapshot(False)

    def snapshot(self, render=True):
        self.check()
        return {'sim_time': self.tick * .05, 'tick': self.tick,
                'pose': {'x': self.x, 'y': 0., 'z': .03, 'yaw': 0.},
                'velocity': {'linear_m_s': self.v, 'angular_rad_s': self.w},
                'command': {'linear_m_s': self.v, 'angular_rad_s': self.w},
                'wheel_velocity_rad_s': [self.v / .028] * 4,
                'trajectory': [{'x': self.x, 'y': 0.}], 'distance_m': abs(self.x),
                'images': {'front': 'frame', 'onboard': 'frame'}}

    def close(self):
        self.check()


@pytest.fixture
def driver(tmp_path, monkeypatch):
    monkeypatch.setattr(mobile_api, 'asset_info', lambda *args: {'source': 'lux3d', 'kind': 'robot_visual'})
    manager = MobileManager(SimpleNamespace(load=lambda: None), tmp_path, env_factory=ClockEnv)
    yield manager
    manager.close_all()


def test_watchdog_stops_lost_key_up_and_sequence_blocks_late_heartbeat(driver):
    first = driver.create(CHASSIS, 'stable-request')
    session = driver.get(first['id'])
    session.call('command', {'linear_m_s': .15, 'angular_rad_s': 0., 'seq': 10})
    eventually(lambda: session.snapshot()['command']['linear_m_s'] > .05)
    eventually(lambda: session.snapshot()['requested_command']['linear_m_s'] == 0.)
    eventually(lambda: session.snapshot()['command']['linear_m_s'] == 0.)
    session.call('command', {'linear_m_s': .1, 'angular_rad_s': 0., 'seq': 11})
    session.call('stop', {'seq': 20})
    session.call('command', {'linear_m_s': .2, 'angular_rad_s': 0., 'seq': 19})
    snapshot = session.snapshot()
    assert snapshot['requested_command'] == {'linear_m_s': 0., 'angular_rad_s': 0.}
    assert snapshot['control_mode'] == 'manual'


def test_same_creation_receipt_never_creates_second_driver(driver):
    first = driver.create(CHASSIS, 'same')
    assert driver.create(CHASSIS, 'same')['id'] == first['id']
    with pytest.raises(MobileConflict, match='绑定其他底盘'):
        driver.create('imported/another', 'same')
    assert len(driver._sessions) == 1
    driver.create(CHASSIS, 'second')
    with pytest.raises(MobileConflict, match='两个'):
        driver.create(CHASSIS, 'third')


def test_late_stop_cannot_cancel_newer_motion_command(driver):
    session = driver.get(driver.create(CHASSIS, 'late-stop')['id'])
    session.call('command', {'linear_m_s': .1, 'angular_rad_s': .2, 'seq': 30})
    old_stop = session.call('stop', {'seq': 20})
    assert old_stop['requested_command'] == {'linear_m_s': .1, 'angular_rad_s': .2}
    new_stop = session.call('stop', {'seq': 31})
    assert new_stop['requested_command'] == {'linear_m_s': 0., 'angular_rad_s': 0.}


def test_route_stops_after_viewer_connection_lost(driver):
    session = driver.get(driver.create(CHASSIS, 'route-disconnect')['id'])
    session.call('route')
    assert session.snapshot(touch=False)['control_mode'] == 'route'
    session._last_seen = time.monotonic() - mobile_api.ROUTE_HEARTBEAT_S - .2

    def disconnected():
        value = session.snapshot(touch=False)
        return value if value['route_status'] == 'connection_lost' else None

    snapshot = eventually(disconnected)
    assert snapshot['control_mode'] == 'manual'
    assert snapshot['requested_command'] == {'linear_m_s': 0., 'angular_rad_s': 0.}
    assert snapshot['status'] != 'closed' and session._alive


def test_recording_pairs_before_state_with_actual_limited_action(driver):
    session = driver.get(driver.create(CHASSIS, 'record')['id'])
    session.call('record', {'active': True})
    session.call('command', {'linear_m_s': .2, 'angular_rad_s': 0., 'seq': 1})
    eventually(lambda: session.snapshot()['recording']['frames'] >= 5)
    with pytest.raises(MobileConflict, match='结束当前数据记录'):
        session.call('reset')
    snapshot = session.call('record', {'active': False})
    dataset = snapshot['dataset']
    directory = driver.lab / 'datasets/mobile' / dataset['id']
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['images_recorded'] is False
    assert manifest['status'] == 'completed'
    arrays = np.load(directory / 'data.npz')
    assert arrays['base_pose'].shape == (dataset['frames'], 7)
    assert arrays['command'].shape == (dataset['frames'], 2)
    assert np.isfinite(arrays['command']).all()
    # The following state's x equals this pre-step x plus the recorded A_t.
    np.testing.assert_allclose(np.diff(arrays['base_pose'][:, 0]), arrays['command'][:-1, 0] * .05)
    np.testing.assert_allclose(np.diff(arrays['time']), .05)
    assert np.any(arrays['requested_command'][:, 0] > arrays['command'][:, 0])


def test_idle_session_closes_and_finalizes_recording(driver):
    session = driver.get(driver.create(CHASSIS, 'idle')['id'])
    session.call('record', {'active': True})
    eventually(lambda: session.snapshot()['recording']['frames'] >= 2)
    session._last_seen = time.monotonic() - mobile_api.IDLE_S - 1
    eventually(lambda: session.snapshot(touch=False)['status'] == 'closed')
    assert not session._alive
    snapshot = session.snapshot(touch=False)
    path = driver.lab / 'datasets/mobile' / snapshot['dataset']['id'] / 'manifest.json'
    assert json.loads(path.read_text(encoding='utf-8'))['reason'] == 'idle_timeout'


def test_empty_recording_exports_consistent_array_shapes(tmp_path):
    directory = tmp_path / 'mobile_empty'
    directory.mkdir()
    result = mobile_api._finish_recording(directory, {}, [])
    assert result['frames'] == 0
    arrays = np.load(directory / 'data.npz')
    for field, width in {'base_pose': 7, 'base_velocity': 6, 'wheel_qpos': 4,
                         'wheel_qvel': 4, 'arm_qpos': 7, 'command': 2, 'requested_command': 2}.items():
        assert arrays[field].shape == (0, width)
    assert arrays['time'].shape == arrays['control_mode'].shape == (0,)


def test_backend_restart_finalizes_durable_journal(tmp_path):
    directory = tmp_path / 'datasets/mobile/mobile_interrupted'
    directory.mkdir(parents=True)
    (directory / 'manifest.json').write_text(json.dumps({'id': directory.name, 'status': 'recording'}), encoding='utf-8')
    row = {'time': 0., 'base_pose': [0, 0, .03, 1, 0, 0, 0], 'base_velocity': [0] * 6,
           'wheel_qpos': [0] * 4, 'wheel_qvel': [0] * 4, 'arm_qpos': [0] * 7,
           'command': [.0175, 0], 'requested_command': [.1, 0], 'control_mode': 'manual'}
    # An interrupted trailing write must not lose the preceding complete frame.
    (directory / 'frames.jsonl').write_text(json.dumps(row) + '\n{"time":', encoding='utf-8')
    MobileManager(SimpleNamespace(load=lambda: None), tmp_path, env_factory=ClockEnv)
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['status'] == 'interrupted' and manifest['reason'] == 'backend_restarted'
    assert manifest['frames'] == 1
    assert np.load(directory / 'data.npz')['command'].shape == (1, 2)


def test_real_http_session_renders_and_drives_with_contact_wheels(tmp_path):
    if not (LAB / 'assets' / CHASSIS / 'visual.obj').exists():
        pytest.skip('Existing Aholo chassis is required for the real physics check')
    destination = tmp_path / 'assets/robot'
    destination.mkdir(parents=True)
    for path in (LAB / 'assets/robot').glob('*.obj'):
        shutil.copyfile(path, destination / path.name)
    shutil.copytree(LAB / 'assets' / CHASSIS, tmp_path / 'assets' / CHASSIS)
    app = FastAPI()
    manager = register_mobile(app, SimpleNamespace(load=build), tmp_path)
    with TestClient(app) as client:
        create = client.post('/api/v1/mobile/sessions', json={'client_request_id': 'real-one', 'chassis_asset': CHASSIS})
        assert create.status_code == 200, create.text
        initial = create.json()
        path = '/api/v1/mobile/sessions/' + initial['id']
        assert all(initial['images'][name].startswith('data:image/jpeg;base64,') for name in ('front', 'onboard'))
        for index in range(4):
            assert client.post(path + '/command', json={'seq': index, 'linear_m_s': .15, 'angular_rad_s': 0}).status_code == 200
            time.sleep(.13)
        moving = client.get(path).json()
        assert moving['pose']['x'] - initial['pose']['x'] > .015
        assert max(abs(value) for value in moving['wheel_velocity_rad_s']) > 1
        assert client.post(path + '/stop', json={'seq': 100}).status_code == 200
        assert client.post(path + '/command', json={'seq': 90, 'linear_m_s': .2, 'angular_rad_s': 0}).status_code == 200
        stopped = client.get(path).json()
        assert stopped['requested_command']['linear_m_s'] == 0.
        assert client.post(path + '/command', json={'seq': 101, 'linear_m_s': 'NaN', 'angular_rad_s': 0}).status_code == 422
        assert client.delete(path).json()['status'] == 'closed'
    assert not manager.get(initial['id'])._alive
