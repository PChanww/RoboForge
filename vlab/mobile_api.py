"""Live wheel-drive sessions, with one physics/render owner per session.

Only cached frames cross into request threads. A short command watchdog stops
manual motion after a lost key-up/network connection; recordings store S_t and
the actual acceleration-limited command applied for the following control step.
"""
from __future__ import annotations

import json
import math
import queue
import threading
import time
import uuid
from concurrent.futures import Future
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from .duck_demo import CHASSIS, asset_info
from .scene_assets import valid_id

DT = .05
WATCHDOG_S = .45
ROUTE_HEARTBEAT_S = 1.5
IDLE_S = 90.
SESSION_S = 600.


def _now():
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def _plain(value):
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _finish_recording(directory, manifest, rows, *, status='completed', reason='user_stopped'):
    dimensions = {'time': (), 'base_pose': (7,), 'base_velocity': (6,),
                  'wheel_qpos': (4,), 'wheel_qvel': (4,), 'arm_qpos': (7,),
                  'command': (2,), 'requested_command': (2,)}
    arrays = {key: np.asarray([row[key] for row in rows], dtype=np.float64).reshape((len(rows), *shape))
              for key, shape in dimensions.items()}
    arrays['control_mode'] = np.asarray([row['control_mode'] for row in rows], dtype='U8')
    temp = directory / 'data.npz.tmp'
    with temp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temp.replace(directory / 'data.npz')
    manifest.update(status=status, reason=reason, finished=_now(), frames=len(rows),
                    duration_s=round(len(rows) * DT, 6),
                    sim_time_start=rows[0]['time'] if rows else None,
                    sim_time_end=rows[-1]['time'] + DT if rows else None)
    _atomic_json(directory / 'manifest.json', manifest)
    rid = directory.name
    return {'id': rid, 'frames': len(rows), 'status': status,
            'manifest_url': f'/api/v1/mobile/datasets/{rid}/manifest',
            'npz_url': f'/api/v1/mobile/datasets/{rid}/data.npz'}


class MobileConflict(ValueError):
    pass


class MobileSession:
    def __init__(self, config, lab_root, chassis_asset, *, env_factory=None):
        self.id = 'drive_' + uuid.uuid4().hex[:16]
        self.lab = Path(lab_root)
        self.asset_ref = chassis_asset
        self._config = config
        self._env_factory = env_factory
        self._cache_lock = threading.Lock()
        self._cache = {'id': self.id, 'status': 'ready', 'images': {},
                       'recording': {'active': False, 'frames': 0}, 'control_mode': 'manual'}
        self._queue = queue.Queue()
        self._ready = Future()
        self._alive = True
        self._last_seen = self._created = time.monotonic()
        self._last_command = 0.
        self._last_seq = -1
        self._desired = (0., 0.)
        self._mode = 'manual'
        self._route_status = None
        self._route = []
        self._record = None
        self._dataset = None
        self._thread = threading.Thread(target=self._run, name='mobile-' + self.id[6:], daemon=True)
        self._thread.start()
        try:
            self._ready.result(timeout=60)
        except Exception:
            self._alive = False
            raise

    def snapshot(self, *, touch=True):
        with self._cache_lock:
            if touch:
                self._last_seen = time.monotonic()
            return dict(self._cache)

    def call(self, operation, payload=None):
        if not self._alive:
            raise MobileConflict('该驾驶会话已结束，请开始新会话。')
        self._last_seen = time.monotonic()
        future = Future()
        self._queue.put((operation, payload, future))
        return future.result(timeout=30)

    def _publish(self, env, *, status=None, error=None, render=True):
        snapshot = _plain(env.snapshot(render=render))
        if not snapshot.get('images'):
            snapshot['images'] = self.snapshot(touch=False).get('images', {})
        moving = abs(snapshot.get('velocity', {}).get('linear_m_s', 0)) > .005
        moving |= abs(snapshot.get('velocity', {}).get('angular_rad_s', 0)) > .02
        moving |= any(abs(value) > 1e-5 for value in self._desired) or self._mode == 'route'
        snapshot.update(id=self.id, status=status or ('running' if moving else 'ready'),
                        control_mode=self._mode, route_status=self._route_status,
                        recording={'active': self._record is not None,
                                   'frames': len(self._record['rows']) if self._record else 0},
                        dataset=self._dataset,
                        requested_command={'linear_m_s': self._desired[0], 'angular_rad_s': self._desired[1]},
                        command_timeout_s=WATCHDOG_S,
                        asset=_plain(getattr(env, 'asset', getattr(env, 'asset_info', {}))))
        if error:
            snapshot['error'] = error
        with self._cache_lock:
            self._cache = snapshot
        return dict(snapshot)

    def _start_record(self, env):
        if self._record:
            return
        rid = 'mobile_' + uuid.uuid4().hex[:16]
        directory = self.lab / 'datasets/mobile' / rid
        directory.mkdir(parents=True)
        manifest = {
            'id': rid, 'kind': 'mobile_driving', 'schema_version': '1.0',
            'status': 'recording', 'created': _now(), 'session_id': self.id,
            'frequency_hz': 20, 'physics': {
                'base': 'unconstrained free rigid body, four velocity-driven contact wheels',
                'pose_updates': 'MuJoCo physics only; no runtime base teleportation',
                'parameters': 'estimated wheel geometry, friction and mass; not real-world calibrated',
                'controller': 'manual differential drive or fixed waypoint demonstration; no obstacle planner',
            },
            'asset': _plain(getattr(env, 'asset', getattr(env, 'asset_info', {}))),
            'fields': {
                'time': 'S_t simulation time in seconds',
                'base_pose': 'S_t [x,y,z,qw,qx,qy,qz] in world frame',
                'base_velocity': 'S_t freejoint qvel [vx,vy,vz,wx,wy,wz], MuJoCo freejoint convention',
                'wheel_qpos': 'S_t four wheel joint angles, radians',
                'wheel_qvel': 'S_t four wheel angular velocities, rad/s',
                'arm_qpos': 'S_t articulated arm joint positions; held during driving',
                'command': 'A_t actual acceleration-limited [linear m/s, angular rad/s] for next 50 ms',
                'requested_command': 'requested [linear m/s, angular rad/s] before acceleration limiting',
                'control_mode': 'manual or route',
            },
            'images_recorded': False,
            'camera_note': 'Live cameras are displayed; this driving dataset contains motion states and commands only.',
            'action_alignment': 'S_t is sampled before the matching A_t is applied for one control step',
        }
        _atomic_json(directory / 'manifest.json', manifest)
        self._record = {'directory': directory, 'manifest': manifest, 'rows': [],
                        'stream': (directory / 'frames.jsonl').open('w', encoding='utf-8', buffering=1)}

    def _end_record(self, reason='user_stopped', status='completed'):
        if not self._record:
            return
        record = self._record
        record['stream'].close()
        self._dataset = _finish_recording(record['directory'], record['manifest'], record['rows'],
                                          status=status, reason=reason)
        self._record = None

    def _record_step(self, state, command, requested):
        if not self._record:
            return
        row = {key: _plain(state[key]) for key in
               ('time', 'base_pose', 'base_velocity', 'wheel_qpos', 'wheel_qvel', 'arm_qpos')}
        row['time'] = float(row['time'])
        row.update(command=[float(command['linear_m_s']), float(command['angular_rad_s'])],
                   requested_command=list(requested), control_mode=self._mode)
        if not all(np.isfinite(np.asarray(value)).all() for key, value in row.items() if key != 'control_mode'):
            raise ValueError('驾驶状态含非有限数值，采集已停止。')
        self._record['stream'].write(json.dumps(row, ensure_ascii=False) + '\n')
        self._record['rows'].append(row)

    def _route_command(self, snapshot):
        pose = snapshot['pose']
        now = snapshot['sim_time']
        if now - self._route_start > 60:
            self._mode, self._route_status = 'manual', 'timeout'
            return (0., 0.)
        while self._route:
            dx, dy = self._route[0][0] - pose['x'], self._route[0][1] - pose['y']
            distance = math.hypot(dx, dy)
            if distance > .045:
                heading = math.atan2(dy, dx) - pose['yaw']
                heading = math.atan2(math.sin(heading), math.cos(heading))
                linear = min(.14, distance * .9) * max(0., math.cos(heading))
                if abs(heading) > .65:
                    linear = 0.
                return linear, float(np.clip(heading * 2.8, -1.2, 1.2))
            self._route.pop(0)
        self._mode, self._route_status = 'manual', 'completed'
        return 0., 0.

    def _operation(self, env, operation, payload):
        if operation == 'command':
            if payload['seq'] > self._last_seq:
                self._last_seq = payload['seq']
                self._last_command = time.monotonic()
                self._desired = payload['linear_m_s'], payload['angular_rad_s']
                self._mode, self._route_status = 'manual', None
        elif operation == 'stop':
            if payload and payload.get('seq') is not None:
                if payload['seq'] < self._last_seq:
                    return self._publish(env, render=False)
                self._last_seq = max(self._last_seq, payload['seq'])
            self._desired = (0., 0.)
            self._last_command = 0.
            self._mode, self._route_status = 'manual', 'stopped' if self._mode == 'route' else None
        elif operation == 'route':
            self._route = [[.6, 0.], [.6, .5], [0., .5], [0., 0.]]
            self._route_start = self.snapshot(touch=False).get('sim_time', 0.)
            self._mode, self._route_status = 'route', 'running'
        elif operation == 'reset':
            if self._record:
                raise MobileConflict('请先结束当前数据记录，再复位机器人。')
            self._desired, self._mode, self._route_status = (0., 0.), 'manual', None
            self._last_command = 0.
            env.reset()
        elif operation == 'record':
            if payload['active']:
                self._start_record(env)
            else:
                self._end_record()
        elif operation == 'close':
            self._desired, self._mode = (0., 0.), 'manual'
            self._end_record(reason=payload or 'session_closed')
            self._alive = False
        else:
            raise ValueError('未知驾驶操作。')
        return self._publish(env, status='closed' if not self._alive else None, render=False)

    def _run(self):
        env = None
        try:
            if self._env_factory is None:
                from .mobile import MobileEnv
                factory = MobileEnv
            else:
                factory = self._env_factory
            env = factory(self._config, self.lab, self.asset_ref)
            self._publish(env)
            self._ready.set_result(True)
            deadline = time.monotonic()
            while self._alive:
                while True:
                    try:
                        operation, payload, future = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    try:
                        result = self._operation(env, operation, payload)
                        future.set_result(result)
                    except Exception as exc:
                        future.set_exception(exc)
                    if not self._alive:
                        break
                if not self._alive:
                    break
                now = time.monotonic()
                if now - self._last_seen > IDLE_S or now - self._created > SESSION_S:
                    reason = 'idle_timeout' if now - self._last_seen > IDLE_S else 'session_time_limit'
                    self._operation(env, 'close', reason)
                    break
                if self._mode == 'route':
                    if now - self._last_seen > ROUTE_HEARTBEAT_S:
                        self._mode, self._route_status = 'manual', 'connection_lost'
                        self._desired = (0., 0.)
                    else:
                        self._desired = self._route_command(self.snapshot(touch=False))
                elif now - self._last_command > WATCHDOG_S:
                    self._desired = (0., 0.)
                state = env.state() if self._record else None
                requested = self._desired
                applied = env.step(*requested, dt=DT)
                command = applied.get('command', applied)
                if state is not None:
                    self._record_step(state, command, requested)
                self._publish(env, render=True)
                deadline += DT
                delay = deadline - time.monotonic()
                # A renderer stall must not trigger an unbounded catch-up loop.
                if delay < -.25:
                    deadline = time.monotonic()
                    delay = 0.
                if delay > 0:
                    time.sleep(min(delay, DT))
        except Exception as exc:
            message = f'{type(exc).__name__}: {exc}'
            if not self._ready.done():
                self._ready.set_exception(exc)
            try:
                self._end_record(reason=message, status='interrupted')
            except Exception:
                pass
            with self._cache_lock:
                self._cache.update(status='failed', error=message, recording={'active': False, 'frames': 0}, dataset=self._dataset)
        finally:
            self._alive = False
            if env is not None:
                env.close()
            while True:
                try:
                    _, _, future = self._queue.get_nowait()
                except queue.Empty:
                    break
                if not future.done():
                    future.set_exception(MobileConflict('驾驶会话已结束。'))

    def close(self):
        if self._alive:
            try:
                self.call('close')
            finally:
                self._thread.join(timeout=10)
        return self.snapshot(touch=False)


class MobileManager:
    def __init__(self, configs, lab_root, *, env_factory=None):
        self.configs, self.lab, self.env_factory = configs, Path(lab_root), env_factory
        self._lock = threading.RLock()
        self._sessions, self._requests = {}, {}
        self._recover_recordings()

    def _recover_recordings(self):
        for path in (self.lab / 'datasets/mobile').glob('mobile_*/manifest.json'):
            try:
                manifest = json.loads(path.read_text(encoding='utf-8'))
                if manifest.get('status') != 'recording':
                    continue
                rows = []
                journal = path.parent / 'frames.jsonl'
                if journal.exists():
                    for line in journal.read_text(encoding='utf-8').splitlines():
                        try:
                            rows.append(json.loads(line))
                        except ValueError:
                            break
                _finish_recording(path.parent, manifest, rows, status='interrupted', reason='backend_restarted')
            except (OSError, ValueError, KeyError):
                continue

    def create(self, chassis_asset, request_id):
        with self._lock:
            if request_id in self._requests:
                sid, ref = self._requests[request_id]
                if ref != chassis_asset:
                    raise MobileConflict('提交ID已绑定其他底盘；请创建新的驾驶请求。')
                return self._sessions[sid].snapshot()
            if sum(session._alive for session in self._sessions.values()) >= 2:
                raise MobileConflict('最多同时打开两个驾驶会话，请先结束已有会话。')
            item = asset_info(self.lab, chassis_asset, '四轮底盘 · Aholo')
            if item.get('source') != 'lux3d' or item.get('kind') != 'robot_visual':
                raise MobileConflict('请使用已导入的 Aholo 机器人底盘外观。')
            session = MobileSession(self.configs.load(), self.lab, chassis_asset, env_factory=self.env_factory)
            self._sessions[session.id] = session
            self._requests[request_id] = session.id, chassis_asset
            return session.snapshot()

    def get(self, sid):
        valid_id(sid)
        with self._lock:
            if sid not in self._sessions:
                raise KeyError(sid)
            return self._sessions[sid]

    def close_all(self):
        with self._lock:
            sessions = list(self._sessions.values())
        for session in sessions:
            session.close()


class CreateMobileBody(BaseModel):
    model_config = ConfigDict(extra='forbid')
    chassis_asset: str = CHASSIS
    client_request_id: str = Field(min_length=1, max_length=100)


class DriveBody(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    linear_m_s: float = Field(ge=-.22, le=.22)
    angular_rad_s: float = Field(ge=-1.8, le=1.8)
    seq: int = Field(ge=0, le=2**53-1)


class RecordBody(BaseModel):
    active: bool


class StopBody(BaseModel):
    seq: int | None = Field(default=None, ge=0, le=2**53-1)


def register_mobile(app, configs, lab_root):
    manager = MobileManager(configs, lab_root)

    def invoke(fn, *args):
        try:
            return fn(*args)
        except KeyError:
            raise HTTPException(404, '驾驶会话不存在，后端重启后请开始新会话。')
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(409, str(exc))

    @app.get('/api/v1/mobile/capabilities')
    def capabilities():
        assets = []
        for path in sorted((Path(lab_root) / 'assets/imported').glob('*/manifest.json')):
            try:
                ref = 'imported/' + path.parent.name
                item = asset_info(Path(lab_root), ref, '四轮底盘 · Aholo' if ref == CHASSIS else path.parent.name)
                if item['source'] == 'lux3d' and item['kind'] == 'robot_visual':
                    assets.append(item)
            except (ValueError, KeyError, OSError):
                continue
        assets.sort(key=lambda item: item['id'] != CHASSIS)
        return {'assets': assets, 'default_asset': CHASSIS,
                'limits': {'linear_m_s': .22, 'angular_rad_s': 1.8},
                'notes': ['真实四轮接触驱动，底盘自由运动；机械臂保持收拢姿态。',
                          'Aholo 底盘外观复用；轮组尺寸、质量和摩擦采用演示估计，未做实物标定。',
                          '示例路线是固定航点演示，暂不包含自动避障或抓取搬运。',
                          '驾驶记录包含20 Hz状态和动作；实时相机画面暂不写入驾驶数据。',
                          '手动控制450 ms无新指令会刹停；无人查看90秒或会话满10分钟自动结束。']}

    @app.post('/api/v1/mobile/sessions')
    def create(body: CreateMobileBody):
        return invoke(manager.create, body.chassis_asset, body.client_request_id)

    @app.get('/api/v1/mobile/sessions/{sid}')
    def read(sid: str):
        return invoke(lambda: manager.get(sid).snapshot())

    @app.post('/api/v1/mobile/sessions/{sid}/command')
    def command(sid: str, body: DriveBody):
        return invoke(lambda: manager.get(sid).call('command', body.model_dump()))

    @app.post('/api/v1/mobile/sessions/{sid}/stop')
    def stop(sid: str, body: StopBody | None = None):
        return invoke(lambda: manager.get(sid).call('stop', body.model_dump() if body else None))

    @app.post('/api/v1/mobile/sessions/{sid}/route')
    def route(sid: str):
        return invoke(lambda: manager.get(sid).call('route'))

    @app.post('/api/v1/mobile/sessions/{sid}/reset')
    def reset(sid: str):
        return invoke(lambda: manager.get(sid).call('reset'))

    @app.post('/api/v1/mobile/sessions/{sid}/record')
    def record(sid: str, body: RecordBody):
        return invoke(lambda: manager.get(sid).call('record', body.model_dump()))

    @app.delete('/api/v1/mobile/sessions/{sid}')
    def close(sid: str):
        return invoke(lambda: manager.get(sid).close())

    def dataset_path(rid, filename):
        valid_id(rid)
        if not rid.startswith('mobile_'):
            raise HTTPException(404, '驾驶数据不存在。')
        path = Path(lab_root) / 'datasets/mobile' / rid / filename
        if not path.is_file():
            raise HTTPException(404, '驾驶数据尚未保存。')
        return path

    @app.get('/api/v1/mobile/datasets/{rid}/manifest')
    def manifest(rid: str):
        return FileResponse(invoke(dataset_path, rid, 'manifest.json'), media_type='application/json', filename=f'{rid}.json')

    @app.get('/api/v1/mobile/datasets/{rid}/data.npz')
    def data(rid: str):
        return FileResponse(invoke(dataset_path, rid, 'data.npz'), media_type='application/octet-stream', filename=f'{rid}.npz')

    @app.on_event('shutdown')
    def shutdown_mobile():
        manager.close_all()

    return manager
