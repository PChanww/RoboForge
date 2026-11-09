"""Gymnasium wrapper over PlatformEnv. Distinguishes terminated vs truncated."""
from __future__ import annotations

import base64
from io import BytesIO

import gymnasium as gym
import numpy as np
from PIL import Image

from .engine import PlatformEnv
from .schema import ProjectConfig


def _decode(jpeg_b64, shape):
    if jpeg_b64 is None:
        return np.zeros(shape, dtype=np.uint8)
    return np.asarray(Image.open(BytesIO(base64.b64decode(jpeg_b64))).convert("RGB"), dtype=np.uint8)


class VLabGymEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, config: ProjectConfig, lab_root, robot_id=None, controller_id=None, render_mode="rgb_array"):
        super().__init__()
        self.env = PlatformEnv(config, lab_root, robot_id, controller_id)
        self.render_mode = render_mode
        c = self.env.controller
        self.action_space = gym.spaces.Box(low=np.array([r[0] for r in c.action_range], dtype=np.float32),
                                           high=np.array([r[1] for r in c.action_range], dtype=np.float32),
                                           dtype=np.float32)
        self._state_dim = len(self.env._joints) * 2 + 3 + 1
        spaces = {"state": gym.spaces.Box(-np.inf, np.inf, (self._state_dim,), np.float32)}
        self._img_shapes = {}
        for s in config.sensors:
            w, h = s.resolution
            self._img_shapes[s.id] = (h, w, 3)
            spaces[s.id] = gym.spaces.Box(0, 255, (h, w, 3), np.uint8)
        self.observation_space = gym.spaces.Dict(spaces)

    def _obs_vec(self, obs):
        st = obs["state"]
        vec = ([st["joint_pos"][n] for n in self.env._joints]
               + [st["joint_vel"][n] for n in self.env._joints]
               + st["eef_position"] + [st["gripper"] if st["gripper"] is not None else 0.0])
        out = {"state": np.asarray(vec, dtype=np.float32)}
        for cid, shape in self._img_shapes.items():
            frame = obs["images"].get(cid)
            out[cid] = _decode(frame["jpeg_b64"] if frame else None, shape)
        return out

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.env.reset(seed)
        return self._obs_vec(obs), {"schema_version": obs["schema_version"], "tick": obs["tick"]}

    def step(self, action):
        obs = self.env.step(np.asarray(action, dtype=float).tolist())
        m = self.env.metrics()
        reward = 1.0 if m["success"] else 0.0
        return self._obs_vec(obs), reward, m["terminated"], m["truncated"], m

    def render(self):
        return self.env.render("front", 800, 540)

    def close(self):
        self.env.close()