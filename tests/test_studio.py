"""Task studio acceptance: text intent, isolation, contact-driven execution."""
import json
from pathlib import Path
import sys
import tempfile

import mujoco
import numpy as np

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
from vlab.engine import PlatformEnv
from vlab.make_default_config import build
from vlab.store import ConfigStore
from vlab.studio import StudioSpec, build_config, parse_prompt


def test_intents():
    cases = [
        ("在分拣工作台上，把红色方块搬运到右侧托盘，随机摆位，采集3条示范。", ("sorting", "pick_place", "block", "red", "right", 3)),
        ("在简洁工作台上，把绿色方块沿桌面推到中间目标区，采集两条示范。", ("workbench", "push", "block", "green", "center", 2)),
        ("在实验台上，把蓝色圆柱放到左侧托盘，随机摆位，采集十三次。", ("lab", "pick_place", "cylinder", "blue", "left", 13)),
    ]
    for prompt, expected in cases:
        plan = parse_prompt(prompt)
        assert plan["ready"], plan
        spec = plan["spec"]
        assert tuple(spec[k] for k in ("scene", "task", "asset", "color", "goal", "episodes")) == expected
    for prompt in ("把杯子里的水倒入另一个杯子，采集3条", "在厨房把红色方块放到右边，采集1条", "把蓝色圆柱推到左边", "把方块搬到托盘，采集30条", "让机器人开门"):
        assert not parse_prompt(prompt)["ready"], prompt
    fixed = parse_prompt("把杯子放到右侧托盘，固定摆位，采集五条，种子42")["spec"]
    assert not fixed["randomize"] and fixed["seed"] == 42 and fixed["episodes"] == 5


def test_physics():
    results = []
    with tempfile.TemporaryDirectory(prefix="roboforge_studio_") as tmp:
        store = ConfigStore(Path(tmp) / "projects")
        store.save(build())
        original = store.load().model_dump_json()
        for task, asset in (("pick_place", "block"), ("pick_place", "cylinder"), ("push", "block")):
            for goal in ("left", "center", "right"):
                for seed in (0, 7):
                    spec = StudioSpec(task=task, asset=asset, goal=goal, seed=seed, scene="lab", color="blue")
                    cfg = build_config(spec, store, "experiment")
                    assert cfg.scene.seed == seed and cfg.project_id != "default"
                    assert cfg.task.object_color_rgb == [.18, .44, .83]
                    cfg.sensors = []
                    env = PlatformEnv(cfg, LAB)
                    try:
                        expected_geom = mujoco.mjtGeom.mjGEOM_BOX if asset == "block" else mujoco.mjtGeom.mjGEOM_CYLINDER
                        assert env.model.geom("cup_contact").type[0] == expected_geom
                        max_height = 0
                        for _ in range(200):
                            env.step(env.servo_expert_action())
                            max_height = max(max_height, env.data.xpos[env._judge.cup, 2])
                            if env.metrics()["success"]:
                                break
                        assert env.metrics()["success"], (task, asset, goal, seed, env.metrics())
                        assert env.episode_time() <= cfg.task.time_limit_s
                        if task == "push":
                            assert max_height < env._asset_height / 2 + .004, "push secretly lifts the object"
                            assert not env.metrics()["grasped"]
                        else:
                            assert env.metrics()["lifted"] and env.metrics()["tray_contact"]
                        results.append([task, asset, goal, seed, env.scheduler.tick])
                    finally:
                        env.close()
        # Negative control: place the closed fingertips ABOVE the object during
        # the push phase. Without contact the object must stay at its spawn.
        cfg = build_config(StudioSpec(task="push"), store)
        cfg.sensors = []
        env = PlatformEnv(cfg, LAB)
        try:
            for _ in range(200):
                action = env.servo_expert_action()
                if env._push_phase >= 2:
                    action[2] = float(np.clip((.13 - env._target[2]) / env.controller.max_delta_m, -1, 1))
                env.step(action)
            assert not env.metrics()["success"]
            assert np.linalg.norm(env.data.xpos[env._judge.cup, :2] - env._judge.start_xyz[:2]) < .005
        finally:
            env.close()
        assert store.load().model_dump_json() == original, "planning changed the default project"
        blocked = StudioSpec(unsupported=["unsupported action"])
        try:
            build_config(blocked, store)
            raise AssertionError("unsupported requirement allowed execution")
        except ValueError:
            pass
    print(json.dumps({"successful_runs": results, "no_contact_negative": "passed", "default_project_isolation": "passed"}))


if __name__ == "__main__":
    test_intents()
    test_physics()
