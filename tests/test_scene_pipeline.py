"""Scene pipeline cross-module verification (stages 1-3).

Checks transforms/quaternion order, environment+collision compile, drop/rest/contact,
visual-collision alignment, scene entering sensor RGB, seed reproducibility, and
old-default-task regression.
"""
import sys
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
from vlab.engine import PlatformEnv
from vlab.schema import ProjectConfig, SceneInstance, SceneCollider
from vlab import transforms


def base():
    return ProjectConfig.model_validate_json((LAB / "configs" / "default.json").read_text(encoding="utf-8"))


def scene_cfg():
    c = base()
    c.scene.environment = SceneInstance(asset_id="demo_workspace", revision=1)
    c.scene.use_default_table = False
    c.scene.colliders = [SceneCollider(name="scene_table", shape="box",
                                       full_size_m=[0.8, 0.6, 0.05], position=[0, 0, -0.025])]
    return c


def main():
    out = {}
    # transforms: xyzw<->wxyz and compose + uniform scale
    q = [0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)]  # 90deg about Z, xyzw
    wxyz = transforms.quat_xyzw_to_wxyz(q)
    out["quat_wxyz_first"] = abs(wxyz[0] - np.cos(np.pi / 4)) < 1e-9
    R = transforms.quat_xyzw_to_mat(q)
    out["quat_rot90z"] = np.allclose(R @ [1, 0, 0], [0, 1, 0], atol=1e-9)
    T = transforms.compose(transforms.transform_matrix([1, 2, 3], [0, 0, 0, 1], 2.0),
                           transforms.transform_matrix([1, 0, 0], [0, 0, 0, 1], 1.0))
    out["compose_scale_translate"] = np.allclose(T[:3, 3], [3, 2, 3])

    # environment + collision compile, drop/rest/contact
    env = PlatformEnv(scene_cfg(), LAB)
    out["env_info"] = env._env_info
    m = env.metrics()
    h = env._asset_height
    out["cup_rest_on_scene_table"] = abs(m["object_xyz"][2] - h / 2) < 0.01
    names = set()
    for ct in env.data.contact:
        for g in (ct.geom1, ct.geom2):
            n = env.model.geom(int(g)).name
            if n:
                names.add(n)
    out["cup_table_contact"] = {"cup_contact", "scene_table"} <= names
    img_scene = env.render("front", 160, 120).copy()
    env.reset(0)
    p1 = env.metrics()["object_xyz"]
    env.reset(0)
    p2 = env.metrics()["object_xyz"]
    out["seed_reproducible"] = np.allclose(p1, p2)
    env.close()

    # scene enters sensor RGB (differs from default empty table)
    envd = PlatformEnv(base(), LAB)
    img_default = envd.render("front", 160, 120).copy()
    envd.close()
    out["scene_changes_rgb"] = not np.array_equal(img_scene, img_default)

    # old default task regression (servo grasp)
    env = PlatformEnv(base(), LAB)
    env.reset(0)
    succ = False
    for _ in range(200):
        env.step(env.servo_expert_action())
        if env.metrics()["success"]:
            succ = True
            break
    out["default_task_regression"] = succ
    env.close()
    assert all(v for k, v in out.items() if isinstance(v, bool)), out
    print(out)


if __name__ == "__main__":
    main()