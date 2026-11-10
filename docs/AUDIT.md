# RoboForge — Audit and experiment results

**English** | [简体中文](AUDIT.zh-CN.md) | [README](../README.md)

This document collects the recorded experiment results, validation evidence, implementation limitations, and historical results previously included in the README. It is based on existing reports; moving this documentation did not rerun the experiments. Results describe their recorded configuration and code version.

## Results and limitations

The following results come from the saved experiment reports. Scripted-expert performance and learned-policy performance are reported separately.

| Experiment | Recorded result | Evidence |
| --- | --- | --- |
| Current 5R arm, scripted pick-and-place expert, 4 cup types × 10 seeds | **40/40 successful** | [Arm audit](../experiments/20261109-arm-audit/REPORT.md) |
| Same motion path with jaws held open | **0/40 successful** | [Arm audit](../experiments/20261109-arm-audit/REPORT.md) |
| Sampled task workspace, self-collision / analytical IK error | 0 detected / 0.0 mm kinematic error in that test | [Arm audit](../experiments/20261109-arm-audit/REPORT.md) |
| Camera at 30 FPS with control at 20 Hz | 271 frames over 9 s; mean interval 0.03333 s | [Core report](../experiments/20261109-platform-core/REPORT.md), `tests/test_engine.py` |
| Expert dataset | 21 episodes / 3,297 frames; official LeRobot loader read-back passed | [Data and training report](../experiments/20261110-data-training/REPORT.md) |
| State BC policy | Training, checkpoint reload, and evaluation completed; **0/11 closed-loop successes** | [Data and training report](../experiments/20261110-data-training/REPORT.md) |
| Exported bundle in a fresh process | Expert succeeded; learned policy ran without completing the task | [Deployment report](../experiments/20261110-deploy-export/REPORT.md) |

Current constraints:

- The 5R arm uses fixed-orientation analytical IK. It does not offer arbitrary six-dimensional end-effector pose control.
- The state BC baseline fails closed-loop evaluation. The report identifies timing/phase mismatch with the scripted demonstrations as a likely contributor.
- SmolVLA ran on the historical gantry but did not complete pick-and-place. The new arm's observation/action adaptation and SmolVLA fine-tuning remain pending.
- Real-hardware transfer has not been validated. URDF/USD export, Isaac Lab, ROS 2/vendor SDK, and ONNX adapters remain pending.
- Auxiliary-model structured import has been checked; a live auxiliary-model API call has not been validated.
- Controller plugins execute trusted local code. Remote-code isolation and resource limits are not implemented.
- Scheduling follows simulation time; real-time wall-clock pacing is not implemented.
- The platform interfaces support configuration-derived joint sets, while some collection/training scripts still target the current 5R dataset.

## Evidence index

| Stage | Scope | Report |
| --- | --- | --- |
| A | Arm collision/IK audit, scripted grasping, and negative controls | [Arm audit](../experiments/20261109-arm-audit/REPORT.md) |
| B | Configurations, timing, sessions, controller ownership, and Gymnasium | [Platform core](../experiments/20261109-platform-core/REPORT.md) |
| C | Scene editor, saved revisions, and applied camera/joint changes | [Frontend](../experiments/20261109-frontend/REPORT.md) |
| D | Dataset read-back, alignment checks, training/reload, and policy evaluation | [Data and training](../experiments/20261110-data-training/REPORT.md) |
| E | Bundle export, second robot, MJCF, and adapter status | [Deployment and export](../experiments/20261110-deploy-export/REPORT.md) |

The reports document implementation coverage and remaining failures. Passing a pipeline check does not establish learned task success or real-hardware transfer.

## Historical gantry results

Historical **30/30** programmatic-cup and **10/10** Lux3D-cup expert results belong to the old gantry. The Lux3D cup came from task **5018305**; the arm from task **5025807**. Historical SmolVLA runs issued real actions but did not complete grasping. See [baseline report](../evidence/baseline-results.json), [comparison video](../evidence/baseline-comparison.mp4), [Lux3D cup results](../evidence/lux3d-physics.json), and [SmolVLA trace](../evidence/lux3d-smolvla.json).
