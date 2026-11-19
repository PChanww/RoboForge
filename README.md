# RoboForge

**English** | [简体中文](README.zh-CN.md)

A configurable robotics simulation and learning workbench. Generate or import 3D assets, configure robot joints, physics, cameras, and controllers, then collect demonstrations, evaluate policies, train baselines, and export deployment artifacts.

> Generate assets → configure joints and sensors → connect a controller → run a task → collect data / train → evaluate / export

RoboForge currently uses MuJoCo for physics and a React + React Three Fiber editor. Dataset conversion and learning scripts integrate with LeRobot and PyTorch.

The built-in task is tabletop pick-and-place with a two-finger gripper. The five-axis arm and the prismatic gantry use different controller configurations. See the [platform scope](docs/PLATFORM_SCOPE.zh-CN.md) for asset, robot, sensor, and collection requirements.

## Current capabilities

- **Versioned projects.** Persist robot link trees, joints, actuators, cameras, controllers, and tasks with a `schema_version`. Saving creates a configuration revision that is applied when a session is rebuilt.
- **Two robot configurations.** A Lux3D-generated, fixed-base arm with five revolute joints, plus a Cartesian gantry configuration for exercising a different embodiment.
- **Interactive editing.** Select objects in the scene tree, edit joint/link/camera/task properties, save and apply changes, and inspect backend-authoritative body poses.
- **Configurable sensors and timing.** World-mounted and link-mounted cameras with configurable pinhole intrinsics, resolution, frame rate, and delay. Physics, control, camera capture, and policy timing use a simulation-clock scheduler with capture/availability timestamps.
- **Controller and policy interfaces.** `joint_position` and `cartesian_delta_ik` modes where supported, HTTP/WebSocket observation–action loops, and local Python controller plugins. Sessions have separate environments and explicit control ownership.
- **Data and learning scripts.** Expert demonstration collection, LeRobot dataset conversion/read-back, data checks, aligned replay, state-based behavior cloning (BC), checkpoint reload, and closed-loop evaluation.
- **Export.** Configuration/asset/weight bundles with an I/O contract and runner, plus MJCF export. The runner currently requires the project's `vlab` package and Python dependencies.

The browser's inspection camera is separate from the sensor cameras used by a policy. The current UI is primarily Chinese; the documentation is available in both languages.

Experiment results, validation evidence, and implementation limitations are documented in the [audit report](docs/AUDIT.md). To pick up development, start with the [migration handoff](MIGRATION_HANDOFF.zh-CN.md) (honest current status, known defects, usage, git info).

## Quick start

Run these commands from the project directory, `vla-lab`. The prepared host uses Windows/PowerShell and Python 3.13 for simulation. The frontend requires Node.js/npm. Camera rendering requires a working OpenGL context.

### Use the prepared workspace

Start the backend in one terminal:

```powershell
..\..\work\venv\Scripts\python.exe -m uvicorn vlab.api:app --host 127.0.0.1 --port 8767
```

Start the frontend in another terminal:

```powershell
cd frontend
npm ci
npm run dev
```

Open [http://localhost:5173](http://localhost:5173). Vite proxies `/api` and `/assets` to the backend on port 8767.

Select a joint or camera in the scene tree, change its properties, and click **保存并应用** (Save and apply). Click **运行专家** (Run expert) to run the scripted controller.

### Set up a new simulation environment

From `vla-lab`, create and activate a Python 3.13 environment:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn vlab.api:app --host 127.0.0.1 --port 8767
```

Then start the frontend as above. If you change the backend port, update `frontend/vite.config.ts`.

The core requirements cover simulation, the API, and API tests. LeRobot/PyTorch training, asset generation, and MP4 replay have additional requirements described below.

## Project layout

```text
vlab/                Configuration-driven engine, API, sessions, scheduler, Gymnasium wrapper
frontend/            React, TypeScript, Vite, React Three Fiber editor
configs/             Default project and second-robot configuration
projects/            Persisted project revisions
assets/              Robot meshes, generated objects, and provenance
asset_pipeline/      Aholo generation, GLB/ZIP import, and arm mesh binding
plugins/             Example local controller plugin
scripts/             Collection, conversion, training, evaluation, and export tools
tests/               Engine and API checks
datasets/            Local staging data, model weights, replay, and reports
deploy/              Generated deployment bundles and MJCF
experiments/         Dated experiment reports and validation evidence
integrations/        Historical SmolVLA client and runtime instructions
sim_env.py, arm.py    Arm/task implementation used by the original prototype
server.py            Legacy HTTP server, default port 8765
index.html           Legacy frontend
evidence/            Historical gantry experiment artifacts
```

Generated datasets and deployment bundles may be absent from a fresh checkout. See `.gitignore` and regenerate them with the scripts when needed.

## Platform API

The current backend runs on port **8767**. Interactive endpoint schemas are available at [http://127.0.0.1:8767/docs](http://127.0.0.1:8767/docs) after startup.

See the [model integration and asset binding guide (Chinese)](docs/MODEL_INTEGRATION.zh-CN.md) for joint and camera mappings, action contracts, and a minimal HTTP client.

| Method and path | Purpose |
| --- | --- |
| `GET /api/v1/capabilities` | Engine version, robot capabilities, and supported controller/sensor/task descriptions |
| `GET /api/v1/config` | Read the current project configuration |
| `POST /api/v1/config/validate` | Validate schema and physical construction |
| `PUT /api/v1/config` | Validate and save a configuration revision |
| `GET /api/v1/config/versions` | List immutable revisions |
| `GET /api/v1/config/versions/{version}` | Read a revision |
| `POST /api/v1/sessions` | Create an isolated environment session |
| `POST /api/v1/sessions/{sid}/reset` | Reset a session |
| `POST /api/v1/sessions/{sid}/control/acquire` | Acquire a control token |
| `POST /api/v1/sessions/{sid}/control/release` | Release control |
| `POST /api/v1/sessions/{sid}/action` | Execute one action using the token and current tick |
| `POST /api/v1/sessions/{sid}/action_block` | Execute an action block |
| `GET /api/v1/sessions/{sid}/observation` | Read named joint state, camera frames, and timing information |
| `GET /api/v1/sessions/{sid}/result` | Read observations and task/termination metrics |
| `GET /api/v1/sessions/{sid}/frame.jpg` | Read a camera frame |
| `DELETE /api/v1/sessions/{sid}` | Close a session |
| `WS /api/v1/ws/{sid}` | Control acquisition, actions, and observations over WebSocket |

Stale actions and conflicting control ownership return HTTP 409. Invalid action dimensions or out-of-range values return HTTP 400. Supported action modes depend on the robot configuration.

The Gymnasium interface is `VLabGymEnv` in `vlab/gym_env.py`, with separate `terminated` and `truncated` flags.

## Collect, train, and evaluate

Run simulation-side commands from `vla-lab` with the simulation Python environment:

```powershell
python scripts/collect_expert.py
python scripts/replay_episode.py --episode 19
```

Collection writes the predefined 21-episode plan to `datasets/staging`. Replay requires `ffmpeg` on PATH and writes an MP4 under `datasets/replay`.

Use a separate learning environment for conversion and training. The recorded WSL setup used Python 3.11, LeRobot 0.4.4, PyTorch 2.6.0+cu124, and Transformers 4.57.3. These packages are separate from the core `requirements.txt`; the scripts also use pandas and a Parquet engine such as pyarrow.

Run the following from the project directory in that environment:

```bash
python scripts/lerobot_writer.py --staging datasets/staging --out datasets/lerobot/expert21
python scripts/data_checks.py --lab .
python scripts/train_bc.py --lab .
python scripts/eval_policy.py --group all
```

`eval_policy.py` can also run in the simulation environment using the exported NumPy weights. Optional privileged-state diagnostics are available:

```bash
python scripts/train_priv.py --lab .
python scripts/eval_policy.py --priv --group heldout_spawn
```

`train_bc.py` uses episodes 0–14 for training, 15–18 for held-out spawn validation, and 19–20 for held-out asset evaluation. It saves PyTorch checkpoints and NumPy weights under `datasets/models`. Existing `datasets/dagger/*.npz` files are included when present; record those inputs when comparing runs.

These commands generate or update local data, checkpoints, and reports. The current BC model is a state-based MLP. SmolVLA inference instructions and the historical projection limitations are documented in [integrations/README.md](integrations/README.md).

## Export and extension points

```powershell
python scripts/export_deploy.py
python scripts/export_mjcf.py
python scripts/test_second_robot.py
python scripts/test_aux_plugin.py
```

Deployment export expects existing BC weights and training/evaluation reports. It writes `deploy/lux3d_5r-v4`, including configuration, assets, weights, an I/O manifest, and `run.py`, then checks expert and policy execution in fresh processes.

For a manual bundle run from `vla-lab`, use a simulation Python environment and make the project package available:

```powershell
$env:VLAB_LAB = (Get-Location).Path
python deploy/lux3d_5r-v4/run.py --controller expert --ticks 200 --report deploy/lux3d_5r-v4/manual-expert.json
```

Moving the bundle to another machine also requires the `vlab` package and compatible Python dependencies. MJCF export references mesh files by name; copy the required meshes alongside the exported model.

- **Assets.** See [asset generation and import](asset_pipeline/README.md). Existing Lux3D provenance is stored in `assets/robot/source.json` and `assets/imported/lux3d-ivory/manifest.json`. API generation uses the optional Aholo SDK and `AHOLO_API_KEY`.
- **Local controllers.** A plugin exposes `create_controller(config)` and implements `reset(obs)` and `act(obs)`. See `vlab/plugins.py` and `plugins/example_openloop.py`.
- **Auxiliary model.** `vlab/aux_model.py` reads `AUX_MODEL_BASE_URL` and `AUX_MODEL_API_KEY`. Without them it reports `waiting_for_configuration`; structured candidate validation is available. Model selection is configured through the adapter.

Keep credentials in the local environment and out of source files, browser code, logs, and screenshots.

## Development checks

```powershell
python tests/test_engine.py
python tests/test_api.py
cd frontend
npm run build
```

## Scenes (environment + collision)

Project scenes bind an environment revision with an instance transform and a default table or explicit static collision proxies. Calibrated triangle meshes enter MuJoCo RGB observations. The tray and manipulable objects have separate collision geometry. Import local `.glb` or `.ply` and inspect the detected mesh, point-cloud, or Gaussian type before selecting how to bind it.

- API: `POST /api/v1/scene-assets/import`, `GET /api/v1/scene-assets[/{id}]`, `POST /api/v1/scene-assets/{id}/validate`, `POST /api/v1/projects/{id}/bind-scene`, `POST/GET /api/v1/generation-jobs`.
- UI: Asset center → import scene; `/scenes/:id/edit` → validate / bind to project.
- Optional World SDK (`python -m pip install -r requirements-world.txt`). Set `AHOLO_API_KEY` in the backend environment. Successful downloads are registered as scene revisions. Gaussian backgrounds support fixed-front hybrid RGB compositing with separate physics proxies. Use MuJoCo RGB for the dual-camera collection path.
- Scene imports apply `unit_scale` to mesh coordinates. Physical use requires confirmed units and either the default table or explicit collision proxies. Project routes load their own configuration and session.
- Visual collect/train/eval jobs accept selected project/dataset/model IDs. Collection and evaluation save configuration snapshots; conversion refuses to overwrite datasets. New visual checkpoints include their input/action contract; legacy checkpoints require explicit CLI opt-in.
- Docs: [scene pipeline](docs/SCENE_PIPELINE.zh-CN.md) · [validation record](docs/SCENE_VALIDATION.zh-CN.md).

## Conversational generation workflow

A connected loop turns a natural-language request into a trainable scene:
chat (optional) or direct form/JSON → versioned `AssetSpec`/`PhysicsSpec` with per-field provenance →
deterministic compile preview → explicit confirm → Lux3D single-asset / World background generation
(idempotent, node-level recovery) → import validation → `SceneRecipe` assembly → real-MuJoCo validation →
atomic apply to a new project version → simulation / collection.

- The LLM only understands and suggests; programs validate, orchestrate, and land. Estimated mass/friction/scale
  stay `model_inferred/unverified` and never become "measured". Unknown dimensions yield an uncalibrated visual
  asset with no physics binding until processed.
- Chat requires an auxiliary model endpoint and credential. It shows `waiting_for_configuration` until configured. Form, JSON, and local import remain available, and the backend handles provider credentials.
- Two scene modes: `composed_scene` (parameterized workbench + explicit static proxies + manipulable object; works
  without World/calibration) and `background_scene` (World appearance + explicit proxies; gaussian requires the
  restricted fixed-front `hybrid` renderer). Remote generation success ≠ imported ≠ trainable.
- API under `/api/v1/generation/*` and `/api/v1/scene-recipes/*`; UI: generation workbench, task detail, and a
  RecipePanel that validates then atomically applies to a project version with read-back.
- Collection manifests record task-asset mesh/manifest hashes, generation source, recipe/spec revision,
  physics basis, cameras, and `renderer_version`.

See [generation workflow](docs/GENERATION_WORKFLOW.zh-CN.md) and the
[round-4 test/capability report](docs/ROUND4_GENERATION_REPORT.zh-CN.md).

## Legacy prototype

`server.py` defaults to port **8765**. The platform backend on port 8767 also provides legacy `/api/*` compatibility routes.

| Route | Legacy contract |
| --- | --- |
| `GET /api/assets` | Available cup assets |
| `POST /api/reset` | `asset` and `seed`; returns a new episode |
| `GET /api/observation?images=1` | End-effector XYZ, gripper opening, instruction, and two Base64-encoded 256×256 JPEGs |
| `POST /api/step` | `episode`, `tick`, and `action: [dx, dy, dz, grip]` |
| `GET /api/result` | Contact, lift, placement, and task metrics |

All four legacy action values are in `[-1, 1]`. Translation increments are at most 0.012 m per step, with workspace clipping. Grip −1 closes and +1 opens. Each step advances 0.05 s of simulation time. A stale episode/tick returns HTTP 409.

`python evaluate.py --no-video` invokes the legacy evaluator against the current `sim_env.py`; its results depend on the checked-out implementation and should be recorded as a new run.
