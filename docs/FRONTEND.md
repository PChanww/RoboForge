# Frontend architecture and operation

**English** | [简体中文](FRONTEND.zh-CN.md)

This document describes the structure, page boundaries, data contracts and bilingual behaviour of
`frontend/`. It is written for anyone changing the interface or debugging what a page shows.
The end-to-end operating manual for the transport task lives in
[the VLA workbench guide](../frontend/VLA_WORKBENCH.md) (Chinese).

## Running it

```powershell
cd frontend
npm ci
npm run dev -- --host 127.0.0.1 --port 5173
```

Vite proxies `/api` and `/assets` to the backend (default `http://127.0.0.1:8767`, override with
`VLAB_API_ORIGIN`). `npm run build` runs `tsc -b` first and only then bundles, so a type error fails
the build instead of shipping; output goes to `frontend/dist/`.

## Stack

React 18 + TypeScript 5 + Vite 6. 3D uses `@react-three/fiber` and `drei` on top of three.js.
State lives in two places: workbench pages use local `useState`, the scene editor uses a `zustand`
store (`src/store.ts`). Charts use `echarts`. Styling is hand-written CSS
(`workspace.css` plus one CSS file per module); Tailwind only supplies utility classes.

## Two shells

The app has two different layouts, chosen by route in `src/App.tsx`:

| Shell | Routes | Notes |
| --- | --- | --- |
| Transport workbench | `/transport` | Stand-alone full-screen app with its own eight-page sidebar; does not use the generic shell |
| Generic workbench | `/projects`, `/datasets`, `/training`, `/evaluations`, `/generation`, `/library`, `/deployments`, `/settings` | Fixed left rail, top bar, content area |

Several stand-alone pages exist outside both shells: `/drive` (mobile demo), `/create` (scenario
authoring), `/studio` (task studio), `/` (collection demo).

The transport workbench is separate because it has its own information architecture: eight pages
ordered describe → prepare → verify → inspect → collect, which is a different mental model from the
resource-management navigation of the generic shell. Both shells carry the language switch.

## The eight transport pages

The active page is kept in `?view=<id>`. Only the current page renders, and switching pages never
creates a new experiment.

| Page | `view` | Content |
| --- | --- | --- |
| Workspace | `home` | Round summary, flow progress, entry points |
| Task | `task` | Describe the action order, check the execution plan, start the run |
| Models | `models` | Choose or generate the object and bin, staged progress, rotating preview, save decision |
| Policy | `policy` | Expert baseline vs VLA selection, connection state, weight identity, adapter limits |
| Simulation | `simulation` | Both cameras, current phase, robot state, inference counters, stop and save |
| Results | `results` | Verdict, per-subtask evidence, failure phase, reach constraint, data checks |
| Datasets | `datasets` | Sample labels, history, NPZ / config / full experiment downloads |
| Settings | `settings` | Bin map, speed, belt speed, seed, size, mass, friction, inference parameters |

### State and persistence

The workbench keeps "the current experiment" in `localStorage`, so a refresh or a page switch never
re-submits:

| Key | Content |
| --- | --- |
| `roboforge.transport.receipt.v1` | `{ client_request_id, spec, run_id }` — the submission identity of the current run |
| `roboforge.transport.draft.v1` | Task draft that has not been submitted |
| `roboforge.transport.reviews.v1` | Per-asset confirmation receipts, so a decision is never asked twice |

`client_request_id` is an idempotency key: resending the same submission cannot create a second run.
Past experiments open from `/transport?view=datasets`, or directly through the deep link
`/transport?run=<run id>&view=<page>`.

### Polling

A live run is polled through `GET /api/v1/transport/runs/{id}`: every 350 ms on the simulation page,
900 ms elsewhere, and polling stops once the run reaches a terminal state. A failed poll never clears
the frames already on screen; it marks the connection offline and retries after 3 seconds.

## Data the UI consumes

The interface only shows fields the backend actually returned. It does not fill in defaults or guess
missing values.

**Run record** (`runs/{id}`): `status`, `spec`, `policy_identity`, `policy_diagnostics`, `plan`,
`episodes[]`, `preview`, `configuration`, `dataset`, `error`.

**Preview snapshot** (`preview`): `tick`, `sim_time`, `images.front/wrist`, `phase`, `phase_label`,
`pose`, `metrics`, `target_xy`, `planned_route`.

**Evaluation report** (`runs/{id}/evaluation`): `verdict` (`code/label/reason/scope`), `summary`,
`episodes[]` (`subtasks`, `phase_timeline`, `failure`, `data_quality`, `reach_evidence`,
`reach_limit`, `metrics`), `data_quality`, `downloads`, `limitations`.

The reachability fields come from the explicit constraint layer: `reach_limited_steps` (how many
steps were projected) and `reach_limit` (step count, first affected frame, largest target
deviation). The UI renders them as "reach constraint N steps · X mm" and states explicitly that
such steps do not change the success verdict.

## Requests and error handling

The transport workbench uses its own `request()` in `src/transport/TransportDemo.tsx` rather than
`src/api.ts`:

- 12 s default timeout, combined from `AbortSignal.timeout` and the caller's signal;
- non-2xx throws `HttpError(status, detail)`, so the UI can tell "invalid input" from "backend down";
- timeouts surface as "backend response timed out", network failures as "cannot reach the local
  transport service"; neither clears data already on screen;
- on 4xx the user's input is preserved and the same action is retried, which avoids duplicate
  submissions.

## Bilingual UI (Chinese / English)

The interface switches between Chinese and English:

- `t(chineseSource, params?)` — the Chinese source string **is** the dictionary key, and the English
  dictionary only maps Chinese → English. Missing English falls back to the Chinese source, so an
  untranslated page never shows a bare key or an empty label.
- Language state comes from `LocaleProvider`, persists in `localStorage`, and keeps
  `document.documentElement.lang` in sync.
- The switch lives in the generic shell top bar and in the transport workbench top bar.
- Strings with values use `t('已保存 {n} 帧', { n })`; values are never concatenated into a dictionary
  entry.

Adding copy: write the Chinese text in the component, wrap it in `t()`, then add one English entry.
Missing English never blocks a build.

## Accessibility and narrow screens

- Controls are real `button` / `a` / `input` elements with `aria-label`; state changes announce
  through `role="status"`.
- At ≤ 570 px the left navigation collapses to an icon rail and data tables scroll inside their own
  container instead of stretching the page.
- Heading levels are fixed: one `h1` per page, cards use `h2`.

## Checks

```powershell
cd frontend
npm run build        # tsc -b + vite build
```

After UI changes, check two widths: desktop (~1400 px) and 390 px. The established approach in this
project is to host the page inside a fixed 390 px iframe to get a real narrow viewport, instead of
resizing the browser window.

## Directory map

```text
src/
  App.tsx                 Routes and the generic shell
  api.ts                  Request wrapper for the generic workbench
  store.ts                zustand store for the scene editor
  transport/              Transport workbench (main application)
    TransportDemo.tsx     Eight-page workbench, polling, downloads
    TransportEvaluations.tsx  Evaluation page
    PolicyGuide.tsx       Policy guidance
    transport.css         Workbench styles
  components/             Generic workbench pages and panels
  studio/ scenario/ demo/ mobile/   Additional stand-alone pages
  workspace.css           Generic shell styles
```

## Known limitations

- `TransportDemo.tsx` is a single component of roughly 90 KB and the pages share most of their state.
  Splitting it per page is the sensible next refactor.
- Camera frames are base64 JPEG returned inside every poll, so larger frames make polling heavier.
- The frontend caches no experiment data; history is re-read from the backend every time.
- The 3D preview needs a WebGL context; without one the page only shows the placeholder.
