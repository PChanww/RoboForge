# 前端架构与操作说明

[English](FRONTEND.md) | **简体中文**

本文件描述 `frontend/` 的技术结构、页面边界、数据契约和双语机制。面向需要修改界面或排查页面行为的人。
搬运任务的端到端操作手册见 [VLA 仿真验证工作台](../frontend/VLA_WORKBENCH.md)。

## 运行

```powershell
cd frontend
npm ci
npm run dev -- --host 127.0.0.1 --port 5173
```

Vite 把 `/api` 与 `/assets` 代理到后端（默认 `http://127.0.0.1:8767`，可用环境变量 `VLAB_API_ORIGIN` 覆盖）。
生产构建 `npm run build` 先跑 `tsc -b` 再做类型检查后的打包，输出到 `frontend/dist/`。

## 技术栈

React 18 + TypeScript 5 + Vite 6。三维部分用 `@react-three/fiber` 与 `drei`（three.js）。
状态管理分两层：工作台页面用本地 `useState`，场景编辑器用 `zustand`（`src/store.ts`）。
图表用 `echarts`，样式以手写 CSS 为主（`workspace.css` 与各模块 CSS），Tailwind 仅提供工具类。

## 两套外壳

应用有两种互不相同的布局，由 `src/App.tsx` 的路由直接决定：

| 外壳 | 路由 | 说明 |
| --- | --- | --- |
| 搬运工作台 | `/transport` | 独立全屏应用，自带左侧八页导航，不使用通用外壳 |
| 通用工作台 | `/projects`、`/datasets`、`/training`、`/evaluations`、`/generation`、`/library`、`/deployments`、`/settings` | 左侧固定导航 + 顶栏 + 内容区 |

另有若干独立单页：`/drive`（移动端演示）、`/create`（场景创作）、`/studio`（任务工作台）、`/`（采集演示）。

搬运工作台单独成页的原因是它有自己的信息架构：八个页面按「描述 → 备料 → 验证 → 看结果 → 取数据」排列，
和通用工作台的资源管理导航不是同一套心智模型。两套外壳都带语言切换。

## 搬运工作台的八个页面

页面通过 `?view=<id>` 保留，只渲染当前页，切页不会创建新实验。

| 页面 | `view` | 主要内容 |
| --- | --- | --- |
| 工作空间 | `home` | 本轮摘要、流程进度与各模块入口 |
| 任务要求 | `task` | 描述动作顺序、检查执行方案、启动实验 |
| 模型资产 | `models` | 选择或生成本轮物品与箱体，分阶段进度、旋转预览与保存决策 |
| 执行策略 | `policy` | 专家基线与 VLA 策略选择、连接状态、权重身份与适配限制 |
| 仿真执行 | `simulation` | 双相机画面、当前阶段、机器人状态、推理统计与停止保存 |
| 可行性结果 | `results` | 本轮结论、逐个子任务证据、失败阶段、可达性约束与数据校验 |
| 数据集 | `datasets` | 样本标签、历史实验、NPZ／配置／完整实验包下载 |
| 参数设置 | `settings` | 箱位地图、速度、带速、种子、尺寸、质量、摩擦与推理参数 |

### 状态与持久化

工作台把「当前实验」记在 `localStorage`，因此刷新或切页不会重新提交：

| 键 | 内容 |
| --- | --- |
| `roboforge.transport.receipt.v1` | `{ client_request_id, spec, run_id }`，当前实验的提交凭据 |
| `roboforge.transport.draft.v1` | 尚未提交的任务草稿 |
| `roboforge.transport.reviews.v1` | 每个资产的确认回执，避免重复确认 |

`client_request_id` 是幂等键：同一次提交重复发送不会产生第二条实验。历史实验可以从
`/transport?view=datasets` 打开，或用深链 `/transport?run=<实验 ID>&view=<页面>` 直接进入。

### 轮询

运行中的实验由 `useEffect` 轮询 `GET /api/v1/transport/runs/{id}`：
仿真执行页 350 ms，其它页面 900 ms，实验进入终态后停止轮询。
轮询失败不清空已有画面，只把连接状态标为离线并在 3 秒后重试。

## 前端消费的数据契约

界面只显示后端实际返回的字段，缺字段时不补默认值、不推测。

**运行记录**（`runs/{id}`）：`status`、`spec`、`policy_identity`、`policy_diagnostics`、
`plan`、`episodes[]`、`preview`、`configuration`、`dataset`、`error`。

**预览快照**（`preview`）：`tick`、`sim_time`、`images.front/wrist`、`phase`、`phase_label`、
`pose`、`metrics`、`target_xy`、`planned_route`。

**评价报告**（`runs/{id}/evaluation`）：`verdict`（`code/label/reason/scope`）、`summary`、
`episodes[]`（`subtasks`、`phase_timeline`、`failure`、`data_quality`、`reach_evidence`、`reach_limit`、`metrics`）、
`data_quality`、`downloads`、`limitations`。

其中可达性字段来自显式约束层：`reach_limited_steps`（被投影的步数）与
`reach_limit`（步数、首次触发帧、最大目标偏差）。界面把它显示为「可达性约束 N 步 · X mm」，
并明确说明这类步骤不改变成功判定。

## 请求与错误处理

搬运工作台用自己的 `request()`（`src/transport/TransportDemo.tsx`），不经过 `src/api.ts`：

- 默认 12 秒超时，可用 `AbortSignal.timeout` 与调用方 `signal` 合并；
- 非 2xx 抛 `HttpError(status, detail)`，界面据此区分「参数不合法」与「后端不可用」；
- 超时转成「后端响应超时」，网络错误转成「无法连接本地搬运服务」，两类都不清空已有数据；
- 4xx 时保留用户输入，提示用同一操作重试，避免重复提交。

## 双语（中／英）

界面支持中文与英文切换，实现约定如下：

- 文案函数 `t(中文原文, 参数?)`：中文原文即词条键，英文词典只维护「中文 → 英文」映射。
  找不到英文时回退中文原文，因此未翻译的页面不会显示 key，也不会显示空白。
- 语言状态由 `LocaleProvider` 提供，保存在 `localStorage`，并同步 `document.documentElement.lang`。
- 切换入口在通用外壳顶栏与搬运工作台顶栏。
- 带变量的文案用 `t('已保存 {n} 帧', { n })` 形式，不在词条里拼接字符串。

新增文案的流程：先在界面里写中文，再用 `t()` 包起来，最后在英文词典补一条；缺英文不阻塞构建。

## 可访问性与窄屏

- 交互控件使用真实 `button`／`a`／`input`，带 `aria-label`；状态变化用 `role="status"` 播报。
- 窄屏（≤ 570 px）下左侧导航收成图标列，数据表在自身容器内横向滚动，不撑破页面。
- 页面标题层级固定：每个页面一个 `h1`，卡片用 `h2`。

## 自检

```powershell
cd frontend
npm run build        # tsc -b + vite build
```

改动界面后至少检查两种宽度：桌面（约 1400 px）与 390 px。项目里的历史做法是用 iframe 固定宽 390 px
承载页面，得到真实窄屏视口，而不用缩放窗口模拟。

## 目录结构

```text
src/
  App.tsx                 路由与通用外壳
  api.ts                  通用工作台的请求封装
  store.ts                场景编辑器的 zustand store
  transport/              搬运工作台（主应用）
    TransportDemo.tsx     八页工作台、轮询、下载
    TransportEvaluations.tsx  评测页
    PolicyGuide.tsx       策略说明
    transport.css         工作台样式
  components/             通用工作台的页面与面板
  studio/ scenario/ demo/ mobile/   其它独立页面
  workspace.css           通用外壳样式
```

## 已知限制

- 搬运工作台是单文件组件（约 90 KB），页面之间共享大量状态；继续扩展前建议按页面拆分。
- 相机画面是 JPEG base64，整帧随轮询返回；画面越大轮询越重。
- 前端不缓存实验数据，历史实验每次都从后端读取。
- 三维预览依赖 WebGL，无 GPU 上下文的环境只能看到占位画面。
