/**
 * Closed-loop evaluation page.
 *
 * Copy rule: module-level maps keep the Chinese labels they map to; `t()` is
 * applied where the label is rendered, so switching language re-renders instead
 * of freezing the first locale at import time.
 */
import { useEffect, useMemo, useRef, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { useI18n } from '../i18n'
import './transport-evaluations.css'

type PolicyIdentity = {
  id?: string
  label?: string
  kind?: string
  learned_policy?: boolean
  privileged?: boolean
  training_job_id?: string | null
  requested_policy_id?: string | null
  device?: string | null
  device_name?: string | null
  cuda_version?: string | null
  adapter?: string | null
}
type RunSummary = { episodes: number; successes: number; frames: number }
type HistoryRun = {
  id: string
  status: string
  task_kind: string
  created: string | null
  instruction?: string
  experiment_name?: string | null
  policy_identity: PolicyIdentity
  summary?: RunSummary
}
type Quality = { valid: boolean; issues: string[] }
type Phase = { phase: number; label: string; start_s: number; end_s: number; frames: number }
type ReachLimit = { steps: number; first_frame: number; max_deviation_m: number }
type Episode = {
  index: number
  seed: number | null
  success: boolean
  frames: number
  duration_s: number
  metrics?: Record<string, unknown>
  reach_evidence?: string
  reach_limit?: ReachLimit | null
  phase_timeline: Phase[]
  subtasks: { stage: string; label: string; completed: boolean; evidence: { time_s?: number } | null }[]
  failure: { stage: string; reason: string; last_bilateral_contact_s: number | null } | null
}
type EvaluationReport = {
  run_id: string
  verdict: { code: string; label: string; reason: string }
  controller: PolicyIdentity
  action_contract: { dim: number }
  policy_diagnostics: { inference_count?: number; inference_s?: number; executed_model_actions?: number }
  task_kind: string
  experiment_name?: string | null
  summary: RunSummary
  episodes: Episode[]
  data_quality: Quality
  downloads: { report_url: string }
}

const TASK_LABEL: Record<string, string> = {
  deliver: '送入箱子', pick_return: '拿起放回', bin_retrieve: '入箱再取出', bin_return: '入箱取出再返回',
}
const STATUS_LABEL: Record<string, string> = {
  succeeded: '通过', failed: '未通过', running: '运行中', queued: '排队中', generating: '生成资产',
  awaiting_asset_review: '待确认资产', stopping: '正在停止', cancelled: '已停止', interrupted: '已中断',
}
const VERDICT_TONE: Record<string, string> = {
  passed: 'good', failed: 'bad', invalid_data: 'bad', pending: 'pending', incomplete: 'neutral',
}
const TERMINAL = ['succeeded', 'failed', 'cancelled', 'interrupted']

function useFormat() {
  const { locale } = useI18n()
  return useMemo(() => {
    const tag = locale === 'en' ? 'en-US' : 'zh-CN'
    return {
      count: new Intl.NumberFormat(tag),
      date: new Intl.DateTimeFormat(tag, {
        month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', timeZone: 'Asia/Shanghai',
      }),
    }
  }, [locale])
}

function formatDate(value: string | null, format: Intl.DateTimeFormat) {
  const timestamp = value ? Date.parse(value) : NaN
  return Number.isFinite(timestamp) ? format.format(timestamp) : ''
}

function statusTone(status: string) {
  if (status === 'succeeded') return 'good'
  if (status === 'failed') return 'bad'
  return TERMINAL.includes(status) ? 'neutral' : 'pending'
}

function matchesStatus(status: string, filter: string) {
  if (filter === 'active') return !TERMINAL.includes(status)
  if (filter === 'incomplete') return status === 'cancelled' || status === 'interrupted'
  return filter === 'all' || status === filter
}

async function evaluationRequest<T>(path: string, signal: AbortSignal): Promise<T> {
  const response = await fetch(`/api/v1/transport/${path}`, { signal })
  if (response.ok) return response.json() as Promise<T>
  const body = await response.json().catch(() => null)
  throw new Error(typeof body?.detail === 'string' ? body.detail : `HTTP ${response.status}`)
}

type Result<T> = { path: string; data: T; error?: never } | { path: string; error: string; data?: never }

function useEvaluationRequest<T>(path: string | null, revision: number) {
  const [result, setResult] = useState<Result<T> | null>(null)
  useEffect(() => {
    if (!path) return
    const controller = new AbortController()
    setResult(null)
    evaluationRequest<T>(path, controller.signal).then(data => {
      if (!controller.signal.aborted) setResult({ path, data })
    }).catch((error: unknown) => {
      if (!controller.signal.aborted) setResult({ path, error: error instanceof Error ? error.message : 'network' })
    })
    return () => controller.abort()
  }, [path, revision])
  return result?.path === path ? result : null
}

function PhaseTimeline({ phases }: { phases: Phase[] }) {
  const { t } = useI18n()
  const { count } = useFormat()
  const end = phases[phases.length - 1].end_s
  return <details className="te-timeline" open>
    <summary>{t('仿真阶段')} <span>{t('{n} 段', { n: phases.length })}</span></summary>
    <div className="te-time-rail" aria-hidden="true">
      {phases.map((phase, index) => <span key={`${phase.phase}-${phase.start_s}`}
        data-color={index % 3} style={{ flexGrow: Math.max(0.001, phase.end_s - phase.start_s) }} />)}
    </div>
    <div className="te-time-axis" aria-hidden="true"><span>{phases[0].start_s.toFixed(2)} s</span><span>{end.toFixed(2)} s</span></div>
    <ol className="te-time-list">
      {phases.map((phase, index) => <li key={`${phase.phase}-${phase.start_s}`}>
        <span className="te-time-label"><i aria-hidden="true" data-color={index % 3} />{t(phase.label)}</span>
        <span className="te-number">{phase.start_s.toFixed(2)}–{phase.end_s.toFixed(2)} s</span>
        <span>{t('{n} 帧', { n: count.format(phase.frames) })}</span>
      </li>)}
    </ol>
  </details>
}

function ReportDetail({ report, run }: { report: EvaluationReport; run?: HistoryRun }) {
  const { t } = useI18n()
  const format = useFormat()
  const [episodeIndex, setEpisodeIndex] = useState(0)
  const { verdict, controller, summary, policy_diagnostics: diagnostics } = report
  const episode = report.episodes.find(item => item.index === episodeIndex) ?? report.episodes[0]
  const duration = report.episodes.reduce((total, item) => total + (Number.isFinite(item.duration_s) ? item.duration_s : 0), 0)
  const checkpoint = controller.training_job_id
  const actionCount = controller.learned_policy ? diagnostics.executed_model_actions : report.action_contract.dim
  const hardware = [controller.device, controller.device_name, controller.cuda_version ? `CUDA ${controller.cuda_version}` : ''].filter(Boolean).join(' · ')
  const reason = (value: string) => {
    const reach = /^Unreachable: target distance ([\d.]+) m exceeds reach ([\d.]+) m$/.exec(value)
    return reach ? t('目标距离 {d} m 超过机械臂可达范围 {r} m。', { d: reach[1], r: reach[2] }) : value
  }

  return <>
    <section className="te-report" aria-labelledby="te-report-title">
      <header className="te-verdict" data-tone={VERDICT_TONE[verdict.code] ?? 'neutral'}>
        <div className="te-verdict-meta">
          <span className="te-muted">{report.experiment_name || run?.experiment_name || t(TASK_LABEL[report.task_kind] ?? report.task_kind)}</span>
          {run?.created && <time dateTime={run.created}>{formatDate(run.created, format.date)}</time>}
        </div>
        <h2 id="te-report-title"><span className="te-verdict-mark" aria-hidden="true" />{t(verdict.label)}</h2>
        <p>{reason(verdict.reason)}</p>
      </header>
      <dl className="te-metrics">
        <div><dt>{t('通过回合')}</dt><dd>{summary.episodes ? <>{summary.successes}<small>/{summary.episodes}</small></> : '—'}</dd></div>
        <div><dt>{t('采集帧数')}</dt><dd>{format.count.format(summary.frames)}</dd></div>
        <div><dt>{t('仿真时长')}</dt><dd>{report.episodes.length ? <>{duration.toFixed(2)}<small> s</small></> : '—'}</dd></div>
        <div><dt>{controller.learned_policy ? t('已执行模型动作') : t('动作维度')}</dt><dd>{typeof actionCount === 'number' ? format.count.format(actionCount) : '—'}</dd></div>
      </dl>
      <div className="te-policy">
        <div>
          <div className="te-meta">{t('执行策略')}</div>
          <strong>{controller.label || controller.id || t('未记录策略')}</strong>
          {hardware && <p>{hardware}</p>}
          {typeof diagnostics.inference_count === 'number' && diagnostics.inference_count > 0 && <p>
            {t('{n} 次推理', { n: format.count.format(diagnostics.inference_count) })}{typeof diagnostics.inference_s === 'number' ? t(' · 累计 {s} s', { s: diagnostics.inference_s.toFixed(2) }) : ''}
          </p>}
        </div>
        <div className="te-policy-links">
          {checkpoint && <Link to={`/training/${encodeURIComponent(checkpoint)}`}>{t('训练记录')} <span aria-hidden="true">↗</span></Link>}
          <code title={report.run_id}>{report.run_id}</code>
        </div>
      </div>
    </section>

    <section className="te-card te-episode" aria-labelledby="te-episodes-title">
      <div className="te-section-title">
        <h3 id="te-episodes-title">{t('回合证据')}</h3>
        {report.episodes.length > 1 && <label className="te-episode-select">
          <span className="sr-only">{t('选择回合')}</span>
          <select value={episode?.index ?? ''} onChange={event => setEpisodeIndex(Number(event.target.value))}>
            {report.episodes.map(item => <option key={item.index} value={item.index}>
              {t('第 {n} 回合 · {result}', { n: item.index + 1, result: item.success ? t('通过') : t('未通过') })}
            </option>)}
          </select>
        </label>}
      </div>
      {episode ? <>
        <div className="te-episode-meta">
          <span className="te-badge" data-tone={episode.success ? 'good' : 'bad'}>{episode.success ? t('通过') : t('未通过')}</span>
          <span>{t('{n} 帧', { n: format.count.format(episode.frames) })}</span>
          {episode.seed != null && <span>{t('种子 {n}', { n: episode.seed })}</span>}
          {(() => {
            const steps = episode.metrics?.reach_limited_steps
            const count = typeof steps === 'number' ? steps : episode.reach_limit?.steps
            if (typeof count !== 'number' || count <= 0) return null
            const max = episode.reach_limit?.max_deviation_m
            const suffix = typeof max === 'number' ? t(' · {mm} mm', { mm: Math.round(max * 1000) }) : ''
            return <span className="te-badge" data-tone="warn" title={t('机械臂目标超出可达范围后被投影到可执行位置')}>
              {t('可达性约束 {n} 步', { n: count })}{suffix}
            </span>
          })()}
        </div>
        {episode.failure && <div className="te-failure">
          <strong>{t('最后记录阶段 · {stage}', { stage: t(episode.failure.stage) })}</strong>
          {episode.failure.reason !== verdict.reason && <p>{reason(episode.failure.reason)}</p>}
          {episode.failure.last_bilateral_contact_s != null && <p>{t('最后一次双侧接触 {s} s', { s: episode.failure.last_bilateral_contact_s.toFixed(2) })}</p>}
        </div>}
        {episode.subtasks.length > 0 && <ol className="te-subtasks" aria-label={t('任务步骤')}>
          {episode.subtasks.map(step => <li key={step.stage} data-completed={step.completed}>
            <span aria-hidden="true">{step.completed ? '✓' : '○'}</span><strong>{t(step.label)}</strong>
            <small>{step.completed ? t('完成') : t('未完成')}{typeof step.evidence?.time_s === 'number' ? ` · ${step.evidence.time_s.toFixed(2)} s` : ''}</small>
          </li>)}
        </ol>}
        {episode.phase_timeline.length > 0 && <PhaseTimeline phases={episode.phase_timeline} />}
      </> : <p className="te-empty">{t('尚无已记录的回合。')}</p>}
    </section>

    <section className="te-card te-quality" aria-labelledby="te-quality-title">
      <div className="te-section-title">
        <h3 id="te-quality-title">{t('数据校验')}</h3>
        {summary.episodes > 0 || !report.data_quality.valid
          ? <span className="te-badge" data-tone={report.data_quality.valid ? 'good' : 'bad'}>{report.data_quality.valid ? t('校验通过') : t('发现问题')}</span>
          : <span className="te-badge" data-tone="neutral">{t('暂无回合')}</span>}
      </div>
      {report.data_quality.issues.length > 0 && <ul>{report.data_quality.issues.map((issue, index) => <li key={`${issue}-${index}`}>{t(issue)}</li>)}</ul>}
      <div className="te-quality-footer">
        <p>{t('评测结果适用于本次记录的场景、配置与随机种子。')}</p>
        <a href={report.downloads.report_url} download>{t('报告 JSON')} <span aria-hidden="true">↓</span></a>
      </div>
    </section>
  </>
}

export default function TransportEvaluations() {
  const { t } = useI18n()
  const format = useFormat()
  const [params, setParams] = useSearchParams()
  const checkpoint = params.get('checkpoint')?.trim() || ''
  const requestedRun = params.get('run')?.trim() || ''
  const [search, setSearch] = useState('')
  const [status, setStatus] = useState('all')
  const [source, setSource] = useState('all')
  const [revision, setRevision] = useState(0)
  const detail = useRef<HTMLDivElement>(null)
  const history = useEvaluationRequest<{ runs: HistoryRun[] }>('history', revision)
  const runs = history?.data?.runs ?? []
  const query = search.trim().toLowerCase()
  const policyName = (policy: PolicyIdentity) => policy.label || policy.id || t('未记录策略')
  const filtered = runs.filter(run => {
    const policy = run.policy_identity
    if (checkpoint && policy.training_job_id !== checkpoint && policy.requested_policy_id !== `mobile-${checkpoint}`) return false
    if (source === 'learned' && !policy.learned_policy && policy.kind !== 'unverified_vla') return false
    if (source === 'expert' && policy.kind !== 'privileged_expert') return false
    if (!matchesStatus(run.status, status)) return false
    return !query || [run.experiment_name, run.id, policyName(policy), run.instruction, TASK_LABEL[run.task_kind] ?? run.task_kind, policy.training_job_id]
      .some(value => value?.toLowerCase().includes(query))
  })
  const hasFilters = Boolean(query || status !== 'all' || source !== 'all')
  const showExplicitRun = requestedRun && (!hasFilters || filtered.some(run => run.id === requestedRun))
  const selected = (showExplicitRun ? requestedRun : '') || filtered[0]?.id
  const selectedMeta = runs.find(run => run.id === selected)
  const report = useEvaluationRequest<EvaluationReport>(selected ? `runs/${encodeURIComponent(selected)}/evaluation` : null, revision)

  function selectRun(id: string) {
    const next = new URLSearchParams(params)
    next.set('run', id)
    setParams(next)
    if (window.matchMedia('(max-width: 780px)').matches) detail.current?.scrollIntoView({ block: 'start' })
  }

  function clearCheckpoint() {
    const next = new URLSearchParams(params)
    next.delete('checkpoint')
    setParams(next, { replace: true })
  }

  const startRun = checkpoint
    ? `/transport?view=policy&checkpoint=${encodeURIComponent(`mobile-${checkpoint}`)}`
    : '/transport?view=task'

  return <div className="te-workbench">
    <div className="te-top">
      <div>
        <h2>{t('闭环评测')}</h2>
        <p>{t('查看移动搬运的任务判定、模型动作和采集数据。')}</p>
      </div>
      <div className="te-actions">
        <button type="button" className="te-button" onClick={() => setRevision(value => value + 1)}>{t('刷新记录')}</button>
        <Link className="te-button te-primary" to={startRun}>{checkpoint ? t('使用检查点仿真') : t('前往搬运仿真')} <span aria-hidden="true">↗</span></Link>
      </div>
    </div>

    {checkpoint && <div className="te-checkpoint">
      <span>{t('检查点筛选')} <code>{checkpoint}</code></span>
      <Link to={`/training/${encodeURIComponent(checkpoint)}`}>{t('训练记录')}</Link>
      <button type="button" onClick={clearCheckpoint}>{t('清除筛选')}</button>
    </div>}

    <div className="te-layout">
      <section className="te-history" aria-labelledby="te-history-title">
        <div className="te-section-title"><h3 id="te-history-title">{t('实验记录')}</h3>{history?.data && <span>{t('{n} 条', { n: filtered.length })}</span>}</div>
        <div className="te-filters">
          <label className="te-search"><span className="sr-only">{t('搜索实验记录')}</span>
            <input type="search" value={search} onChange={event => setSearch(event.target.value)} placeholder={t('搜索任务、策略或实验 ID')} />
          </label>
          <label><span className="sr-only">{t('运行状态')}</span><select value={status} onChange={event => setStatus(event.target.value)}>
            <option value="all">{t('全部状态')}</option><option value="succeeded">{t('通过')}</option><option value="failed">{t('未通过')}</option>
            <option value="active">{t('进行中')}</option><option value="incomplete">{t('停止或中断')}</option>
          </select></label>
          <label><span className="sr-only">{t('策略类型')}</span><select value={source} onChange={event => setSource(event.target.value)}>
            <option value="all">{t('全部策略')}</option><option value="learned">{t('学习策略')}</option><option value="expert">{t('专家基线')}</option>
          </select></label>
        </div>
        {!history && <p className="te-empty" role="status">{t('正在读取实验记录…')}</p>}
        {history?.error && <p className="te-empty te-error" role="alert">{t('无法读取实验记录。')}{history.error}</p>}
        {history?.data && (filtered.length > 0 ? <ul className="te-run-list">
          {filtered.map(run => <li key={run.id}>
            <button type="button" className="te-run" aria-current={selected === run.id ? 'true' : undefined} onClick={() => selectRun(run.id)}>
              <div className="te-run-top"><strong>{run.experiment_name || policyName(run.policy_identity)}</strong>
                <span className="te-badge" data-tone={statusTone(run.status)}>{t(STATUS_LABEL[run.status] ?? run.status)}</span></div>
              <div className="te-run-middle"><span>{t(TASK_LABEL[run.task_kind] ?? run.task_kind)}{run.experiment_name ? ` · ${policyName(run.policy_identity)}` : ''}</span>
                {run.summary && run.summary.episodes > 0 && <span>{t('{a}/{b} 回合通过', { a: run.summary.successes, b: run.summary.episodes })}</span>}</div>
              <div className="te-run-bottom"><code title={run.id}>{run.id}</code>{run.created && <time dateTime={run.created}>{formatDate(run.created, format.date)}</time>}</div>
            </button>
          </li>)}
        </ul> : <div className="te-empty">
          <p>{hasFilters ? t('没有符合当前筛选条件的记录。') : checkpoint ? t('此检查点暂无实验记录。') : t('尚无移动搬运实验记录。')}</p>
          {hasFilters && <button type="button" className="te-text-button" onClick={() => { setSearch(''); setStatus('all'); setSource('all') }}>{t('重置列表筛选')}</button>}
        </div>)}
      </section>

      <div className="te-detail" ref={detail}>
        {selected && !report && <p className="te-card te-empty" role="status">{t('正在读取评测报告…')}</p>}
        {report?.error && <div className="te-card te-empty te-error" role="alert">
          <p>{t('无法读取评测报告。')}{report.error}</p>
          <button type="button" className="te-text-button" onClick={() => setRevision(value => value + 1)}>{t('重新读取')}</button>
        </div>}
        {report?.data && <ReportDetail key={report.data.run_id} report={report.data} run={selectedMeta} />}
        {!selected && history?.data && <div className="te-card te-empty te-idle">
          <h3>{checkpoint ? t('查看该检查点的闭环结果') : t('选择实验查看评测')}</h3>
          <p>{checkpoint ? t('完成一次搬运仿真后，实验报告会出现在列表中。') : t('列表中显示最近保存的移动搬运实验。')}</p>
          {checkpoint && !hasFilters && <Link className="te-button" to={startRun}>{t('使用检查点仿真')} <span aria-hidden="true">↗</span></Link>}
        </div>}
      </div>
    </div>
  </div>
}
