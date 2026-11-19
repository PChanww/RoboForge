import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api'
import ExperimentWorkspace from '../components/ExperimentWorkspace'
import './scenario.css'

type Asset = { id: string; name: string }
type Spec = {
  environment: string; robot: string; object: string; operation: string
  environment_mode: 'generate' | 'reuse' | 'none'; robot_mode: 'generate' | 'reuse'; object_mode: 'generate' | 'reuse'
  environment_asset: string | null; chassis_asset: string; object_asset: string
  robot_mount: 'four_wheel' | 'fixed_arm'; episodes: number; seed: number
  belt_speed_m_s: number; continuous: boolean; random_pose: boolean
}
type Capabilities = { configured: boolean; defaults: Spec; notes: string[]; objects: Asset[]; chassis: Asset[]; environments: Asset[] }
type Plan = { ready: boolean; spec: Spec; summary: string; steps: string[]; warnings: string[]; issues: string[]; generation_count: number; parser: string }
type Role = 'environment' | 'robot' | 'object'
type Run = {
  id: string; status: 'generating' | 'assembling' | 'simulating' | 'succeeded' | 'failed' | 'blocked' | 'cancelled'
  spec: Spec; steps: { role: Role; status: string; asset_id?: string; session_id?: string; job_id?: string; error?: string }[]
  project_id?: string; job_id?: string; dataset_id?: string; error?: string; created: string
  plan?: Plan
}
type Receipt = { client_request_id: string; spec: Spec; run_id?: string }
type Episode = { index: number; seed: number; frames: number; success: boolean; duration_s: number }
type Manifest = { episodes: Episode[]; summary: { episodes: number; successes: number; frames: number }; [key: string]: unknown }
type Preview = { images: Record<string, string | null>; tick: number; sim_time: number; episode_index?: number; stage?: string; completed: Episode[]; metrics?: { belt_speed_m_s?: number } }
const DRAFT_KEY = 'roboforge.scenario.draft.v1'
const RECEIPT_KEY = 'roboforge.scenario.receipt.v1'
const DEFAULT: Spec = {
  environment: '仓库分拣区，后方有低矮货架和规整的储物箱，中央为机器人保留空旷作业区。',
  robot: '紧凑四轮机器人底盘，绿色与白色外壳、四个黑色轮胎，顶部安装机械臂。',
  object: '黄色塑料小鸭，橙色嘴，圆润光滑的单个完整刚体。',
  operation: '传送带不停，物品随机位置和初始姿态，机器人跟随抓取放入旁边的收纳箱，采集3次。',
  environment_mode: 'generate', robot_mode: 'generate', object_mode: 'generate', environment_asset: null,
  chassis_asset: 'imported/four-wheel-robot-chassis-82e8e5', object_asset: 'imported/yellow-plastic-duck-a08e3e',
  robot_mount: 'four_wheel', episodes: 3, seed: 0, belt_speed_m_s: .025, continuous: true, random_pose: true,
}
const EMPTY: Preview = { images: {}, tick: 0, sim_time: 0, completed: [] }
const runLabels: Record<Run['status'], string> = { generating: '生成场景资产', assembling: '组装与检查场景', simulating: '仿真与采集中', succeeded: '实验完成', failed: '实验未完成', blocked: '需要处理', cancelled: '已停止' }
const stepLabels: Record<string, string> = { pending: '等待生成', queued: '等待处理', submitting: '正在提交', generating: '正在生成', remote_processing: 'Aholo 正在生成', downloading: '下载模型', importing: '导入模型', validating: '检查模型', succeeded: '模型就绪', ready: '模型就绪', reused: '复用已有模型', reuse: '复用已有模型', skipped: '无需生成', none: '未添加', failed: '处理失败', submit_unknown: '提交结果待确认', blocked: '需要处理', cancelled: '已停止' }
const roleLabels: Record<Role, string> = { environment: '环境道具', robot: '机器人外观', object: '抓取物品' }
function readLocal<T>(key: string): T | null { try { return JSON.parse(localStorage.getItem(key) ?? 'null') } catch { return null } }
function active(run: Run | null) { return !!run && ['generating', 'assembling', 'simulating'].includes(run.status) }
function errorText(error: unknown) { return error instanceof Error ? error.message : String(error) }
class HttpError extends Error { constructor(public status: number, message: string) { super(message) } }
async function request<T>(path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  const response = await fetch('/api/v1/' + path, { method: body === undefined ? 'GET' : 'POST', signal,
    ...(body === undefined ? {} : { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }) })
  const result = await response.json().catch(() => null)
  if (!response.ok) {
    const detail = result?.detail ?? result?.error ?? result
    throw new HttpError(response.status, typeof detail === 'string' ? detail : JSON.stringify(detail) || `请求失败 (${response.status})`)
  }
  return result as T
}

export default function ScenarioWorkbench() {
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null)
  const [spec, setSpec] = useState<Spec>(() => readLocal<Receipt>(RECEIPT_KEY)?.spec ?? readLocal<Spec>(DRAFT_KEY) ?? DEFAULT)
  const [receipt, setReceipt] = useState<Receipt | null>(() => readLocal<Receipt>(RECEIPT_KEY))
  const [plan, setPlan] = useState<Plan | null>(null)
  const [run, setRun] = useState<Run | null>(null)
  const [preview, setPreview] = useState<Preview>(EMPTY)
  const [manifest, setManifest] = useState<Manifest | null>(null)
  const [camera, setCamera] = useState('front')
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')
  const [previewError, setPreviewError] = useState('')
  const [pollVersion, setPollVersion] = useState(0)
  const submission = useRef(false)
  const [connectionVersion, setConnectionVersion] = useState(0)
  const hasSavedDraft = useRef(!!readLocal<Spec>(DRAFT_KEY) || !!readLocal<Receipt>(RECEIPT_KEY))
  const unknownSubmission = !!receipt && !receipt.run_id
  const restoring = !!receipt?.run_id && !run
  const locked = !!run || unknownSubmission || restoring || !!busy
  const runId = receipt?.run_id

  useEffect(() => {
    let disposed = false
    request<Capabilities>('scenarios/capabilities').then(result => {
      if (disposed) return
      setCapabilities(result)
      if (!hasSavedDraft.current) { setSpec(result.defaults); hasSavedDraft.current = true }
    }).catch(err => { if (!disposed) setError('无法连接场景服务：' + errorText(err)) })
    return () => { disposed = true }
  }, [connectionVersion])
  useEffect(() => { localStorage.setItem(DRAFT_KEY, JSON.stringify(spec)) }, [spec])

  function saveReceipt(next: Receipt) { localStorage.setItem(RECEIPT_KEY, JSON.stringify(next)); setReceipt(next) }
  function change<K extends keyof Spec>(key: K, value: Spec[K]) {
    const phrases: Partial<Record<keyof Spec, RegExp>> = {
      episodes: /(?:采集|收集|录制|记录)\s*[\d零一二三四五六七八九十两]+\s*(?:条|次|组)/gi,
      seed: /(?:seed|种子)\s*[:：=]?\s*\d+/gi,
      belt_speed_m_s: /(?:带速|传送带速度|belt\s*speed)\s*[:：=为]?\s*\d+(?:\.\d+)?\s*(?:cm\/s|m\/s|厘米每秒|米每秒|厘米\/秒|米\/秒)/gi,
      continuous: /传送带不停|不停带|不停|持续输送|连续输送|continuous|暂停传送带|到位停止|停带|节拍/gi,
      random_pose: /随机初始姿态|固定初始姿态|固定姿态|不要随机|不随机|直立|随机姿态|不同姿态/gi,
    }
    setSpec(current => ({ ...current, [key]: value,
      ...(phrases[key] ? { operation: current.operation.replace(phrases[key]!, '').replace(/([，,；;])\s*[，,；;]/g, '$1').replace(/^[，,；;\s]+|[，,；;\s]+$/g, '') } : {}) }))
    setPlan(null); setError('')
  }

  useEffect(() => {
    if (!runId) return
    let disposed = false
    let timer: ReturnType<typeof setTimeout>
    const controller = new AbortController()
    const poll = async (first = false) => {
      try {
        const result = await request<Run>(`scenarios/${runId}${first ? '' : '/poll'}`, first ? undefined : {}, controller.signal)
        if (disposed) return
        setRun(result); setSpec(result.spec); if (result.plan) setPlan(result.plan); setError('')
        if (active(result)) timer = setTimeout(() => void poll(), 3000)
      } catch (err) {
        if (disposed) return
        setError('暂时无法读取实验进度：' + errorText(err))
        timer = setTimeout(() => void poll(true), 5000)
      }
    }
    void poll(true)
    return () => { disposed = true; controller.abort(); clearTimeout(timer) }
  }, [runId, pollVersion])

  const jobId = run?.job_id
  const running = active(run)
  useEffect(() => {
    if (!jobId) return
    let disposed = false
    let timer: ReturnType<typeof setTimeout>
    const controller = new AbortController()
    const poll = async () => {
      const start = performance.now()
      try {
        const result = await request<Preview>(`jobs/${jobId}/preview`, undefined, controller.signal)
        if (disposed) return
        setPreview(previous => previous.tick === result.tick && previous.episode_index === result.episode_index && previous.completed.length === result.completed.length ? previous : result)
        setPreviewError('')
        if (running) timer = setTimeout(poll, Math.max(0, 50 - (performance.now() - start)))
      } catch (err) {
        if (disposed) return
        if (!(err instanceof HttpError && err.status === 404)) setPreviewError('相机画面暂不可用，正在重试。')
        if (running) timer = setTimeout(poll, 500)
      }
    }
    void poll()
    return () => { disposed = true; controller.abort(); clearTimeout(timer) }
  }, [jobId, running])

  useEffect(() => {
    if (run?.status !== 'succeeded' || !run.dataset_id) return
    let disposed = false
    let timer: ReturnType<typeof setTimeout>
    const load = async () => {
      try { const result = await api.dataset('collection__' + run.dataset_id); if (!disposed) setManifest(result) }
      catch (err) { if (!disposed) { setError('数据清单尚未读取完成：' + errorText(err)); timer = setTimeout(load, 3000) } }
    }
    void load()
    return () => { disposed = true; clearTimeout(timer) }
  }, [run?.status, run?.dataset_id])

  async function checkPlan() {
    setBusy('plan'); setError('')
    try { const result = await request<Plan>('scenarios/plan', spec); setPlan(result); setSpec(result.spec) }
    catch (err) { setError(errorText(err)) }
    finally { setBusy('') }
  }
  async function submit(resume = false) {
    if (submission.current || (!resume && !plan?.ready)) return
    submission.current = true; setBusy('submit'); setError('')
    const next = resume && receipt ? receipt : { client_request_id: crypto.randomUUID(), spec: plan!.spec }
    saveReceipt(next)
    try {
      const result = await request<Run>('scenarios', { spec: next.spec, client_request_id: next.client_request_id })
      saveReceipt({ ...next, run_id: result.id }); setRun(result); setSpec(result.spec)
      setPreview(EMPTY); setManifest(null); setPreviewError(''); setPollVersion(value => value + 1)
    } catch (err) {
      if (err instanceof HttpError && [400, 409, 422].includes(err.status)) {
        localStorage.removeItem(RECEIPT_KEY); setReceipt(null)
        setError('提交被拒绝：' + errorText(err))
      } else setError('提交结果尚未确认。请恢复这次提交，页面会沿用同一个请求：' + errorText(err))
    } finally { submission.current = false; setBusy('') }
  }
  async function control(action: 'stop' | 'retry') {
    if (!run || busy) return
    setBusy(action); setError('')
    try { setRun(await request<Run>(`scenarios/${run.id}/${action}`, {})); setPollVersion(value => value + 1) }
    catch (err) { setError(errorText(err)) }
    finally { setBusy('') }
  }
  function reset() {
    localStorage.removeItem(RECEIPT_KEY); setReceipt(null); setRun(null); setPlan(null)
    setPreview(EMPTY); setManifest(null); setError(''); setPreviewError('')
  }
  function downloadManifest() {
    if (!manifest) return
    const url = URL.createObjectURL(new Blob([JSON.stringify(manifest, null, 2)], { type: 'application/json' }))
    const link = document.createElement('a'); link.href = url; link.download = `${run?.dataset_id ?? 'scenario'}-manifest.json`; link.click()
    setTimeout(() => URL.revokeObjectURL(url), 1000)
  }
  const rows = manifest?.episodes ?? preview.completed
  const successes = rows.filter(row => row.success).length
  const image = preview.images[camera]
  const generationCount = plan?.generation_count ?? 0
  const roles: Role[] = ['environment', 'robot', 'object']
  const workflowIndex = run ? run.status === 'generating' ? 1 : run.status === 'assembling' ? 2 : 3 : plan ? 1 : 0

  return <ExperimentWorkspace title="整场景生成" description="描述场景、机器人外观和任务目标。"
    actions={<button className="workspace-cta workspace-cta--quiet" onClick={() => setConnectionVersion(value => value + 1)}>
      {capabilities ? '刷新连接' : '重新连接'}
    </button>}>
    <div className="sc-app">
      <div className="sc-main">
        <div className="sc-workflow">{['描述需求', '生成资产', '组装场景', '仿真与采集'].map((label, index) => <div key={label} className={index === workflowIndex ? 'active' : index < workflowIndex ? 'done' : ''}><span>{index < workflowIndex ? '✓' : `0${index + 1}`}</span>{label}{index < 3 && <i/>}</div>)}</div>
        {error && <div className="sc-alert" role="alert"><strong>操作提示</strong><span>{error}</span><button onClick={() => setError('')} aria-label="关闭提示">×</button></div>}
        {unknownSubmission && <div className="sc-recovery"><div><strong>这次提交已保存在浏览器中</strong><p>恢复会复用同一个请求，确认已有任务和生成进度。</p></div><button disabled={!!busy} onClick={() => void submit(true)}>{busy === 'submit' ? '正在恢复…' : '恢复这次提交'}</button></div>}
        <div className="sc-layout"><section className="sc-intent sc-card"><div className="sc-card-title"><h2>场景要求</h2><p>外形要求直接交给 Aholo；动作按检查后的执行参数运行。</p></div>
          <div className="sc-requirements">
            <section className="sc-requirement"><div><h3>环境</h3><select aria-label="环境来源" value={spec.environment_mode} disabled={locked} onChange={e => change('environment_mode', e.target.value as Spec['environment_mode'])}><option value="generate">按要求生成道具</option><option value="reuse">复用环境道具</option><option value="none">简洁作业区</option></select></div>
              <textarea aria-label="环境要求" value={spec.environment} maxLength={1500} disabled={locked || spec.environment_mode === 'none'} onChange={e => change('environment', e.target.value)} placeholder="例如：仓库分拣区，后方货架，中央留出空旷作业区"/>
              {spec.environment_mode === 'reuse' && <label className="sc-field">已有环境道具<select aria-label="已有环境道具" disabled={locked} value={spec.environment_asset ?? ''} onChange={e => change('environment_asset', e.target.value || null)}><option value="">请选择环境道具</option>{capabilities?.environments.map(asset => <option key={asset.id} value={asset.id}>{asset.name}</option>)}</select></label>}
              <small>生成环境道具并组装在作业区后方，工作区保持空旷；当前不生成完整房间。</small>
            </section>
            <section className="sc-requirement"><div><h3>机器人</h3><select aria-label="机器人来源" value={spec.robot_mode} disabled={locked || spec.robot_mount === 'fixed_arm'} onChange={e => change('robot_mode', e.target.value as Spec['robot_mode'])}><option value="reuse">复用机器人外观</option><option value="generate">按要求生成外观</option></select></div>
              <textarea aria-label="机器人要求" value={spec.robot} maxLength={1500} disabled={locked} onChange={e => change('robot', e.target.value)} placeholder="例如：绿色四轮底盘，顶部安装机械臂"/>
              {spec.robot_mode === 'reuse' && spec.robot_mount === 'four_wheel' && <label className="sc-field">已有底盘<select aria-label="已有机器人底盘" value={spec.chassis_asset} disabled={locked} onChange={e => change('chassis_asset', e.target.value)}>{capabilities?.chassis.map(asset => <option key={asset.id} value={asset.id}>{asset.name}</option>)}</select></label>}
              <small>{spec.robot_mount === 'fixed_arm' ? '固定机械臂沿用现有机构，不生成底盘外观。' : '生成底盘外观，机械臂与抓取动作沿用现有机构。'}</small>
            </section>
            <section className="sc-requirement"><div><h3>物品</h3><select aria-label="物品来源" value={spec.object_mode} disabled={locked} onChange={e => change('object_mode', e.target.value as Spec['object_mode'])}><option value="reuse">复用已有物品</option><option value="generate">按要求生成物品</option></select></div>
              <textarea aria-label="物品要求" value={spec.object} maxLength={1500} disabled={locked} onChange={e => change('object', e.target.value)} placeholder="例如：红色塑料小鸭，黄色嘴，完整刚体"/>
              {spec.object_mode === 'reuse' && <label className="sc-field">已有抓取物品<select aria-label="已有抓取物品" value={spec.object_asset} disabled={locked} onChange={e => change('object_asset', e.target.value)}>{capabilities?.objects.map(asset => <option key={asset.id} value={asset.id}>{asset.name}</option>)}</select></label>}
              <small>当前适合夹爪能容纳的小型单体。新造型的抓取结果会如实记录。</small>
            </section>
            <section className="sc-requirement"><div><h3>工作状态与任务</h3><span className="sc-chip">抓取 → 入箱</span></div>
              <textarea aria-label="工作状态要求" value={spec.operation} maxLength={1500} disabled={locked} onChange={e => change('operation', e.target.value)} placeholder="例如：传送带不停，随机姿态，抓取入箱，采集3次"/>
              <small>描述中可提出带速、随机姿态和采集数量，检查方案后核对执行参数。</small>
            </section>
          </div>
          <div className="sc-settings"><div className="sc-settings-title"><h3>执行参数</h3><span>可直接调整</span></div><div className="sc-settings-grid">
            <label className="sc-field">机器人工作方式<select aria-label="机器人工作方式" value={spec.robot_mount} disabled={locked} onChange={e => {
              const mount = e.target.value as Spec['robot_mount']
              setSpec(current => ({ ...current, robot_mount: mount, ...(mount === 'fixed_arm' ? { robot_mode: 'reuse' as const } : {}) })); setPlan(null); setError('')
            }}><option value="four_wheel">四轮底盘停靠作业</option><option value="fixed_arm">固定机械臂</option></select></label>
            <label className="sc-field">传送带速度 <span>cm/s</span><input aria-label="传送带速度" type="number" min={1} max={4} step={.5} value={Number((spec.belt_speed_m_s * 100).toFixed(2))} disabled={locked} onInput={e => change('belt_speed_m_s', Math.max(1, Math.min(4, Number(e.currentTarget.value) || 1)) / 100)} onChange={() => {}}/></label>
            <label className="sc-field">采集次数 <span>Episodes</span><input aria-label="场景采集次数" type="number" min={1} max={20} value={spec.episodes} disabled={locked} onInput={e => change('episodes', Math.max(1, Math.min(20, Math.trunc(Number(e.currentTarget.value) || 1))))} onChange={() => {}}/></label>
            <label className="sc-field">起始随机种子<input aria-label="场景随机种子" type="number" min={0} max={999999} value={spec.seed} disabled={locked} onInput={e => change('seed', Math.max(0, Math.min(999999, Math.trunc(Number(e.currentTarget.value) || 0))))} onChange={() => {}}/></label>
          </div><div className="sc-toggles"><label><input type="checkbox" checked={spec.continuous} disabled={locked} onChange={e => change('continuous', e.target.checked)}/>传送带全程不停</label><label><input type="checkbox" checked={spec.random_pose} disabled={locked} onChange={e => change('random_pose', e.target.checked)}/>随机初始姿态</label></div><p className="sc-parameter-note">修改参数会清除工作描述中对应的设置，最终以检查后的方案为准。</p></div>
          <div className="sc-intent-actions"><button className="sc-primary" disabled={locked || !capabilities || spec.operation.trim().length < 2} onClick={() => void checkPlan()}>{busy === 'plan' ? '正在检查…' : '检查执行方案'}<span aria-hidden="true">→</span></button><small>检查方案不会提交在线生成。</small></div>
        </section>
        <div className="sc-right"><section className="sc-plan sc-card"><div className="sc-card-title"><h2>{run ? runLabels[run.status] : plan ? '把要求变成可执行方案' : '先看看这次怎么做'}</h2></div>
          {plan ? <div className="sc-plan-content"><p className="sc-plan-summary">{plan.summary}</p><dl className="sc-plan-facts"><div><dt>环境</dt><dd>{plan.spec.environment_mode === 'none' ? '简洁作业区' : plan.spec.environment_mode === 'generate' ? '生成环境道具并放入场景' : '复用环境道具'}</dd></div><div><dt>机器人</dt><dd>{plan.spec.robot_mount === 'four_wheel' ? '四轮底盘停靠 + 机械臂' : '固定机械臂'}</dd></div><div><dt>物品</dt><dd>{plan.spec.object_mode === 'generate' ? '生成新物品' : '复用已有物品'}</dd></div><div><dt>执行</dt><dd>{plan.spec.continuous ? '不停带跟随' : '节拍输送'} · {Number((plan.spec.belt_speed_m_s * 100).toFixed(2))} cm/s · {plan.spec.episodes} 次</dd></div></dl>
            {!!plan.steps.length && <ol className="sc-plan-steps">{plan.steps.map((step, index) => <li key={index}><span>{String(index + 1).padStart(2, '0')}</span>{step}</li>)}</ol>}
            {plan.issues.length > 0 && <div className="sc-notices sc-issues"><strong>这些要求还不能执行</strong><ul>{plan.issues.map((issue, index) => <li key={index}>{issue}</li>)}</ul></div>}
            {plan.warnings.length > 0 && <div className="sc-notices"><ul>{plan.warnings.map((warning, index) => <li key={index}>{warning}</li>)}</ul></div>}
            <div className="sc-cost"><span>本次在线生成<strong>{generationCount}<small>件资产</small></strong></span><p>{generationCount ? '点击下方按钮后使用 Aholo 额度。生成需要等待，刷新可恢复进度。' : '使用已有资产，组装后直接进入仿真。'}</p></div>
            <button className="sc-primary sc-full" disabled={locked || !plan.ready || plan.issues.length > 0 || (generationCount > 0 && !capabilities?.configured)} onClick={() => void submit()}>{busy === 'submit' ? '正在提交…' : '生成并开始仿真'}<span aria-hidden="true">↗</span></button>
            {generationCount > 0 && capabilities && !capabilities.configured && <p className="sc-unavailable">后端尚未配置 Aholo，当前可复用已有资产。</p>}
          </div> : <div className="sc-plan-empty"><span>✧</span><p>{run ? '实验已恢复。可以查看下方的资产进度、相机和采集结果。' : '分别描述四项要求，再检查执行方案。你会看到需要生成的资产，以及实际执行的参数。'}</p><small>当前用结构化参数和规则检查任务；不会为不支持的动作生成虚假结果。</small></div>}
          {run && <div className="sc-run-actions">{active(run) && <button className="sc-secondary" disabled={!!busy} onClick={() => void control('stop')}>{busy === 'stop' ? '正在停止…' : '停止本次实验'}</button>}{['failed', 'blocked'].includes(run.status) && <button className="sc-secondary" disabled={!!busy} onClick={() => void control('retry')}>{busy === 'retry' ? '正在恢复…' : '恢复已有任务'}</button>}{!active(run) && <button className="sc-secondary" disabled={!!busy} onClick={reset}>调整要求，开始新实验</button>}{run.error && <p className="sc-run-error" role="alert">{run.error}</p>}{active(run) && <small>停止会结束后续处理和采集；已提交的模型生成仍会使用额度。</small>}</div>}
          {restoring && <div className="sc-plan-empty"><p>正在恢复上次实验…</p></div>}
        </section>
        <section className="sc-assets sc-card"><div className="sc-section-head"><h2>场景资产</h2><span>{run ? '实时进度' : '等待方案'}</span></div>{roles.map(role => {
          const step = run?.steps.find(item => item.role === role)
          const status = step?.status.toLowerCase()
          const mode = role === 'environment' ? spec.environment_mode : role === 'robot' ? spec.robot_mode : spec.object_mode
          return <div className="sc-asset-row" key={role}><span className={'sc-asset-icon ' + (status && ['succeeded', 'ready', 'reused', 'reuse', 'skipped'].includes(status) ? 'ready' : '')}>{role === 'environment' ? '▦' : role === 'robot' ? '⌁' : '◇'}</span><div><strong>{roleLabels[role]}</strong><small>{status ? stepLabels[status] ?? '处理中' : mode === 'generate' ? '按要求生成' : mode === 'none' ? '无需生成' : '复用已有模型'}</small>{step?.error && <p>{step.error}</p>}</div>{step?.session_id && <Link to={`/generation/${step.session_id}`} title={`查看${roleLabels[role]}生成任务`}>查看 ↗</Link>}</div>
        })}</section>
        {capabilities?.notes?.length ? <div className="sc-scope"><strong>当前实验范围</strong>{capabilities.notes.map((note, index) => <p key={index}>{note}</p>)}</div> : null}
        </div></div>
        <section className="sc-simulation sc-card"><div className="sc-section-head"><div><span className={'sc-live-dot ' + (running ? 'running' : '')}/><h2>仿真现场</h2></div><span>{run ? runLabels[run.status] : '准备生成场景'}</span></div>
          <div className="sc-camera">{image ? <img src={'data:image/jpeg;base64,' + image} alt={camera === 'front' ? '场景前视相机画面' : '场景腕部相机画面'}/> : <div className="sc-camera-empty"><span>◇</span><strong>{run?.status === 'generating' ? '场景资产正在生成' : run?.status === 'assembling' ? '正在组装真实仿真场景' : jobId ? '正在等待相机画面' : '你的工作现场将在这里出现'}</strong><p>{previewError || (run ? '生成和导入完成后自动进入仿真，无需再次操作。' : '检查方案并开始实验后，查看双相机与机器人操作。')}</p></div>}<div className="sc-camera-overlay"><span>{camera === 'front' ? '前视相机' : '腕部相机'}</span><span>{preview.sim_time.toFixed(2)} s · Tick {preview.tick}</span></div></div>
          <div className="sc-camera-controls"><div><button className={camera === 'front' ? 'selected' : ''} onClick={() => setCamera('front')}>前视相机</button><button className={camera === 'wrist' ? 'selected' : ''} onClick={() => setCamera('wrist')}>腕部相机</button></div><span>{jobId ? `${preview.episode_index === undefined ? '' : `第 ${preview.episode_index + 1} / ${run?.spec.episodes} 次 · `}${preview.stage ?? '准备仿真'}${preview.metrics?.belt_speed_m_s === undefined ? '' : ` · 带速 ${(preview.metrics.belt_speed_m_s * 100).toFixed(1)} cm/s`}` : '生成需要等待，任务会保存在后端。'}</span></div>
        </section>
        <section className="sc-results sc-card"><div className="sc-section-head"><div><h2>本次采集</h2>{run && <span className={'sc-result-status ' + run.status}>{runLabels[run.status]}</span>}</div><button className="sc-secondary" disabled={!manifest} onClick={downloadManifest}>↓ 下载清单</button></div><div className="sc-stats"><div><span>完成 / 计划</span><strong>{String(rows.length).padStart(2, '0')}<small>/ {String(run?.spec.episodes ?? spec.episodes).padStart(2, '0')}</small></strong></div><div><span>已记录帧数</span><strong>{(manifest?.summary.frames ?? rows.reduce((total, row) => total + row.frames, 0)).toLocaleString()}<small>帧</small></strong></div><div><span>成功率</span><strong>{rows.length ? Math.round(successes / rows.length * 100) : '—'}<small>{rows.length ? '%' : ''}</small></strong></div><div><span>数据内容</span><p>双相机 · 关节状态 · 控制动作<br/>物体姿态 · 传送带状态 · 时间戳</p></div></div>
          {rows.length ? <div className="sc-table-wrap"><table><thead><tr><th>样本</th><th>种子</th><th>数据帧数</th><th>仿真时长</th><th>结果</th><th>数据</th></tr></thead><tbody>{rows.map(row => <tr key={row.index}><td>Episode {String(row.index + 1).padStart(3, '0')}</td><td>Seed {row.seed}</td><td>{row.frames} 帧</td><td>{row.duration_s.toFixed(2)} s</td><td><span className={'sc-outcome ' + (row.success ? '' : 'failed')}>{row.success ? '✓ 成功' : '未成功'}</span></td><td>{manifest && run?.dataset_id ? <a href={`/api/v1/datasets/collection__${encodeURIComponent(run.dataset_id)}/episodes/${row.index}`}>下载 NPZ ↗</a> : '正在保存'}</td></tr>)}</tbody></table></div> : <div className="sc-results-empty"><span>▤</span><strong>{run?.status === 'simulating' ? '正在等待第一条完整示范' : '让需求产生可检查的机器人数据'}</strong><p>每次操作都会记录实际结果，抓取失败也会保留在数据中。</p></div>}
        </section>
      </div>
    </div>
  </ExperimentWorkspace>
}
