import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api'
import './studio.css'

type Spec = { prompt: string; scene: string; task: string; asset: string; color: string; goal: string; episodes: number; seed: number; randomize: boolean; unsupported: string[] }
type Plan = { spec: Spec; ready: boolean; summary: string; defaults: string[]; parser: string }
type Episode = { index: number; asset: string; seed: number; frames: number; duration_s: number; success: boolean }
type Job = { id: string; status: string; created: string; error?: string; params: { dataset_id?: string; studio_spec?: Spec } }
type RobotState = { eef_position?: number[]; gripper?: number; joint_vel?: Record<string, number> }
type Snapshot = { tick: number; sim_time: number; episode_index?: number; images: Record<string, string>; completed: Episode[]; state?: RobotState; stage?: string; metrics?: { contacts: number; success: boolean; lifted: boolean } }
type Manifest = { episodes: Episode[]; summary: { frames: number; episodes: number; successes: number }; expert_version?: string }
type Trace = { frame_time: number[]; eef_pos: number[][]; gripper: number[]; action: number[][] }
const EMPTY: Snapshot = { tick: 0, sim_time: 0, images: {}, completed: [] }
const INITIAL: Spec = { prompt: '', scene: 'sorting', task: 'pick_place', asset: 'block', color: 'red', goal: 'right', episodes: 3, seed: 0, randomize: true, unsupported: [] }
const assets: Record<string, string> = { block: '方块', cylinder: '圆柱', standard: '标准杯', tall: '高杯', wide: '宽口杯' }
const scenes: Record<string, string> = { workbench: '简洁工作台', sorting: '分拣工作台', lab: '实验工作台' }
const colors: Record<string, string> = { red: '红色', blue: '蓝色', green: '绿色', orange: '橙色', violet: '紫色' }
const colorHex: Record<string, string> = { red: '#d4402e', blue: '#2e70d4', green: '#33a16e', orange: '#ed8530', violet: '#8c5cc4' }
const statuses: Record<string, string> = { queued: '等待执行', running: '正在采集', succeeded: '采集完成', failed: '执行异常', cancelled: '已停止', interrupted: '已中断' }
const examples = [
  { label: '分拣与搬运', icon: 'box', text: '在分拣工作台上，把红色方块搬运到右侧托盘，随机摆位，采集3条示范。', detail: '方块 · 抓取放置 · 3 条' },
  { label: '接触式推移', icon: 'move', text: '在简洁工作台上，把绿色方块沿桌面推到中间目标区，采集2条示范。', detail: '方块 · 桌面推移 · 2 条' },
  { label: '实验台搬运', icon: 'flask', text: '在实验台上，把蓝色圆柱放到左侧托盘，随机摆位，采集3条示范。', detail: '圆柱 · 抓取放置 · 3 条' },
]
const active = (job: Job | null) => !!job && ['queued', 'running'].includes(job.status)
const errorText = (err: unknown) => err instanceof Error ? err.message : String(err)
async function request<T>(path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  const response = await fetch('/api/v1/' + path, { method: body === undefined ? 'GET' : 'POST', signal,
    headers: body === undefined ? undefined : { 'Content-Type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body), cache: 'no-store' })
  if (!response.ok) { const data = await response.json().catch(() => ({})); throw new Error(typeof data.detail === 'string' ? data.detail : `请求未完成 (${response.status})`) }
  return response.json()
}
function Glyph({ name, size = 18 }: { name: string; size?: number }) {
  const paths: Record<string, React.ReactNode> = {
    box: <><path d="m12 3 9 5v8l-9 5-9-5V8l9-5Z"/><path d="m3 8 9 5 9-5m-9 5v8M7.5 5.5l9 5"/></>,
    grid: <><rect x="3" y="3" width="7" height="7" rx="2"/><rect x="14" y="3" width="7" height="7" rx="2"/><rect x="3" y="14" width="7" height="7" rx="2"/><rect x="14" y="14" width="7" height="7" rx="2"/></>,
    data: <><ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v7c0 4 16 4 16 0V5M4 12v7c0 4 16 4 16 0v-7"/></>,
    arrow: <path d="M5 12h14m-5-5 5 5-5 5"/>,
    move: <><path d="M4 12h16m-4-4 4 4-4 4M8 8l-4 4 4 4"/><path d="M12 4v16"/></>,
    flask: <><path d="M9 3h6m-5 0v6l-6 10a1 1 0 0 0 1 2h14a1 1 0 0 0 1-2L14 9V3M7 15h10"/></>,
    play: <path d="m9 5 11 7-11 7V5Z"/>, stop: <rect x="6" y="6" width="12" height="12" rx="2"/>,
    check: <path d="m5 12 4 4L19 6"/>, download: <><path d="M12 3v12m-5-5 5 5 5-5M4 17v4h16v-4"/></>,
    edit: <><path d="m15 4 5 5-11 11H4v-5L15 4ZM12 7l5 5"/></>,
    camera: <><rect x="3" y="6" width="18" height="15" rx="3"/><circle cx="12" cy="13" r="4"/><path d="m8 6 1-3h6l1 3"/></>,
    plus: <path d="M12 5v14M5 12h14"/>, info: <><circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7v1"/></>,
  }
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name] ?? paths.box}</svg>
}
function Sparkline({ values, cursor, label, unit }: { values: number[]; cursor: number; label: string; unit: string }) {
  const lo = Math.min(...values), hi = Math.max(...values)
  const points = values.map((v, i) => `${i / Math.max(1, values.length - 1) * 280},${42 - (v - lo) / Math.max(.001, hi - lo) * 32}`).join(' ')
  return <div className="st-chart"><div><span>{label}</span><b>{values[cursor]?.toFixed(1)}<small>{unit}</small></b></div><svg viewBox="0 0 280 48" preserveAspectRatio="none"><path d="M0 43H280" stroke="#dce2dd"/><polyline points={points} fill="none" stroke="#39866c" strokeWidth="1.8"/><path d={`M${cursor / Math.max(1, values.length - 1) * 280} 2v43`} stroke="#b37c4b" strokeDasharray="3 3"/></svg></div>
}

export default function TaskStudio() {
  const [prompt, setPrompt] = useState('')
  const [plan, setPlan] = useState<Plan | null>(null)
  const [spec, setSpec] = useState<Spec>(INITIAL)
  const [connected, setConnected] = useState(false)
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')
  const [previewError, setPreviewError] = useState('')
  const [preview, setPreview] = useState<{ images: Record<string, string>; state?: RobotState }>({ images: {} })
  const [previewing, setPreviewing] = useState(false)
  const [snapshot, setSnapshot] = useState<Snapshot>(EMPTY)
  const [jobId, setJobId] = useState<string | null>(() => localStorage.getItem('roboforge.studio.job'))
  const [job, setJob] = useState<Job | null>(null)
  const [manifest, setManifest] = useState<Manifest | null>(null)
  const [history, setHistory] = useState<Job[]>([])
  const [camera, setCamera] = useState('front')
  const [mode, setMode] = useState<'preview' | 'live' | 'replay'>(() => localStorage.getItem('roboforge.studio.job') ? 'live' : 'preview')
  const [tab, setTab] = useState<'workbench' | 'data'>('workbench')
  const [logs, setLogs] = useState<string[]>([])
  const [showLogs, setShowLogs] = useState(false)
  const [selected, setSelected] = useState(0)
  const [trace, setTrace] = useState<Trace | null>(null)
  const [frame, setFrame] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [replayLoading, setReplayLoading] = useState(false)
  const stopped = useRef(false)
  const requestVersion = useRef(0)
  const running = active(job)
  const restoring = !!jobId && !job
  const locked = running || restoring || !!busy
  const rows = manifest?.episodes ?? snapshot.completed
  const total = job?.params.studio_spec?.episodes ?? spec.episodes
  const dataset = job?.params.dataset_id ? 'collection__' + job.params.dataset_id : null

  async function refreshConnection() {
    try { await api.getConfig(); setConnected(true); const data = await api.jobs(); setHistory(data.jobs.filter((j: Job) => j.params.studio_spec).slice(0, 6)) }
    catch { setConnected(false) }
  }
  useEffect(() => { void refreshConnection(); const timer = setInterval(() => void refreshConnection(), 15000); return () => clearInterval(timer) }, [])

  useEffect(() => {
    if (!jobId) return
    let disposed = false
    let timer: ReturnType<typeof setTimeout>
    stopped.current = false
    const poll = async () => {
      try {
        const next = await api.job(jobId) as Job
        if (disposed) return
        setJob(next); setConnected(true); stopped.current = !active(next)
        if (next.params.studio_spec) setSpec(next.params.studio_spec)
        if (!active(next)) {
          void refreshConnection()
          const log = await api.jobLogs(jobId); if (!disposed) setLogs(log.lines)
          if (next.status === 'succeeded' && next.params.dataset_id) {
            const result = await api.dataset('collection__' + next.params.dataset_id)
            if (!disposed) setManifest(result as Manifest)
          } else if (['failed', 'interrupted'].includes(next.status)) {
            if (!disposed) { setError(next.error || '作业未正常结束，请查看执行日志。'); setShowLogs(true) }
          }
        }
        if (!disposed && active(next)) timer = setTimeout(poll, 800)
      } catch (err) { if (!disposed) {
        if (errorText(err).startsWith('404:')) {
          localStorage.removeItem('roboforge.studio.job'); setJobId(null); setJob(null); setMode('preview')
          setError('上次实验记录已不存在，可以重新创建实验。')
        } else { setError('暂时无法读取作业：' + errorText(err)); timer = setTimeout(poll, 2500) }
      } }
    }
    void poll()
    return () => { disposed = true; clearTimeout(timer) }
  }, [jobId])

  useEffect(() => {
    if (!jobId) return
    let disposed = false
    let timer: ReturnType<typeof setTimeout>
    const controller = new AbortController()
    const poll = async () => {
      const last = stopped.current, start = performance.now()
      try {
        const next = await request<Snapshot>(`jobs/${jobId}/preview`, undefined, controller.signal)
        if (disposed) return
        setSnapshot(previous => previous.tick === next.tick && previous.episode_index === next.episode_index && previous.completed.length === next.completed.length ? previous : next)
        if (!last) timer = setTimeout(poll, Math.max(0, 16 - (performance.now() - start)))
      } catch { if (!disposed && !last) timer = setTimeout(poll, 1000) }
    }
    void poll()
    return () => { disposed = true; controller.abort(); clearTimeout(timer) }
  }, [jobId])

  useEffect(() => {
    if (!connected || mode !== 'preview' || spec.unsupported.length) return
    let disposed = false
    const version = ++requestVersion.current
    const controller = new AbortController()
    const timer = setTimeout(async () => {
      setPreviewing(true); setPreviewError('')
      try { const next = await request<{ images: Record<string, string>; state: RobotState }>('studio/preview', spec, controller.signal); if (!disposed && version === requestVersion.current) setPreview(next) }
      catch (err) { if (!disposed) setPreviewError(errorText(err)) }
      finally { if (!disposed) setPreviewing(false) }
    }, 400)
    return () => { disposed = true; controller.abort(); clearTimeout(timer) }
  }, [spec, connected, mode])

  useEffect(() => {
    if (!playing || !trace) return
    const timer = setInterval(() => setFrame(current => { if (current >= trace.frame_time.length - 1) { setPlaying(false); return current } return current + 1 }), 50)
    return () => clearInterval(timer)
  }, [playing, trace])

  async function generate(text = prompt) {
    if (!text.trim()) return
    setPrompt(text); setBusy('plan'); setError(''); setPreviewError(''); setPlaying(false)
    try {
      const result = await request<Plan>('studio/plan', { prompt: text })
      setPlan(result); setSpec(result.spec); setMode('preview'); setTab('workbench')
      setJobId(null); setJob(null); setManifest(null); setSnapshot(EMPTY); setTrace(null); setPreview({ images: {} }); setShowLogs(false)
      localStorage.removeItem('roboforge.studio.job')
    } catch (err) { setError(errorText(err)) }
    finally { setBusy('') }
  }
  function edit<K extends keyof Spec>(key: K, value: Spec[K]) {
    setSpec(current => ({ ...current, [key]: value })); setMode('preview')
  }
  async function execute() {
    setBusy('execute'); setError(''); setPlaying(false)
    try {
      const created = await request<Job>('studio/execute', spec)
      setJob(created); setJobId(created.id); setManifest(null); setSnapshot(EMPTY); setTrace(null); setMode('live'); setTab('workbench'); setShowLogs(false)
      localStorage.setItem('roboforge.studio.job', created.id)
    } catch (err) { setError(errorText(err)) }
    finally { setBusy('') }
  }
  async function stop() {
    if (!job) return
    setBusy('stop')
    try { setJob(await api.stopJob(job.id)); stopped.current = true }
    catch (err) { setError(errorText(err)) }
    finally { setBusy('') }
  }
  async function replay(index: number) {
    if (!dataset) return
    setSelected(index); setPlaying(false); setFrame(0); setTrace(null); setMode('replay'); setReplayLoading(true); setError('')
    try { setTrace(await request<Trace>(`studio/datasets/${dataset}/episodes/${index}/trace`)) }
    catch (err) { setError(errorText(err)); setTrace(null) }
    finally { setReplayLoading(false) }
  }
  function openJob(item: Job) {
    if (locked) return
    if (item.id === jobId) { setTab('data'); return }
    setJob(item); setJobId(item.id); setSpec(item.params.studio_spec!); setPrompt(item.params.studio_spec!.prompt); setPlan(null)
    setManifest(null); setSnapshot(EMPTY); setTrace(null); setPlaying(false); setMode('live'); setTab('data'); setError(''); setShowLogs(false)
    localStorage.setItem('roboforge.studio.job', item.id)
  }
  async function toggleLogs() {
    if (!job) return
    setShowLogs(value => !value)
    try { const data = await api.jobLogs(job.id); setLogs(data.lines) } catch (err) { setError(errorText(err)) }
  }
  function downloadManifest() {
    if (!manifest) return
    const url = URL.createObjectURL(new Blob([JSON.stringify(manifest, null, 2)], { type: 'application/json' }))
    const a = document.createElement('a'); a.href = url; a.download = `${job?.params.dataset_id}-manifest.json`; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000)
  }

  const currentSpec = mode === 'preview' ? spec : job?.params.studio_spec ?? spec
  const images = mode === 'preview' ? preview.images : snapshot.images
  const replayImage = mode === 'replay' && trace && dataset ? `/api/v1/studio/datasets/${dataset}/episodes/${selected}/frames/${frame}/${camera}` : ''
  const image = replayImage || (images[camera] ? 'data:image/jpeg;base64,' + images[camera] : '')
  const state = mode === 'preview' ? preview.state : snapshot.state
  const eef = mode === 'replay' && trace ? trace.eef_pos[frame] : state?.eef_position
  const gripper = mode === 'replay' && trace ? trace.gripper[frame] : state?.gripper
  const time = mode === 'replay' && trace ? trace.frame_time[frame] : mode === 'live' ? snapshot.sim_time : 0
  const isPush = currentSpec.task === 'push'
  const steps = isPush ? ['准备推移', '下降对齐', '接触推移', '撤离并等待稳定'] : ['接近物体', '减速下降', '夹紧与接触确认', '抬升', '搬运', '放置', '松开', '撤离']
  const completed = rows.length
  const successes = rows.filter(row => row.success).length
  const blocked = spec.unsupported.length > 0 || (spec.task === 'push' && spec.asset !== 'block')
  const workflowStep = mode === 'preview' ? plan ? 1 : 0 : mode === 'replay' || manifest ? 3 : 2

  return <div className="st-app">
    <aside className="st-sidebar">
      <Link className="st-brand" to="/"><span className="st-brand-icon"><Glyph name="box" size={25}/></span><span>RoboForge<small>ROBOT LEARNING STUDIO</small></span></Link>
      <button className="st-new" disabled={locked} onClick={() => { setPlan(null); setSpec(INITIAL); setPrompt(''); setJobId(null); setJob(null); setManifest(null); setSnapshot(EMPTY); setTrace(null); setMode('preview'); setTab('workbench'); setPlaying(false); setError(''); localStorage.removeItem('roboforge.studio.job') }}><Glyph name="plus"/>新建实验<kbd>N</kbd></button>
      <span className="st-nav-label">工作空间</span>
      <nav><button className={tab === 'workbench' ? 'selected' : ''} onClick={() => setTab('workbench')}><Glyph name="grid"/>任务工作台<span>01</span></button><button className={tab === 'data' ? 'selected' : ''} onClick={() => setTab('data')}><Glyph name="data"/>数据与回放<span>{completed.toString().padStart(2, '0')}</span></button></nav>
      <div className="st-history"><span className="st-nav-label">最近实验 <span>{history.length}</span></span>{history.length ? history.map(item => <button key={item.id} disabled={locked} onClick={() => openJob(item)} className={item.id === jobId ? 'current' : ''}><i className={item.status}/><span>{assets[item.params.studio_spec!.asset]} · {item.params.studio_spec!.task === 'push' ? '桌面推移' : '搬运放置'}<small>{new Date(item.created).toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' })} · {statuses[item.status]}</small></span><span>↗</span></button>) : <p>完成的实验会留在这里。</p>}</div>
      <div className="st-sidebar-bottom"><span className="st-lab-icon">RF</span><div>本地实验室<small>MuJoCo · 物理仿真</small></div><Link to="/projects" title="打开原有平台">↗</Link></div>
    </aside>

    <div className="st-body">
      <header className="st-topbar"><div>工作空间 <span>/</span> <b>{tab === 'data' ? '数据与回放' : '任务工作台'}</b></div><div><span className="st-parser-badge">本地任务解析</span><button className={'st-connection ' + (connected ? 'online' : '')} onClick={() => void refreshConnection()}><i/>{connected ? '后端已连接' : '后端未连接'}<span>↻</span></button></div></header>
      <main className="st-main">
        <div className="st-heading"><div><span className="st-kicker">FROM INTENT TO ROBOT DATA</span><h1>{tab === 'data' ? '每一次示范，都可以回看。' : '把你的想法，变成机器人任务。'}</h1><p>{tab === 'data' ? '回到采集时刻，检查画面、动作和执行结果。' : '描述场景、物体和操作目标，在仿真中验证，再收集学习数据。'}</p></div><span className="st-version">STUDIO <b>01</b></span></div>
        <div className="st-workflow">{['提出需求', '检查方案', '执行与采集', '检查数据'].map((label, i) => <div key={label} className={i === workflowStep ? 'active' : i < workflowStep ? 'done' : ''}><span>{i < workflowStep ? <Glyph name="check" size={12}/> : `0${i + 1}`}</span>{label}{i < 3 && <i/>}</div>)}</div>
        {error && <div className="st-alert" role="alert"><Glyph name="info"/><span>{error}</span><button onClick={() => setError('')} aria-label="关闭错误">×</button></div>}
        <div className={'st-workspace ' + (tab === 'data' ? 'st-data-layout' : '')}>
          {tab === 'workbench' ? <section className="st-intent-card">
            <div className="st-card-title"><span className="st-title-icon"><Glyph name="edit"/></span><h2>你想让机器人做什么？</h2><span>01</span></div>
            <p className="st-intent-hint">场景 + 操作对象 + 任务目标 + 采集数量</p>
            <div className="st-composer"><textarea aria-label="任务需求" placeholder={'例如：在分拣台上，把红色方块放到右侧托盘，随机摆位，采集 3 条示范。'} value={prompt} disabled={locked} maxLength={2000} onChange={e => setPrompt(e.target.value)} onKeyDown={e => { if ((e.ctrlKey || e.metaKey) && e.key === 'Enter' && !locked) void generate() }}/><div><span>{prompt.length}/2000 <kbd>Ctrl ↵</kbd></span><button disabled={locked || !connected || prompt.trim().length < 2} onClick={() => void generate()}>{busy === 'plan' ? '正在解析…' : plan ? '重新生成方案' : '生成任务方案'}<Glyph name="arrow" size={16}/></button></div></div>
            {!plan && !job ? <><div className="st-section-caption">也可以从一个想法开始 <span>TRY AN EXAMPLE</span></div><div className="st-examples">{examples.map(example => <button key={example.label} disabled={locked || !connected} onClick={() => void generate(example.text)}><span className={'st-example-icon ' + example.icon}><Glyph name={example.icon} size={22}/></span><span><b>{example.label}</b><small>{example.detail}</small></span><Glyph name="arrow" size={17}/></button>)}</div><div className="st-scope"><Glyph name="info" size={16}/><p>当前支持桌面场景、方块 / 圆柱 / 杯子，以及搬运放置和方块推移。复杂要求会提示需要补充的能力。</p></div></> : <div className="st-plan-result">
              <div className="st-plan-heading"><span>{blocked ? '需求需要调整' : '任务方案已就绪'}</span>{!blocked && <Glyph name="check" size={16}/>}</div>
              {blocked ? <ul className="st-unsupported">{spec.unsupported.map(item => <li key={item}>{item}</li>)}{spec.task === 'push' && spec.asset !== 'block' && <li>推移任务请选择方块。</li>}</ul> : <p className="st-plan-summary">在{scenes[spec.scene]}中，将{colors[spec.color]}{assets[spec.asset]}{spec.task === 'push' ? '沿桌面推到' : '搬运至'}{{ left: '左侧', center: '中央', right: '右侧' }[spec.goal]}{spec.task === 'push' ? '目标区' : '托盘'}，记录 {spec.episodes} 条示范。</p>}
              <div className="st-plan-fields"><label>场景<select aria-label="场景" value={spec.scene} disabled={locked} onChange={e => edit('scene', e.target.value)}>{Object.entries(scenes).map(([key, value]) => <option key={key} value={key}>{value}</option>)}</select></label><label>操作类型<select aria-label="操作类型" value={spec.task} disabled={locked} onChange={e => edit('task', e.target.value)}><option value="pick_place">抓取与放置</option><option value="push">桌面推移</option></select></label><label>操作对象<select aria-label="操作对象" value={spec.asset} disabled={locked} onChange={e => edit('asset', e.target.value)}>{Object.entries(assets).map(([key, value]) => <option key={key} value={key}>{value}</option>)}</select></label><label>目标位置<select aria-label="目标位置" value={spec.goal} disabled={locked} onChange={e => edit('goal', e.target.value)}><option value="left">左侧</option><option value="center">中央</option><option value="right">右侧</option></select></label><label>采集数量<input aria-label="采集数量" type="number" min={1} max={20} disabled={locked} value={spec.episodes} onChange={e => edit('episodes', Math.min(20, Math.max(1, Math.trunc(+e.target.value || 1))))}/></label><label>起始种子<input aria-label="起始种子" type="number" min={0} max={999999} disabled={locked} value={spec.seed} onChange={e => edit('seed', Math.min(999999, Math.max(0, Math.trunc(+e.target.value || 0))))}/></label></div>
              <div className="st-color-row"><span>物体颜色</span>{Object.entries(colorHex).map(([key, color]) => <button key={key} disabled={locked} aria-label={colors[key]} aria-pressed={spec.color === key} className={spec.color === key ? 'active' : ''} style={{ background: color }} onClick={() => edit('color', key)}>{spec.color === key && <Glyph name="check" size={12}/>}</button>)}</div>
              <label className="st-checkbox"><input type="checkbox" checked={spec.randomize} disabled={locked} onChange={e => edit('randomize', e.target.checked)}/>每条示范使用不同摆位<span>可复现的随机种子</span></label>
              {!!plan?.defaults.length && <details className="st-defaults"><summary>使用了 {plan.defaults.length} 项默认设置</summary>{plan.defaults.map(item => <p key={item}>{item}</p>)}</details>}
              <button className={'st-execute ' + (running ? 'stop' : '')} disabled={!!busy || restoring || (!running && (!connected || blocked || previewing || !!previewError))} onClick={() => void (running ? stop() : execute())}><Glyph name={running ? 'stop' : 'play'} size={17}/>{busy === 'execute' ? '正在启动…' : busy === 'stop' ? '正在停止…' : running ? '停止本次采集' : '执行任务并采集'}{!running && <span>{spec.episodes} 条示范</span>}</button>
              <p className="st-execute-note">{running ? '正在运行；本次参数已锁定。' : '确认方案后执行，画面与数据来自同一次仿真。'}</p>
            </div>}
          </section> : <section className="st-data-card"><div className="st-card-title"><span className="st-title-icon"><Glyph name="data"/></span><h2>本次数据集</h2><span>{completed.toString().padStart(2, '0')}</span></div><p className="st-intent-hint">{job ? statuses[job.status] : '从一次实验开始积累数据。'}</p><div className="st-data-summary"><div><b>{completed}<small> / {total}</small></b><span>已完成示范</span></div><div><b>{completed ? Math.round(successes / completed * 100) + '%' : '—'}</b><span>任务成功率</span></div></div><div className="st-episode-list">{rows.map(row => <button key={row.index} disabled={!manifest || replayLoading} className={mode === 'replay' && selected === row.index ? 'active' : ''} onClick={() => void replay(row.index)}><span className="st-episode-number">{String(row.index + 1).padStart(2, '0')}</span><span><b>Episode {String(row.index + 1).padStart(3, '0')}</b><small>{row.frames} 帧 · {row.duration_s.toFixed(2)} s · Seed {row.seed}</small></span><span className={row.success ? 'st-success' : 'st-failure'}>{row.success ? '成功' : '未成功'}</span><Glyph name="play" size={14}/></button>)}</div>{!rows.length && <div className="st-empty-data"><Glyph name="data" size={32}/><b>{running ? '等待第一条完整示范' : '还没有采集样本'}</b><p>执行任务后，每条示范都会显示在这里。</p><button onClick={() => setTab('workbench')}>回到任务工作台 <Glyph name="arrow" size={14}/></button></div>}<button className="st-manifest" disabled={!manifest} onClick={downloadManifest}><Glyph name="download" size={16}/>下载数据清单 JSON</button><p className="st-data-help">选择一条已完成示范，回放画面并检查对应的动作与状态。正在采集时，可先查看已有结果。</p></section>}

          <section className="st-view-card">
            <div className="st-view-header"><div><Glyph name="box" size={18}/><h2>{mode === 'replay' ? `Episode ${String(selected + 1).padStart(3, '0')} · 样本回放` : '场景与执行'}</h2><span className={'st-view-tag ' + (running && mode === 'live' ? 'live' : '')}>{mode === 'replay' ? 'RECORDED' : mode === 'live' ? running ? 'LIVE' : 'LAST FRAME' : 'SCENE PREVIEW'}</span></div><span>{scenes[currentSpec.scene]}</span></div>
            <div className="st-viewport">{image ? <img className="st-main-image" src={image} alt={`${camera === 'front' ? '前视' : '腕部'}相机${mode === 'replay' ? '回放' : '仿真'}画面`}/> : <div className="st-view-empty"><Glyph name="box" size={48}/><h3>{!connected ? '等待连接仿真环境' : blocked ? '先调整任务要求' : mode === 'live' ? '正在准备仿真环境' : '正在构建场景预览'}</h3><p>{!connected ? '启动后端后，点击右上角检查连接。' : blocked ? '修改左侧需求，再重新生成方案。' : '相机画面准备完成后会自动显示。'}</p></div>}<div className="st-view-top-overlay"><span><i/>{mode === 'replay' ? '样本数据' : 'MuJoCo 物理仿真'}</span><span>{camera === 'front' ? 'CAM 01 / FRONT' : 'CAM 02 / WRIST'}</span></div><div className="st-view-bottom-overlay"><span><i style={{ background: colorHex[currentSpec.color] }}/>{assets[currentSpec.asset]}<b>→</b>{isPush ? '桌面目标区' : '目标托盘'}</span><span>{time.toFixed(2)} <small>s</small> / {mode === 'replay' ? `FRAME ${frame + 1}` : `TICK ${mode === 'preview' ? 0 : snapshot.tick}`}</span></div>{previewing && mode === 'preview' && <div className="st-updating"><i/>正在更新场景</div>}</div>
            {previewError && mode === 'preview' && <div className="st-preview-error">场景预览未完成：{previewError}</div>}
            <div className="st-view-toolbar"><div className="st-camera-tabs"><button className={camera === 'front' ? 'active' : ''} onClick={() => setCamera('front')}><Glyph name="camera" size={15}/>前视相机</button><button className={camera === 'wrist' ? 'active' : ''} onClick={() => setCamera('wrist')}><Glyph name="camera" size={15}/>腕部相机</button></div><span>{mode === 'replay' ? '观测与动作逐帧对应' : mode === 'live' && running ? `第 ${(snapshot.episode_index ?? 0) + 1} / ${total} 条示范` : '两路相机 · 20 Hz 记录'}</span></div>
            {mode === 'replay' ? <div className="st-replay-controls"><button disabled={!trace || replayLoading} onClick={() => { if (trace && frame === trace.frame_time.length - 1) setFrame(0); setPlaying(value => !value) }} aria-label={playing ? '暂停回放' : '播放回放'}><Glyph name={playing ? 'stop' : 'play'} size={17}/></button><input aria-label="回放进度" type="range" min={0} max={Math.max(0, (trace?.frame_time.length ?? 1) - 1)} value={frame} disabled={!trace || replayLoading} onChange={e => { setPlaying(false); setFrame(+e.target.value) }}/><span>{replayLoading ? '读取中…' : `${frame + 1} / ${trace?.frame_time.length ?? 0}`}</span>{dataset && <a href={`/api/v1/datasets/${dataset}/episodes/${selected}`} title="下载此条示范 NPZ"><Glyph name="download" size={17}/></a>}</div> : <div className="st-stage"><div><span className="st-section-caption">{mode === 'live' && job ? statuses[job.status] : '任务执行流程'}</span><strong>{mode === 'live' ? snapshot.stage ?? '等待相机首帧' : isPush ? '接触物体，在桌面上完成推移' : '逐渐夹紧，搬运，再平稳放置'}</strong></div><div className="st-stage-steps">{steps.map((step, i) => <span key={step} title={step} className={mode === 'live' && step === snapshot.stage ? 'active' : ''}><i/>{isPush || i % 2 === 0 || i === steps.length - 1 ? step : ''}</span>)}</div></div>}
            <div className="st-telemetry"><div><span>末端位置 <small>m</small></span><strong>{eef ? eef.map(v => v.toFixed(3)).join(' / ') : '— / — / —'}</strong><small>X / Y / Z</small></div><div><span>夹爪开度</span><strong>{gripper === undefined ? '—' : Math.round(gripper * 100)}<small>%</small></strong><div className="st-grip-track"><i style={{ width: `${(gripper ?? 0) * 100}%` }}/></div></div><div><span>{mode === 'replay' ? '控制动作 · 归一化' : '当前接触'}</span><strong className="st-action-value">{mode === 'replay' && trace ? trace.action[frame].map(v => v.toFixed(2)).join(' / ') : mode === 'live' ? `${snapshot.metrics?.contacts ?? 0} 侧接触` : '等待执行'}</strong><small>{mode === 'replay' ? 'ΔX / ΔY / ΔZ / GRIP' : '物理接触反馈'}</small></div></div>
            {mode === 'replay' && trace && <div className="st-charts"><Sparkline values={trace.eef_pos.map(point => point[2] * 1000)} cursor={frame} label="末端高度" unit="mm"/><Sparkline values={trace.gripper.map(value => value * 100)} cursor={frame} label="夹爪开度" unit="%"/></div>}
          </section>
        </div>

        <section className="st-output"><div className="st-output-heading"><div><span className="st-kicker">COLLECTION OUTPUT</span><h2>从任务到数据，保留完整过程。</h2></div><div>{job && <button className="st-text-button" onClick={() => void toggleLogs()}>{showLogs ? '收起日志' : '执行日志'}</button>}<button className="st-output-link" onClick={() => setTab(tab === 'data' ? 'workbench' : 'data')}>{tab === 'data' ? '任务工作台' : '查看数据与回放'}<Glyph name="arrow" size={16}/></button></div></div><div className="st-output-stats"><div><Glyph name="data" size={20}/><span><b>{completed.toString().padStart(2, '0')}<small> / {total.toString().padStart(2, '0')}</small></b><span>已完成示范</span></span></div><div><Glyph name="grid" size={20}/><span><b>{rows.reduce((n, row) => n + row.frames, 0).toLocaleString()}<small>帧</small></b><span>已完成样本中的观测与动作</span></span></div><div><Glyph name="check" size={20}/><span><b>{completed ? Math.round(successes / completed * 100) : '—'}<small>{completed ? '%' : ''}</small></b><span>任务成功率 · {successes} 条成功</span></span></div><div className="st-output-progress"><span><b>{job ? statuses[job.status] : '等待第一次实验'}</b><small>{Math.round(completed / total * 100)}%</small></span><div><i style={{ width: `${completed / total * 100}%` }}/></div><small>{manifest ? 'NPZ 样本与 JSON 清单已保存' : running ? '每条示范完成后更新统计' : '相机图像 / 状态 / 动作 / 时间戳'}</small></div></div>{showLogs && <pre className="st-log">{logs.join('\n') || '暂无执行日志。'}</pre>}</section>
        <footer className="st-footer"><span><i/>需求 → 场景 → 任务 → 数据</span><span>专家控制 · 简化刚体模型 · 本地运行</span><Link to="/collect">打开基础采集 Demo ↗</Link></footer>
      </main>
    </div>
  </div>
}
