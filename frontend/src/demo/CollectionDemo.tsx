import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api'
import { collectDemo, listDemoAssets, previewDemo, type DemoAsset } from './backend'
import { collectDucks, duckCatalog, previewDuck, defaultChassis } from './duckBackend'
import ModelGeneration from './ModelGeneration'
import './collection.css'

type Episode = { index: number; seed: number; frames: number; success: boolean; duration_s: number; asset?: string }
type Job = { id: string; status: string; params: { dataset_id?: string; plan?: { asset: string; seed: number }[]; provenance?: { chassis?: string } }; error?: string }
type Manifest = { episodes: Episode[]; summary: { episodes: number; frames: number; successes: number }; [key: string]: unknown }
type Preview = { episode_index?: number; tick: number; sim_time: number; images: Record<string, string | null>; completed: Episode[]; stage?: string; metrics?: { belt_speed_m_s?: number; belt_travel_m?: number } }
const emptyPreview: Preview = { tick: 0, sim_time: 0, images: {}, completed: [] }
const statusNames: Record<string, string> = { queued: '等待执行', running: '采集中', succeeded: '采集完成', failed: '采集失败', cancelled: '已停止', interrupted: '已中断' }
const isActive = (job: Job | null) => !!job && ['running', 'queued'].includes(job.status)
const message = (error: unknown) => error instanceof Error ? error.message : String(error)

function Icon({ name, size = 20 }: { name: 'cube' | 'play' | 'grid' | 'data' | 'arrow' | 'check' | 'download' | 'stop'; size?: number }) {
  const paths = {
    cube: <><path d="m12 3 9 5v8l-9 5-9-5V8l9-5Z" /><path d="m3 8 9 5 9-5M12 13v8M7.5 5.5l9 5" /></>,
    play: <path d="m9 5 11 7-11 7V5Z" />,
    grid: <><rect x="4" y="4" width="6" height="6" rx="1" /><rect x="14" y="4" width="6" height="6" rx="1" /><rect x="4" y="14" width="6" height="6" rx="1" /><rect x="14" y="14" width="6" height="6" rx="1" /></>,
    data: <><ellipse cx="12" cy="6" rx="8" ry="3" /><path d="M4 6v6c0 4 16 4 16 0V6M4 12v6c0 4 16 4 16 0v-6" /></>,
    arrow: <path d="M5 12h14m-5-5 5 5-5 5" />,
    check: <path d="m5 12 4 4L19 6" />,
    download: <path d="M12 3v12m-5-5 5 5 5-5M4 16v5h16v-5" />,
    stop: <rect x="6" y="6" width="12" height="12" rx="1" />,
  }
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name]}</svg>
}

export default function CollectionDemo({ duckSorting = false }: { duckSorting?: boolean }) {
  const storageKey = duckSorting ? 'roboforge.duck-demo.v2.job' : 'roboforge.aholo-demo.job'
  const [connected, setConnected] = useState(false)
  const [checking, setChecking] = useState(false)
  const [assets, setAssets] = useState<DemoAsset[]>([])
  const [asset, setAsset] = useState(() => duckSorting ? localStorage.getItem('roboforge.duck-object') ?? '' : '')
  const [chassis, setChassis] = useState(() => localStorage.getItem('roboforge.duck-chassis') ?? defaultChassis)
  const [chassisAssets, setChassisAssets] = useState<DemoAsset[]>([])
  const [count, setCount] = useState(1)
  const [seed, setSeed] = useState(0)
  const [job, setJob] = useState<Job | null>(null)
  const [jobId, setJobId] = useState<string | null>(() => localStorage.getItem(storageKey))
  const [preview, setPreview] = useState<Preview>(emptyPreview)
  const [sceneImages, setSceneImages] = useState<Record<string, string>>({})
  const [previewing, setPreviewing] = useState(false)
  const [previewError, setPreviewError] = useState('')
  const [previewRevision, setPreviewRevision] = useState(0)
  const previewStopped = useRef(false)
  const [camera, setCamera] = useState('front')
  const [logs, setLogs] = useState<string[]>([])
  const [manifest, setManifest] = useState<Manifest | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [selected, setSelected] = useState<number | null>(null)
  const [showLogs, setShowLogs] = useState(false)
  const datasetRef = useRef<HTMLDivElement>(null)
  const mounted = useRef(true)
  const running = isActive(job)
  const locked = running || busy || (!!jobId && !job)

  async function checkConnection() {
    setChecking(true)
    try {
      const data = duckSorting ? await duckCatalog() : null
      const catalog = data ? data.objects : await listDemoAssets()
      if (mounted.current) {
        if (data) {
          setChassisAssets(data.chassis)
          setChassis(current => data.chassis.some(item => item.id === current) ? current : defaultChassis)
        }
        setConnected(true); setAssets(catalog); setPreviewRevision(n => n + 1)
        setAsset(current => catalog.some(item => item.id === current) ? current : catalog[0]?.id || '')
      }
    }
    catch { if (mounted.current) setConnected(false) }
    finally { if (mounted.current) setChecking(false) }
  }
  useEffect(() => {
    mounted.current = true
    void checkConnection()
    return () => { mounted.current = false }
  }, [])

  useEffect(() => {
    if (!connected || !asset || jobId) return
    let cancelled = false
    setPreviewing(true); setPreviewError(''); setSceneImages({})
    const timer = setTimeout(async () => {
      try {
        const images = await (duckSorting ? previewDuck(asset, seed, chassis) : previewDemo(asset, seed))
        if (!cancelled) setSceneImages(images)
      } catch (err) { if (!cancelled) setPreviewError(message(err)) }
      finally { if (!cancelled) setPreviewing(false) }
    }, 200)
    return () => { cancelled = true; clearTimeout(timer) }
  }, [connected, asset, chassis, seed, jobId, previewRevision])

  useEffect(() => {
    if (!jobId) return
    let cancelled = false
    previewStopped.current = false
    let timer: ReturnType<typeof setTimeout>
    const poll = async () => {
      try {
        const [next, log] = await Promise.all([api.job(jobId), api.jobLogs(jobId)])
        if (cancelled) return
        setJob(next); setLogs(log.lines); setConnected(true); setError('')
        if (next.params.plan?.length) {
          setAsset(next.params.plan[0].asset); setSeed(next.params.plan[0].seed); setCount(next.params.plan.length)
          if (next.params.provenance?.chassis) setChassis(next.params.provenance.chassis)
        }
        previewStopped.current = !isActive(next)
        if (next.status === 'succeeded' && next.params.dataset_id) {
          const result = await api.dataset('collection__' + next.params.dataset_id)
          if (!cancelled) setManifest(result as Manifest)
        }
        if (['failed', 'interrupted'].includes(next.status)) {
          setError(next.error || '采集未完成，请查看作业日志。'); setShowLogs(true)
        }
        if (!cancelled && ['queued', 'running'].includes(next.status)) timer = setTimeout(poll, 1000)
      } catch (err) {
        if (!cancelled) {
          if (message(err).startsWith('404:')) {
            localStorage.removeItem(storageKey); setJobId(null); setJob(null)
          }
          setError('获取采集状态失败：' + message(err)); timer = setTimeout(poll, 3000)
        }
      }
    }
    void poll()
    return () => { cancelled = true; clearTimeout(timer) }
  }, [jobId])

  useEffect(() => {
    if (!jobId) return
    let cancelled = false
    let timer: ReturnType<typeof setTimeout>
    const controller = new AbortController()
    const pollPreview = async () => {
      const started = performance.now()
      // Fetch a final snapshot even if the job finishes while this request runs.
      const finalSnapshot = previewStopped.current
      try {
        const response = await fetch(`/api/v1/jobs/${encodeURIComponent(jobId)}/preview`, {
          cache: 'no-store', signal: controller.signal,
        })
        if (!response.ok) throw new Error(`相机读取失败 (${response.status})`)
        const snapshot: Preview = await response.json()
        if (cancelled) return
        setPreview(previous => previous.tick === snapshot.tick
          && previous.episode_index === snapshot.episode_index
          && previous.completed.length === snapshot.completed.length ? previous : snapshot)
        if (!finalSnapshot) timer = setTimeout(pollPreview, Math.max(0, 16 - (performance.now() - started)))
      } catch {
        if (!cancelled && !finalSnapshot) timer = setTimeout(pollPreview, 1000)
      }
    }
    void pollPreview()
    return () => { cancelled = true; clearTimeout(timer); controller.abort() }
  }, [jobId])

  const rows = manifest?.episodes ?? preview.completed
  const total = job ? job.params.plan?.length ?? count : count
  const frames = rows.reduce((n, row) => n + row.frames, 0)
  const successes = rows.filter(row => row.success).length
  const progress = Math.min(100, rows.length / total * 100)
  const status = job ? statusNames[job.status] ?? job.status : '准备就绪'
  const detail = rows.find(row => row.index === selected)
  const currentAsset = assets.find(item => item.id === asset)
  const currentChassis = chassisAssets.find(item => item.id === chassis)
  const image = jobId ? preview.images[camera] : sceneImages[camera]

  function clearLastRun() {
    setJob(null); setJobId(null); setManifest(null); setPreview(emptyPreview); setSelected(null); setShowLogs(false)
    localStorage.removeItem(storageKey)
  }

  async function start() {
    setError(''); setSelected(null)
    setBusy(true)
    try {
      if (!currentAsset) throw new Error('请先选择后端已有的 Aholo 物品。')
      const created = await (duckSorting ? collectDucks(currentAsset, count, seed, chassis) : collectDemo(currentAsset, count, seed))
      setManifest(null); setLogs([]); setPreview(emptyPreview); setShowLogs(false)
      setJob(created); setJobId(created.id); localStorage.setItem(storageKey, created.id); setConnected(true)
    } catch (err) { setError(message(err)) }
    finally { setBusy(false) }
  }
  async function stop() {
    if (!job) return
    setBusy(true)
    try { setJob(await api.stopJob(job.id)) } catch (err) { setError(message(err)) }
    finally { setBusy(false) }
  }
  function downloadManifest() {
    if (!manifest) return
    const url = URL.createObjectURL(new Blob([JSON.stringify(manifest, null, 2)], { type: 'application/json' }))
    const link = document.createElement('a'); link.href = url; link.download = `${job?.params.dataset_id ?? 'collection'}-manifest.json`; link.click()
    setTimeout(() => URL.revokeObjectURL(url), 1000)
  }

  return <div className="rf-app">
    <aside className="rf-sidebar">
      <Link to="/" className="rf-brand"><span className="rf-brand-symbol"><Icon name="cube" size={24} /></span>RoboForge<span className="rf-beta">LAB</span></Link>
      <div className="rf-workspace"><span className="rf-workspace-avatar">R</span><div>本地工作空间<small>机器人学习实验室</small></div><span className="rf-workspace-dots">···</span></div>
      <span className="rf-nav-caption">工作空间</span>
      <nav className="rf-nav"><button className="rf-nav-active" onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}><Icon name="grid" />采集工作台<span>01</span></button><button onClick={() => datasetRef.current?.scrollIntoView({ behavior: 'smooth' })}><Icon name="data" />本次数据集<span>{rows.length.toString().padStart(2, '0')}</span></button></nav>
      <div className="rf-sidebar-note"><span className="rf-note-mark">↗</span><strong>从一次抓取开始</strong><p>把机器人的每次观察与动作，变成下一次学习的起点。</p><Link to="/projects">打开完整平台 <Icon name="arrow" size={15} /></Link></div>
      <div className="rf-sidebar-bottom"><span className="rf-avatar">RF</span><div>RoboForge Demo<small>数据采集 · v0.1</small></div></div>
    </aside>
    <div className="rf-body">
      <header className="rf-topbar"><div>工作空间<span>/</span><strong>采集工作台</strong></div><button className={'rf-connection ' + (connected ? 'connected' : '')} onClick={checkConnection} disabled={checking}><i />{checking ? '检查连接…' : connected ? '后端已连接' : '后端未连接'}<span>↻</span></button></header>
      <main className="rf-main">
        <div className="rf-page-title"><div><span className="rf-eyebrow">AHOLO ASSETS / ROBOFORGE SIMULATION</span><h1>{duckSorting ? '传送带不停，机器人跟随抓取' : '让生成的物品，在仿真里动起来'}<span>。</span></h1><p>{duckSorting ? '描述并生成你的模型，跟随移动目标夹取，越过箱沿后平稳入箱。' : '选择后端已有的 Aholo 物品，观看机械臂操作，记录一组真实的仿真数据。'}</p></div><span className="rf-version">{duckSorting ? '不停带跟随 / 003' : '后端效果演示 / 001'}</span></div>
        {duckSorting && <Link className="rf-scenario-entry" to="/create"><span><strong>描述完整场景，生成后直接仿真</strong><small>环境 · 机器人 · 物品 · 工作状态</small></span><span>创建新场景 ↗</span></Link>}
        {duckSorting && <Link className="rf-scenario-entry" to="/drive"><span><strong>驾驶四轮移动机器人</strong><small>真实轮地接触 · 手动驾驶 · 示例路线 · 运动数据</small></span><span>让机器人动起来 ↗</span></Link>}
        <div className="rf-mode-bar"><span className="rf-task-badge">Aholo 模型 × MuJoCo 仿真</span><p>{duckSorting ? '不停带输送 → 跟随夹取 → 抬高越沿 → 箱内松爪；底盘停靠作业。' : '复用后端已经生成的模型，直接预览与采集。'}</p></div>
        {duckSorting && <ModelGeneration locked={locked} onUse={async (id, target) => {
          const data = await duckCatalog()
          const list = target === 'object' ? data.objects : data.chassis
          if (!list.some(item => item.id === id)) throw new Error('模型尚未出现在可用目录；请查看完整生成任务的导入状态和尺寸。')
          clearLastRun(); setAssets(data.objects); setChassisAssets(data.chassis)
          if (target === 'object') { setAsset(id); localStorage.setItem('roboforge.duck-object', id) }
          else { setChassis(id); localStorage.setItem('roboforge.duck-chassis', id) }
          setPreviewRevision(n => n + 1)
        }} />}
        <div className="rf-workbench">
          <section className="rf-scene-card">
            <div className="rf-card-heading"><div><span className="rf-live-dot" />{duckSorting ? '传送带分拣' : '抓取与放置'} <span className="rf-tag">{duckSorting ? 'DUCK SORTING' : 'PICK & PLACE'}</span></div><span className="rf-scene-label">{running ? '采集中 · 实时画面' : job ? '本次采集的最后一帧' : '初始场景 · 后端渲染'}</span></div>
            <div className="rf-camera-stage">{image ? <img src={`data:image/jpeg;base64,${image}`} alt={camera === 'front' ? '前视相机采集画面' : '腕部相机采集画面'} /> : <div className="rf-camera-empty"><Icon name="cube" size={48} /><strong>{!connected ? '等待连接后端' : previewError ? '场景预览未完成' : running ? '正在构建仿真环境…' : previewing ? '正在载入 Aholo 模型…' : !assets.length ? '后端暂无可用的 Aholo 物品' : '正在准备相机画面…'}</strong><p>{previewError || '画面由 RoboForge 后端的 MuJoCo 相机直接输出。'}</p></div>}<div className="rf-camera-overlay"><span>{camera === 'front' ? 'FRONT / 前视相机' : 'WRIST / 腕部相机'}</span><span>{(jobId ? preview.sim_time : 0).toFixed(2)} s · Tick {jobId ? preview.tick : 0}</span></div></div>
            <div className="rf-camera-controls"><div className="rf-mode-tabs"><button className={camera === 'front' ? 'active' : ''} onClick={() => setCamera('front')}>前视相机</button><button className={camera === 'wrist' ? 'active' : ''} onClick={() => setCamera('wrist')}>腕部相机</button></div><span>{running && preview.episode_index !== undefined ? `正在采集第 ${preview.episode_index + 1} / ${total} 次${duckSorting ? ' · ' + (preview.stage ?? '准备输送') + ' · 带速 ' + ((preview.metrics?.belt_speed_m_s ?? 0) * 100).toFixed(1) + ' cm/s' : ''}` : duckSorting ? (preview.stage ?? '准备输送') + ' · 带速 ' + ((preview.metrics?.belt_speed_m_s ?? 0) * 100).toFixed(1) + ' cm/s' : '画面与数据来自同一个仿真环境'}</span></div>
            <div className="rf-task-strip"><span className="rf-task-icon"><Icon name="cube" size={22} /></span><div><strong>将{currentAsset?.name.split(' · ')[0] ?? '物品'}放入{duckSorting ? '机器人旁的收纳箱' : '目标托盘'}</strong><p>{duckSorting ? '跟随目标' : '接近目标'} <b>→</b> 夹取抬升 <b>→</b> {duckSorting ? '松爪入箱' : '搬运放置'}</p></div><span className="rf-task-badge">专家示范</span></div>
            {currentAsset && <div className="rf-asset-source"><span>AHOLO / LUX3D</span><strong>{duckSorting ? '物体' : '生成任务'} #{currentAsset.generation_id}{duckSorting && <> · 底盘 #{currentChassis?.generation_id ?? '—'}</>}</strong><p>模型高度 {Math.round(currentAsset.height_m * 100)} cm · 已导入后端</p><small>{duckSorting ? '生成模型的凸包参与碰撞；先抬高越过箱沿，再到箱内松爪。质量和摩擦尚未实物标定。' : '使用生成模型的网格外观；抓取碰撞采用圆柱代理，颜色为材质均值。'}</small></div>}
          </section>
          <section className="rf-settings-card"><div className="rf-section-label"><span>采集设置</span><span className="rf-step-number">STEP 01</span></div><h2>准备一组示范</h2><p className="rf-settings-intro">{duckSorting ? '改变位置与初始姿态，采集可复现的分拣示范。' : '先从简单、可重复的任务开始。'}</p>
            <label className="rf-field">后端 Aholo 物品<select aria-label="后端 Aholo 物品" value={asset} disabled={locked || !assets.length} onChange={e => { clearLastRun(); setAsset(e.target.value); if (duckSorting) localStorage.setItem('roboforge.duck-object', e.target.value) }}>{!assets.length && <option value="">等待载入物品…</option>}{assets.map(item => <option key={item.id} value={item.id}>{item.name}</option>)}</select></label>
            {duckSorting && <label className="rf-field">四轮底盘<select aria-label="四轮底盘" value={chassis} disabled={locked} onChange={e => {
              clearLastRun(); setChassis(e.target.value); localStorage.setItem('roboforge.duck-chassis', e.target.value)
            }}>{chassisAssets.map(item => <option key={item.id} value={item.id}>{item.name}</option>)}</select></label>}
            <label className="rf-field">采集次数 <span>Episodes</span><div className="rf-count-control"><button aria-label="减少采集次数" disabled={locked || count <= 1} onClick={() => setCount(n => n - 1)}>−</button><input aria-label="采集次数" type="number" min={1} max={20} value={count} disabled={locked} onChange={e => setCount(Math.max(1, Math.min(20, Math.trunc(Number(e.target.value) || 1))))} /><button aria-label="增加采集次数" disabled={locked || count >= 20} onClick={() => setCount(n => n + 1)}>+</button></div></label>
            <label className="rf-field">起始随机种子<input aria-label="起始随机种子" type="number" min={0} max={999999} disabled={locked} value={seed} onChange={e => { clearLastRun(); setSeed(Math.max(0, Math.min(999999, Math.trunc(Number(e.target.value) || 0)))) }} /><small>{duckSorting ? '每次递增，改变位置、朝向与初始倾斜；姿态随后按物理接触演化。' : '每次递增，用于生成可复现的物体摆位。'}</small></label>
            <div className="rf-recording"><span>记录内容</span><div><span><Icon name="check" size={13} />相机图像</span><span><Icon name="check" size={13} />关节状态</span><span><Icon name="check" size={13} />控制动作</span><span><Icon name="check" size={13} />时间戳</span>{duckSorting && <><span><Icon name="check" size={13} />物体姿态</span><span><Icon name="check" size={13} />传送带状态</span></>}</div></div>
            <button className={'rf-start ' + (running ? 'rf-stop' : '')} disabled={busy || (!running && (!connected || locked || !currentAsset || (!jobId && (previewing || !!previewError))))} onClick={running ? stop : start}><Icon name={running ? 'stop' : 'play'} size={18} />{busy ? '正在提交…' : running ? '停止本次采集' : '开始采集'}{!running && <Icon name="arrow" size={18} />}</button>
            <p className="rf-button-hint">{connected ? duckSorting ? '真实接触仿真 · 状态反馈专家控制器' : 'Aholo 已有资产 · 后端专家控制器' : '请先启动后端，然后点击右上角检查连接'}</p>
          </section>
        </div>
        {error && <div className="rf-error" role="alert"><strong>操作未完成</strong><span>{error}</span><button onClick={() => setError('')} aria-label="关闭错误提示">×</button></div>}
        <section className="rf-results" ref={datasetRef}>
          <div className="rf-result-header"><div><span className="rf-eyebrow">COLLECTION OUTPUT</span><h2>本次采集<span className={'rf-status ' + (running ? 'running' : '')}><i />{status}</span></h2></div><div className="rf-result-actions">{job && <button onClick={() => setShowLogs(v => !v)}>{showLogs ? '收起日志' : '查看日志'}</button>}<button disabled={!manifest} onClick={downloadManifest}><Icon name="download" size={15} />下载清单</button></div></div>
          <div className="rf-stats"><div><span>已完成 / 计划</span><strong>{rows.length.toString().padStart(2, '0')}<small>/ {total.toString().padStart(2, '0')}</small></strong></div><div><span>已记录帧数</span><strong>{frames.toLocaleString()}<small>帧</small></strong></div><div><span>成功率</span><strong>{rows.length ? Math.round(successes / rows.length * 100) : '—'}<small>{rows.length ? '%' : ''}</small></strong></div><div className="rf-progress-stat"><div><span>采集进度</span><b>{Math.round(progress)}%</b></div><div className="rf-progress-track"><span style={{ width: progress + '%' }} /></div><small>{job?.params.dataset_id ?? '等待创建采集作业'}</small></div></div>
          <div className="rf-table-wrap"><table className="rf-table"><thead><tr><th>样本编号</th><th>物体摆位种子</th><th>数据帧数</th><th>仿真时长</th><th>任务结果</th><th /></tr></thead><tbody>{rows.map(row => <tr key={row.index} className={selected === row.index ? 'selected' : ''}><td><span className="rf-episode-icon">▤</span>Episode {String(row.index + 1).padStart(3, '0')}</td><td>Seed {row.seed}</td><td>{row.frames} 帧</td><td>{row.duration_s.toFixed(2)} s</td><td><span className={'rf-outcome ' + (!row.success ? 'failed' : '')}>{row.success ? '✓ 成功' : '未成功'}</span></td><td><button aria-label={`查看 Episode ${row.index + 1} 详情`} onClick={() => setSelected(selected === row.index ? null : row.index)}>详情 ↗</button></td></tr>)}</tbody></table>{!rows.length && <div className="rf-empty"><Icon name="data" size={28} /><strong>{running ? '正在等待第一条完整示范' : '第一条机器人示范，从这里开始'}</strong><p>{running ? '每完成一次操作，样本就会显示在这里。' : '设置采集次数，点击上方按钮开始。'}</p></div>}</div>
          {detail && <div className="rf-detail"><strong>Episode {detail.index + 1} · 采集记录</strong><p>随机种子 {detail.seed} · {detail.frames} 组观测与动作 · {detail.success ? '任务成功' : '任务未成功'} · episodes/ep{String(detail.index).padStart(3, '0')}.npz</p>{manifest && job?.params.dataset_id && <a className="rf-download-link" href={`/api/v1/datasets/collection__${encodeURIComponent(job.params.dataset_id)}/episodes/${detail.index}`}>下载此条数据 NPZ ↗</a>}</div>}
          {manifest && <div className="rf-detail"><strong>数据已保存</strong><p>datasets/collections/{job?.params.dataset_id}/</p></div>}
          {showLogs && <pre className="rf-logs">{logs.join('\n') || '正在等待后端日志…'}</pre>}
        </section>
        <footer className="rf-footer"><span><span className="rf-footer-mark">◇</span> ROBOFORGE <span>让机器人在数据中学习。</span></span><span>本地仿真采集 · NPZ + JSON manifest</span></footer>
      </main>
    </div>
  </div>
}
