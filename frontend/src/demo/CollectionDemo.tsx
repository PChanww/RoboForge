import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api'
import { collectDemo, listDemoAssets, previewDemo, type DemoAsset } from './backend'
import { collectDucks, duckCatalog, previewDuck, defaultChassis } from './duckBackend'
import ModelGeneration from './ModelGeneration'
import ExperimentWorkspace from '../components/ExperimentWorkspace'
import './collection.css'

type Episode = { index: number; seed: number; frames: number; success: boolean; duration_s: number; asset?: string }
type Job = { id: string; status: string; params: { dataset_id?: string; plan?: { asset: string; seed: number }[]; provenance?: { chassis?: string } }; error?: string }
type Manifest = { episodes: Episode[]; summary: { episodes: number; frames: number; successes: number }; [key: string]: unknown }
type Preview = { episode_index?: number; tick: number; sim_time: number; images: Record<string, string | null>; completed: Episode[]; stage?: string; metrics?: { belt_speed_m_s?: number; belt_travel_m?: number } }
type SavedFrame = Awaited<ReturnType<typeof api.datasetPreview>>['frames'][number]
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
  const [unavailablePreviewJob, setUnavailablePreviewJob] = useState<string | null>(null)
  const [savedPreview, setSavedPreview] = useState<{ dataset: string; episode: number; frame: SavedFrame | null } | null>(null)
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
  const savedDataset = manifest && job?.params.dataset_id ? 'collection__' + job.params.dataset_id : null
  const savedEpisode = selected ?? (manifest ? manifest.episodes[manifest.episodes.length - 1]?.index : undefined)
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
        const missingRoute = response.status === 404
          && (await response.json().catch(() => null))?.detail === 'Not Found'
        if (missingRoute || response.status === 405 || response.status === 501) {
          if (!cancelled) setUnavailablePreviewJob(jobId)
          return
        }
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

  useEffect(() => {
    if (!savedDataset || savedEpisode === undefined) return
    let cancelled = false
    api.datasetPreview(savedDataset, savedEpisode)
      .then(({ frames }) => {
        if (!cancelled) setSavedPreview({ dataset: savedDataset, episode: savedEpisode, frame: frames[frames.length - 1] ?? null })
      })
      .catch(() => {
        if (!cancelled) setSavedPreview({ dataset: savedDataset, episode: savedEpisode, frame: null })
      })
    return () => { cancelled = true }
  }, [savedDataset, savedEpisode, previewRevision])

  const rows = manifest?.episodes ?? preview.completed
  const total = job ? job.params.plan?.length ?? count : count
  const frames = rows.reduce((n, row) => n + row.frames, 0)
  const progress = Math.min(100, rows.length / total * 100)
  const status = job ? statusNames[job.status] ?? job.status : '准备就绪'
  const detail = rows.find(row => row.index === selected)
  const currentAsset = assets.find(item => item.id === asset)
  const currentChassis = chassisAssets.find(item => item.id === chassis)
  const liveUnavailable = !!jobId && unavailablePreviewJob === jobId
  const savedResult = savedPreview?.dataset === savedDataset && savedPreview.episode === savedEpisode ? savedPreview : null
  const savedFrame = savedResult?.frame
  const liveImage = jobId ? preview.images[camera] : null
  const image = savedFrame ? savedFrame.images[camera] : liveImage ?? sceneImages[camera]
  const cameraName = camera === 'front' ? '前视相机' : '腕部相机'
  const cameraLabel = savedFrame && savedResult ? `采集回放 · 第 ${savedResult.episode + 1} 条`
    : liveImage ? running ? '采集中 · 实时画面' : '最近采集画面'
      : image ? running ? '采集中 · 显示采集前预览' : '采集前预览' : '相机预览'
  const cameraClock = savedFrame ? `${savedFrame.sim_time.toFixed(2)} s · 第 ${savedFrame.index + 1} 帧`
    : liveImage ? `${preview.sim_time.toFixed(2)} s · Tick ${preview.tick}` : null

  function emptyCameraMessage() {
    if (savedFrame) return `该样本没有${cameraName}画面`
    if (savedDataset) return savedResult ? '无法读取已保存的相机画面' : '正在读取采集画面…'
    if (running) return liveUnavailable ? '当前后端不提供实时画面，采集完成后可查看回放' : '正在等待采集画面…'
    if (job) return '本次采集暂无可查看的画面'
    if (!connected) return '等待连接后端'
    if (previewError) return '场景预览未完成'
    if (previewing) return '正在载入场景…'
    return assets.length ? '正在准备相机画面…' : '后端暂无可用的 Aholo 物品'
  }

  function clearLastRun() {
    setJob(null); setJobId(null); setManifest(null); setPreview(emptyPreview); setSavedPreview(null); setSelected(null); setShowLogs(false)
    localStorage.removeItem(storageKey)
  }

  async function start() {
    setError(''); setSelected(null)
    setBusy(true)
    try {
      if (!currentAsset) throw new Error('请先选择后端已有的 Aholo 物品。')
      const created = await (duckSorting ? collectDucks(currentAsset, count, seed, chassis) : collectDemo(currentAsset, count, seed))
      setManifest(null); setLogs([]); setPreview(emptyPreview); setSavedPreview(null); setShowLogs(false)
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

  return <ExperimentWorkspace
    title={duckSorting ? '传送带分拣' : '数据采集'}
    description={duckSorting ? '跟随移动目标，采集抓取和放置示范。' : '选择物体和随机种子，记录相机图像、关节状态与控制动作。'}
    actions={<>
      <button className="workspace-cta workspace-cta--quiet" onClick={checkConnection} disabled={checking}>
        {checking ? '检查连接…' : connected ? '刷新连接' : '重新连接'}
      </button>
      <button className="workspace-cta workspace-cta--quiet" onClick={() => datasetRef.current?.scrollIntoView({ behavior: 'smooth' })}>本次采集 ↓</button>
    </>}
  >
    <div className="rf-app">
      <div className="rf-main">
        {duckSorting && !connected && !checking && <p className="experiment-unavailable" role="status">
          当前后端未提供传送带分拣服务。 <Link to="/collect">前往机械臂数据采集 →</Link>
        </p>}
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
            <div className="rf-card-heading">
              <div><span className="rf-live-dot" />仿真画面</div>
              <span className="rf-scene-label">{cameraLabel}</span>
            </div>
            <div className="rf-camera-stage">
              {image ? <img src={`data:image/jpeg;base64,${image}`} alt={`${cameraName}画面`} /> : (
                <div className="rf-camera-empty">
                  <Icon name="cube" size={48} />
                  <strong>{emptyCameraMessage()}</strong>
                  {!jobId && previewError && <p>{previewError}</p>}
                </div>
              )}
              <div className="rf-camera-overlay"><span>{cameraName}</span>{cameraClock && <span>{cameraClock}</span>}</div>
            </div>
            <div className="rf-camera-controls"><div className="rf-mode-tabs"><button className={camera === 'front' ? 'active' : ''} onClick={() => setCamera('front')}>前视相机</button><button className={camera === 'wrist' ? 'active' : ''} onClick={() => setCamera('wrist')}>腕部相机</button></div><span>{running && preview.episode_index !== undefined ? `正在采集第 ${preview.episode_index + 1} / ${total} 次${duckSorting ? ' · ' + (preview.stage ?? '准备输送') + ' · 带速 ' + ((preview.metrics?.belt_speed_m_s ?? 0) * 100).toFixed(1) + ' cm/s' : ''}` : duckSorting ? (preview.stage ?? '准备输送') + ' · 带速 ' + ((preview.metrics?.belt_speed_m_s ?? 0) * 100).toFixed(1) + ' cm/s' : ''}</span></div>
            <div className="rf-task-strip"><span className="rf-task-icon"><Icon name="cube" size={22} /></span><div><strong>将{currentAsset?.name.split(' · ')[0] ?? '物品'}放入{duckSorting ? '机器人旁的收纳箱' : '目标托盘'}</strong></div><span className="rf-task-badge">专家示范</span></div>
            {currentAsset && <div className="rf-asset-source"><span>模型来源</span><strong>{duckSorting ? '物体' : '生成任务'} #{currentAsset.generation_id}{duckSorting && <> · 底盘 #{currentChassis?.generation_id ?? '—'}</>}</strong><p>模型高度 {Math.round(currentAsset.height_m * 100)} cm · 已导入后端</p><small>{duckSorting ? '生成模型的凸包参与碰撞；先抬高越过箱沿，再到箱内松爪。质量和摩擦尚未实物标定。' : '使用生成模型的网格外观；抓取碰撞采用圆柱代理，颜色为材质均值。'}</small></div>}
          </section>
          <section className="rf-settings-card"><h2>采集设置</h2>
            <label className="rf-field">抓取物体<select aria-label="抓取物体" value={asset} disabled={locked || !assets.length} onChange={e => { clearLastRun(); setAsset(e.target.value); if (duckSorting) localStorage.setItem('roboforge.duck-object', e.target.value) }}>{!assets.length && <option value="">{checking ? '加载物品…' : '暂无可用物品'}</option>}{assets.map(item => <option key={item.id} value={item.id}>{item.name}</option>)}</select></label>
            {duckSorting && <label className="rf-field">四轮底盘<select aria-label="四轮底盘" value={chassis} disabled={locked} onChange={e => {
              clearLastRun(); setChassis(e.target.value); localStorage.setItem('roboforge.duck-chassis', e.target.value)
            }}>{chassisAssets.map(item => <option key={item.id} value={item.id}>{item.name}</option>)}</select></label>}
            <label className="rf-field">采集次数<div className="rf-count-control"><button aria-label="减少采集次数" disabled={locked || count <= 1} onClick={() => setCount(n => n - 1)}>−</button><input aria-label="采集次数" type="number" min={1} max={20} value={count} disabled={locked} onChange={e => setCount(Math.max(1, Math.min(20, Math.trunc(Number(e.target.value) || 1))))} /><button aria-label="增加采集次数" disabled={locked || count >= 20} onClick={() => setCount(n => n + 1)}>+</button></div></label>
            <label className="rf-field">起始随机种子<input aria-label="起始随机种子" type="number" min={0} max={999999} disabled={locked} value={seed} onChange={e => { clearLastRun(); setSeed(Math.max(0, Math.min(999999, Math.trunc(Number(e.target.value) || 0)))) }} /><small>{duckSorting ? '每次递增，改变位置、朝向与初始倾斜；姿态随后按物理接触演化。' : '每次递增，用于生成可复现的物体摆位。'}</small></label>
            <button className={'rf-start ' + (running ? 'rf-stop' : '')} disabled={busy || (!running && (!connected || locked || !currentAsset || (!jobId && (previewing || !!previewError))))} onClick={running ? stop : start}><Icon name={running ? 'stop' : 'play'} size={18} />{busy ? '正在提交…' : running ? '停止本次采集' : '开始采集'}{!running && <Icon name="arrow" size={18} />}</button>
          </section>
        </div>
        {error && <div className="rf-error" role="alert"><strong>操作未完成</strong><span>{error}</span><button onClick={() => setError('')} aria-label="关闭错误提示">×</button></div>}
        <section className="rf-results" ref={datasetRef}>
          <div className="rf-result-header"><div><h2>本次采集<span className={'rf-status ' + (running ? 'running' : '')}><i />{status}</span></h2></div><div className="rf-result-actions">{savedDataset && <Link className="rf-download-link" to={`/datasets/${encodeURIComponent(savedDataset)}`}>打开数据集 →</Link>}{job && <button onClick={() => setShowLogs(v => !v)}>{showLogs ? '收起日志' : '查看日志'}</button>}<button disabled={!manifest} onClick={downloadManifest}><Icon name="download" size={15} />下载清单</button></div></div>
          <div className="rf-stats"><div><span>已完成 / 计划</span><strong>{rows.length.toString().padStart(2, '0')}<small>/ {total.toString().padStart(2, '0')}</small></strong></div><div><span>已记录帧数</span><strong>{frames.toLocaleString()}<small>帧</small></strong></div><div className="rf-progress-stat"><div><span>采集进度</span><b>{Math.round(progress)}%</b></div><div className="rf-progress-track"><span style={{ width: progress + '%' }} /></div></div></div>
          <div className="rf-table-wrap"><table className="rf-table"><thead><tr><th>样本编号</th><th>物体摆位种子</th><th>数据帧数</th><th>仿真时长</th><th>任务结果</th><th /></tr></thead><tbody>{rows.map(row => <tr key={row.index} className={selected === row.index ? 'selected' : ''}><td><span className="rf-episode-icon">▤</span>Episode {String(row.index + 1).padStart(3, '0')}</td><td>Seed {row.seed}</td><td>{row.frames} 帧</td><td>{row.duration_s.toFixed(2)} s</td><td><span className={'rf-outcome ' + (!row.success ? 'failed' : '')}>{row.success ? '✓ 成功' : '未成功'}</span></td><td><button aria-label={`查看 Episode ${row.index + 1} 详情`} onClick={() => setSelected(selected === row.index ? null : row.index)}>详情 ↗</button></td></tr>)}</tbody></table>{!rows.length && <div className="rf-empty"><Icon name="data" size={28} /><strong>{running ? '正在等待第一条完整示范' : '暂无采集记录'}</strong></div>}</div>
          {detail && <div className="rf-detail"><strong>Episode {detail.index + 1} · 采集记录</strong><p>随机种子 {detail.seed} · {detail.frames} 组观测与动作 · {detail.success ? '任务成功' : '任务未成功'} · episodes/ep{String(detail.index).padStart(3, '0')}.npz</p>{manifest && job?.params.dataset_id && <a className="rf-download-link" href={`/api/v1/datasets/collection__${encodeURIComponent(job.params.dataset_id)}/episodes/${detail.index}`}>下载此条数据 NPZ ↗</a>}</div>}
          {showLogs && <pre className="rf-logs">{logs.join('\n') || '正在等待后端日志…'}</pre>}
        </section>

      </div>
    </div>
  </ExperimentWorkspace>
}
