import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import './mobile.css'

type Asset = { id: string; name: string }
type Capabilities = { assets: Asset[]; default_asset: string; limits: { linear_m_s: number; angular_rad_s: number }; notes: string[] }
type Receipt = { chassis_asset: string; client_request_id: string; id?: string }
type Snapshot = {
  id: string; status: 'ready' | 'running' | 'failed' | 'closed'; epoch: number; sim_time: number; tick: number
  pose: { x: number; y: number; z: number; yaw: number }
  velocity: { linear_m_s: number; angular_rad_s: number }
  command: { linear_m_s: number; angular_rad_s: number }; wheel_velocity_rad_s: number[]
  images: { front?: string; onboard?: string }; trajectory: { x: number; y: number }[]; distance_m: number
  recording: { active: boolean; frames: number }
  dataset?: { id: string; frames: number; manifest_url: string; npz_url: string }
  control_mode: 'manual' | 'route'; route_status?: string; error?: string; asset?: Asset
}
type Direction = 'forward' | 'backward' | 'left' | 'right'
const RECEIPT_KEY = 'roboforge.mobile.receipt.v1'
const KEY_DIRECTION: Record<string, Direction> = { KeyW: 'forward', ArrowUp: 'forward', KeyS: 'backward', ArrowDown: 'backward', KeyA: 'left', ArrowLeft: 'left', KeyD: 'right', ArrowRight: 'right' }
const STATUS = { ready: '仿真就绪', running: '仿真运行中', closed: '会话已结束', failed: '仿真需要处理' }
const ROUTE_STATUS: Record<string, string> = { running: '正在依次驶向固定路点', completed: '示例路线已完成', stopped: '示例路线已停止', timeout: '示例路线超时，机器人已停车', connection_lost: '画面连接中断，示例路线已停止' }
function saved(): Receipt | null { try { return JSON.parse(localStorage.getItem(RECEIPT_KEY) ?? 'null') } catch { return null } }
class HttpError extends Error { constructor(public status: number, message: string) { super(message) } }
async function request<T>(path: string, body?: unknown, method?: string, keepalive = false): Promise<T> {
  const response = await fetch('/api/v1/mobile/' + path, { method: method ?? (body === undefined ? 'GET' : 'POST'), keepalive,
    ...(body === undefined ? {} : { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }) })
  const value = await response.json().catch(() => null)
  if (!response.ok) { const message = value?.detail ?? value?.error; throw new HttpError(response.status, typeof message === 'string' ? message : `请求失败 (${response.status})`) }
  return value as T
}
function message(error: unknown) { return error instanceof Error ? error.message : String(error) }
function inputTarget(target: EventTarget | null) { return target instanceof HTMLElement && (['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName) || target.isContentEditable) }
function imageSource(value?: string) { return value ? value.startsWith('data:') ? value : 'data:image/jpeg;base64,' + value : undefined }

export default function MobileDemo() {
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null)
  const [receipt, setReceipt] = useState<Receipt | null>(saved)
  const [asset, setAsset] = useState(() => saved()?.chassis_asset ?? '')
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null)
  const [camera, setCamera] = useState<'front' | 'onboard'>('front')
  const [speed, setSpeed] = useState(.12)
  const [turnSpeed, setTurnSpeed] = useState(1.2)
  const [held, setHeld] = useState<Set<Direction>>(new Set())
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')
  const [online, setOnline] = useState(false)
  const [retry, setRetry] = useState(0)
  const pressed = useRef(new Map<string, Direction>())
  const sequence = useRef(Date.now() * 1000)
  const current = useRef({ id: receipt?.id, active: false, speed, turnSpeed })
  const mounted = useRef(true)
  const sessionId = receipt?.id
  const active = !!snapshot && ['ready', 'running'].includes(snapshot.status)
  const driving = active && online && !busy
  current.current = { id: sessionId, active: driving, speed, turnSpeed }

  function apply(next: Snapshot, force = false) {
    if (!mounted.current) return
    setSnapshot(previous => {
      if (previous?.id === next.id) {
        const before = previous.epoch ?? 0, after = next.epoch ?? 0
        if (after < before || (after === before && !force && previous.tick > next.tick)) return previous
      }
      return next
    })
  }
  function persist(next: Receipt | null) { if (next) localStorage.setItem(RECEIPT_KEY, JSON.stringify(next)); else localStorage.removeItem(RECEIPT_KEY); setReceipt(next) }
  function nextSequence() { sequence.current = Math.max(sequence.current + 1, Date.now() * 1000); return sequence.current }
  function refreshHeld() { setHeld(new Set(pressed.current.values())) }
  async function brake(keepalive = false) {
    pressed.current.clear(); if (mounted.current) refreshHeld()
    const id = current.current.id
    if (!id) return
    // A newer zero command prevents an older in-flight heartbeat from restarting movement.
    await Promise.allSettled([
      request(`sessions/${id}/command`, { linear_m_s: 0, angular_rad_s: 0, seq: nextSequence() }, undefined, keepalive),
      request(`sessions/${id}/stop`, { seq: nextSequence() }, undefined, keepalive),
    ])
  }
  function emit() {
    const live = current.current
    if (!live.id || !live.active || !pressed.current.size) return
    const values = new Set(pressed.current.values())
    const forward = Number(values.has('forward')) - Number(values.has('backward'))
    const turn = Number(values.has('left')) - Number(values.has('right'))
    void request(`sessions/${live.id}/command`, { linear_m_s: forward * live.speed, angular_rad_s: turn * live.turnSpeed, seq: nextSequence() }).catch(err => {
      if (!mounted.current) return
      setError('驾驶指令未送达，已请求停车：' + message(err)); setOnline(false); void brake()
    })
  }
  function press(source: string, direction: Direction) { if (!current.current.active) return; pressed.current.set(source, direction); refreshHeld(); emit() }
  function release(source: string) { if (!pressed.current.delete(source)) return; refreshHeld(); if (pressed.current.size) emit(); else void brake() }

  useEffect(() => {
    mounted.current = true
    let disposed = false
    request<Capabilities>('capabilities').then(value => { if (!disposed) { setCapabilities(value); setAsset(previous => previous || value.default_asset); setError('') } }).catch(err => { if (!disposed) setError('无法连接移动仿真服务：' + message(err)) })
    return () => { disposed = true }
  }, [retry])

  useEffect(() => {
    if (!sessionId) return
    let disposed = false
    let timer: ReturnType<typeof setTimeout>
    let first = true
    const poll = async () => {
      const started = performance.now()
      try {
        const result = await request<Snapshot>(`sessions/${sessionId}`)
        if (disposed) return
        apply(result); setOnline(true)
        if (first) { first = false; setError(''); await brake() }
        if (['ready', 'running'].includes(result.status)) timer = setTimeout(poll, Math.max(15, 65 - (performance.now() - started)))
        else { persist(null); setOnline(false); if (result.error) setError(result.error) }
      } catch (err) {
        if (disposed) return
        setOnline(false); void brake()
        if (err instanceof HttpError && [404, 410].includes(err.status)) {
          setError('上次会话已经结束。已有下载结果会保留，可以启动新会话。')
          setSnapshot(previous => previous ? { ...previous, status: 'closed' } : null)
          persist(null)
        } else { setError('画面连接中断，正在恢复；底盘会自动停车。' + message(err)); timer = setTimeout(poll, 1200) }
      }
    }
    void poll()
    return () => { disposed = true; clearTimeout(timer); void brake(true) }
  }, [sessionId, retry])

  useEffect(() => {
    const down = (event: KeyboardEvent) => {
      if (inputTarget(event.target)) return
      if (event.code === 'Escape' || event.code === 'Space') { if (current.current.id) { event.preventDefault(); void brake() } return }
      const direction = KEY_DIRECTION[event.code]
      if (direction && current.current.active) { event.preventDefault(); if (!event.repeat) press('key:' + event.code, direction) }
    }
    const up = (event: KeyboardEvent) => { if (KEY_DIRECTION[event.code]) { if (pressed.current.has('key:' + event.code)) event.preventDefault(); release('key:' + event.code) } }
    const loseFocus = () => { void brake(true) }
    const hidden = () => { if (document.hidden) loseFocus() }
    const pointerUp = () => { for (const source of [...pressed.current.keys()]) if (source.startsWith('pointer:')) release(source) }
    const heartbeat = setInterval(emit, 150)
    window.addEventListener('keydown', down); window.addEventListener('keyup', up)
    window.addEventListener('blur', loseFocus); window.addEventListener('pagehide', loseFocus)
    document.addEventListener('visibilitychange', hidden); window.addEventListener('pointerup', pointerUp); window.addEventListener('pointercancel', pointerUp)
    return () => {
      clearInterval(heartbeat); window.removeEventListener('keydown', down); window.removeEventListener('keyup', up)
      window.removeEventListener('blur', loseFocus); window.removeEventListener('pagehide', loseFocus); document.removeEventListener('visibilitychange', hidden)
      window.removeEventListener('pointerup', pointerUp); window.removeEventListener('pointercancel', pointerUp)
      mounted.current = false; void brake(true)
    }
  }, [])

  async function start() {
    if (busy || !asset) return
    setBusy('start'); setError('')
    const next = receipt && !receipt.id ? receipt : { chassis_asset: asset, client_request_id: crypto.randomUUID() }
    persist(next)
    try {
      const result = await request<Snapshot>('sessions', next)
      persist({ ...next, id: result.id }); apply(result, true); setOnline(true)
    } catch (err) {
      setError('启动未完成：' + message(err) + '。恢复会使用同一请求，不重复创建会话。')
      if (err instanceof HttpError && [400, 409, 422].includes(err.status)) persist(null)
    } finally { setBusy('') }
  }
  async function control(action: 'stop' | 'reset' | 'route' | 'record' | 'close') {
    if (!sessionId || busy) return
    setBusy(action); setError('')
    try {
      if (action !== 'record') await brake()
      if (action === 'close') {
        const result = await request<Snapshot>(`sessions/${sessionId}`, undefined, 'DELETE')
        if (result?.id) apply(result, true); else setSnapshot(previous => previous ? { ...previous, status: 'closed' } : null)
        persist(null); setOnline(false)
      } else {
        const result = await request<Snapshot>(`sessions/${sessionId}/${action}`, action === 'record' ? { active: !snapshot?.recording.active } : {})
        apply(result, action === 'reset'); setOnline(true)
      }
    } catch (err) { setError('操作未完成：' + message(err)) } finally { setBusy('') }
  }

  const pose = snapshot?.pose ?? { x: 0, y: 0, z: 0, yaw: 0 }
  const velocity = snapshot?.velocity ?? { linear_m_s: 0, angular_rad_s: 0 }
  const points = snapshot?.trajectory ?? []
  const trajectory = points.map(point => `${160 + point.x * 65},${112 - point.y * 65}`).join(' ')
  const robotPoint = { x: 160 + pose.x * 65, y: 112 - pose.y * 65 }
  const dataset = snapshot?.dataset
  const status = snapshot ? STATUS[snapshot.status] : receipt?.id ? '正在恢复会话' : '等待启动'
  const directionButton = (direction: Direction, symbol: string, label: string, keyLabel: string) => <button key={direction} className={'md-direction md-dir-' + direction + (held.has(direction) ? ' pressed' : '')} aria-label={label} disabled={!driving} onPointerDown={event => { event.preventDefault(); event.currentTarget.setPointerCapture(event.pointerId); press('pointer:' + direction, direction) }} onPointerUp={() => release('pointer:' + direction)} onPointerCancel={() => release('pointer:' + direction)} onLostPointerCapture={() => release('pointer:' + direction)}><span aria-hidden="true">{symbol}</span><strong>{label}</strong><small>{keyLabel}</small></button>

  return <div className="md-app">
    <aside className="md-sidebar"><Link to="/" className="md-brand"><span>◇</span> RoboForge <small>LAB</small></Link><div className="md-lab"><span>RF</span><div>本地实验室<small>从机器人外形到真实运动</small></div></div><nav><Link className="selected" to="/drive"><span>⌁</span>移动机器人<i>01</i></Link><Link to="/transport"><span>⇄</span>移动抓取搬运<i>↗</i></Link><Link to="/create"><span>✧</span>场景生成工作台<i>↗</i></Link><Link to="/"><span>▦</span>传送带 Demo<i>↗</i></Link></nav><div className="md-sidebar-note"><span>AHOLO × MUJOCO</span><strong>给机器人移动的能力。</strong><p>真实轮地接触、底盘驱动与相机画面，记录每一次驾驶。</p></div><span className="md-sidebar-foot">ROBOFORGE · MOBILE LAB</span></aside>
    <div className="md-body"><header className="md-topbar"><span>工作空间 <b>/</b> 移动机器人实验室</span><button onClick={() => setRetry(value => value + 1)} aria-label="重新连接移动仿真"><i className={online ? 'online' : ''}/>{online ? '仿真已连接' : sessionId ? '连接中' : '本地仿真'}<span aria-hidden="true">↻</span></button></header>
      <main className="md-main"><div className="md-heading"><div><span className="md-kicker">MOBILE ROBOT / LIVE SIMULATION</span><h1>让机器人，<em>动起来。</em></h1><p>驾驶四轮机器人，观察真实运动，把轨迹和控制指令记录成数据。</p></div><div className="md-mode"><span/>真实物理驱动</div></div>
        {error && <div className="md-alert" role="alert"><span>{error}</span><button onClick={() => setError('')} aria-label="关闭提示">×</button></div>}
        <div className="md-layout"><div className="md-column-left"><section className="md-view md-card"><div className="md-section-head"><div><i className={online && active ? 'live' : ''}/><h2>仿真现场</h2><span>AHOLO × MUJOCO</span></div><strong>{status}</strong></div><div className="md-camera">{imageSource(snapshot?.images[camera]) ? <img src={imageSource(snapshot?.images[camera])} alt={camera === 'front' ? '移动机器人前视相机画面' : '移动车载相机画面'}/> : <div className="md-camera-empty"><span>⌁</span><strong>{busy === 'start' ? '正在创建真实仿真场景' : sessionId ? '等待后端相机画面' : '你的移动机器人将在这里启动'}</strong><p>复用已有 Aholo 底盘，点击右侧启动，无需重新生成模型。</p></div>}<div className="md-camera-overlay"><span>{camera === 'front' ? 'FRONT / 场地视角' : 'ONBOARD / 车载视角'}</span><span>{(snapshot?.sim_time ?? 0).toFixed(2)} s · Tick {snapshot?.tick ?? 0}</span></div>{snapshot?.recording.active && <span className="md-recording"><i/>REC · {snapshot.recording.frames} 帧</span>}</div><div className="md-camera-tabs"><div><button className={camera === 'front' ? 'selected' : ''} onClick={() => setCamera('front')}>前视相机</button><button className={camera === 'onboard' ? 'selected' : ''} onClick={() => setCamera('onboard')}>车载相机</button></div><span>{snapshot?.control_mode === 'route' ? '示例路线执行中' : held.size ? '手动驾驶中' : '松开控制即停车'}</span></div></section>
          <section className="md-telemetry md-card"><div><span>实际前进速度</span><strong>{(velocity.linear_m_s * 100).toFixed(1)}<small>cm/s</small></strong></div><div><span>实际转弯速度</span><strong>{velocity.angular_rad_s.toFixed(2)}<small>rad/s</small></strong></div><div><span>累计路程</span><strong>{(snapshot?.distance_m ?? 0).toFixed(2)}<small>m</small></strong></div><div><span>当前位置</span><strong className="md-position">{pose.x.toFixed(2)}, {pose.y.toFixed(2)}<small>m</small></strong></div></section>
          <section className="md-record md-card"><div className="md-section-head"><div><h2>记录移动数据</h2><span>20 HZ · 状态与控制</span></div><span>{snapshot?.recording.active ? '正在记录' : dataset ? '数据已保存' : '准备记录'}</span></div><div className="md-record-body"><div><strong>{(snapshot?.recording.active ? snapshot.recording.frames : dataset?.frames ?? 0).toLocaleString()}<small>帧</small></strong><p>底盘姿态 · 实际速度 · 四轮转速<br/>驾驶指令 · 时间戳 · 模型来源<br/>相机仅实时查看，不写入本次记录</p></div><div className="md-record-actions"><button className={snapshot?.recording.active ? 'md-record-stop' : 'md-primary'} disabled={!active || !online || !!busy} onClick={() => void control('record')}>{busy === 'record' ? '正在保存…' : snapshot?.recording.active ? '结束记录并保存' : '开始记录数据'}</button><div><a aria-disabled={!dataset} className={!dataset ? 'disabled' : ''} href={dataset?.manifest_url} download onClick={event => { if (!dataset) event.preventDefault() }}>JSON 清单 ↓</a><a aria-disabled={!dataset} className={!dataset ? 'disabled' : ''} href={dataset?.npz_url} download onClick={event => { if (!dataset) event.preventDefault() }}>NPZ 数据 ↓</a></div></div></div></section>
        </div><div className="md-column-right"><section className="md-setup md-card"><div className="md-card-title"><span>01 / ROBOT</span><h2>四轮移动机器人</h2></div><div className="md-setup-body"><label>Aholo 底盘外观<select aria-label="移动机器人底盘" value={asset} disabled={!!receipt || active || !!busy} onChange={event => setAsset(event.target.value)}>{!capabilities && <option value="">正在读取资产…</option>}{capabilities?.assets.map(item => <option key={item.id} value={item.id}>{item.name}</option>)}</select></label><p>复用已有模型；机械臂保持收拢，先验证底盘移动。</p>{!active ? <button className="md-primary md-full" disabled={!!busy || !capabilities || !asset || !!receipt?.id} onClick={() => void start()}>{busy === 'start' ? '正在启动…' : receipt && !receipt.id ? '恢复这次启动' : '启动移动仿真'}<span aria-hidden="true">↗</span></button> : <div className="md-ready"><span/>当前会话已就绪 <small>{snapshot.id.slice(0, 12)}</small></div>}</div></section>
          <section className="md-controls md-card"><div className="md-card-title"><span>02 / DRIVE</span><h2>你来驾驶</h2><p>按住按钮，或使用 WASD / 方向键。</p></div><div className="md-pad">{directionButton('forward', '↑', '前进', 'W / ↑')}{directionButton('left', '↶', '左转', 'A / ←')}<button className="md-brake" aria-label="急停" disabled={!active || !online} onPointerDown={event => { event.preventDefault(); void brake() }} onClick={() => void brake()}><span aria-hidden="true">■</span><strong>停车</strong><small>SPACE</small></button>{directionButton('right', '↷', '右转', 'D / →')}{directionButton('backward', '↓', '后退', 'S / ↓')}</div><div className="md-sliders"><label>前进速度<strong>{(speed * 100).toFixed(0)} <small>cm/s</small></strong><input aria-label="移动速度" type="range" min={.04} max={capabilities?.limits.linear_m_s ?? .22} step={.01} value={speed} disabled={!capabilities || !!busy} onChange={event => { setSpeed(Number(event.target.value)); void brake() }}/></label><label>转弯速度<strong>{turnSpeed.toFixed(1)} <small>rad/s</small></strong><input aria-label="转弯速度" type="range" min={.3} max={capabilities?.limits.angular_rad_s ?? 1.8} step={.1} value={turnSpeed} disabled={!capabilities || !!busy} onChange={event => { setTurnSpeed(Number(event.target.value)); void brake() }}/></label></div><p className="md-control-note">松开、切换窗口或断开连接后，机器人自动减速停车。</p><div className="md-control-actions"><button disabled={!driving || snapshot?.recording.active} onClick={() => void control('reset')}>{busy === 'reset' ? '复位中…' : '复位机器人'}</button><button disabled={!driving} onClick={() => void control('route')}>{busy === 'route' ? '准备路线…' : '运行示例路线'}<span aria-hidden="true">↗</span></button></div><p className="md-route-note">示例路线使用固定路点控制，尚不包含自主导航或避障。{snapshot?.route_status && <span>{ROUTE_STATUS[snapshot.route_status] ?? snapshot.route_status}</span>}</p></section>
          <section className="md-map md-card"><div className="md-section-head"><div><h2>实际运动轨迹</h2></div><span>航向 {(pose.yaw * 180 / Math.PI).toFixed(0)}°</span></div><svg viewBox="0 0 320 224" role="img" aria-label="后端实际位置轨迹"><defs><pattern id="mobile-grid" width="32.5" height="32.5" patternUnits="userSpaceOnUse"><path d="M32.5 0H0V32.5" fill="none" stroke="#e5ebde" strokeWidth=".7"/></pattern></defs><rect x="0" y="0" width="320" height="224" fill="#f8faf4"/><rect x="10" y="10" width="300" height="204" fill="url(#mobile-grid)"/><path d="M160 10V214M10 112H310" stroke="#d4dfca" strokeDasharray="3 4"/><polyline points={trajectory} fill="none" stroke="#719966" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>{active || snapshot ? <g transform={`translate(${robotPoint.x} ${robotPoint.y}) rotate(${-pose.yaw * 180 / Math.PI})`}><circle r="9" fill="#335f45" stroke="#fff" strokeWidth="2"/><path d="M13 0L4 -4V4Z" fill="#c9dc9e"/></g> : null}<text x="16" y="207" fontSize="9" fill="#9ba98d">网格间距 0.5 m</text><text x="280" y="105" fontSize="9" fill="#9ba98d">+X</text><text x="168" y="24" fontSize="9" fill="#9ba98d">+Y</text></svg><div className="md-map-caption"><span>真实状态轨迹</span><span>{points.length} 个轨迹点</span></div></section>
        </div></div><section className="md-notes"><strong>当前实验范围</strong><p>四轮接触地面驱动，机械臂随底盘移动；本轮聚焦移动，不执行抓取。外观来自 Aholo，轮地接触和动力参数为演示估计值。</p>{capabilities?.notes?.map((note, index) => <p key={index}>{note}</p>)}{active && <button disabled={!!busy} onClick={() => void control('close')}>{busy === 'close' ? '正在结束…' : '结束会话并保存记录'}</button>}</section><footer className="md-footer"><span>◇ ROBOFORGE · 从运动到数据</span><Link to="/create">返回场景生成工作台 ↗</Link></footer>
      </main></div>
  </div>
}
