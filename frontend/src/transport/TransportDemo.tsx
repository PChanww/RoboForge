import { Component, useEffect, useRef, useState, type ReactNode } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { Canvas } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js'
import PolicyGuide from './PolicyGuide'
import './transport.css'

type Asset = { id: string; name: string }
type Point = [number, number]
type Parameters = { height_m?: number | null; width_m?: number | null; depth_m?: number | null; mass_kg?: number | null; friction?: number | null }
type Spec = { experiment_name: string; object_name: string; bin_name: string; instruction: string; task_kind?: 'deliver' | 'pick_return' | 'bin_retrieve' | 'bin_return'; task_sequence?: string[]; controller: 'expert' | 'vla'; policy_id: string | null; vla_replan_steps: number; vla_max_inference_s: number; object_asset: string; chassis_asset: string; episodes: number; seed: number; belt_speed_m_s: number; retreat_distance_m: number; random_pose: boolean; target_xy: Point | null; speed_scale: number; object_mode: 'reuse' | 'generate'; object_description: string; object_parameters: Parameters; object_parameter_sources: Record<string, string>; bin_mode: 'builtin' | 'generate' | 'reuse'; bin_description: string; bin_asset: string | null; bin_size_m: [number, number, number]; bin_color: string }
type Capabilities = { objects: Asset[]; chassis: Asset[]; bins?: Asset[]; defaults: Spec; notes: string[]; configured?: boolean; suggestion?: { configured: boolean; state: string } }
type PolicyIdentity = { label?: string; learned_policy?: boolean; adapter?: string; device?: string | null; device_name?: string | null; training_model?: string }
type PolicyCapability = { id: string; label: string; kind: 'vla' | 'learned_bc'; ready: boolean; state: 'ready' | 'offline' | 'loading' | 'error'; action_schema: string; action_dim: number; adapter: string | null; limitations: string[]; identity?: PolicyIdentity | null; error?: string }
type PolicyCapabilities = { vla_connected: boolean; policies: PolicyCapability[]; default_policy_id: string | null; notes: string[] }
type Navigation = { ready: boolean; issues: string[]; planned_route: Point[]; dock_xy: Point }
type Plan = { ready: boolean; spec: Spec; summary: string; steps: string[]; task_sequence?: string[]; issues: string[]; warnings: string[]; generation_count: number; navigation?: Navigation }
type Episode = { index: number; seed: number; frames: number; duration_s: number; success: boolean; metrics?: Record<string, unknown>; reach_evidence?: string; reach_limit?: ReachLimit | null }
type ReachLimit = { steps: number; first_frame: number; max_deviation_m: number }
type Preview = { tick: number; epoch?: number; sim_time: number; images: Record<string, string | null>; phase?: number | string; phase_label?: string; metrics?: Record<string, unknown>; pose?: { x: number; y: number; z: number; yaw: number }; base_velocity?: number[]; belt_speed_m_s?: number; transport_distance_m?: number; distance_m?: number; target_xy?: Point; goal_distance_m?: number; planned_route?: Point[] }
type AssetDecision = 'save' | 'use_once' | 'reject'
type AssetReview = { name?: string; role: 'object' | 'bin'; asset_id: string; session_id?: string; status: 'pending' | 'accepted' | 'rejected'; decision: AssetDecision | null; saved: boolean; storage: 'cache' | 'library'; saved_at?: string; description: string; parameters: Parameters; preview_url?: string; preview_format?: 'glb' | 'obj'; preview: { obj_url: string; glb_url?: string | null }; package_url: string; download_url?: string }
type ReviewReceipt = { decision: AssetDecision; client_request_id: string }
type HistoryRow = { experiment_name?: string; id: string; status: Run['status']; instruction?: string; task_kind?: string; policy_identity?: PolicyIdentity; summary?: { episodes: number; successes: number; frames: number }; created: string; updated: string; error?: string | null }
type Evaluation = { run_id: string; status: string; verdict: { code: 'passed' | 'failed' | 'pending' | 'incomplete' | 'invalid_data'; label: string; scope: string; reason: string }; controller: { kind: string; label: string; learned_policy: boolean }; summary: { episodes: number; successes: number; frames: number; success_rate: number }; episodes: { index: number; seed: number; success: boolean; frames: number; duration_s: number; task_kind?: string; program_complete?: boolean; task_events?: Record<string, unknown>[]; subtasks?: { stage: string; label: string; completed: boolean; evidence: Record<string, unknown> | null }[]; phase_timeline: { phase: number; label: string; start_s: number; end_s: number; frames: number }[]; failure: { stage: string; phase: number; reason: string; last_bilateral_contact_s: number | null } | null; reach_evidence?: string; reach_limit?: ReachLimit | null; metrics?: Record<string, unknown>; data_quality: { valid: boolean; issues: string[] } }[]; data_quality: { valid: boolean; issues: string[] }; policy: { vla_connected: boolean; transport_action_dim: number; legacy_action_dim: number; notes: string[] }; downloads: { report_url: string; contract_url: string; archive_url: string }; limitations: string[] }
type Run = { id: string; status: 'queued' | 'generating' | 'awaiting_asset_review' | 'running' | 'stopping' | 'succeeded' | 'failed' | 'cancelled' | 'interrupted'; stop_requested?: boolean; spec: Spec; plan?: Plan; preview?: Preview | null; policy_identity?: PolicyIdentity; policy_diagnostics?: { inference_count?: number; executed_model_actions?: number; last_inference_s?: number; inference_pending?: boolean }; asset_steps?: { role: string; status: string; asset_id?: string; session_id?: string; error?: string }[]; asset_reviews?: AssetReview[]; episodes: Episode[]; current_episode?: number; dataset?: { id: string; manifest_url: string; episodes: { index: number; npz_url: string }[] }; configuration?: { sha256: string; file: string; url: string }; error?: string; created: string; updated: string }
type Suggestion = { parameters: Parameters; source: string; notes?: string[]; provenance?: Record<string, { source: string }> }
type Receipt = { client_request_id: string; spec: Spec; run_id?: string }
const RECEIPT_KEY = 'roboforge.transport.receipt.v1'
const DRAFT_KEY = 'roboforge.transport.draft.v1'
const REVIEW_KEY = 'roboforge.transport.reviews.v1'
const DEFAULT: Spec = { experiment_name: '', object_name: '', bin_name: '', instruction: '从不停运行的传送带上抓取所选物品，移动到目标箱旁，将物品放入箱内。', controller: 'expert', policy_id: null, vla_replan_steps: 5, vla_max_inference_s: 60, object_asset: 'imported/yellow-plastic-duck-a08e3e', chassis_asset: 'imported/four-wheel-robot-chassis-82e8e5', episodes: 1, seed: 3, belt_speed_m_s: .025, retreat_distance_m: .4, random_pose: true, target_xy: null, speed_scale: 1.6, object_mode: 'reuse', object_description: '黄色塑料小鸭，圆润单体、适合夹爪抓取。', object_parameters: {}, object_parameter_sources: {}, bin_mode: 'builtin', bin_description: '一个顶部敞开的青绿色收纳箱，四周完整，箱底平整，无盖。', bin_asset: null, bin_size_m: [.21, .22, .056], bin_color: '#2E8F85' }
const POLICY_STATE: Record<PolicyCapability['state'], string> = { ready: '推理服务已就绪', offline: '推理服务未连接', loading: '推理模型加载中', error: '推理服务异常' }
const TASK_STEP: Record<string, string> = { pick_from_belt: '从传送带抓起', transport_to_bin: '移动到箱子旁', place_in_bin: '放入箱内', regrasp_from_bin: '从箱内再次抓起', lifted_from_bin: '抬出箱沿并保持夹持', return_along_route: '沿原路线返回', return_to_belt: '放回传送带' }
const STATUS: Record<Run['status'], string> = { queued: '正在组装场景', generating: '正在生成场景资产', awaiting_asset_review: '等待你确认模型', running: '仿真验证与采集中', stopping: '正在停止并保存', succeeded: '本次验证通过', failed: '本次验证未通过', cancelled: '实验已停止', interrupted: '实验已中断' }
const ASSET_STATUS: Record<string, string> = { pending: '等待生成', submitting: '正在提交', generating: 'Aholo 生成中', queued: '等待处理', remote_processing: 'Aholo 生成中', downloading: '下载模型', importing: '导入模型', validating: '检查模型', succeeded: '模型就绪', reused: '复用已有模型', ready: '模型就绪', builtin: '参数化箱体', skipped: '参数化箱体', failed: '处理失败', submit_unknown: '提交结果待确认', cancelled: '已停止' }
const PARAMS: { key: keyof Parameters; label: string; unit: string; factor: number; min: number; max: number; step: number }[] = [
  { key: 'height_m', label: '高度', unit: 'cm', factor: 100, min: 2, max: 7.5, step: .1 },
  { key: 'width_m', label: '宽度', unit: 'cm', factor: 100, min: 1.5, max: 8.5, step: .1 },
  { key: 'depth_m', label: '深度', unit: 'cm', factor: 100, min: 1.5, max: 8.5, step: .1 },
  { key: 'mass_kg', label: '质量', unit: 'g', factor: 1000, min: 5, max: 150, step: 1 },
  { key: 'friction', label: '摩擦系数', unit: 'μ', factor: 1, min: .2, max: 3, step: .1 },
]
const MAP = { left: -1.2, right: 1.2, bottom: -.6, top: 1.5, width: 640, height: 480, pad: 30 }
function read<T>(key: string): T | null { try { return JSON.parse(localStorage.getItem(key) ?? 'null') } catch { return null } }
function normalize(value?: Partial<Spec> | null): Spec { return { ...DEFAULT, ...value, object_parameters: { ...(value?.object_parameters ?? {}) }, object_parameter_sources: { ...(value?.object_parameter_sources ?? {}) }, bin_size_m: value?.bin_size_m ?? DEFAULT.bin_size_m } }
function active(run: Run | null) { return !!run && ['queued', 'generating', 'awaiting_asset_review', 'running', 'stopping'].includes(run.status) }
function errorText(error: unknown) { return error instanceof Error ? error.message : String(error) }
function runErrorText(message: string) {
  const reach = /^Unreachable: target distance ([\d.]+) m exceeds reach ([\d.]+) m$/.exec(message)
  return reach ? `策略给出的目标距离为 ${reach[1]} m，超出机械臂的 ${reach[2]} m 可达范围。` : message
}
class HttpError extends Error { constructor(public status: number, message: string) { super(message) } }
async function request<T>(path: string, body?: unknown, signal?: AbortSignal, timeoutMs = 12000): Promise<T> {
  const timeout = AbortSignal.timeout(timeoutMs)
  try {
    const response = await fetch('/api/v1/transport/' + path, {
      method: body === undefined ? 'GET' : 'POST', signal: signal ? AbortSignal.any([signal, timeout]) : timeout,
      ...(body === undefined ? {} : { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }),
    })
    const value = await response.json().catch(err => { if (timeout.aborted || signal?.aborted) throw err; return null })
    if (!response.ok) {
      const detail = value?.detail ?? value?.error
      throw new HttpError(response.status, typeof detail === 'string' ? detail : JSON.stringify(detail) || `后端请求失败（HTTP ${response.status}）`)
    }
    if (value == null) throw new Error('后端没有返回可读取的数据。')
    return value as T
  } catch (error) {
    if (timeout.aborted && !signal?.aborted) throw new Error('后端响应超时，请检查本地服务后重试。')
    if (error instanceof TypeError && !signal?.aborted) throw new Error('无法连接本地搬运服务。')
    throw error
  }
}
function source(image?: string | null) { return image ? image.startsWith('data:') ? image : 'data:image/jpeg;base64,' + image : undefined }
function metricNumber(metrics: Preview['metrics'], ...names: string[]) { for (const name of names) if (typeof metrics?.[name] === 'number') return metrics[name] as number; return undefined }
function policyDevice(identity?: PolicyIdentity | null) { return [identity?.device, identity?.device_name].filter(Boolean).join(' · ') }
function reachOf(source?: { metrics?: Record<string, unknown>; reach_limit?: ReachLimit | null } | null) {
  const steps = source?.metrics?.reach_limited_steps
  const count = typeof steps === 'number' ? steps : source?.reach_limit?.steps
  if (typeof count !== 'number' || count <= 0) return null
  const max = source?.reach_limit?.max_deviation_m
  return { steps: count, deviationMm: typeof max === 'number' ? Math.round(max * 1000) : undefined }
}
function eventEvidence(event: Record<string, unknown> | null) {
  if (!event) return '尚无完成事件'
  return [typeof event.time_s === 'number' ? `${event.time_s.toFixed(2)} s` : '', typeof event.contacts === 'number' ? `${event.contacts} 侧夹爪接触` : '', event.controlled_release === true ? '受控松爪' : '', event.tray_contact === true ? '箱底支撑' : '', event.roller_contact === true ? '传送带支撑' : '', typeof event.object_low_z === 'number' ? `物体最低点 ${(event.object_low_z * 100).toFixed(1)} cm` : '', typeof event.stable_ticks === 'number' ? `稳定 ${event.stable_ticks} 帧` : ''].filter(Boolean).join(' · ') || '已记录物理完成事件'
}

const VIEWS = [
  { id: 'home', icon: '⌂', label: '工作空间' },
  { id: 'task', icon: '✎', label: '任务要求' },
  { id: 'models', icon: '◇', label: '模型资产' },
  { id: 'policy', icon: '⟁', label: '执行策略' },
  { id: 'simulation', icon: '◉', label: '仿真执行' },
  { id: 'results', icon: '✓', label: '可行性结果' },
  { id: 'datasets', icon: '▤', label: '数据集' },
  { id: 'settings', icon: '⚙', label: '参数设置' },
] as const
type View = typeof VIEWS[number]['id']
function roleLabel(role: string) { return role === 'bin' ? '目标收纳箱' : '抓取物品' }
function releaseModel(object: THREE.Object3D) {
  object.traverse(node => {
    if (!(node instanceof THREE.Mesh)) return
    node.geometry.dispose()
    for (const material of Array.isArray(node.material) ? node.material : [node.material]) {
      for (const value of Object.values(material)) if (value instanceof THREE.Texture) value.dispose()
      material.dispose()
    }
  })
}
class PreviewBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false }
  static getDerivedStateFromError() { return { failed: true } }
  render() { return this.state.failed ? <div className="tt-model-fallback">当前浏览器无法打开 3D 预览，请下载模型包检查。</div> : this.props.children }
}
function AssetPreview({ asset }: { asset: AssetReview }) {
  const [model, setModel] = useState<THREE.Group | null>(null)
  const [message, setMessage] = useState('正在加载模型…')
  const [textured, setTextured] = useState(false)
  const glb = asset.preview_format === 'obj' ? null : asset.preview?.glb_url ?? (asset.preview_format === 'glb' ? asset.preview_url : null)
  const obj = asset.preview?.obj_url ?? (asset.preview_format === 'obj' ? asset.preview_url : null)
  useEffect(() => {
    let disposed = false
    let loaded: THREE.Group | null = null
    setModel(null); setMessage('正在加载模型…')
    async function load() {
      let content: THREE.Group
      let original = false
      try {
        if (!glb) throw new Error('没有 GLB 预览')
        content = (await new GLTFLoader().loadAsync(glb)).scene
        original = true
      } catch {
        if (!obj) throw new Error('没有可用的模型预览文件')
        content = await new OBJLoader().loadAsync(obj)
        content.rotation.x = -Math.PI / 2
        content.traverse(node => {
          if (node instanceof THREE.Mesh) node.material = new THREE.MeshStandardMaterial({ color: '#9bb5a0', roughness: .65, side: THREE.DoubleSide })
        })
      }
      content.updateMatrixWorld(true)
      const bounds = new THREE.Box3().setFromObject(content)
      const center = bounds.getCenter(new THREE.Vector3())
      const span = Math.max(...bounds.getSize(new THREE.Vector3()).toArray())
      if (!Number.isFinite(span) || span <= 0) { releaseModel(content); throw new Error('模型尺寸无效') }
      const group = new THREE.Group()
      group.add(content); content.position.sub(center); group.scale.setScalar(1.7 / span)
      loaded = group
      if (disposed) { releaseModel(group); return }
      setTextured(original); setModel(group); setMessage('')
    }
    void load().catch(err => { if (!disposed) setMessage('3D 预览未加载：' + errorText(err)) })
    return () => { disposed = true; if (loaded) releaseModel(loaded) }
  }, [glb, obj])
  return <div className="tt-model-preview" role="region" aria-label={`${roleLabel(asset.role)}三维预览`}>
    {model ? <PreviewBoundary key={asset.asset_id}><Canvas dpr={[1, 1.5]} camera={{ position: [2.5, 1.7, 2.8], fov: 38 }} gl={{ antialias: true, preserveDrawingBuffer: true }}>
      <color attach="background" args={['#edf2e9']}/><ambientLight intensity={1.7}/><directionalLight position={[3, 5, 4]} intensity={3}/><directionalLight position={[-3, 1, -3]} intensity={1.2}/>
      <primitive object={model}/><OrbitControls makeDefault target={[0, 0, 0]} minDistance={1.2} maxDistance={7} enablePan={false}/>
    </Canvas></PreviewBoundary> : <div className="tt-model-fallback">{message}</div>}
    <span className="tt-model-hint">{model ? textured ? '原始生成外观 · 拖动旋转 / 滚轮缩放' : 'OBJ 结构预览 · 拖动旋转 / 滚轮缩放' : '可通过下方模型包查看原文件'}</span>
  </div>
}
function AssetProgress({ step, review }: { step: NonNullable<Run['asset_steps']>[number]; review?: AssetReview }) {
  const labels = ['提交', '生成', '下载', '导入', '待确认', '已确认']
  const positions: Record<string, number> = { pending: 0, submitting: 0, queued: 1, generating: 1, remote_processing: 1, downloading: 2, importing: 3, validating: 3, succeeded: 4, ready: 4, awaiting_review: 4 }
  const current = review?.status === 'accepted' ? 5 : positions[step.status] ?? 0
  const failed = ['failed', 'submit_unknown', 'cancelled'].includes(step.status)
  const reused = ['reused', 'builtin', 'skipped'].includes(step.status)
  return <div className="tt-generation-item"><div><strong>{roleLabel(step.role)}</strong><span>{review?.status === 'pending' ? '等待你的意见' : review?.saved ? '已保存到本地资产库' : review?.decision === 'use_once' ? '仅用于本次实验' : ASSET_STATUS[step.status] ?? step.status}</span></div>
    {reused ? <p className="tt-helper">{step.status === 'reused' ? '使用已有模型，无需再次审阅。' : '使用参数化箱体，无需生成。'}</p> : <ol className={'tt-generation-progress ' + (failed ? 'failed' : '')} aria-label={`${roleLabel(step.role)}生成阶段`}>{labels.map((label, index) => <li key={label} className={index < current ? 'done' : index === current ? 'current' : ''}><span>{index < current ? '✓' : index + 1}</span><small>{label}</small></li>)}</ol>}
    {step.error && <p className="tt-run-error">{step.error}</p>}
    {step.session_id && <Link className="tt-text-link" to={`/generation/${step.session_id}`}>查看生成记录 ↗</Link>}
  </div>
}

function mapX(x: number) { return MAP.pad + (x - MAP.left) / (MAP.right - MAP.left) * (MAP.width - 2 * MAP.pad) }
function mapY(y: number) { return MAP.pad + (MAP.top - y) / (MAP.top - MAP.bottom) * (MAP.height - 2 * MAP.pad) }
function pointsLine(points: Point[]) { return points.filter(point => point.length === 2 && point.every(Number.isFinite)).map(([x, y]) => `${mapX(x)},${mapY(y)}`).join(' ') }
function WorkspaceMap({ target, size, color, pose, route, trajectory, disabled, onPick }: { target: Point; size: Spec['bin_size_m']; color: string; pose?: Preview['pose']; route: Point[]; trajectory: Point[]; disabled: boolean; onPick: (point: Point) => void }) {
  function pick(event: React.MouseEvent<SVGSVGElement>) {
    if (disabled) return
    const matrix = event.currentTarget.getScreenCTM()
    if (!matrix) return
    const point = new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse())
    const x = Math.max(MAP.left, Math.min(MAP.right, MAP.left + (point.x - MAP.pad) / (MAP.width - 2 * MAP.pad) * (MAP.right - MAP.left)))
    const y = Math.max(MAP.bottom, Math.min(MAP.top, MAP.top - (point.y - MAP.pad) / (MAP.height - 2 * MAP.pad) * (MAP.top - MAP.bottom)))
    onPick([Math.round(x * 100) / 100, Math.round(y * 100) / 100])
  }
  const actual = pose ?? { x: 0, y: .205, z: 0, yaw: 0 }
  const width = size[0] / (MAP.right - MAP.left) * (MAP.width - 2 * MAP.pad)
  const height = size[1] / (MAP.top - MAP.bottom) * (MAP.height - 2 * MAP.pad)
  return <svg className={'tt-map ' + (disabled ? 'locked' : '')} viewBox={`0 0 ${MAP.width} ${MAP.height}`} role="img" aria-label="目标箱位置地图，点击可放置目标箱" onClick={pick}>
    <defs><pattern id="tt-map-grid" width="48.33" height="40" patternUnits="userSpaceOnUse"><path d="M 48.33 0 L 0 0 0 40" fill="none" stroke="#dfe8db" strokeWidth=".8"/></pattern></defs>
    <rect x={MAP.pad} y={MAP.pad} width={MAP.width - MAP.pad * 2} height={MAP.height - MAP.pad * 2} rx="8" fill="url(#tt-map-grid)" stroke="#d4dfcd"/>
    <line x1={mapX(0)} y1={MAP.pad} x2={mapX(0)} y2={MAP.height - MAP.pad} stroke="#c3d2ba" strokeDasharray="4 5"/><line x1={MAP.pad} y1={mapY(0)} x2={MAP.width - MAP.pad} y2={mapY(0)} stroke="#c3d2ba" strokeDasharray="4 5"/>
    <rect x={mapX(-.34)} y={mapY(.012)} width={mapX(.34) - mapX(-.34)} height={mapY(-.14) - mapY(.012)} rx="5" fill="#8fa59e" stroke="#687f77"/><path d={`M${mapX(-.24)} ${mapY(-.064)}h21m-7-5 7 5-7 5 M${mapX(.08)} ${mapY(-.064)}h21m-7-5 7 5-7 5`} stroke="#edf3ed" fill="none" strokeWidth="2"/><text x={mapX(0)} y={mapY(-.14) + 19} textAnchor="middle">持续运行的传送带</text>
    {route.length > 1 && <polyline points={pointsLine(route)} fill="none" stroke="#b7be74" strokeWidth="3" strokeDasharray="7 5" strokeLinejoin="round"/>}{trajectory.length > 1 && <polyline points={pointsLine(trajectory)} fill="none" stroke="#4e8661" strokeWidth="3" strokeLinejoin="round"/>}
    <g transform={`translate(${mapX(target[0])},${mapY(target[1])})`}><rect x={-width / 2} y={-height / 2} width={width} height={height} fill={color} fillOpacity=".22" stroke={color} strokeWidth="4" rx="3"/><text y={-height / 2 - 9} textAnchor="middle" className="tt-map-target">目标箱</text></g>
    <circle cx={mapX(0)} cy={mapY(.205)} r="4" fill="#a7b39a"/><text x={mapX(0) - 11} y={mapY(.205) - 12} textAnchor="end">起点</text><g transform={`translate(${mapX(actual.x)},${mapY(actual.y)}) rotate(${-actual.yaw * 180 / Math.PI})`}><rect x="-12" y="-16" width="24" height="32" rx="6" fill="#285c43" stroke="#fff" strokeWidth="2"/><path d="m-5 6 5 5 5-5" fill="none" stroke="#d9edbd" strokeWidth="2"/></g>
    <text x={MAP.width - MAP.pad} y="17" textAnchor="end">+Y ↑ · 单位 m</text><text x={MAP.pad} y={MAP.height - 9}>X −1.2</text><text x={MAP.width - MAP.pad} y={MAP.height - 9} textAnchor="end">+1.2 →</text>
  </svg>
}

export default function TransportDemo() {
  const [searchParams, setSearchParams] = useSearchParams()
  const view = (VIEWS.some(item => item.id === searchParams.get('view')) ? searchParams.get('view') : 'home') as View
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null)
  const [policies, setPolicies] = useState<PolicyCapabilities | null>(null)
  const [policyError, setPolicyError] = useState('')
  const [policyChecking, setPolicyChecking] = useState(false)
  const [policyCheckedAt, setPolicyCheckedAt] = useState<string | null>(null)
  const [receipt, setReceipt] = useState<Receipt | null>(() => read(RECEIPT_KEY))
  const [spec, setSpec] = useState<Spec>(() => normalize(read<Receipt>(RECEIPT_KEY)?.spec ?? read<Spec>(DRAFT_KEY)))
  const [plan, setPlan] = useState<Plan | null>(null)
  const [run, setRun] = useState<Run | null>(null)
  const [camera, setCamera] = useState('front')
  const [busy, setBusy] = useState('')
  const [downloading, setDownloading] = useState('')
  const [downloadProgress, setDownloadProgress] = useState<{ received: number; total: number } | null>(null)
  const [error, setError] = useState('')
  const [online, setOnline] = useState(false)
  const [connecting, setConnecting] = useState(true)
  const [retry, setRetry] = useState(0)
  const [suggestion, setSuggestion] = useState<Suggestion | null>(null)
  const [trajectory, setTrajectory] = useState<Point[]>([])
  const [evaluation, setEvaluation] = useState<Evaluation | null>(null)
  const [evaluationError, setEvaluationError] = useState('')
  const [history, setHistory] = useState<HistoryRow[] | null>(null)
  const [historyError, setHistoryError] = useState('')
  const [trainingDatasets, setTrainingDatasets] = useState<Set<string> | null>(null)
  const [trainingCatalogError, setTrainingCatalogError] = useState(false)
  const [checkpointNotice, setCheckpointNotice] = useState('')
  const [reviewReceipts, setReviewReceipts] = useState<Record<string, ReviewReceipt>>(() => read(REVIEW_KEY) ?? {})
  const trajectoryKey = useRef('')
  const submission = useRef(false)
  const hasDraft = useRef(!!read(DRAFT_KEY) || !!read(RECEIPT_KEY))
  const runId = receipt?.run_id
  const locked = !!busy || !!run || !!receipt
  const running = active(run)
  const preview = run?.preview
  const target: Point = preview?.target_xy ?? spec.target_xy ?? [.12, .44 + spec.retreat_distance_m]
  const waitingReview = run?.status === 'awaiting_asset_review'
  const selectedPolicy = policies?.policies?.find(policy => policy.id === (spec.policy_id ?? policies.default_policy_id))
  const currentModel = spec.controller === 'expert' ? 'expert' : selectedPolicy?.identity?.training_model ?? selectedPolicy?.id
  const policyReady = spec.controller === 'expert' || !!selectedPolicy?.ready && !policyError && !policyChecking
  const controllerLabel = run?.policy_identity?.label ?? (spec.controller === 'vla' ? selectedPolicy?.label ?? 'VLA 策略' : '状态专家基线')
  const checkpoint = searchParams.get('checkpoint')
  const requestedPolicy = policies?.policies.find(policy => policy.id === checkpoint)
  const canOpenHistory = !busy && !running && (!receipt || !!run)
  const simulationSource = VIEWS.find(item => item.id !== 'simulation' && item.id === searchParams.get('from'))
  const simulationBackView = simulationSource?.id ?? 'task'
  function viewParams(next: View) {
    const params = new URLSearchParams(checkpoint ? { view: next, checkpoint } : { view: next })
    if (next === 'simulation') params.set('from', view === 'simulation' ? simulationBackView : view)
    return params
  }
  function go(next: View) { setSearchParams(viewParams(next)) }

  useEffect(() => {
    if (!requestedPolicy || locked) return
    hasDraft.current = true
    setSpec(current => ({ ...current, controller: 'vla', policy_id: requestedPolicy.id }))
    setPlan(null)
    setCheckpointNotice(`已选用 ${requestedPolicy.label}`)
    setSearchParams({ view: 'policy' }, { replace: true })
  }, [locked, requestedPolicy, setSearchParams])

  // Deep link: /transport?run=<id> opens that experiment without starting a new
  // one. Keeps the requested view so a result page can be shared or reloaded.
  const linkedRun = searchParams.get('run')?.trim() || ''
  useEffect(() => {
    if (!linkedRun || linkedRun === receipt?.run_id || running) return
    let cancelled = false
    void (async () => {
      try {
        const result = await request<Run>(`runs/${linkedRun}?compact=true`)
        if (cancelled) return
        saveReceipt({ run_id: result.id, spec: normalize(result.spec), client_request_id: `link-${result.id}` })
        setRun(result); setSpec(normalize(result.spec)); setPlan(result.plan ?? null)
        setEvaluation(null); setRetry(value => value + 1)
      } catch (err) {
        if (!cancelled) setError('无法读取链接指向的实验：' + errorText(err))
      }
    })()
    return () => { cancelled = true }
  }, [linkedRun, receipt?.run_id, running])

  useEffect(() => {
    const controller = new AbortController()
    setConnecting(true)
    request<Capabilities>('capabilities', undefined, controller.signal).then(value => {
      if (controller.signal.aborted) return
      setCapabilities(value); setOnline(true)
      setError(current => current.startsWith('无法读取仿真配置') ? '' : current)
      if (!hasDraft.current) { setSpec(normalize(value.defaults)); hasDraft.current = true }
    }).catch(err => { if (!controller.signal.aborted) { setOnline(false); setError('无法读取仿真配置：' + errorText(err)) } })
      .finally(() => { if (!controller.signal.aborted) setConnecting(false) })
    return () => controller.abort()
  }, [retry])
  useEffect(() => {
    let disposed = false
    setPolicyChecking(true)
    request<PolicyCapabilities>('policy-capabilities').then(value => {
      if (disposed) return
      setPolicies(value); setPolicyError(''); setPolicyCheckedAt(new Date().toLocaleTimeString('zh-CN'))
    }).catch(err => { if (!disposed) { setPolicyError(errorText(err)); setPolicies(null) } }).finally(() => { if (!disposed) setPolicyChecking(false) })
    return () => { disposed = true }
  }, [retry, view === 'policy'])
  useEffect(() => { localStorage.setItem(DRAFT_KEY, JSON.stringify(spec)) }, [spec])
  useEffect(() => { localStorage.setItem(REVIEW_KEY, JSON.stringify(reviewReceipts)) }, [reviewReceipts])
  useEffect(() => {
    if (view !== 'datasets') return
    let disposed = false
    request<{ runs: HistoryRow[] }>('history').then(value => { if (!disposed) { setHistory(value.runs); setHistoryError('') } }).catch(err => { if (!disposed) setHistoryError(errorText(err)) })
    request<{ datasets: { id: string }[] }>('training').then(value => {
      if (disposed) return
      setTrainingDatasets(new Set(value.datasets.map(row => row.id)))
      setTrainingCatalogError(false)
    }).catch(() => {
      if (disposed) return
      setTrainingDatasets(null)
      setTrainingCatalogError(true)
    })
    return () => { disposed = true }
  }, [view, retry, run?.status])
  useEffect(() => {
    setEvaluation(null); setEvaluationError('')
    if (!run || running || !['results', 'datasets'].includes(view)) return
    const controller = new AbortController()
    request<Evaluation>(`runs/${run.id}/evaluation`, undefined, controller.signal, 20000)
      .then(value => { if (!controller.signal.aborted) setEvaluation(value) })
      .catch(err => { if (!controller.signal.aborted) setEvaluationError(errorText(err)) })
    return () => controller.abort()
  }, [run?.id, running, view, retry])
  useEffect(() => {
    if (!run || active(run)) return
    let disposed = false
    request<Capabilities>('capabilities').then(value => { if (!disposed) setCapabilities(value) }).catch(() => {})
    return () => { disposed = true }
  }, [run?.id, run?.status])
  useEffect(() => {
    const pose = preview?.pose
    if (!pose) return
    const key = `${run?.id}:${preview?.epoch}:${run?.current_episode}`
    if (trajectoryKey.current !== key) { trajectoryKey.current = key; setTrajectory([[pose.x, pose.y]]); return }
    setTrajectory(current => {
      const last = current[current.length - 1]
      return !last || Math.hypot(pose.x - last[0], pose.y - last[1]) > .004 ? [...current.slice(-1199), [pose.x, pose.y]] : current
    })
  }, [preview?.tick, preview?.epoch, preview?.pose, run?.id, run?.current_episode])
  function saveReceipt(value: Receipt | null) { if (value) localStorage.setItem(RECEIPT_KEY, JSON.stringify(value)); else localStorage.removeItem(RECEIPT_KEY); setReceipt(value) }
  function change<K extends keyof Spec>(key: K, value: Spec[K]) {
    const phrases: Partial<Record<keyof Spec, RegExp>> = {
      episodes: /(?:采集|收集|录制|记录)\s*[\d零一二三四五六七八九十两]+\s*(?:条|次|组)/gi,
      seed: /(?:seed|种子)\s*[:：=]?\s*\d+/gi,
      belt_speed_m_s: /(?:带速|传送带速度)\s*[:：=为]?\s*\d+(?:\.\d+)?\s*(?:cm\/s|m\/s|厘米每秒|米每秒|厘米\/秒|米\/秒)/gi,
      retreat_distance_m: /(?:向后移动|后退|向后行驶)\s*\d+(?:\.\d+)?\s*(?:厘米|公分|米|cm|m)/gi,
      random_pose: /随机初始姿态|随机姿态|不同姿态|固定初始姿态|固定姿态|不要随机|不随机|直立/gi,
      target_xy: /(?:坐标|位置|箱子|箱体|目标)\s*[:：为]?\s*[（(]\s*-?\d+(?:\.\d+)?\s*[,，]\s*-?\d+(?:\.\d+)?\s*[）)]|x\s*[:=：]\s*-?\d+(?:\.\d+)?\s*[,，;；\s]+y\s*[:=：]\s*-?\d+(?:\.\d+)?/gi,
    }
    setSpec(current => {
      const next = { ...current, [key]: value, ...(key === 'instruction' ? { task_kind: 'deliver' as const, task_sequence: [] } : {}), ...(key === 'object_asset' ? { object_parameters: {}, object_parameter_sources: {} } : {}), ...(phrases[key] ? { instruction: current.instruction.replace(phrases[key]!, key === 'retreat_distance_m' ? '向后移动' : key === 'target_xy' ? '目标箱' : '').replace(/([，,；;])\s*[，,；;]/g, '$1') } : {}) }
      if (key === 'object_mode' && value === 'reuse' && !capabilities?.objects.some(asset => asset.id === current.object_asset)) {
        next.object_asset = capabilities?.objects[0]?.id ?? DEFAULT.object_asset
        next.object_parameters = {}; next.object_parameter_sources = {}
      }
      if (key === 'bin_mode' && value === 'reuse' && !capabilities?.bins?.some(asset => asset.id === current.bin_asset)) {
        next.bin_asset = capabilities?.bins?.[0]?.id ?? null
        if (!next.bin_asset) next.bin_mode = 'builtin'
      }
      return next
    })
    if (key === 'object_asset' || key === 'object_description') setSuggestion(null)
    if (key === 'controller' || key === 'policy_id') setCheckpointNotice('')
    setPlan(null); setError('')
  }
  function parameter(key: keyof Parameters, value: string, factor: number) {
    change('object_parameters', { ...spec.object_parameters, [key]: value === '' ? null : Number(value) / factor })
    setSpec(current => {
      const sources = { ...current.object_parameter_sources }
      if (value === '') delete sources[key]
      else sources[key] = 'user_explicit'
      return { ...current, object_parameter_sources: sources }
    })
  }
  function boxSize(index: number, value: string) { const next = [...spec.bin_size_m] as Spec['bin_size_m']; next[index] = Number(value) / 100; change('bin_size_m', next) }
  async function suggest() {
    if (locked) return
    setBusy('suggest'); setError('')
    try {
      const current = Object.fromEntries(Object.entries(spec.object_parameters).filter(([key, value]) => value != null && (!spec.object_parameter_sources[key] || spec.object_parameter_sources[key] === 'user_explicit')))
      const result = await request<Suggestion>('suggest-object', { description: spec.object_description, current_parameters: current }, undefined, 30000)
      const sources = Object.fromEntries(Object.entries(result.provenance ?? {}).map(([key, value]) => [key, value.source]))
      const explicitSources = Object.fromEntries(Object.keys(current).map(key => [key, 'user_explicit']))
      setSuggestion(result); setSpec(value => ({ ...value, object_parameters: { ...result.parameters, ...current }, object_parameter_sources: { ...sources, ...explicitSources } })); setPlan(null)
    } catch (err) { setError('参数建议未完成：' + errorText(err)) } finally { setBusy('') }
  }
  useEffect(() => {
    if (!runId) return
    let disposed = false
    let timer: ReturnType<typeof setTimeout>
    const controller = new AbortController()
    const poll = async () => {
      try {
        const result = await request<Run>(`runs/${runId}?compact=true`, undefined, controller.signal, 8000)
        if (disposed) return
        setRun(previous => previous?.id === result.id && previous.updated > result.updated ? previous : result)
        setSpec(normalize(result.spec)); if (result.plan) setPlan(result.plan); setOnline(true)
        setError(value => value.startsWith('暂时无法读取实验进度') ? '' : value)
        if (active(result)) {
          const delay = result.status === 'running' && view === 'simulation' ? 350 : 900
          timer = setTimeout(poll, delay)
        }
      } catch (err) {
        if (disposed) return
        if (err instanceof HttpError && err.status === 404) {
          saveReceipt(null); setRun(null); setPlan(null); setOnline(true)
          setError('没找到这次实验的已保存记录。任务设置仍在，可以重新检查方案。')
          return
        }
        setOnline(false); setError('暂时无法读取实验进度，正在恢复：' + errorText(err))
        timer = setTimeout(poll, 3000)
      }
    }
    void poll()
    return () => { disposed = true; controller.abort(); clearTimeout(timer) }
  }, [runId, retry, view])
  async function inspect() {
    if (locked) return
    if (!policyReady) { setError('所选学习策略尚未就绪，请在执行策略页刷新连接状态。'); return }
    setBusy('plan'); setError('')
    try { const result = await request<Plan>('plan', spec); setPlan(result); setSpec(normalize(result.spec)); setOnline(true) } catch (err) { setError('方案检查未完成：' + errorText(err)) } finally { setBusy('') }
  }
  async function submit() {
    if (submission.current || run || busy || (!receipt && !plan?.ready)) return
    if (!receipt && !policyReady) { setError('所选学习策略尚未就绪，请在执行策略页刷新连接状态。'); return }
    submission.current = true; setBusy('submit'); setError('')
    const value = receipt ?? { spec: plan!.spec, client_request_id: crypto.randomUUID() }
    saveReceipt(value)
    try {
      const result = await request<Run>('runs', { spec: value.spec, client_request_id: value.client_request_id })
      saveReceipt({ ...value, run_id: result.id }); setRun(result); setSpec(normalize(result.spec)); if (result.plan) setPlan(result.plan); setOnline(true)
      go(['generating', 'awaiting_asset_review'].includes(result.status) || (result.status === 'queued' && (result.plan?.generation_count ?? plan?.generation_count ?? 0) > 0) ? 'models' : 'simulation')
    } catch (err) {
      if (err instanceof HttpError && [400, 409, 422, 507].includes(err.status)) saveReceipt(null)
      setError('启动未完成：' + errorText(err))
    } finally { submission.current = false; setBusy('') }
  }
  async function stop() {
    if (!runId || busy) return
    setBusy('stop'); setError('')
    try {
      const result = await request<Run>(`runs/${runId}/stop`, {}, undefined, 7000)
      setRun(previous => previous?.id === result.id && previous.updated > result.updated ? previous : result)
    } catch (err) {
      setError('尚未确认是否停止：' + errorText(err))
      setRetry(value => value + 1)
    } finally { setBusy('') }
  }
  function startNewExperiment(policy?: PolicyCapability) {
    if (!canOpenHistory) return
    setSpec(current => {
      const object = capabilities?.objects.find(asset => asset.id === current.object_asset) ?? capabilities?.objects[0]
      const bin = capabilities?.bins?.find(asset => asset.id === current.bin_asset)
      return { ...current, experiment_name: '', object_name: '', bin_name: '', object_mode: 'reuse', object_asset: object?.id ?? DEFAULT.object_asset, ...(object?.id !== current.object_asset ? { object_parameters: {}, object_parameter_sources: {} } : {}), bin_mode: bin ? 'reuse' : 'builtin', bin_asset: bin?.id ?? null, ...(policy ? { controller: 'vla', policy_id: policy.id } : {}) }
    })
    saveReceipt(null); setRun(null); setPlan(null); setError(''); setCamera('front'); setTrajectory([]); setEvaluation(null); setSuggestion(null); trajectoryKey.current = ''
    setCheckpointNotice(policy ? `已选用 ${policy.label}` : '')
    setSearchParams({ view: policy ? 'policy' : 'task' }, { replace: !!policy })
  }
  function fresh() { startNewExperiment() }
  async function reviewAsset(asset: AssetReview, decision: AssetDecision) {
    if (!runId || busy || asset.status !== 'pending') return
    const key = `${runId}:${asset.role}`
    const saved = reviewReceipts[key]
    if (saved && saved.decision !== decision) return
    const receipt = saved ?? { decision, client_request_id: crypto.randomUUID() }
    localStorage.setItem(REVIEW_KEY, JSON.stringify({ ...reviewReceipts, [key]: receipt }))
    setReviewReceipts(current => ({ ...current, [key]: receipt })); setBusy(`review:${asset.role}`); setError('')
    try {
      const result = await request<Run>(`runs/${runId}/assets/${asset.role}/review`, receipt)
      setRun(previous => previous?.id === result.id && previous.updated > result.updated ? previous : result)
      setReviewReceipts(current => { const next = { ...current }; delete next[key]; localStorage.setItem(REVIEW_KEY, JSON.stringify(next)); return next })
      setRetry(value => value + 1)
      if (result.status !== 'awaiting_asset_review' && decision !== 'reject') go('simulation')
    } catch (err) {
      if (err instanceof HttpError && [400, 409, 422].includes(err.status)) setReviewReceipts(current => { const next = { ...current }; delete next[key]; return next })
      setError('模型意见尚未确认，请使用同一操作重试：' + errorText(err))
    } finally { setBusy('') }
  }
  async function openHistory(id: string) {
    if (!canOpenHistory) return
    setBusy('history'); setError('')
    try {
      const result = await request<Run>(`runs/${id}?compact=true`)
      saveReceipt({ run_id: result.id, spec: normalize(result.spec), client_request_id: `history-${result.id}` })
      setRun(result); setSpec(normalize(result.spec)); setPlan(result.plan ?? null); setEvaluation(null); setRetry(value => value + 1); go('simulation')
    } catch (err) { setError('无法读取历史实验：' + errorText(err)) } finally { setBusy('') }
  }
  async function downloadArtifact(path: string, filename: string, format: 'zip' | 'json' | 'npz' = 'zip') {
    if (downloading) return
    setDownloading(path); setDownloadProgress(null); setError('')
    try {
      const parts: Blob[] = []
      const chunkSize = 8 * 1024 * 1024
      let offset = 0, total = 0, validator = ''
      do {
        const response = await fetch(path, { headers: { Range: `bytes=${offset}-${offset + chunkSize - 1}`, ...(validator ? { 'If-Range': validator } : {}) }, cache: 'no-store' })
        if (!response.ok || response.status === 204) {
          const detail = await response.json().catch(() => null)
          throw new Error(typeof detail?.detail === 'string' ? detail.detail : `文件请求失败 (${response.status})`)
        }
        const chunk = await response.blob()
        if (response.status === 206) {
          const range = /^bytes (\d+)-(\d+)\/(\d+)$/.exec(response.headers.get('Content-Range') ?? '')
          if (!range || Number(range[1]) !== offset || Number(range[2]) < offset || Number(range[2]) >= Number(range[3]) || !Number.isSafeInteger(Number(range[3])) || Number(range[2]) - offset + 1 !== chunk.size || total && total !== Number(range[3])) throw new Error('下载分段不完整，请重新下载。')
          const currentValidator = response.headers.get('ETag') ?? response.headers.get('Last-Modified') ?? ''
          if (validator && validator !== currentValidator) throw new Error('文件在下载期间发生变化，请重新下载。')
          validator = currentValidator; total = Number(range[3]); offset = Number(range[2]) + 1
        } else if (response.status === 200 && offset === 0) {
          total = chunk.size; offset = total
        } else throw new Error('服务器无法继续分段下载，请重新下载。')
        parts.push(chunk); setDownloadProgress({ received: offset, total })
      } while (offset < total)
      const blob = new Blob(parts, { type: format === 'json' ? 'application/json' : format === 'npz' ? 'application/octet-stream' : 'application/zip' })
      if (blob.size !== total) throw new Error('下载文件长度不完整，请重新下载。')
      if (format === 'json') {
        try { JSON.parse(await blob.text()) } catch { throw new Error('服务器没有返回有效的 JSON 文件，请稍后重试。') }
      } else {
        const magic = new Uint8Array(await blob.slice(0, 2).arrayBuffer())
        if (magic[0] !== 80 || magic[1] !== 75) throw new Error('服务器没有返回有效的数据包，请稍后重试。')
      }
      const url = URL.createObjectURL(blob)
      const anchor = document.createElement('a')
      anchor.href = url; anchor.download = filename; document.body.append(anchor); anchor.click(); anchor.remove()
      setTimeout(() => URL.revokeObjectURL(url), 60000)
    } catch (err) { setError('下载未完成：' + errorText(err)) } finally { setDownloading(''); setDownloadProgress(null) }
  }
  async function recover() {
    if (!runId || busy) return
    setBusy('recover'); setError('')
    try { const result = await request<Run>(`runs/${runId}/retry`, {}); setRun(result); setRetry(value => value + 1) } catch (err) { setError('恢复未完成：' + errorText(err)) } finally { setBusy('') }
  }

  const rows = run?.episodes ?? []
  const frames = rows.reduce((sum, row) => sum + row.frames, 0)
  const successes = rows.filter(row => row.success).length
  const contacts = metricNumber(preview?.metrics, 'contacts', 'grasp_contacts')
  const retreat = metricNumber(preview?.metrics, 'travelled_distance_m', 'distance_m', 'retreated_distance_m') ?? preview?.distance_m ?? preview?.transport_distance_m
  const remaining = metricNumber(preview?.metrics, 'goal_distance_m') ?? preview?.goal_distance_m
  const image = source(preview?.images?.[camera])
  const stage = preview?.phase_label ?? (run?.status === 'generating' ? '按要求生成资产' : run?.status === 'queued' ? '组装场景与检查物理' : run?.status === 'succeeded' ? '完整流程结束' : '等待仿真开始')
  const dataset = rows.length ? run?.dataset : undefined
  const generationCount = plan?.generation_count ?? Number(spec.object_mode === 'generate') + Number(spec.bin_mode === 'generate')
  const route = preview?.planned_route ?? plan?.navigation?.planned_route ?? []
  const pendingReviews = run?.asset_reviews?.filter(asset => asset.status === 'pending') ?? []
  const flowStage = run ? ['generating', 'awaiting_asset_review'].includes(run.status) ? 1 : active(run) ? 2 : 3 : 0
  const heading: Record<View, [string, string]> = {
    home: ['描述一个任务，开始一次验证。', '依次准备任务、模型与策略，检查执行结果，再获取采集数据。'],
    simulation: ['查看任务，正在怎样执行。', '只关注现场相机、当前阶段和机器人状态。'],
    results: ['查看这次任务的可行性结果。', '检查阶段是否完成、失败原因与数据质量。'],
    policy: ['选择由谁，执行这次任务。', '配置视觉语言动作策略，或使用状态专家基线进行对照。'],
    task: ['先描述，需要验证什么。', '明确抓取对象、目的地和采集数量，再检查执行范围。'],
    models: ['先看模型，再决定使用。', '新外形生成后暂停，逐个确认是否保存到本地资产库。'],
    datasets: ['把验证过程，变成数据。', '保留成功与失败样本，导出相机、状态、动作和可行性报告。'],
    settings: ['为这次验证，设置条件。', '机器人、物性、箱体尺寸和目标位置集中在这里。'],
  }
  const cameraPanel = <section className="tt-camera-card tt-card"><div className="tt-section-head"><div><i className={run?.status === 'running' ? 'live' : ''}/><h2>任务验证现场</h2><span>MUJOCO / LIVE</span></div><strong>{run ? STATUS[run.status] : '等待任务'}</strong></div>
    <div className="tt-camera">{image ? <img src={image} alt={camera === 'front' ? '移动搬运前视相机画面' : camera === 'wrist' ? '移动搬运腕部相机画面' : '移动搬运车载相机画面'}/> : <div className="tt-camera-empty"><span>◉</span><strong>{waitingReview ? '模型已生成，等待你的意见' : run?.status === 'generating' ? '正在生成本轮场景模型' : run ? '正在准备后端仿真画面' : '从一个明确的具身任务开始'}</strong><p>{waitingReview ? '前往模型资产页旋转查看，选择保存或仅本次使用。' : run?.status === 'generating' ? '模型资产页显示提交、生成、下载与导入阶段。' : '当前支持不停带抓取、移动搬运与放入目标箱的组合验证。'}</p>{waitingReview || run?.status === 'generating' ? <button className="tt-camera-action" onClick={() => go('models')}>查看模型与进度 →</button> : !run && <button className="tt-camera-action" onClick={() => go('task')}>描述验证任务 →</button>}</div>}
      <div className="tt-camera-overlay"><span>{camera === 'front' ? 'FRONT / 全局前视' : camera === 'wrist' ? 'WRIST / 腕部视角' : 'ONBOARD / 车载视角'}</span><span>{(preview?.sim_time ?? 0).toFixed(2)} s · Tick {preview?.tick ?? 0}</span></div>{run?.status === 'running' && <div className="tt-live-phase"><i/>{stage}</div>}</div>
    <div className="tt-camera-tabs"><div><button className={camera === 'front' ? 'selected' : ''} onClick={() => setCamera('front')}>前视相机</button><button className={camera === 'wrist' ? 'selected' : ''} onClick={() => setCamera('wrist')}>腕部相机</button>{preview?.images?.onboard && <button className={camera === 'onboard' ? 'selected' : ''} onClick={() => setCamera('onboard')}>车载相机</button>}</div><span>{stage}</span></div>
  </section>
  const metricsPanel = <section className="tt-metrics tt-card"><div><span>距停靠点</span><strong>{remaining === undefined ? '—' : remaining.toFixed(2)}<small>m</small></strong></div><div><span>夹爪接触</span><strong>{contacts ?? '—'}<small>/ 2 侧</small></strong></div><div><span>累计移动</span><strong>{retreat === undefined ? '—' : retreat.toFixed(2)}<small>m</small></strong></div><div><span>底盘位置</span><strong className="tt-position">{preview?.pose ? `${preview.pose.x.toFixed(2)}, ${preview.pose.y.toFixed(2)}` : '—'}<small>m</small></strong></div></section>
  const reportPanel = <section className="tt-evaluation tt-card"><div className="tt-section-head"><div><h2>可行性结果</h2><span>当前条件 / 当前策略</span></div>{evaluation && <a href={evaluation.downloads.report_url} download aria-disabled={!!downloading} onClick={event => { event.preventDefault(); void downloadArtifact(`/api/v1/transport/runs/${run?.id}/evaluation`, `${run?.id}_evaluation.json`, 'json') }}>下载验证报告 ↓</a>}</div>
    <div className="tt-evaluation-body">{run ? <><div className={'tt-verdict ' + (evaluation?.verdict.code ?? 'pending')}><span>{evaluation?.verdict.code === 'passed' ? '✓' : evaluation?.verdict.code === 'failed' || evaluation?.verdict.code === 'invalid_data' ? '!' : '◷'}</span><div><strong>{evaluation?.verdict.label ?? STATUS[run.status]}</strong><p>{evaluation?.verdict.reason ?? run.error ?? (waitingReview ? '确认本轮所有新模型后，才会进入仿真。' : '结果按真实执行与数据完整性判断。')}</p></div></div>
      {evaluation?.verdict.scope && <p className="tt-report-scope">{evaluation.verdict.scope}</p>}
      {evaluation?.episodes.map(episode => <div className="tt-episode-report" key={episode.index}>
        <div><strong>Episode {String(episode.index + 1).padStart(3, '0')}</strong><span>{episode.success ? '验证通过' : '验证未通过'} · {episode.duration_s.toFixed(2)} s</span></div>
        {episode.failure && <p className="tt-failure-reason"><b>失败阶段：{episode.failure.stage}</b>{episode.failure.reason}</p>}
        {(() => { const reach = reachOf(episode); return reach ? <p className="tt-reach-note"><b>可达性约束 {reach.steps} 步</b>机械臂目标超出可达范围后被投影到可执行位置{reach.deviationMm !== undefined ? `，最大偏差 ${reach.deviationMm} mm` : ''}。这类步骤由物理可达性限制，不改变成功判定，也不代表策略完成了抓取。</p> : null })()}
        {episode.phase_timeline.length > 0 && <div className="tt-phase-timeline">{episode.phase_timeline.map(item => <span key={item.phase} className={episode.failure?.phase === item.phase ? 'failed' : ''} title={`${item.start_s.toFixed(2)}–${item.end_s.toFixed(2)} s`}>{item.label}</span>)}</div>}
        {(run.spec.task_kind || !!episode.task_events?.length) && !!episode.subtasks?.length && <div className="tt-subtask-report"><strong>{episode.program_complete ? '任务程序已完整执行' : '任务程序尚未完成'}<small>{episode.subtasks.filter(task => task.completed).length} / {episode.subtasks.length} 步</small></strong>{episode.subtasks.map(task => <div className={task.completed ? 'complete' : ''} key={task.stage}><span>{task.completed ? '✓' : '○'}</span><div><strong>{task.label}</strong><p>{eventEvidence(task.evidence)}</p></div></div>)}</div>}
        {!episode.data_quality.valid && <p className="tt-failure-reason">数据检查：{episode.data_quality.issues.join('；')}</p>}
      </div>)}
      {evaluation && evaluation.summary.episodes > 0 && <div className={'tt-quality ' + (evaluation.data_quality.valid ? '' : 'invalid')}><strong>{evaluation.data_quality.valid ? '数据检查通过' : '数据质量需要检查'}</strong><span>{evaluation.summary.frames} 帧 · {evaluation.summary.successes}/{evaluation.summary.episodes} 条通过</span>{evaluation.data_quality.issues.map(issue => <p key={issue}>{issue}</p>)}</div>}
      {evaluation && evaluation.summary.episodes === 0 && <div className="tt-quality"><strong>等待采集数据</strong><span>尚未检查数据质量</span></div>}
      {run.error && !evaluation?.episodes.some(episode => episode.failure) && <p className="tt-failure-reason">{run.error}</p>}
      {evaluationError && <p className="tt-helper">报告暂时不可读取：{evaluationError}<button className="tt-inline-button" onClick={() => setRetry(value => value + 1)}>重新读取</button></p>}
    </> : <div className="tt-empty-note"><strong>验证会回答：这次任务完成了吗？</strong><p>展示失败阶段、接触与移动情况，并检查采集数据。结果只适用于本轮配置。</p></div>}
    <p className="tt-policy-disclosure">本轮执行策略：{evaluation?.controller.label ?? '等待后端确认'}。可行性结论仅对应本轮策略和场景条件。</p>
    {evaluation?.limitations.length ? <details className="tt-report-limitations"><summary>查看验证边界</summary>{evaluation.limitations.map(item => <p key={item}>{item}</p>)}</details> : null}</div>
  </section>
  const planPanel = <section className="tt-plan tt-card"><div className="tt-card-title"><span>EXECUTION PLAN</span><h2>{run ? STATUS[run.status] : '检查执行范围与方案'}</h2></div>{!run && <div className="tt-plan-check"><button className="tt-secondary tt-full" disabled={locked || !capabilities || !spec.instruction.trim() || !policyReady} onClick={() => void inspect()}>{busy === 'plan' ? '正在检查…' : '检查执行方案'}<span aria-hidden="true">→</span></button><p className="tt-plan-controller">执行策略：{controllerLabel}<button onClick={() => go('policy')}>查看策略 →</button></p>{!policyReady && <p className="tt-helper">VLA 服务尚未就绪，连接确认后可检查与启动。</p>}</div>}
    {plan ? <div className="tt-plan-body"><p className="tt-plan-summary">{plan.summary}</p>{!!plan.task_sequence?.length && <div className="tt-program" aria-label="后端确认的任务顺序">{plan.task_sequence.map((step, index) => <span key={`${step}-${index}`}>{index + 1}. {TASK_STEP[step] ?? step}</span>)}</div>}<ol>{plan.steps.map((step, index) => <li key={step}><span>{String(index + 1).padStart(2, '0')}</span>{step}</li>)}</ol>{plan.issues.length > 0 && <div className="tt-notices tt-issues"><strong>当前无法执行</strong>{plan.issues.map(issue => <p key={issue}>{issue}</p>)}</div>}
      <div className="tt-reuse"><span>本次在线生成<strong>{generationCount}<small>件资产</small></strong></span><p>{generationCount ? '使用 Aholo 额度生成；生成后先等待你的模型意见。' : '复用已有资产，直接进行仿真验证。'}</p></div>
      {!run && <button className="tt-primary tt-full" disabled={!plan.ready || !policyReady || !!busy || !!receipt?.run_id || (generationCount > 0 && !capabilities?.configured)} onClick={() => void submit()}>{busy === 'submit' ? '正在提交…' : generationCount ? '生成模型，等待确认' : '开始仿真验证与采集'}<span aria-hidden="true">↗</span></button>}
      {waitingReview && <button className="tt-primary tt-full" onClick={() => go('models')}>确认本轮模型 <span>→</span></button>}
      {running && <button className="tt-stop tt-full tt-spaced" disabled={!!busy || run?.stop_requested || run?.status === 'stopping'} onClick={() => void stop()}>{busy === 'stop' || run?.status === 'stopping' ? '正在停止并保存…' : waitingReview ? '放弃本轮实验' : '停止实验并保存数据'}</button>}
      {run && !running && ['failed', 'interrupted'].includes(run.status) && !run.configuration && !run.episodes.length && <button className="tt-secondary tt-full tt-spaced" disabled={!!busy} onClick={() => void recover()}>恢复资产处理</button>}
      {run && !running && <button className="tt-secondary tt-full tt-spaced" onClick={fresh}>调整要求，开始新实验</button>}
      {plan.warnings.length > 0 && <details className="tt-plan-warnings"><summary>执行范围与注意事项</summary>{plan.warnings.map(warning => <p key={warning}>{warning}</p>)}</details>}
      {run?.error && <p className="tt-run-error">{runErrorText(run.error)}</p>}</div> : <div className="tt-plan-empty"><span>◉</span><p>检查目标可达性、物品参数和生成数量，再决定是否开始。</p><small>检查方案不会创建在线模型生成任务。</small></div>}
  </section>
  const taskPanel = <section className="tt-requirements tt-card"><div className="tt-card-title"><span>TASK DESCRIPTION</span><h2>你希望验证什么任务？</h2><p>直接描述动作先后顺序、对象和目的地。检查后会列出实际执行步骤。</p></div><div className="tt-form"><label className="tt-field">实验名称 <small>选填</small><input type="text" placeholder="例如：小鸭抓取 · 默认箱位" value={spec.experiment_name} maxLength={80} disabled={locked} onChange={event => change('experiment_name', event.target.value)}/></label>
    <label className="tt-field tt-spaced">任务要求<textarea aria-label="移动搬运任务要求" value={spec.instruction} disabled={locked} maxLength={1500} onChange={event => change('instruction', event.target.value)}/></label>
    <div className="tt-task-examples"><span>试试这些任务</span>{[
      ['抓取入箱', '从不停移动的传送带上抓取物品，移动到指定箱子旁并放入，采集1条。'],
      ['抓起后放回', '从不停移动的传送带上抓起物品，抬起确认后，沿原路径放回传送带，采集1条。'],
      ['入箱后取出', '从不停移动的传送带上抓取物品，先放到箱子里，然后从箱子里拿出来，抬到箱子上方并保持夹持，采集1条。'],
      ['入箱取回并返带', '从不停移动的传送带上抓取物品，先放到箱子里，然后从箱子拿出，沿原路放回传送带，采集1条。'],
    ].map(([label, instruction]) => <button key={label} disabled={locked} onClick={() => change('instruction', instruction)}>{label} ↗</button>)}</div><p className="tt-helper">示例会填入描述，你可以继续修改。执行内容以后端检查确认的步骤为准。</p></div></section>
  const objectModelPanel = <section className="tt-object tt-card"><div className="tt-card-title"><span>OBJECT MODEL</span><h2>传送带上放什么？</h2><p>复用资产，或描述一个适合夹爪的小型刚体。</p></div><div className="tt-form"><div className="tt-segments"><button className={spec.object_mode === 'reuse' ? 'selected' : ''} disabled={locked} onClick={() => change('object_mode', 'reuse')}>已有物品</button><button className={spec.object_mode === 'generate' ? 'selected' : ''} disabled={locked} onClick={() => change('object_mode', 'generate')}>生成新物品</button></div>
    {spec.object_mode === 'reuse' && <label className="tt-field tt-spaced">抓取物品<select aria-label="移动搬运物品" disabled={locked || !capabilities} value={spec.object_asset} onChange={event => change('object_asset', event.target.value)}>{capabilities?.objects.map(item => <option key={item.id} value={item.id}>{item.name}</option>)}</select></label>}
    {spec.object_mode === 'generate' && <label className="tt-field tt-spaced">物品模型名称 <small>选填</small><input type="text" placeholder="例如：蓝色积木" value={spec.object_name} maxLength={80} disabled={locked} onChange={event => change('object_name', event.target.value)}/></label>}
    <label className="tt-field tt-spaced">{spec.object_mode === 'generate' ? '新物品外形与参数要求' : '物品参数要求'}<textarea aria-label="物品外形和参数要求" disabled={locked} maxLength={1500} value={spec.object_description} onChange={event => change('object_description', event.target.value)}/></label>
    <p className="tt-helper">{spec.object_mode === 'generate' ? '生成后先展示模型。你可以保存到本地资产库、仅本次使用，或放弃本轮。' : '已有物品保留原外形。更改外形请选择“生成新物品”。'}</p><button className="tt-secondary tt-full tt-spaced" onClick={() => go('settings')}>设置尺寸、质量与摩擦 →</button></div></section>
  const binModelPanel = <section className="tt-bin tt-card"><div className="tt-card-title"><span>BIN MODEL</span><h2>目的地箱子长什么样？</h2><p>箱体物理尺寸与颜色在参数设置中调整。</p></div><div className="tt-form"><label className="tt-field">箱子外观方式<select aria-label="箱子外观方式" value={spec.bin_mode} disabled={locked} onChange={event => change('bin_mode', event.target.value as Spec['bin_mode'])}><option value="builtin">参数化开口箱</option><option value="generate">Aholo 生成新外观</option><option value="reuse">复用已生成外观</option></select></label>
    {spec.bin_mode === 'reuse' && <label className="tt-field tt-spaced">已有箱子外观<select aria-label="已有箱子外观" value={spec.bin_asset ?? ''} disabled={locked} onChange={event => change('bin_asset', event.target.value || null)}><option value="">请选择开口箱模型</option>{capabilities?.bins?.map(item => <option key={item.id} value={item.id}>{item.name}</option>)}</select></label>}
    {spec.bin_mode === 'generate' && <><label className="tt-field tt-spaced">箱子模型名称 <small>选填</small><input type="text" placeholder="例如：青绿色收纳箱" value={spec.bin_name} maxLength={80} disabled={locked} onChange={event => change('bin_name', event.target.value)}/></label>
    <label className="tt-field tt-spaced">箱子外观要求<textarea aria-label="箱子外观要求" value={spec.bin_description} maxLength={1500} disabled={locked} onChange={event => change('bin_description', event.target.value)}/></label></>}
    <p className="tt-helper">生成网格用于外观，仿真按照指定尺寸构建敞口碰撞箱体。</p></div></section>
  const progressPanel = run?.asset_steps?.length ? <section className="tt-assets tt-card"><div className="tt-section-head"><div><h2>模型生成进度</h2><span>按实际阶段更新</span></div><strong>{STATUS[run.status]}</strong></div><div className="tt-generation-list">{run.asset_steps.map(step => <AssetProgress key={step.role} step={step} review={run.asset_reviews?.find(asset => asset.role === step.role)}/>)}</div></section> : null
  const reviewsPanel = run?.asset_reviews?.length ? <section className="tt-reviews"><div className="tt-review-intro"><span>MODEL REVIEW</span><h2>{waitingReview ? `还有 ${pendingReviews.length} 个模型等待你的意见` : '本轮模型与保存状态'}</h2><p>先确认外观与结构。所有新模型获准使用后，仿真才会开始；保存操作会加入本机资产库。</p></div>{run.asset_reviews.map(asset => {
    const receipt = reviewReceipts[`${run.id}:${asset.role}`]
    return <article className="tt-review-card tt-card" key={asset.role}><div className="tt-section-head"><div><h2 className="tt-custom-name">{asset.name || roleLabel(asset.role)}</h2></div><strong className={asset.saved ? 'tt-saved' : ''}>{asset.saved ? '已保存到本地资产库' : asset.status === 'rejected' ? '已放弃' : asset.decision === 'use_once' ? '仅本次使用 · 缓存保留' : '待确认 · 尚未保存到资产库'}</strong></div><AssetPreview asset={asset}/><div className="tt-review-details"><p>{asset.description}</p><div className="tt-review-parameters">{asset.role === 'object' ? PARAMS.filter(item => asset.parameters[item.key] != null).map(item => <span key={item.key}>{item.label}<strong>{Number((asset.parameters[item.key]! * item.factor).toFixed(2))} {item.unit}</strong></span>) : <span>本轮箱体尺寸<strong>{spec.bin_size_m.map(value => Number((value * 100).toFixed(1))).join(' × ')} cm</strong></span>}</div><p className="tt-helper">预览展示生成外观；仿真会按本轮参数标定尺寸，并使用后端碰撞代理。</p>
      {asset.saved_at && <p className="tt-storage-note">本地保存于 {new Date(asset.saved_at).toLocaleString('zh-CN')}</p>}<div className="tt-review-downloads"><a href={asset.download_url ?? asset.package_url} download aria-disabled={!!downloading} onClick={event => { event.preventDefault(); void downloadArtifact(asset.download_url ?? asset.package_url, `${run.id}-${asset.role}.zip`) }}>下载模型包 ZIP ↓</a>{asset.session_id && <Link to={`/generation/${asset.session_id}`}>生成记录 ↗</Link>}</div>
      {asset.status === 'pending' && <>{receipt && <div className="tt-notices">上次意见的提交结果尚未确认，请重试相同操作。刷新不会自动接受模型。</div>}<div className="tt-review-actions"><button className="tt-primary" disabled={!!busy || !!receipt && receipt.decision !== 'save'} onClick={() => void reviewAsset(asset, 'save')}>{busy === `review:${asset.role}` ? '正在提交意见…' : receipt?.decision === 'save' ? '重试保存并使用' : '保存到本地资产库并使用'}</button><button className="tt-secondary" disabled={!!busy || !!receipt && receipt.decision !== 'use_once'} onClick={() => void reviewAsset(asset, 'use_once')}>{receipt?.decision === 'use_once' ? '重试仅本次使用' : '仅本次使用'}</button><button className="tt-reject" disabled={!!busy || !!receipt && receipt.decision !== 'reject'} onClick={() => void reviewAsset(asset, 'reject')}>{receipt?.decision === 'reject' ? '重试放弃' : '放弃本轮'}</button></div></>}
    </div></article>
  })}</section> : null
  const policyPanel = <div className="tt-policy-view"><section className="tt-card"><div className="tt-card-title"><h2>选择执行策略</h2></div><div className="tt-policy-choices">
    <button className={'tt-policy-choice ' + (spec.controller === 'expert' ? 'selected' : '')} aria-pressed={spec.controller === 'expert'} disabled={locked} onClick={() => { change('controller', 'expert'); change('policy_id', null) }}><h3>状态专家基线</h3><p>适合先验证场景和采集示范。</p><small>{spec.controller === 'expert' ? '✓ 本轮已选择' : '选择基线'} →</small></button>
    <button className={'tt-policy-choice ' + (spec.controller === 'vla' ? 'selected' : '')} aria-pressed={spec.controller === 'vla'} disabled={locked} onClick={() => { change('controller', 'vla'); change('policy_id', spec.policy_id ?? policies?.default_policy_id ?? policies?.policies?.[0]?.id ?? null) }}><h3>学习策略</h3><p>由右侧选中的 SmolVLA 或移动 BC 生成动作。</p><small>{spec.controller === 'vla' ? '✓ 本轮已选择' : '选择学习策略'} →</small></button>
  </div>{locked && <p className="tt-policy-locked">本轮策略已随实验锁定。{run && !running && <button className="tt-inline-button" onClick={fresh}>开始新实验以调整</button>}</p>}<div className="tt-policy-next"><button className="tt-secondary" onClick={() => go('settings')}>推理与采集参数 →</button><button className="tt-primary" onClick={() => go('task')}>返回任务检查方案 →</button></div></section>
    <section className="tt-card tt-policy-connection"><div className="tt-section-head"><div><h2>策略推理连接</h2><span>读取后端实际状态</span></div><button className="tt-inline-button" disabled={policyChecking} onClick={() => setRetry(value => value + 1)}>{policyChecking ? '正在检查…' : '刷新连接 ↻'}</button></div><div className="tt-policy-details">
      <label className="tt-field">推理模型<select aria-label="推理模型" value={spec.controller === 'vla' ? spec.policy_id ?? policies?.default_policy_id ?? '' : ''} disabled={locked || !policies?.policies?.length} onChange={event => { change('policy_id', event.target.value || null); change('controller', 'vla') }}><option value="" disabled>{policies?.policies?.length ? '选择模型并启用学习策略' : '等待后端模型信息'}</option>{policies?.policies?.map(policy => <option key={policy.id} value={policy.id}>{policy.label}{policy.ready ? '' : '（未就绪）'}</option>)}</select></label>
      <div className={'tt-policy-state ' + (selectedPolicy?.ready && !policyError && !policyChecking ? 'ready' : '')} role="status"><i/><div><strong>{policyChecking ? '正在查询推理服务' : policyError ? '连接状态读取失败' : selectedPolicy ? POLICY_STATE[selectedPolicy.state] ?? selectedPolicy.state : '尚无可用的推理模型信息'}</strong><p>{policyError || (policyCheckedAt ? `最近检查 ${policyCheckedAt}` : '通过后端检查服务状态后，再决定是否启动。')}</p>{selectedPolicy?.error && <p>{selectedPolicy.error}</p>}</div></div>
      {selectedPolicy && <><dl className="tt-policy-facts"><div><dt>模型身份</dt><dd>{selectedPolicy.identity?.label ?? selectedPolicy.label}</dd></div>{policyDevice(selectedPolicy.identity) && <div><dt>推理设备</dt><dd aria-label="VLA 推理设备">{policyDevice(selectedPolicy.identity)}</dd></div>}<div><dt>动作接口</dt><dd>{selectedPolicy.action_schema} · {selectedPolicy.action_dim} 维</dd></div><div><dt>适配方式</dt><dd>{selectedPolicy.adapter ?? '等待服务报告'}</dd></div></dl>{selectedPolicy.limitations?.length > 0 && <div className="tt-policy-limitations"><strong>本模型的执行范围</strong>{selectedPolicy.limitations.map(item => <p key={item}>{item}</p>)}</div>}</>}
      {policies?.notes?.length ? <div className="tt-policy-notes">{policies.notes.map(note => <p key={note}>{note}</p>)}</div> : null}
    </div></section><PolicyGuide current={currentModel} /></div>
  const mapPanel = <section className="tt-map-card tt-card"><div className="tt-section-head"><div><h2>目标箱位置</h2><span>俯视场地图</span></div><strong>{locked ? '本轮位置已锁定' : '点击空地设置箱子中心'}</strong></div><WorkspaceMap target={target} size={spec.bin_size_m} color={spec.bin_color} pose={preview?.pose} route={route} trajectory={trajectory} disabled={locked} onPick={point => change('target_xy', point)}/>
    <div className="tt-map-controls"><label className="tt-field">目标 X <small>m</small><input aria-label="目标箱 X 坐标" type="number" min={MAP.left} max={MAP.right} step={.05} value={target[0]} disabled={locked} onChange={event => change('target_xy', [Number(event.target.value), target[1]])}/></label><label className="tt-field">目标 Y <small>m</small><input aria-label="目标箱 Y 坐标" type="number" min={MAP.bottom} max={MAP.top} step={.05} value={Number(target[1].toFixed(2))} disabled={locked} onChange={event => change('target_xy', [target[0], Number(event.target.value)])}/></label><button className="tt-secondary" disabled={locked} onClick={() => change('target_xy', null)}>恢复后方位置</button></div><div className="tt-map-legend"><span><i className="route"/>后端规划路线</span><span><i/>真实行驶轨迹</span></div><p className="tt-map-help">后端检查完整箱体边界、障碍重叠与可达路线。不可执行的位置会在方案中说明。</p></section>
  const robotSettings = <section className="tt-card"><div className="tt-card-title"><span>ROBOT & COLLECTION</span><h2>机器人与采集条件</h2></div><div className="tt-form"><div className="tt-form-grid tt-compact-grid"><label className="tt-field">机器人底盘<select aria-label="移动搬运底盘" disabled={locked || !capabilities} value={spec.chassis_asset} onChange={event => change('chassis_asset', event.target.value)}>{capabilities?.chassis.map(item => <option key={item.id} value={item.id}>{item.name}</option>)}</select></label><label className="tt-field">传送带速度 <small>cm/s</small><input aria-label="移动搬运带速" type="number" min={1} max={3} step={.1} disabled={locked} value={Number((spec.belt_speed_m_s * 100).toFixed(1))} onChange={event => change('belt_speed_m_s', Number(event.target.value) / 100)}/></label><label className="tt-field">采集次数<input aria-label="移动搬运采集次数" type="number" min={1} max={5} step={1} disabled={locked} value={spec.episodes} onChange={event => change('episodes', Number(event.target.value))}/></label><label className="tt-field">起始随机种子<input aria-label="移动搬运种子" type="number" min={0} max={999999} step={1} disabled={locked} value={spec.seed} onChange={event => change('seed', Number(event.target.value))}/></label></div>
    <label className="tt-speed">动作速度 <strong>{spec.speed_scale.toFixed(1)}×</strong><input aria-label="机器人动作速度" type="range" min={1} max={2} step={.1} value={spec.speed_scale} disabled={locked} onChange={event => change('speed_scale', Number(event.target.value))}/><span>1.0× 稳妥 <span>2.0× 更快</span></span></label><label className="tt-toggle"><input type="checkbox" checked={spec.random_pose} disabled={locked} onChange={event => change('random_pose', event.target.checked)}/>随机物品初始姿态</label></div></section>
  const objectSettings = <section className="tt-card"><div className="tt-card-title"><span>OBJECT PHYSICS</span><h2>抓取物品参数</h2><p>手动填写优先，空白沿用资产。自动建议未经实物测量。</p></div><div className="tt-form"><div className="tt-parameter-heading"><strong>尺寸与物性</strong><button disabled={locked || !spec.object_description.trim()} onClick={() => void suggest()}>{busy === 'suggest' ? '正在建议…' : '自动补全空白参数 ✧'}</button></div><div className="tt-parameters">{PARAMS.map(item => <label className="tt-field" key={item.key}>{item.label}<small>{item.unit}</small><input aria-label={`物品${item.label}`} type="number" min={item.min} max={item.max} step={item.step} placeholder="沿用资产" disabled={locked} value={spec.object_parameters[item.key] == null ? '' : Number((spec.object_parameters[item.key]! * item.factor).toFixed(3))} onChange={event => parameter(item.key, event.target.value, item.factor)}/></label>)}</div>
    {suggestion && <div className="tt-suggestion"><strong>{suggestion.source === 'model_inferred' ? 'AI 参数建议' : '规则估计参数'}</strong><span>已保留手动填写值</span>{suggestion.notes?.map(note => <p key={note}>{note}</p>)}</div>}{!suggestion && capabilities?.suggestion && !capabilities.suggestion.configured && <p className="tt-helper">参数建议语言模型尚未配置，自动补全使用规则估计。Aholo 外形生成独立可用。</p>}</div></section>
  const binSettings = <section className="tt-card"><div className="tt-card-title"><span>BIN PHYSICS</span><h2>箱体尺寸与颜色</h2></div><div className="tt-form"><div className="tt-bin-size">{['宽度', '深度', '箱壁高度'].map((label, index) => <label className="tt-field" key={label}>{label}<small>cm</small><input aria-label={`箱子${label}`} type="number" step={.1} min={index === 2 ? 4 : 16} max={index === 2 ? 9 : 30} value={Number((spec.bin_size_m[index] * 100).toFixed(2))} disabled={locked} onChange={event => boxSize(index, event.target.value)}/></label>)}</div><label className="tt-color">箱体颜色<input aria-label="箱体颜色" type="color" value={spec.bin_color} disabled={locked} onChange={event => change('bin_color', event.target.value)}/><span>{spec.bin_color.toUpperCase()}</span></label></div></section>
  const policySettings = <section className="tt-card"><div className="tt-card-title"><span>POLICY INFERENCE</span><h2>VLA 推理参数</h2><p>仅在选择 VLA 模型策略时使用。</p></div><div className="tt-form"><div className="tt-form-grid tt-compact-grid"><label className="tt-field">每次推理执行步数<input aria-label="VLA 每次执行步数" type="number" min={1} max={10} step={1} value={spec.vla_replan_steps} disabled={locked || spec.controller !== 'vla'} onChange={event => change('vla_replan_steps', Number(event.target.value))}/></label><label className="tt-field">单次推理超时 <small>s</small><input aria-label="VLA 推理超时" type="number" min={1} max={120} step={1} value={spec.vla_max_inference_s} disabled={locked || spec.controller !== 'vla'} onChange={event => change('vla_max_inference_s', Number(event.target.value))}/></label></div><button className="tt-secondary tt-full tt-spaced" onClick={() => go('policy')}>查看执行策略与连接 →</button></div></section>
  const dataPanel = <section className="tt-results tt-card"><div className="tt-section-head"><div><h2>验证与采集数据</h2><span>抓取 · 底盘 · 物体 · 相机</span></div>{dataset ? <div className="tt-exports"><a href={dataset.manifest_url} download aria-disabled={!!downloading} onClick={event => { event.preventDefault(); void downloadArtifact(dataset.manifest_url, `${run?.id}_manifest.json`, 'json') }}>下载 JSON 清单 ↓</a>{run?.configuration?.url && <a href={run.configuration.url} download aria-disabled={!!downloading} onClick={event => { event.preventDefault(); void downloadArtifact(run.configuration!.url, `${run.id}_config.json`, 'json') }}>实验配置 ↓</a>}</div> : <span>成功与失败均按实际保存</span>}</div><div className="tt-stats"><div><span>完成 / 计划</span><strong>{String(rows.length).padStart(2, '0')}<small>/ {run?.spec.episodes ?? spec.episodes}</small></strong></div><div><span>已保存帧数</span><strong>{frames.toLocaleString()}<small>帧</small></strong></div><div><span>通过样本</span><strong>{successes}<small>/ {rows.length}</small></strong></div></div>
    {rows.length ? <div className="tt-table-wrap"><table><thead><tr><th>样本</th><th>种子</th><th>数据帧数</th><th>仿真时长</th><th>可达性约束</th><th>实际结果</th><th>数据</th></tr></thead><tbody>{rows.map(row => { const reach = reachOf(row); return <tr key={row.index}><td>Episode {String(row.index + 1).padStart(3, '0')}</td><td>{row.seed}</td><td>{row.frames} 帧</td><td>{row.duration_s.toFixed(2)} s</td><td>{reach ? <span className="tt-reach-badge" title={reach.deviationMm !== undefined ? `最大目标偏差 ${reach.deviationMm} mm` : undefined}>{reach.steps} 步{reach.deviationMm !== undefined ? ` · ${reach.deviationMm} mm` : ''}</span> : <span className="tt-reach-none">无</span>}</td><td><span className={'tt-outcome ' + (row.success ? '' : 'failed')}>{row.success ? '✓ 验证通过' : '验证未通过'}</span></td><td>{dataset?.episodes.find(item => item.index === row.index) ? <a href={dataset.episodes.find(item => item.index === row.index)!.npz_url} download aria-disabled={!!downloading} onClick={event => { event.preventDefault(); void downloadArtifact(dataset.episodes.find(item => item.index === row.index)!.npz_url, `${run?.id}_ep${String(row.index).padStart(3, '0')}.npz`, 'npz') }}>下载 NPZ ↗</a> : '正在保存'}</td></tr> })}</tbody></table></div> : <div className="tt-results-empty"><span>▤</span><strong>{running ? '等待本轮验证产生数据' : '还没有本轮采集数据'}</strong><p>记录状态、7 维动作与双相机，失败也保留。</p></div>}
    {evaluation && <div className="tt-dataset-downloads">{!running && rows.length > 0 && <a className="tt-primary" href={evaluation.downloads.archive_url} download aria-disabled={!!downloading} onClick={event => { event.preventDefault(); void downloadArtifact(evaluation.downloads.archive_url, `${run?.id}_experiment.zip`) }}>下载本轮完整数据包 ZIP ↓</a>}<a href={evaluation.downloads.report_url} download aria-disabled={!!downloading} onClick={event => { event.preventDefault(); void downloadArtifact(`/api/v1/transport/runs/${run?.id}/evaluation`, `${run?.id}_evaluation.json`, 'json') }}>验证报告 JSON ↓</a><a href={evaluation.downloads.contract_url} download aria-disabled={!!downloading} onClick={event => { event.preventDefault(); void downloadArtifact(evaluation.downloads.contract_url + '?inline=true', `${run?.id}_policy-contract.json`, 'json') }}>动作与数据接口约定 ↓</a></div>}
    {dataset && trainingDatasets?.has(dataset.id) && <div className="tt-dataset-downloads">
      <Link className="tt-primary" to={`/training?dataset=${encodeURIComponent(dataset.id)}`}>使用本轮数据训练 →</Link>
    </div>}
  </section>
  const historyPanel = <section className="tt-card tt-history">
    <div className="tt-section-head">
      <div><h2>历史实验</h2><span>查看仿真、评测和可训练示范</span></div>
      <button className="tt-inline-button" onClick={() => setRetry(value => value + 1)}>刷新</button>
    </div>
    {trainingCatalogError && <p className="tt-empty-note" role="status">暂时无法确认哪些记录可用于训练。点击「刷新」重试。</p>}
    {historyError ? <p className="tt-empty-note" role="alert">历史列表暂时不可读取：{historyError}</p>
      : !history ? <p className="tt-empty-note" role="status">正在读取历史实验…</p>
      : history.length ? <ul className="tt-history-list">
        {history.map(item => <li key={item.id} className={`tt-history-item ${item.id === run?.id ? 'selected' : ''}`}>
          <div className="tt-history-meta"><strong className="tt-custom-name">{item.experiment_name || item.id}</strong><span>{STATUS[item.status] ?? item.status}</span></div>
          <p>{item.experiment_name && <>{item.id} · </>}{new Date(item.created).toLocaleString('zh-CN')} · {item.summary?.successes ?? 0}/{item.summary?.episodes ?? 0} 条通过 · {(item.summary?.frames ?? 0).toLocaleString('zh-CN')} 帧</p>
          {item.error && <small>{item.error}</small>}
          <div className="tt-history-actions">
            <button disabled={!canOpenHistory || item.id === run?.id} onClick={() => void openHistory(item.id)}>{item.id === run?.id ? '当前实验' : '打开仿真'}</button>
            <Link to={`/evaluations?run=${encodeURIComponent(item.id)}`}>评测报告</Link>
            {trainingDatasets?.has(item.id) && <Link className="tt-history-train" to={`/training?dataset=${encodeURIComponent(item.id)}`}>用此数据训练 →</Link>}
          </div>
        </li>)}
      </ul> : <p className="tt-empty-note">尚无已保存的实验。</p>}
  </section>
  const executionPanel = <section className="tt-card tt-execution-state"><div className="tt-section-head"><div><h2>当前执行阶段</h2></div><strong>{run ? STATUS[run.status] : '尚未启动'}</strong></div><div className="tt-execution-body"><span className="tt-execution-orbit">{run?.status === 'running' ? '◉' : '◷'}</span><div><strong>{stage}</strong><p>{run ? running ? `第 ${Math.min(run.spec.episodes, (run.current_episode ?? rows.length) + 1)} 条 / ${run.spec.episodes} 条 · ${preview?.sim_time?.toFixed(2) ?? '0.00'} s` : `已完成 ${rows.length} 条 / ${run.spec.episodes} 条 · ${preview?.sim_time?.toFixed(2) ?? '0.00'} s` : '在任务页检查执行方案并启动实验。'}</p><p>执行策略：{controllerLabel}{policyDevice(run?.policy_identity) && <span aria-label="本轮策略执行设备"> · 设备 {policyDevice(run?.policy_identity)}</span>}{run?.policy_diagnostics && <> · 推理 {run.policy_diagnostics.inference_count ?? 0} 次 · 执行动作 {run.policy_diagnostics.executed_model_actions ?? 0} 步{typeof run.policy_diagnostics.last_inference_s === 'number' ? ` · 最近推理 ${run.policy_diagnostics.last_inference_s.toFixed(2)} s` : ''}{(() => { const steps = preview?.metrics?.reach_limited_steps; return typeof steps === 'number' && steps > 0 ? ` · 可达性约束 ${steps} 步` : '' })()}</>}</p>{run?.policy_diagnostics?.inference_pending && <p className="tt-inference-pending" role="status"><i/>正在推理下一组动作</p>}</div>
    {running ? <button className="tt-stop" disabled={!!busy || run?.stop_requested || run?.status === 'stopping'} onClick={() => void stop()}>{busy === 'stop' ? '正在发送停止请求…' : run?.status === 'stopping' ? '正在保存数据…' : '停止并保存'}</button> : run && <button className="tt-secondary" onClick={() => go('results')}>查看可行性结果 →</button>}</div>{run?.error && run.status !== 'stopping' && <p className="tt-run-error">{runErrorText(run.error)}</p>}</section>
  const homePanel = <div className="tt-home"><section className="tt-home-intro tt-card"><div><span>EMBODIED TASK LAB</span><h2>让任务经过一次真实验证。</h2><p>描述动作顺序，准备模型和执行策略。现场、结果与数据各有独立模块。</p></div><button className="tt-primary" onClick={() => go('task')}>{run ? '查看本轮任务' : '描述新任务'} <span>→</span></button></section><div className="tt-home-grid">{[
    { view: 'task' as View, icon: '✎', title: '任务要求', text: spec.instruction, action: '描述与检查' },
    { view: 'models' as View, icon: '◇', title: '模型资产', text: pendingReviews.length ? `${pendingReviews.length} 个模型等待你的意见` : spec.object_mode === 'generate' ? '本轮将生成新物品，随后等待确认' : capabilities?.objects.find(asset => asset.id === spec.object_asset)?.name ?? '选择本轮物品与箱子', action: '准备模型' },
    { view: 'policy' as View, icon: '⟁', title: '执行策略', text: controllerLabel, action: '检查连接' },
    { view: 'simulation' as View, icon: '◉', title: '仿真执行', text: run ? STATUS[run.status] : '尚未启动，先完成执行方案检查', action: '查看现场' },
    { view: 'results' as View, icon: '✓', title: '可行性结果', text: evaluation?.verdict.label ?? '等待本轮验证结果', action: '检查结论' },
    { view: 'datasets' as View, icon: '▤', title: '数据集', text: `${rows.length} 条样本 · ${frames.toLocaleString()} 帧`, action: '采集与导出' },
  ].map(item => <button className="tt-home-card tt-card" key={item.view} onClick={() => go(item.view)}><span>{item.icon}</span><h2>{item.title}</h2><p>{item.text}</p><small>{item.action} →</small></button>)}</div></div>

  return <div className="tt-app">
    <aside className="tt-sidebar"><Link to={`/transport?${viewParams('home')}`} className="tt-brand"><span>◇</span> RoboForge <small>LAB</small></Link><div className="tt-lab"><span>VLA</span><div>仿真验证工作空间<small>任务可行性 · 数据采集 · 模型训练</small></div></div><nav>{VIEWS.map(item => <Link key={item.id} to={`/transport?${viewParams(item.id)}`} className={view === item.id ? 'selected' : ''} aria-label={item.label} title={item.label} aria-current={view === item.id ? 'page' : undefined}><span>{item.icon}</span>{item.label}{item.id === 'models' && pendingReviews.length > 0 ? <i className="tt-nav-count">{pendingReviews.length}</i> : <i>↗</i>}</Link>)}</nav><div className="tt-sidebar-extra"><Link to="/projects">项目工作台 ↗</Link><Link to="/training">训练 ↗</Link><Link to="/evaluations">评测 ↗</Link></div><div className="tt-sidebar-note"><span>AHOLO × MUJOCO</span><strong>先确认，再验证。</strong><p>模型由你决定是否保存，结果由真实仿真产生。</p></div><span className="tt-sidebar-foot">ROBOFORGE · EMBODIED TASK LAB</span></aside>
    <div className="tt-body"><header className="tt-topbar">
      <span><Link className="tt-topbar-back" to={view === 'simulation' ? `/transport?${viewParams(simulationBackView)}` : '/projects'}>
        ← {view === 'simulation' ? `返回${simulationSource?.label ?? '任务要求'}` : '返回项目'}
      </Link><b aria-hidden="true">/</b>{VIEWS.find(item => item.id === view)?.label}</span>
      <button onClick={() => setRetry(value => value + 1)} aria-label="重新连接搬运仿真" disabled={connecting}><i className={online ? 'online' : ''}/>{connecting ? '正在连接后端' : online ? '后端已连接' : '连接失败 · 重试'}<span aria-hidden="true">↻</span></button>
    </header><main className="tt-main">
      <div className="tt-heading"><div><span className="tt-kicker">EMBODIED TASK / VALIDATION & DATA</span><h1>{heading[view][0]}</h1><p>{heading[view][1]}</p></div><span className="tt-mode"><i/>{run ? STATUS[run.status] : '准备新实验'}</span></div>
      {run?.spec.experiment_name && <p className="tt-current-name tt-custom-name">实验 · {run.spec.experiment_name}</p>}

      {view === 'home' && <section className="tt-workflow tt-card tt-task-workflow">{['描述任务', '确认模型', '仿真验证', '可行性结果', '导出数据'].map((name, index) => <div key={name} className={index < flowStage ? 'done' : index === flowStage ? 'active' : index === 4 && dataset ? 'ready' : ''}><span>{index < flowStage ? '✓' : String(index + 1).padStart(2, '0')}</span><strong>{name}</strong>{index < 4 && <i/>}</div>)}</section>}
      {error && <div role="alert" className="tt-alert"><span>{error}</span><button aria-label="关闭提示" onClick={() => setError('')}>×</button></div>}
      {run?.status === 'stopping' && <div className="tt-stop-status" role="status">后端已收到停止请求，正在结束本轮并写入已采集数据。</div>}
      {busy === 'submit' && <div className="tt-stop-status" role="status">正在提交实验，等待后端确认。</div>}
      {downloading && <div className="tt-download-status" role="status"><span className="tt-download-spinner"/>正在准备文件下载，请稍候…<small>{downloadProgress ? `${(downloadProgress.received / 1048576).toFixed(1)} / ${(downloadProgress.total / 1048576).toFixed(1)} MB` : '正在读取文件信息'} · 准备完成后保存到下载位置</small></div>}
      {receipt && !receipt.run_id && busy !== 'submit' && <div className="tt-recovery"><div><strong>这次提交的结果尚未确认</strong><p>使用同一个请求 ID 恢复，避免重复生成或启动实验。</p></div><button disabled={!!busy} onClick={() => void submit()}>恢复这次提交</button></div>}
      {waitingReview && view !== 'models' && <div className="tt-review-banner"><div><strong>{pendingReviews.length} 个新模型等待你的意见</strong><p>当前仿真暂停，刷新和切换页面不会自动接受模型。</p></div><button onClick={() => go('models')}>查看并确认模型 →</button></div>}
      {view === 'policy' && checkpointNotice && <div className="tt-checkpoint-notice" role="status">{checkpointNotice}</div>}
      {view === 'policy' && checkpoint && (locked || !requestedPolicy) && <div className="tt-checkpoint-notice" role="status">
        <div>
          <strong>{requestedPolicy ? `待选用 ${requestedPolicy.label}` : '选择训练检查点'}</strong>
          <p>{policyChecking || !policies && !policyError ? '正在读取策略列表…'
            : !requestedPolicy ? '未找到此检查点，请刷新策略列表或返回训练中心。'
            : running ? '当前实验仍在进行。结束后可用该检查点准备新实验。'
            : receipt && !run ? '请先恢复当前实验，再选择检查点。'
            : '在新实验中使用该检查点。当前实验的结果仍可在历史记录中查看。'}</p>
        </div>
        {requestedPolicy && run && canOpenHistory
          ? <button className="tt-primary" onClick={() => startNewExperiment(requestedPolicy)}>用此模型准备新实验 →</button>
          : !requestedPolicy && policies && <Link to="/training">返回训练中心 →</Link>}
      </div>}
      {view === 'home' && homePanel}
      {view === 'simulation' && <div className="tt-execution-view">{cameraPanel}{executionPanel}{metricsPanel}</div>}
      {view === 'results' && <div className="tt-report-view">{reportPanel}{!run && <button className="tt-secondary" onClick={() => go('task')}>先描述验证任务 →</button>}</div>}
      {view === 'policy' && policyPanel}
      {view === 'task' && <div className="tt-layout"><div className="tt-left">{taskPanel}</div><div className="tt-right">{planPanel}</div></div>}
      {view === 'models' && <div className="tt-layout"><div className="tt-left">{progressPanel}{reviewsPanel}{!run?.asset_reviews?.length && <section className="tt-card tt-empty-note"><strong>新模型生成后将在这里等待确认</strong><p>可以旋转查看原始 3D 外观、下载模型包，并决定保存到资产库或仅本次使用。</p></section>}{run && !running && <button className="tt-secondary" onClick={fresh}>调整要求，开始新实验</button>}</div><div className="tt-right">{objectModelPanel}{binModelPanel}{!run && <button className="tt-secondary tt-full" onClick={() => go('task')}>返回任务检查方案 →</button>}</div></div>}
      {view === 'settings' && <div className="tt-layout"><div className="tt-left">{mapPanel}{objectSettings}</div><div className="tt-right">{robotSettings}{binSettings}{policySettings}<button className="tt-primary tt-full" onClick={() => go('task')}>返回任务并检查方案 →</button></div></div>}
      {view === 'datasets' && <div className="tt-data-view">{dataPanel}{historyPanel}</div>}
      <footer className="tt-footer"><span>{run ? `实验 ${run.id}` : 'ROBOFORGE · 具身任务验证与数据采集'}</span><Link to="/projects">项目工作台 ↗</Link></footer>
    </main></div>
  </div>
}
