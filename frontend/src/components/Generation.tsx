import { useCallback, useEffect, useState } from 'react'
import { Link, useLocation, useNavigate, useParams } from 'react-router-dom'
import { api } from '../api'
import { Button, Field, Section } from './ui'

const ASSET_TYPES = {
  free_visual: { kind: 'static_prop', label: '自由生成 3D 外观', role: 'visual_only', example: '例如：磨砂银色机械爪，带两根可见的弧形夹指' },
  task_object: { kind: 'task_object', label: '任务物体', role: 'dynamic', example: '例如：黄色橡胶鸭，高 6 cm' },
  static_prop: { kind: 'static_prop', label: '静态道具', role: 'static', example: '例如：带侧壁的收纳托盘，宽 25 cm' },
  robot_visual: { kind: 'robot_visual', label: '机械臂外观 / 部件', role: 'visual_only', example: '例如：蓝灰色五轴机械臂，底座、上臂、前臂和腕部分段清楚' },
  environment_splat: { kind: 'environment_splat', label: '场景背景（全景 / 3DGS）', role: 'visual_only', example: '例如：光线均匀的实验室，中间留空' },
  environment_mesh: { kind: 'environment_mesh', label: '场景网格', role: 'visual_only', example: '例如：平整的钢制工作台，宽 1 m' },
} as const
type AssetType = keyof typeof ASSET_TYPES
const ASSET_CHOICES = Object.keys(ASSET_TYPES) as AssetType[]
const ROLE_LABELS = { dynamic: '动态物体', static: '静态物体', visual_only: '仅视觉' }
const ROLES = Object.keys(ROLE_LABELS) as (keyof typeof ROLE_LABELS)[]
const supportsVesselFeatures = (type: AssetType) => type === 'task_object' || type === 'static_prop'

function ProvTag({ p }: { p: any }) {
  if (!p) return null
  const color = p.source === 'user_explicit' ? 'text-accent' : p.source === 'measured' ? 'text-green-400'
    : p.source === 'model_inferred' ? 'text-yellow-300' : 'text-muted'
  return <span className={`text-[10px] ${color}`}>{p.source}/{p.confirmation}/{p.verification}</span>
}

function DimInput({ label, dim, onChange }: { label: string; dim: any; onChange: (v: number | null) => void }) {
  return (
    <Field label={`${label}（米，空=未知）`}>
      <div className="flex items-center gap-2">
        <input type="number" step="0.001" className="!w-24" value={dim?.value_m ?? ''}
          placeholder="未知" onChange={(e) => onChange(e.target.value === '' ? null : Number(e.target.value))} />
        <ProvTag p={dim?.provenance} />
      </div>
    </Field>
  )
}

export function GenerationList() {
  const nav = useNavigate()
  const [sessions, setSessions] = useState<any[]>([])
  const [cap, setCap] = useState<any>(null)
  const [text, setText] = useState('')
  const [assetType, setAssetType] = useState<AssetType>('task_object')
  const [creating, setCreating] = useState(false)
  const [msg, setMsg] = useState('')
  const reload = useCallback(() => { api.genSessions().then((r) => setSessions(r.sessions)).catch((e) => setMsg(e.message)) }, [])
  useEffect(() => { reload(); api.genCapabilities().then(setCap).catch(() => {}) }, [reload])
  const choice = ASSET_TYPES[assetType]
  const freeVisual = assetType === 'free_visual'
  const create = async () => {
    const initialText = text.trim()
    if (creating || (freeVisual && !initialText)) return
    setCreating(true); setMsg('')
    try {
      const s = await api.genCreateSession({ initial_text: initialText || undefined, kind: choice.kind,
        ...(freeVisual && { role: choice.role }) })
      if ((freeVisual || assetType === 'robot_visual') && initialText && !s.asset_spec?.appearance) {
        try {
          await api.genPatchSpec(s.session_id, { expected_revision: s.revision, appearance: initialText })
        } catch {
          nav(`/generation/${s.session_id}`, { state: { notice: '会话已创建。请在规格编辑中保存外观描述后再生成。' } })
          return
        }
      }
      nav(`/generation/${s.session_id}`)
    } catch (e: any) { setMsg(e.message) }
    finally { setCreating(false) }
  }
  return (
    <div className="workspace-page">
      <header className="workspace-page-heading"><h1>资产生成</h1></header>
      {cap && !cap.providers?.lux3d?.configured && !cap.providers?.aholo_world?.configured && (
        <p className="text-xs text-muted mb-3">在线生成需要配置 Aholo。 <Link to="/settings" className="text-accent">打开设置 →</Link></p>
      )}
      <div className="flex gap-2 items-end mb-4 flex-wrap">
        <Field label="资产类型"><select value={assetType} onChange={(e) => setAssetType(e.target.value as AssetType)}>
          {ASSET_CHOICES.map((type) => <option key={type} value={type}>{ASSET_TYPES[type].label}</option>)}</select></Field>
        <Field label={freeVisual ? '描述想生成的外观' : '初始需求（选填）'}>
          <input className="!w-full sm:!w-80" value={text} onChange={(e) => setText(e.target.value)}
            placeholder={choice.example} /></Field>
        <Button onClick={create} disabled={creating || (freeVisual && !text.trim())}>{creating ? '创建中…' : '新建会话'}</Button>
      </div>
      {freeVisual && <p className="text-xs text-muted mb-3">由描述生成单件 3D 网格，可预览和配景。用于物理仿真前需另行标定和绑定。</p>}
      {assetType === 'robot_visual' && <p className="text-xs text-muted mb-3">生成机械臂的外观网格。要在仿真里控制关节，需把部件装配到已有的关节模板。</p>}
      {msg && <div className="text-xs text-yellow-300 mb-2">{msg}</div>}
      <Section title="已有会话">
        <table className="w-full text-xs">
          <thead><tr className="text-muted text-left"><th className="py-1">ID</th><th>状态</th><th>rev</th><th>更新</th></tr></thead>
          <tbody>{sessions.map((s) => (
            <tr key={s.session_id} className="border-t border-edge">
              <td className="py-1"><Link className="text-accent font-mono" to={`/generation/${s.session_id}`}>{s.session_id}</Link></td>
              <td>{s.state}</td><td>{s.revision}</td><td className="text-muted">{s.updated?.slice(0, 19)}</td>
            </tr>))}
            {!sessions.length && <tr><td colSpan={4} className="text-muted py-2">暂无会话</td></tr>}
          </tbody>
        </table>
      </Section>
    </div>
  )
}

export function GenerationWorkbench() {
  const { sessionId } = useParams()
  const { state } = useLocation()
  const [s, setS] = useState<any>(null)
  const [cap, setCap] = useState<any>(null)
  const [msg, setMsg] = useState<string>(state?.notice ?? '')
  const [chat, setChat] = useState('')
  const [busy, setBusy] = useState(false)
  const [promptEdit, setPromptEdit] = useState<string | null>(null)
  const [appearanceDraft, setAppearanceDraft] = useState<string | null>(null)
  const reload = useCallback(() => {
    if (!sessionId) return
    api.genSession(sessionId).then(setS).catch((e) => setMsg(e.message))
  }, [sessionId])
  useEffect(() => { setAppearanceDraft(null) }, [sessionId])
  useEffect(() => { reload(); api.genCapabilities().then(setCap).catch(() => {}) }, [reload])

  const patch = async (fields: any) => {
    if (!s) return
    try { setS(await api.genPatchSpec(s.session_id, { expected_revision: s.revision, ...fields })); return true }
    catch (e: any) { setMsg(e.message); reload() }
    return false
  }
  const send = async () => {
    if (!s || !chat.trim()) return
    setBusy(true)
    try { setS(await api.genMessage(s.session_id, chat, s.revision)); setChat('') }
    catch (e: any) { setMsg(e.message); reload() }
    finally { setBusy(false) }
  }
  const answer = async (qid: string, ans: string) => {
    if (!s) return
    try { setS(await api.genAnswer(s.session_id, qid, ans, s.revision)) } catch (e: any) { setMsg(e.message); reload() }
  }
  const compile = async () => {
    if (!s) return
    try {
      const body = promptEdit !== null ? { prompt_override: promptEdit } : {}
      setS(await api.genCompile(s.session_id, body)); setPromptEdit(null)
    } catch (e: any) { setMsg(e.message) }
  }
  const branch = async () => {
    if (!s) return
    const reason = window.prompt('新分支原因（重新生成会记录父修订与原因）')
    if (!reason) return
    try { setS(await api.genBranch(s.session_id, reason, s.revision)) } catch (e: any) { setMsg(e.message) }
  }

  if (!s) return <div className="p-4 text-xs text-muted">{msg || '加载中…'}</div>
  const spec = s.asset_spec, phys = s.physics_spec
  const kind = spec.kind as Exclude<AssetType, 'free_visual'>
  const robotVisual = kind === 'robot_visual'
  const background = kind === 'environment_splat'
  const freeVisual = kind === 'static_prop' && spec.role === 'visual_only'
  const assetType = freeVisual ? 'free_visual' : kind
  const vesselFeatures = supportsVesselFeatures(assetType)
  const roles = background ? ROLES.filter(role => role === 'visual_only') : robotVisual ? ROLES.filter(role => role !== 'dynamic') : ROLES
  const auxReady = cap?.aux_model?.configured
  const editingAppearance = appearanceDraft !== null && appearanceDraft !== spec.appearance
  const missingFreeProvider = freeVisual && cap?.providers?.lux3d?.configured === false
  return (
    <div className="p-4">
      <Link to="/generation" className="text-xs text-accent">← 生成工作台</Link>
      <h2 className="text-lg text-ink my-2">会话 {s.session_id} <span className="text-xs text-muted">rev {s.revision} · {s.state}</span></h2>
      {msg && <div className="text-xs text-yellow-300 mb-2">{msg}</div>}
      <div className={freeVisual ? '' : 'grid md:grid-cols-2 gap-4'}>
        {!freeVisual && <div>
          <Section title="对话澄清">
            {!auxReady && <div className="text-[11px] text-yellow-300 mb-2 border border-edge rounded p-2 bg-panel2">
              waiting_for_configuration：未配置 AUX_MODEL_BASE_URL/AUX_MODEL_API_KEY，对话规划不可用。
              右侧表单/JSON 与本地导入仍可正常使用；不会用模板冒充模型回复。</div>}
            <div className="border border-edge rounded bg-panel2 p-2 h-64 overflow-auto mb-2 space-y-1">
              {(s.messages || []).map((m: any, i: number) => (
                <div key={i} className="text-xs">
                  <span className={m.role === 'user' ? 'text-accent' : m.role === 'assistant' ? 'text-ink' : 'text-muted'}>
                    [{m.role}{m.kind !== 'text' ? `/${m.kind}` : ''}]</span> {m.content}
                </div>))}
              {!s.messages?.length && <div className="text-xs text-muted">暂无消息</div>}
            </div>
            <div className="flex gap-2">
              <input className="flex-1" value={chat} onChange={(e) => setChat(e.target.value)}
                placeholder={auxReady ? '描述需求…' : '配置辅助模型后可对话；当前请用右侧表单'}
                onKeyDown={(e) => { if (e.key === 'Enter') send() }} />
              <Button disabled={busy || !auxReady} onClick={send}>{busy ? '思考中…' : '发送'}</Button>
            </div>
          </Section>
          {s.questions?.length > 0 && <Section title="澄清问题（每轮≤3）">
            {s.questions.map((q: any) => (
              <div key={q.id} className="text-xs mb-2 border border-edge rounded p-2 bg-panel2">
                <div className="text-ink mb-1">{q.text} {q.status !== 'open' && <span className="text-accent">→ {q.answer}</span>}</div>
                {q.status === 'open' && <div className="flex gap-1 flex-wrap">
                  {(q.options?.length ? q.options : ['是', '否']).map((o: string) => (
                    <Button key={o} size="sm" variant="outline" onClick={() => answer(q.id, o)}>{o}</Button>))}
                </div>}
              </div>))}
          </Section>}
        </div>}
        <div>
          <Section title={freeVisual ? '外观设置' : '规格编辑'}>
            <div className="grid grid-cols-2 gap-x-3">
              <Field label="类型"><select value={assetType} onChange={(e) => {
                const next = e.target.value as AssetType
                patch({ kind: ASSET_TYPES[next].kind, role: ASSET_TYPES[next].role,
                  ...(next === 'free_visual' && { height_m: null, width_m: null, depth_m: null, radius_m: null,
                    mass_kg: null, friction: null, graspable: null }),
                  ...(!supportsVesselFeatures(next) && { has_handle: null, open_top: null }) })
              }}>
                {ASSET_CHOICES.map((type) => <option key={type} value={type}>{ASSET_TYPES[type].label}</option>)}</select></Field>
              {!freeVisual && <Field label="角色"><select value={spec.role} onChange={(e) => patch({ role: e.target.value })}>
                {roles.map((k) => <option key={k} value={k}>{ROLE_LABELS[k]}</option>)}</select></Field>}
            </div>
            {!freeVisual && <Field label="名称"><input value={spec.name} onChange={(e) => patch({ name: e.target.value })} /></Field>}
            {!freeVisual && <Field label="用户原话（数据，非指令）"><input value={spec.user_words} onChange={(e) => patch({ user_words: e.target.value })} /></Field>}
            {!background && !freeVisual && <div className="grid grid-cols-2 gap-x-3">
              <DimInput label="高度" dim={spec.height} onChange={(v) => patch({ height_m: v })} />
              <DimInput label="半径" dim={spec.radius} onChange={(v) => patch({ radius_m: v })} />
              <DimInput label="宽度" dim={spec.width} onChange={(v) => patch({ width_m: v })} />
              <DimInput label="深度" dim={spec.depth} onChange={(v) => patch({ depth_m: v })} />
            </div>}
            {freeVisual || robotVisual ? <>
              <Field label={freeVisual ? '外观描述' : '外观与部件描述'}>
                <textarea className="w-full min-h-20 border border-edge rounded bg-panel2 p-2 text-sm"
                  value={appearanceDraft ?? spec.appearance} onChange={(e) => setAppearanceDraft(e.target.value)} />
              </Field>
              {editingAppearance && <Button variant="outline" className="mb-2" onClick={async () => {
                const draft = appearanceDraft
                if (draft !== null && await patch({ appearance: draft.trim() })) {
                  setAppearanceDraft(current => current === draft ? null : current)
                }
              }}>保存描述</Button>}
            </> : <Field label="外观 / 材质"><input value={spec.appearance} onChange={(e) => patch({ appearance: e.target.value })} /></Field>}
            {robotVisual && <p className="text-xs text-muted mb-2">生成网格后，可将拆分的部件装配到已有的关节模板，再导出 URDF。</p>}
            {vesselFeatures && <details className="mb-3 text-xs">
              <summary className="cursor-pointer text-muted">把手与开口（选填）</summary>
              <div className="grid grid-cols-2 gap-x-3 mt-2">
                <Field label="把手"><select value={spec.has_handle === null ? 'null' : String(spec.has_handle)}
                  onChange={(e) => patch({ has_handle: e.target.value === 'null' ? null : e.target.value === 'true' })}>
                  <option value="null">未说明</option><option value="false">无</option><option value="true">有</option></select></Field>
                <Field label="开口"><select value={spec.open_top === null ? 'null' : String(spec.open_top)}
                  onChange={(e) => patch({ open_top: e.target.value === 'null' ? null : e.target.value === 'true' })}>
                  <option value="null">未说明</option><option value="true">是</option><option value="false">否</option></select></Field>
              </div>
            </details>}
            {kind === 'task_object' && <Field label="可抓取"><select value={phys.graspable === null ? 'null' : String(phys.graspable)}
                onChange={(e) => patch({ graspable: e.target.value === 'null' ? null : e.target.value === 'true' })}>
                <option value="null">未知</option><option value="true">是</option><option value="false">否</option></select></Field>}
            {!robotVisual && !background && !freeVisual && <div className="grid grid-cols-2 gap-x-3">
              <Field label="质量（kg，空=未知）"><input type="number" step="0.01" value={phys.mass?.value_kg ?? ''}
                onChange={(e) => patch({ mass_kg: e.target.value === '' ? null : Number(e.target.value) })} /></Field>
              <Field label="摩擦 [slide,torsion,rolling]"><input value={(phys.friction || []).join(',')}
                placeholder="0.95,0.005,0.0001" onChange={(e) => {
                  const v = e.target.value.split(',').map(Number).filter((n) => Number.isFinite(n))
                  patch({ friction: v.length === 3 ? v : null }) }} /></Field>
            </div>}
            {!freeVisual && <div className="text-[10px] text-muted">估计值标记 model_inferred/unverified；未知尺寸→未标定，处理前不绑定物理。</div>}
          </Section>
        </div>
      </div>
      <Section title={freeVisual ? '生成预览' : '编译预览'}>
        <div className="flex gap-2 mb-2">
          <Button onClick={compile} disabled={editingAppearance || (freeVisual && !spec.appearance.trim())}>{freeVisual ? '预览生成提示词' : '编译 / 刷新预览'}</Button>
          {!freeVisual && <Button variant="outline" onClick={branch}>新建分支</Button>}
          {!freeVisual && s.compiled && <span className="text-xs text-muted self-center">compiler {s.compiled.compiler_version} · {s.compiled.provider} · {s.compiled.template} · hash {s.compiled.spec_hash?.slice(0, 12)}</span>}
        </div>
        {s.compiled ? <div className={freeVisual ? 'text-xs' : 'grid md:grid-cols-2 gap-3 text-xs'}>
          <div>
            <div className="text-accent mb-1">英文提示词（可编辑，编辑后重编译）</div>
            <textarea className="w-full h-28 font-mono text-[11px]" value={promptEdit ?? s.compiled.prompt_en}
              onChange={(e) => setPromptEdit(e.target.value)} />
            {!freeVisual && <>
              <div className="text-accent mt-2 mb-1">中文规格</div>
              <pre className="whitespace-pre-wrap text-[11px] bg-panel2 border border-edge rounded p-2">{s.compiled.spec_zh}</pre>
            </>}
          </div>
          {!freeVisual && <div>
            <div className="text-accent mb-1">Provider 载荷（不含质量/摩擦/关节）</div>
            <pre className="whitespace-pre-wrap text-[11px] bg-panel2 border border-edge rounded p-2">{JSON.stringify(s.compiled.payload, null, 1)}</pre>
            <div className="text-accent mt-2 mb-1">不支持/后处理字段</div>
            <div className="text-[11px] text-muted">{s.compiled.excluded_or_postprocess_fields?.join(', ')}</div>
            <div className="text-accent mt-2 mb-1">警告</div>
            <ul className="text-[11px] text-yellow-300 list-disc pl-4">{s.compiled.warnings?.map((w: string, i: number) => <li key={i}>{w}</li>)}</ul>
          </div>}
        </div> : !freeVisual && <div className="text-xs text-muted">尚未编译。</div>}
      </Section>
      {missingFreeProvider && <p className="text-xs text-muted mb-3">在线生成需要配置 Lux3D。 <Link to="/settings" className="text-accent">打开设置 →</Link></p>}
      <ConfirmPanel key={assetType} s={s} onDone={setS} meshOnly={freeVisual}
        disabled={editingAppearance || missingFreeProvider || (freeVisual && !spec.appearance.trim())} />
      <TasksPanel sessionId={s.session_id} />
      {!freeVisual && <RecipePanel s={s} />}
    </div>
  )
}

const NODE_LABELS: Record<string, string> = {
  submit: '提交', remote: '远程生成', download: '下载', import: '导入', validate: '校验',
}

function ConfirmPanel({ s, onDone, disabled, meshOnly }: {
  s: any; onDone: (s: any) => void; disabled: boolean; meshOnly: boolean
}) {
  const [provider, setProvider] = useState(s.asset_spec.kind === 'environment_splat' ? 'aholo_world' : 'lux3d')
  const [remoteId, setRemoteId] = useState('')
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState('')
  const confirm = async () => {
    setBusy(true); setMsg('')
    try {
      const crid = (crypto as any).randomUUID ? crypto.randomUUID() : `crid-${Date.now()}`
      const r = await api.genConfirm(s.session_id, {
        client_request_id: crid, provider, remote_id: remoteId || undefined,
      })
      onDone(r.session)
      setMsg(`已确认生成：任务 ${r.job.id}（${r.job.state}）。绑定 spec rev ${r.binding.spec_revision} · hash ${r.binding.spec_hash?.slice(0, 12)} · ${r.binding.provider}。`)
    } catch (e: any) { setMsg(e.message) }
    finally { setBusy(false) }
  }
  return (
    <Section title="确认生成">
      <div className="flex gap-2 items-end flex-wrap text-xs">
        {!meshOnly && <Field label="生成服务"><select value={provider} onChange={(e) => setProvider(e.target.value)}>
          <option value="lux3d">Lux3D（单件网格）</option><option value="aholo_world">Aholo World（背景）</option>
        </select></Field>}
        {!meshOnly && <Field label="恢复已有远程任务 ID（可选，避免新建付费任务）">
          <input className="!w-56" value={remoteId} onChange={(e) => setRemoteId(e.target.value)} placeholder="如 3FO4K4RMF54N" /></Field>
        }
        <Button disabled={busy || disabled} onClick={confirm}>{busy ? '提交中…' : '确认生成'}</Button>
      </div>
      {meshOnly ? <>
        {!remoteId.trim() && <p className="text-xs text-muted mt-2">提交会使用 Aholo 生成额度。</p>}
        <details className="mt-2 text-xs">
          <summary className="cursor-pointer text-muted">恢复已有远程任务</summary>
          <Field label="远程任务 ID"><input className="!w-56" value={remoteId} onChange={(e) => setRemoteId(e.target.value)} /></Field>
        </details>
      </>
        : <p className="text-xs text-muted mt-2">新任务会使用对应服务的生成额度。</p>}
      {msg && <div className="text-xs text-yellow-300 mt-1">{msg}</div>}
    </Section>
  )
}

function NodeBar({ nodes }: { nodes: any }) {
  if (!nodes) return null
  return <div className="flex gap-1 flex-wrap">{Object.entries(nodes).map(([n, st]) => (
    <span key={n} className={`text-[10px] px-1.5 py-0.5 rounded border border-edge ${st === 'done' ? 'text-accent' : st === 'pending' ? 'text-muted' : 'text-yellow-300'}`}>
      {NODE_LABELS[n] || n}:{String(st)}</span>))}</div>
}

function TasksPanel({ sessionId }: { sessionId: string }) {
  const [tasks, setTasks] = useState<any[]>([])
  const [msg, setMsg] = useState('')
  const reload = useCallback(() => { api.genTasks(sessionId).then((r) => setTasks(r.tasks)).catch(() => {}) }, [sessionId])
  useEffect(() => { reload(); const t = setInterval(reload, 2500); return () => clearInterval(t) }, [reload])
  const act = async (jid: string, node: string) => {
    try { await api.genRetryNode(sessionId, jid, node); reload() } catch (e: any) { setMsg(e.message) }
  }
  const poll = async (jid: string) => {
    try { await api.genPoll(sessionId, jid); reload() } catch (e: any) { setMsg(e.message) }
  }
  if (!tasks.length) return null
  return (
    <Section title="生成任务（节点进度 / 恢复）">
      {msg && <div className="text-xs text-yellow-300 mb-1">{msg}</div>}
      {tasks.map((j) => (
        <div key={j.id} className="border border-edge rounded p-2 bg-panel2 mb-2 text-xs">
          <div className="flex gap-2 items-center flex-wrap">
            <Link className="text-accent font-mono" to={`/generation/${sessionId}/tasks/${j.id}`}>{j.id}</Link>
            <span className={j.state === 'succeeded' ? 'text-accent' : j.state === 'failed' ? 'text-red-400' : 'text-yellow-300'}>{j.state}</span>
            <span className="text-muted">{j.provider} · {j.slug || j.params?.name || ''}</span>
            {j.remote_id && <span className="text-muted">remote {j.remote_id}</span>}
          </div>
          <div className="my-1"><NodeBar nodes={j.nodes} /></div>
          {j.error && <div className="text-red-400 text-[11px]">{j.error}</div>}
          <div className="flex gap-1 mt-1 flex-wrap">
            {j.remote_id && !['succeeded', 'failed'].includes(j.state) &&
              <Button size="sm" variant="outline" onClick={() => poll(j.id)}>查询进度</Button>}
            {['retryable_error', 'failed'].includes(j.state) && j.remote_id &&
              ['download', 'import', 'validate'].map((n) => (
                <Button key={n} size="sm" variant="outline" onClick={() => act(j.id, n)}>重试{NODE_LABELS[n]}</Button>))}
            {j.state === 'succeeded' && j.asset_id && <>
              <Link className="text-accent text-[11px] self-center" to={`/library/${encodeURIComponent(j.asset_id)}`}>查看资产 →</Link>
              <a className="text-accent text-[11px] self-center" href={api.genBindingGuideUrl(sessionId, j.id)} download>下载绑定指南</a>
            </>}
          </div>
        </div>))}
    </Section>
  )
}

export function GenerationTaskDetail() {
  const { sessionId, jobId } = useParams()
  const [job, setJob] = useState<any>(null)
  const [msg, setMsg] = useState('')
  const reload = useCallback(() => {
    if (!sessionId) return
    api.genTasks(sessionId).then((r) => setJob(r.tasks.find((t: any) => t.id === jobId) || null)).catch((e) => setMsg(e.message))
  }, [sessionId, jobId])
  useEffect(() => { reload(); const t = setInterval(reload, 2500); return () => clearInterval(t) }, [reload])
  const retry = async (node: string) => { try { await api.genRetryNode(sessionId!, jobId!, node); reload() } catch (e: any) { setMsg(e.message) } }
  if (!job) return <div className="p-4 text-xs text-muted">{msg || '加载中…'}</div>
  return (
    <div className="p-4">
      <Link to={`/generation/${sessionId}`} className="text-xs text-accent">← 会话</Link>
      <h2 className="text-lg text-ink my-2">任务 {job.id} <span className="text-xs text-muted">{job.provider} · {job.state}</span></h2>
      {msg && <div className="text-xs text-yellow-300 mb-2">{msg}</div>}
      <Section title="节点依赖与进度">
        <NodeBar nodes={job.nodes} />
        <div className="text-[11px] text-muted mt-2">submit → remote → download → import → validate；远程成功≠已导入≠可训练。</div>
      </Section>
      <Section title="绑定与溯源">
        <pre className="text-[11px] bg-panel2 border border-edge rounded p-2 whitespace-pre-wrap">{JSON.stringify({
          remote_id: job.remote_id, spec_id: job.spec_id, spec_revision: job.spec_revision,
          spec_hash: job.spec_hash, project_id: job.project_id, config_version: job.config_version,
          client_request_id: job.client_request_id, download_sha256: job.download_sha256,
          asset_id: job.asset_id, validation: job.validation, retries: job.retries,
        }, null, 1)}</pre>
      </Section>
      {job.error && <Section title="错误"><div className="text-xs text-red-400">{job.error}</div></Section>}
      <Section title="恢复">
        <div className="flex gap-1 flex-wrap">
          {job.remote_id && ['download', 'import', 'validate'].map((n) => (
            <Button key={n} size="sm" variant="outline" onClick={() => retry(n)}>重试{NODE_LABELS[n]}</Button>))}
          {!job.remote_id && <span className="text-xs text-muted">无已知 remote_id：请先在提供商历史中恢复任务，再填入远程 ID 重新确认；不会盲目重提。</span>}
        </div>
      </Section>
      {job.state === 'succeeded' && job.asset_id && sessionId &&
        <div className="flex gap-3 mt-2 text-xs">
          <Link className="text-accent" to={`/library/${encodeURIComponent(job.asset_id)}`}>查看已入库资产 →</Link>
          <a className="text-accent" href={api.genBindingGuideUrl(sessionId, job.id)} download>下载绑定指南</a>
        </div>}
    </div>
  )
}

const WORKBENCH = { name: 'workbench', shape: 'box', full_size_m: [0.64, 0.43, 0.056], position: [0, 0, -0.028], quaternion_xyzw: [0, 0, 0, 1] }
const TOOLBOX = { name: 'toolbox', shape: 'box', full_size_m: [0.10, 0.10, 0.05], position: [0.25, -0.15, 0.025], quaternion_xyzw: [0, 0, 0, 1] }

function RecipePanel({ s }: { s: any }) {
  const [mode, setMode] = useState('composed_scene')
  const [taskRef, setTaskRef] = useState('builtin/standard')
  const [bgRef, setBgRef] = useState('')
  const [renderer, setRenderer] = useState('mujoco')
  const [useTable, setUseTable] = useState(true)
  const [collidersText, setCollidersText] = useState('[]')
  const [projectId, setProjectId] = useState(s.project_id || 'default')
  const [projects, setProjects] = useState<string[]>(['default'])
  const [recipe, setRecipe] = useState<any>(null)
  const [report, setReport] = useState<any>(null)
  const [applied, setApplied] = useState<any>(null)
  const [msg, setMsg] = useState('')
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    api.projects().then((r) => {
      const ids = (r.projects || []).map((p: any) => p.project_id || p.id || p.name).filter(Boolean)
      setProjects(Array.from(new Set(['default', ...ids])))
    }).catch(() => {})
    api.genTasks(s.session_id).then((r) => {
      const done = (r.tasks || []).find((t: any) => t.state === 'succeeded' && t.asset_id && t.params?.kind === 'task_object')
      if (done) setTaskRef(done.asset_id)
    }).catch(() => {})
  }, [s.session_id])

  const preset = () => { setCollidersText(JSON.stringify([WORKBENCH, TOOLBOX], null, 1)); setUseTable(false) }

  const create = async () => {
    setBusy(true); setMsg(''); setApplied(null)
    try {
      const colliders = JSON.parse(collidersText || '[]')
      const r = await api.recipeFromSession(s.session_id, {
        scene_mode: mode, task_asset_ref: taskRef || null,
        background_ref: mode === 'background_scene' ? (bgRef || null) : null,
        sensor_renderer: renderer, use_default_table: useTable, colliders,
      })
      setRecipe(r)
      const rep = await api.recipeValidate(r.recipe_id, projectId)
      setReport(rep)
    } catch (e: any) { setMsg(e.message) } finally { setBusy(false) }
  }
  const revalidate = async () => {
    if (!recipe) return
    setBusy(true); setMsg('')
    try { setReport(await api.recipeValidate(recipe.recipe_id, projectId)) }
    catch (e: any) { setMsg(e.message) } finally { setBusy(false) }
  }
  const apply = async () => {
    if (!recipe) return
    setBusy(true); setMsg('')
    try {
      const sc = await api.getScene(projectId)
      const out = await api.recipeApply(recipe.recipe_id, { project_id: projectId, expected_project_version: sc.version })
      setApplied(out)
    } catch (e: any) { setMsg(e.message) } finally { setBusy(false) }
  }

  return (
    <Section title="组装 SceneRecipe → 校验 → 应用到项目版本">
      <div className="grid grid-cols-2 gap-3 text-xs">
        <div className="space-y-2">
          <Field label="场景模式">
            <select value={mode} onChange={(e) => { setMode(e.target.value); if (e.target.value === 'background_scene') setRenderer('hybrid') }}>
              <option value="composed_scene">composed_scene（显式代理+参数化工作台，无需 World/标定）</option>
              <option value="background_scene">background_scene（World 外观+显式代理）</option>
            </select>
          </Field>
          <Field label="可操作任务对象 ref（builtin/标准 或 imported/已入库）">
            <input value={taskRef} onChange={(e) => setTaskRef(e.target.value)} placeholder="builtin/standard" />
          </Field>
          {mode === 'background_scene' && <Field label="背景 ref（scenes/&lt;asset_id&gt;/&lt;rev&gt;）">
            <input value={bgRef} onChange={(e) => setBgRef(e.target.value)} placeholder="scenes/world_586c79650960/1" />
          </Field>}
          <Field label="传感器渲染器">
            <select value={renderer} onChange={(e) => setRenderer(e.target.value)}>
              <option value="mujoco">mujoco</option><option value="hybrid">hybrid（高斯背景，仅固定前视）</option>
            </select>
          </Field>
          <label className="flex items-center gap-2 text-xs">
            <input type="checkbox" className="!w-auto" checked={useTable} onChange={(e) => setUseTable(e.target.checked)} />
            使用默认工作台（use_default_table）
          </label>
          <Field label="目标项目">
            <select value={projectId} onChange={(e) => setProjectId(e.target.value)}>
              {projects.map((p) => <option key={p} value={p}>{p}</option>)}
            </select>
          </Field>
        </div>
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-accent">静态代理 / 参数化工作台（SceneCollider JSON）</span>
            <Button size="sm" variant="outline" onClick={preset}>工作台+工具箱预设</Button>
          </div>
          <textarea className="w-full h-40 font-mono text-[11px]" value={collidersText} onChange={(e) => setCollidersText(e.target.value)} />
          <div className="text-[10px] text-muted mt-1">静态道具用显式碰撞代理提供接触；视觉网格不提供物理。支持面 z=0。</div>
        </div>
      </div>
      <div className="flex gap-2 mt-2 flex-wrap">
        <Button onClick={create} disabled={busy}>创建配方并校验</Button>
        {recipe && <Button variant="outline" onClick={revalidate} disabled={busy}>重新校验</Button>}
        {report?.valid && <Button onClick={apply} disabled={busy}>应用到 {projectId}（原子保存）</Button>}
        {recipe && <span className="text-[11px] text-muted self-center">recipe {recipe.recipe_id} · hash {recipe.manifest_hash?.slice(0, 12)}</span>}
      </div>
      {msg && <div className="text-xs text-yellow-300 mt-2">{msg}</div>}
      {report && <div className="mt-2 text-xs">
        <div className={report.valid ? 'text-accent' : 'text-red-400'}>校验：{report.valid ? '通过（真实 MuJoCo 构建+短步稳定）' : '未通过'}</div>
        {!!report.issues?.length && <ul className="text-red-400 list-disc pl-4 text-[11px]">{report.issues.map((i: string, k: number) => <li key={k}>{i}</li>)}</ul>}
        {!!report.warnings?.length && <ul className="text-yellow-300 list-disc pl-4 text-[11px]">{report.warnings.map((w: string, k: number) => <li key={k}>{w}</li>)}</ul>}
        <pre className="text-[11px] bg-panel2 border border-edge rounded p-2 mt-1 whitespace-pre-wrap">{JSON.stringify({
          physics: report.checks?.physics, task_object: report.checks?.task_object,
          background: report.checks?.background, dependency_graph: report.checks?.dependency_graph,
        }, null, 1)}</pre>
      </div>}
      {applied && <div className="mt-2 text-xs border border-edge rounded p-2 bg-panel2">
        <div className="text-accent">已应用：{applied.project_id} v{applied.version}（{applied.validation_status}）· 任务对象 {applied.task_asset}</div>
        <div className="text-[11px] text-muted mt-1">读回：colliders [{(applied.read_back?.colliders || []).map((c: any) => c.name).join(', ')}] · use_default_table={String(applied.read_back?.use_default_table)} · renderer={applied.read_back?.sensor_renderer}</div>
        <div className="flex gap-3 mt-1">
          <Link className="text-accent text-[11px]" to={`/projects/${applied.project_id}/sim`}>进入仿真 →</Link>
          <a className="text-accent text-[11px]" href={api.projectBindingGuideUrl(applied.project_id, applied.version)} download>下载项目绑定指南</a>
          <span className="text-[10px] text-muted">远程/ mock 成功≠可训练；此处以真实 MuJoCo 校验为准。</span>
        </div>
      </div>}
    </Section>
  )
}
