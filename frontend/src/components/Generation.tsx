import { useCallback, useEffect, useRef, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { api } from '../api'
import { Button, Field, Section } from './ui'

const KINDS = ['task_object', 'static_prop', 'robot_visual', 'environment_splat', 'environment_mesh']
const ROLES = ['dynamic', 'static', 'visual_only']

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
  const [kind, setKind] = useState('task_object')
  const [msg, setMsg] = useState('')
  const reload = useCallback(() => { api.genSessions().then((r) => setSessions(r.sessions)).catch((e) => setMsg(e.message)) }, [])
  useEffect(() => { reload(); api.genCapabilities().then(setCap).catch(() => {}) }, [reload])
  const create = async () => {
    try {
      const s = await api.genCreateSession({ initial_text: text || undefined, kind })
      nav(`/generation/${s.session_id}`)
    } catch (e: any) { setMsg(e.message) }
  }
  return (
    <div className="p-4">
      <h2 className="text-lg text-ink mb-2">生成工作台</h2>
      <div className="text-xs text-muted mb-3">
        对话 → 规格 → 生成 → 导入 → 组装场景 → 仿真验证。语言模型只理解与建议，程序校验、编排、落地。
        {cap && <> 辅助模型：<span className={cap.aux_model?.configured ? 'text-accent' : 'text-yellow-300'}>
          {cap.aux_model?.state}</span>（未配置时仍可用表单/JSON/本地导入）。</>}
      </div>
      {cap && <div className="text-[11px] text-muted mb-3 border border-edge rounded p-2 bg-panel2">
        <div className="text-accent mb-1">当前支持范围（已打通完整物理链）</div>
        <div>单体/环境网格：Lux3D（{cap.providers?.lux3d?.configured ? '凭据已配置' : '等待 AHOLO_API_KEY'}，支持文本与参考图）；高斯背景：Aholo World（{cap.providers?.aholo_world?.configured ? '已配置' : '等待配置'}）；本地导入始终可用。</div>
        <div>物理碰撞：box / cylinder / plane + 凹面体素复合盒分解（compound）；SceneRecipe 支持多动态自由刚体（role=dynamic）与参数化桌面高度 support_plane_z。</div>
        <div>机械臂工具链：支持 part_split 部件拆分装配并导出标准 URDF（人形蒙皮绑骨不冒充机械臂关节）。{cap.render_limits?.[0]}。</div>
      </div>}
      <div className="flex gap-2 items-end mb-4 flex-wrap">
        <Field label="资产类型"><select value={kind} onChange={(e) => setKind(e.target.value)}>
          {KINDS.map((k) => <option key={k}>{k}</option>)}</select></Field>
        <Field label="初始需求（可留空，进入后编辑）">
          <input className="!w-96" value={text} onChange={(e) => setText(e.target.value)}
            placeholder="例如：蓝色无把手陶瓷杯，高 9cm" /></Field>
        <Button onClick={create}>新建会话</Button>
      </div>
      {msg && <div className="text-xs text-yellow-300 mb-2">{msg}</div>}
      <Section title="已有会话（刷新/重启后保留）">
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
  const [s, setS] = useState<any>(null)
  const [cap, setCap] = useState<any>(null)
  const [msg, setMsg] = useState('')
  const [chat, setChat] = useState('')
  const [busy, setBusy] = useState(false)
  const [promptEdit, setPromptEdit] = useState<string | null>(null)
  const [draftAnswers, setDraftAnswers] = useState<Record<string, string>>({})
  const [frictionText, setFrictionText] = useState('')
  const [genMode, setGenMode] = useState<'auto' | 'text_to_3d' | 'img_to_3d'>('auto')
  const [taskRev, setTaskRev] = useState(0)
  const reload = useCallback(() => {
    if (!sessionId) return
    api.genSession(sessionId).then(setS).catch((e) => setMsg(e.message))
  }, [sessionId])
  useEffect(() => { reload(); api.genCapabilities().then(setCap).catch(() => {}) }, [reload])
  useEffect(() => {
    setFrictionText((s?.physics_spec?.friction || []).join(','))
  }, [s?.session_id, s?.physics_spec?.friction])

  const patch = async (fields: any) => {
    if (!s) return
    try { setS(await api.genPatchSpec(s.session_id, { expected_revision: s.revision, ...fields })) }
    catch (e: any) { setMsg(e.message); reload() }
  }
  const send = async () => {
    if (!s || !chat.trim()) return
    setBusy(true)
    try { setS(await api.genMessage(s.session_id, chat, s.revision)); setChat('') }
    catch (e: any) { setMsg(e.message); reload() }
    finally { setBusy(false) }
  }
  const answer = async (qid: string, ans: string): Promise<boolean> => {
    if (!s) return false
    try { setS(await api.genAnswer(s.session_id, qid, ans, s.revision)); return true }
    catch (e: any) { setMsg(e.message); reload(); return false }
  }
  const compile = async () => {
    if (!s) return
    try {
      const body: any = { expected_revision: s.revision, generation_mode: genMode }
      if (promptEdit !== null) body.prompt_override = promptEdit
      setS(await api.genCompile(s.session_id, body)); setPromptEdit(null)
    } catch (e: any) {
      setMsg(e.message)
      if (String(e.message || '').includes('409')) reload()
    }
  }
  const branch = async () => {
    if (!s) return
    const reason = window.prompt('新分支原因（重新生成会记录父修订与原因）')
    if (!reason) return
    try { setS(await api.genBranch(s.session_id, reason, s.revision)) } catch (e: any) { setMsg(e.message) }
  }

  if (!s) return <div className="p-4 text-xs text-muted">{msg || '加载中…'}</div>
  const spec = s.asset_spec, phys = s.physics_spec
  const auxReady = cap?.aux_model?.configured
  return (
    <div className="p-4">
      <Link to="/generation" className="text-xs text-accent">← 生成工作台</Link>
      <h2 className="text-lg text-ink my-2">会话 {s.session_id} <span className="text-xs text-muted">rev {s.revision} · {s.state}</span></h2>
      {msg && <div className="text-xs text-yellow-300 mb-2">{msg}</div>}
      <div className="grid grid-cols-2 gap-4">
        <div>
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
                {q.status === 'open' && <div className="space-y-1.5">
                  {q.options?.length > 0 && <div className="flex gap-1 flex-wrap">
                    {q.options.map((o: string) => (
                      <Button key={o} size="sm" variant="outline" onClick={() => answer(q.id, o)}>{o}</Button>))}
                  </div>}
                  <div className="flex gap-1 items-center">
                    <input className="flex-1" value={draftAnswers[q.id] ?? ''}
                      onChange={(e) => setDraftAnswers((prev) => ({ ...prev, [q.id]: e.target.value }))}
                      onKeyDown={async (e) => {
                        const v = (draftAnswers[q.id] ?? '').trim()
                        if (e.key === 'Enter' && v) {
                          if (await answer(q.id, v)) setDraftAnswers((p) => ({ ...p, [q.id]: '' }))
                        }
                      }}
                      placeholder="输入自由文本回答（如：20 cm / 0.15 kg / 无把手）" />
                    <Button size="sm" onClick={async () => {
                      const v = (draftAnswers[q.id] ?? '').trim()
                      if (v && await answer(q.id, v)) setDraftAnswers((p) => ({ ...p, [q.id]: '' }))
                    }}>提交</Button>
                  </div>
                </div>}
              </div>))}
          </Section>}
        </div>
        <div>
          <Section title="规格编辑（直接表单/JSON，跳过对话）">
            <div className="grid grid-cols-2 gap-x-3">
              <Field label="类型"><select value={spec.kind} onChange={(e) => {
                const nk = e.target.value
                if (nk === 'environment_splat' || (nk === 'robot_visual' && spec.role === 'dynamic')) {
                  patch({ kind: nk, role: 'visual_only' })
                } else if (nk === 'static_prop' && spec.role === 'dynamic') {
                  patch({ kind: nk, role: 'static' })
                } else {
                  patch({ kind: nk })
                }
              }}>
                {KINDS.map((k) => <option key={k}>{k}</option>)}</select></Field>
              <Field label="角色"><select value={spec.role} onChange={(e) => patch({ role: e.target.value })}>
                {ROLES.map((k) => <option key={k}>{k}</option>)}</select></Field>
            </div>
            <Field label="名称"><input value={spec.name} onChange={(e) => patch({ name: e.target.value })} /></Field>
            <Field label="用户原话（数据，非指令）"><input value={spec.user_words} onChange={(e) => patch({ user_words: e.target.value })} /></Field>
            <div className="grid grid-cols-2 gap-x-3">
              <DimInput label="高度" dim={spec.height} onChange={(v) => patch({ height_m: v })} />
              <DimInput label="半径" dim={spec.radius} onChange={(v) => patch({ radius_m: v })} />
              <DimInput label="宽度" dim={spec.width} onChange={(v) => patch({ width_m: v })} />
              <DimInput label="深度" dim={spec.depth} onChange={(v) => patch({ depth_m: v })} />
            </div>
            <Field label="外观/材质"><input value={spec.appearance} onChange={(e) => patch({ appearance: e.target.value })} /></Field>
            <Field label="参考图（URL 或上传本地图，用于 Lux3D 图生3D）">
              <div className="space-y-1">
                <div className="flex gap-1 items-center">
                  <input className="flex-1"
                    value={spec.reference_image && String(spec.reference_image).startsWith('data:') ? `[已上传图片 · ${Math.round(String(spec.reference_image).length / 1024)} KB]` : (spec.reference_image || '')}
                    readOnly={!!(spec.reference_image && String(spec.reference_image).startsWith('data:'))}
                    placeholder="https://... 或右侧上传图片"
                    onChange={(e) => patch({ reference_image: e.target.value || null })} />
                  <label className="text-[11px] px-2 py-1 rounded border border-edge bg-panel2 cursor-pointer text-accent hover:bg-panel whitespace-nowrap">
                    上传图片
                    <input type="file" accept="image/png,image/jpeg,image/webp" className="hidden" onChange={(e) => {
                      const f = e.target.files?.[0]
                      if (!f) return
                      if (f.size > 4 * 1024 * 1024) { setMsg('参考图大小不能超过 4 MB'); return }
                      const reader = new FileReader()
                      reader.onload = () => { if (typeof reader.result === 'string') patch({ reference_image: reader.result }) }
                      reader.readAsDataURL(f)
                    }} />
                  </label>
                  {spec.reference_image && <Button size="sm" variant="outline" onClick={() => patch({ reference_image: null })}>清除</Button>}
                </div>
                {spec.reference_image && <img src={spec.reference_image} alt="ref" className="h-14 rounded border border-edge object-contain bg-black/20" />}
              </div>
            </Field>
            <div className="grid grid-cols-3 gap-x-3 text-xs">
              <Field label="把手"><select value={spec.has_handle === null ? 'null' : String(spec.has_handle)}
                onChange={(e) => patch({ has_handle: e.target.value === 'null' ? null : e.target.value === 'true' })}>
                <option value="null">未说明</option><option value="false">无</option><option value="true">有</option></select></Field>
              <Field label="开口"><select value={spec.open_top === null ? 'null' : String(spec.open_top)}
                onChange={(e) => patch({ open_top: e.target.value === 'null' ? null : e.target.value === 'true' })}>
                <option value="null">未说明</option><option value="true">是</option><option value="false">否</option></select></Field>
              <Field label="可抓取"><select value={phys.graspable === null ? 'null' : String(phys.graspable)}
                onChange={(e) => patch({ graspable: e.target.value === 'null' ? null : e.target.value === 'true' })}>
                <option value="null">未知</option><option value="true">是</option><option value="false">否</option></select></Field>
            </div>
            <div className="grid grid-cols-2 gap-x-3">
              <Field label="质量（kg，空=未知）"><input type="number" step="0.01" value={phys.mass?.value_kg ?? ''}
                onChange={(e) => patch({ mass_kg: e.target.value === '' ? null : Number(e.target.value) })} /></Field>
              <Field label="摩擦 [slide,torsion,rolling]"><input value={frictionText}
                placeholder="0.95,0.005,0.0001"
                onChange={(e) => setFrictionText(e.target.value)}
                onBlur={() => {
                  const trimmed = frictionText.trim()
                  if (!trimmed) { if (phys.friction !== null) patch({ friction: null }); return }
                  const parts = trimmed.split(',').map((p) => p.trim()).filter(Boolean)
                  const v = parts.map(Number)
                  if (v.length === 3 && v.every((n) => Number.isFinite(n) && n >= 0)) {
                    patch({ friction: v })
                  } else {
                    setMsg('摩擦格式需为 3 个非负数值，如 0.95,0.005,0.0001')
                    setFrictionText((phys.friction || []).join(','))
                  }
                }}
                onKeyDown={(e) => { if (e.key === 'Enter') (e.target as HTMLInputElement).blur() }} /></Field>
            </div>
            <div className="text-[10px] text-muted">估计值标记 model_inferred/unverified；未知尺寸→未标定，处理前不绑定物理。</div>
          </Section>
        </div>
      </div>
      <Section title="编译预览（确定性、版本化；结构化字段权威）">
        <div className="flex gap-2 mb-2 flex-wrap items-center">
          {spec.reference_image && <select className="text-xs" value={genMode} onChange={(e) => setGenMode(e.target.value as any)}>
            <option value="auto">生成模式：自动（有图优先 img_to_3d）</option>
            <option value="img_to_3d">仅参考图生成 (img_to_3d)</option>
            <option value="text_to_3d">文本+参考图联合 (text_to_3d)</option>
          </select>}
          <Button onClick={compile}>编译 / 刷新预览</Button>
          <Button variant="outline" onClick={branch}>新建分支</Button>
          {s.compiled && <span className="text-xs text-muted self-center">compiler {s.compiled.compiler_version} · {s.compiled.provider} · {s.compiled.template} · hash {s.compiled.spec_hash?.slice(0, 12)}</span>}
        </div>
        {s.compiled ? <div className="grid grid-cols-2 gap-3 text-xs">
          <div>
            <div className="text-accent mb-1">英文提示词（可编辑，编辑后重编译）</div>
            <textarea className="w-full h-28 font-mono text-[11px]" value={promptEdit ?? s.compiled.prompt_en}
              onChange={(e) => setPromptEdit(e.target.value)} />
            <div className="text-accent mt-2 mb-1">中文规格</div>
            <pre className="whitespace-pre-wrap text-[11px] bg-panel2 border border-edge rounded p-2">{s.compiled.spec_zh}</pre>
          </div>
          <div>
            <div className="text-accent mb-1">Provider 载荷（不含质量/摩擦/关节）</div>
            <pre className="whitespace-pre-wrap text-[11px] bg-panel2 border border-edge rounded p-2">{JSON.stringify({
              ...s.compiled.payload,
              ...(typeof s.compiled.payload?.img === 'string' && s.compiled.payload.img.startsWith('data:')
                ? { img: `[data:image base64, ${s.compiled.payload.img.length} chars]` } : {}),
            }, null, 1)}</pre>
            <div className="text-accent mt-2 mb-1">不支持/后处理字段</div>
            <div className="text-[11px] text-muted">{s.compiled.excluded_or_postprocess_fields?.join(', ')}</div>
            <div className="text-accent mt-2 mb-1">警告</div>
            <ul className="text-[11px] text-yellow-300 list-disc pl-4">{s.compiled.warnings?.map((w: string, i: number) => <li key={i}>{w}</li>)}</ul>
          </div>
        </div> : <div className="text-xs text-muted">尚未编译。无语言模型时此处仍显示规格预览，不会渲染空三维画布。</div>}
      </Section>
      <div className="text-xs text-muted">下一步：确认生成 → 入库 → 组装 SceneRecipe → 校验（真实 MuJoCo）→ 应用到项目版本 → 仿真。</div>
      <ConfirmPanel s={s} promptDirty={promptEdit !== null && promptEdit !== s.compiled?.prompt_en} onDone={setS} onReload={reload} onTaskUpdated={() => setTaskRev((n) => n + 1)} />
      <TasksPanel sessionId={s.session_id} onUpdated={() => setTaskRev((n) => n + 1)} />
      <RecipePanel s={s} taskRev={taskRev} />
    </div>
  )
}

const NODE_LABELS: Record<string, string> = {
  submit: '提交', remote: '远程生成', download: '下载', import: '导入', validate: '校验',
}

function ConfirmPanel({ s, promptDirty, onDone, onReload, onTaskUpdated }: { s: any; promptDirty: boolean; onDone: (s: any) => void; onReload: () => void; onTaskUpdated?: () => void }) {
  const [provider, setProvider] = useState(s.compiled?.provider || (s.asset_spec.kind === 'environment_splat' ? 'aholo_world' : 'lux3d'))
  const [remoteId, setRemoteId] = useState('')
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState('')
  const [remoteList, setRemoteList] = useState<any[] | null>(null)
  const [remoteLoading, setRemoteLoading] = useState(false)
  const [remoteErr, setRemoteErr] = useState('')
  useEffect(() => {
    setProvider(s.compiled?.provider || (s.asset_spec.kind === 'environment_splat' ? 'aholo_world' : 'lux3d'))
  }, [s.session_id, s.asset_spec.kind, s.compiled?.provider])

  const compiled = s.compiled
  const reqHash = compiled?.request_hash || compiled?.spec_hash || ''
  const compiledStale = !!s.compiled_stale
  const providerMismatch = !!compiled && compiled.provider !== provider
  const canConfirm = !!compiled && !compiledStale && !promptDirty && !providerMismatch

  const loadRemote = async () => {
    setRemoteLoading(true); setRemoteErr('')
    try {
      const r = await api.genRemoteTasks(provider)
      setRemoteList(r.items || [])
      if (r.error) setRemoteErr(r.error)
    } catch (e: any) {
      setRemoteErr(e.message || '加载失败')
    } finally {
      setRemoteLoading(false)
    }
  }

  const importExistingRemote = async (rid: string) => {
    setBusy(true); setMsg('')
    try {
      const r = await api.genImportRemote(s.session_id, {
        remote_id: rid,
        provider,
        expected_revision: s.revision,
      })
      onDone(r.session)
      onTaskUpdated?.()
      setMsg(`已从账号导入任务 ${rid} → 本地任务 ${r.job?.id}（${r.job?.state}，未消耗新积分）。`)
    } catch (e: any) {
      const errText = String(e.message || '')
      if (/^409:/.test(errText)) onReload()
      setMsg(errText)
    } finally {
      setBusy(false)
    }
  }

  const confirm = async () => {
    if (!canConfirm) return
    setBusy(true); setMsg('')
    const storageKey = `vlab-confirm:${s.session_id}`
    const bindingKey = JSON.stringify([s.session_id, reqHash, compiled.spec_revision, provider, (s.branches || []).length, remoteId || ''])
    let crid = ''
    try {
      const saved = JSON.parse(localStorage.getItem(storageKey) || 'null')
      if (saved && saved.bindingKey === bindingKey && saved.client_request_id) {
        crid = saved.client_request_id
      }
    } catch { /* ignore storage parse error */ }
    if (!crid) {
      crid = (crypto as any).randomUUID ? crypto.randomUUID() : `crid-${Date.now()}-${Math.random().toString(16).slice(2, 8)}`
      try { localStorage.setItem(storageKey, JSON.stringify({ bindingKey, client_request_id: crid })) } catch { /* ignore */ }
    }
    try {
      const r = await api.genConfirm(s.session_id, {
        client_request_id: crid,
        expected_revision: s.revision,
        expected_compiled_revision: compiled.spec_revision,
        expected_request_hash: reqHash,
        provider,
        remote_id: remoteId || undefined,
      })
      if (r.job?.state === 'unconfigured') {
        try { localStorage.removeItem(storageKey) } catch { /* ignore */ }
      }
      onDone(r.session)
      onTaskUpdated?.()
      setMsg(`已确认生成：任务 ${r.job.id}（${r.job.state}）。绑定 spec rev ${r.binding.spec_revision} · hash ${r.binding.spec_hash?.slice(0, 12)} · ${r.binding.provider}。`)
    } catch (e: any) {
      const errText = String(e.message || '')
      if (/^(400|404|422):/.test(errText)) {
        try { localStorage.removeItem(storageKey) } catch { /* ignore */ }
      }
      if (/^409:/.test(errText)) onReload()
      setMsg(errText)
    }
    finally { setBusy(false) }
  }
  return (
    <Section title="确认生成（显式产品交互；模型回复不会自动触发付费生成）">
      <div className="flex gap-2 items-end flex-wrap text-xs">
        <Field label="Provider"><select value={provider} onChange={(e) => { setProvider(e.target.value); setRemoteList(null) }}>
          <option value="lux3d">lux3d（单体）</option><option value="aholo_world">aholo_world（背景）</option>
        </select></Field>
        <Field label="恢复已有远程任务 ID（可选，避免新建付费任务）">
          <input className="!w-56" value={remoteId} onChange={(e) => setRemoteId(e.target.value)} placeholder="如 5887561 或 3FO4K4RMF54N" /></Field>
        <Button disabled={busy || !canConfirm} onClick={confirm}>{busy ? '提交中…' : '确认生成'}</Button>
        <Button size="sm" variant="outline" disabled={remoteLoading} onClick={loadRemote}>
          {remoteLoading ? '查询中…' : '查看 Aholo 账号历史任务（免重扣积分）'}
        </Button>
      </div>
      {remoteErr && <div className="text-[11px] text-yellow-300 mt-1">{remoteErr}</div>}
      {remoteList && <div className="mt-2 border border-edge rounded p-2 bg-panel2 text-xs">
        <div className="flex justify-between items-center mb-1">
          <span className="text-accent">账号远端任务（{provider} · 共 {remoteList.length} 条）</span>
          <button className="text-[11px] text-muted hover:text-ink" onClick={() => setRemoteList(null)}>收起</button>
        </div>
        {remoteList.length === 0 ? <div className="text-muted text-[11px]">暂无远端任务记录</div> : (
          <div className="space-y-1 max-h-44 overflow-auto">
            {remoteList.map((it) => (
              <div key={it.remote_id} className="flex items-center justify-between gap-2 border-t border-edge pt-1">
                <div className="truncate">
                  <span className="font-mono text-accent">{it.remote_id}</span>
                  <span className="text-muted ml-2">{it.task_type || it.scene || ''} {it.name || ''}</span>
                  <span className={`ml-2 ${it.state === 'succeeded' ? 'text-accent' : it.state === 'failed' ? 'text-red-400' : 'text-yellow-300'}`}>{it.state}</span>
                  {it.consumed_credits != null && <span className="text-muted ml-2">{it.consumed_credits} 积分</span>}
                </div>
                <div className="flex gap-1 shrink-0">
                  <Button size="sm" variant="outline" onClick={() => setRemoteId(it.remote_id)}>填入 ID</Button>
                  {it.state !== 'failed' && (
                    <Button size="sm" disabled={busy} onClick={() => importExistingRemote(it.remote_id)}>直接导入本会话</Button>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>}
      {!compiled && <div className="text-[11px] text-yellow-300 mt-1">请先点击上方「编译 / 刷新预览」并审核英文提示词与载荷，然后再确认生成。</div>}
      {compiled && compiledStale && <div className="text-[11px] text-yellow-300 mt-1">规格在编译后已变更，请重新点击「编译 / 刷新预览」。</div>}
      {promptDirty && <div className="text-[11px] text-yellow-300 mt-1">英文提示词有未编译修改，请先点击「编译 / 刷新预览」校验并更新哈希。</div>}
      {providerMismatch && <div className="text-[11px] text-yellow-300 mt-1">当前已编译预览针对 {compiled.provider}；切换 provider 后请重新编译。</div>}
      <div className="text-[10px] text-muted mt-1">同一冻结规格与请求哈希的 client_request_id 会持久保留直至确认明确；重复点击或响应丢失重试不会新建本地任务。未知提交结果不会盲目重提。</div>
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

function TasksPanel({ sessionId, onUpdated }: { sessionId: string; onUpdated?: () => void }) {
  const [tasks, setTasks] = useState<any[]>([])
  const [msg, setMsg] = useState('')
  const [recoverIds, setRecoverIds] = useState<Record<string, string>>({})
  const sigRef = useRef('')
  const reload = useCallback(() => {
    api.genTasks(sessionId).then((r) => {
      const list = r.tasks || []
      setTasks(list)
      const sig = list.map((t: any) => `${t.id}:${t.state}:${t.asset_id || ''}`).join('|')
      if (sig !== sigRef.current) {
        sigRef.current = sig
        onUpdated?.()
      }
    }).catch(() => {})
  }, [sessionId])
  useEffect(() => { reload(); const t = setInterval(reload, 2500); return () => clearInterval(t) }, [reload])
  const act = async (jid: string, node: string) => {
    try { await api.genRetryNode(sessionId, jid, node); reload() } catch (e: any) { setMsg(e.message) }
  }
  const poll = async (jid: string) => {
    try { await api.genPoll(sessionId, jid); reload() } catch (e: any) { setMsg(e.message) }
  }
  const recover = async (jid: string, provider?: string) => {
    const rid = (recoverIds[jid] || '').trim()
    if (!rid) return
    try { await api.genRecoverTask(sessionId, jid, rid, provider); setMsg(`已绑定 remote_id=${rid}`); reload() }
    catch (e: any) { setMsg(e.message) }
  }
  const splitAsset = async (assetId: string) => {
    const slug = assetId.startsWith('imported/') ? assetId.slice('imported/'.length) : assetId
    try {
      const r = await api.partSplitAsset(slug, { segments: 4, axis: 'auto', remote: false })
      setMsg(`部件拆分完成（${r.method}）：共 ${r.part_count} 个部件 (${(r.parts || []).map((p: any) => p.name).join(', ')})`)
    } catch (e: any) { setMsg(e.message) }
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
            <span className="text-muted">{j.provider || 'aholo_world'} · {j.slug || j.params?.name || ''}</span>
            {j.remote_id && <span className="text-muted">remote {j.remote_id}</span>}
          </div>
          <div className="my-1"><NodeBar nodes={j.nodes} /></div>
          {j.error && <div className="text-red-400 text-[11px]">{j.error}</div>}
          <div className="flex gap-1 mt-1 flex-wrap items-center">
            {j.remote_id && !['succeeded', 'failed'].includes(j.state) &&
              <Button size="sm" variant="outline" onClick={() => poll(j.id)}>查询进度</Button>}
            {['retryable_error', 'failed'].includes(j.state) && j.remote_id && j.nodes &&
              ['download', 'import', 'validate'].map((n) => (
                <Button key={n} size="sm" variant="outline" onClick={() => act(j.id, n)}>重试{NODE_LABELS[n]}</Button>))}
            {!j.remote_id && ['submit_unknown', 'interrupted', 'unconfigured', 'retryable_error'].includes(j.state) && <>
              <input className="!w-44" value={recoverIds[j.id] || ''} placeholder="填入已找回的 remote_id"
                onChange={(e) => setRecoverIds((prev) => ({ ...prev, [j.id]: e.target.value }))} />
              <Button size="sm" variant="outline" onClick={() => recover(j.id, j.provider)}>绑定恢复</Button>
            </>}
            {j.state === 'succeeded' && j.asset_id && <>
              <Link className="text-accent text-[11px] self-center" to={`/library/${encodeURIComponent(j.asset_id)}`}>查看资产 →</Link>
              {j.asset_id.startsWith('imported/') && (
                <Button size="sm" variant="outline" onClick={() => splitAsset(j.asset_id)}>部件拆分 (part_split)</Button>
              )}
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
  const [recoverRid, setRecoverRid] = useState('')
  const reload = useCallback(() => {
    if (!sessionId) return
    api.genTasks(sessionId).then((r) => setJob(r.tasks.find((t: any) => t.id === jobId) || null)).catch((e) => setMsg(e.message))
  }, [sessionId, jobId])
  useEffect(() => { reload(); const t = setInterval(reload, 2500); return () => clearInterval(t) }, [reload])
  const retry = async (node: string) => { try { await api.genRetryNode(sessionId!, jobId!, node); reload() } catch (e: any) { setMsg(e.message) } }
  const poll = async () => { try { await api.genPoll(sessionId!, jobId!); reload() } catch (e: any) { setMsg(e.message) } }
  const recover = async () => {
    if (!recoverRid.trim()) return
    try { await api.genRecoverTask(sessionId!, jobId!, recoverRid.trim(), job?.provider); setMsg(`已绑定 remote_id=${recoverRid.trim()}`); reload() }
    catch (e: any) { setMsg(e.message) }
  }
  if (!job) return <div className="p-4 text-xs text-muted">{msg || '加载中…'}</div>
  return (
    <div className="p-4">
      <Link to={`/generation/${sessionId}`} className="text-xs text-accent">← 会话</Link>
      <h2 className="text-lg text-ink my-2">任务 {job.id} <span className="text-xs text-muted">{job.provider || 'aholo_world'} · {job.state}</span></h2>
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
          asset_id: job.asset_id, validation: job.validation, retries: job.retries, audit: job.audit,
        }, null, 1)}</pre>
      </Section>
      {job.error && <Section title="错误"><div className="text-xs text-red-400">{job.error}</div></Section>}
      <Section title="恢复">
        <div className="flex gap-1 flex-wrap items-center">
          {job.remote_id && !['succeeded', 'failed'].includes(job.state) &&
            <Button size="sm" variant="outline" onClick={poll}>查询进度</Button>}
          {job.remote_id && ['retryable_error', 'failed'].includes(job.state) && job.nodes && ['download', 'import', 'validate'].map((n) => (
            <Button key={n} size="sm" variant="outline" onClick={() => retry(n)}>重试{NODE_LABELS[n]}</Button>))}
          {!job.remote_id && ['submit_unknown', 'interrupted', 'unconfigured', 'retryable_error'].includes(job.state) && <>
            <input className="!w-52 text-xs" value={recoverRid} onChange={(e) => setRecoverRid(e.target.value)} placeholder="填入已找回的 remote_id" />
            <Button size="sm" variant="outline" onClick={recover}>在原任务上绑定恢复</Button>
            <span className="text-xs text-muted">不会重提远程生成；保留原规格与 request_id 审计记录。</span>
          </>}
        </div>
      </Section>
      {job.state === 'succeeded' && job.asset_id &&
        <Link className="text-accent text-xs" to={`/library/${encodeURIComponent(job.asset_id)}`}>查看已入库资产 →</Link>}
    </div>
  )
}

const WORKBENCH = { name: 'workbench', shape: 'box', full_size_m: [0.64, 0.43, 0.056], position: [0, 0, -0.028], quaternion_xyzw: [0, 0, 0, 1] }
const TOOLBOX = { name: 'toolbox', shape: 'box', full_size_m: [0.10, 0.10, 0.05], position: [0.25, -0.15, 0.025], quaternion_xyzw: [0, 0, 0, 1] }

function RecipePanel({ s, taskRev = 0 }: { s: any; taskRev?: number }) {
  const [mode, setMode] = useState('composed_scene')
  const [taskRef, setTaskRef] = useState('builtin/standard')
  const [bgRef, setBgRef] = useState('')
  const [renderer, setRenderer] = useState('mujoco')
  const [useTable, setUseTable] = useState(true)
  const [supportZ, setSupportZ] = useState(0)
  const [collidersText, setCollidersText] = useState('[]')
  const [assetNodesText, setAssetNodesText] = useState('[]')
  const [nodesTouched, setNodesTouched] = useState(false)
  const [useAsGoalTray, setUseAsGoalTray] = useState(false)
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
  }, [s.session_id])

  useEffect(() => {
    api.genTasks(s.session_id).then((r) => {
      const tasks = r.tasks || []
      const doneTask = tasks.find((t: any) => t.state === 'succeeded' && t.asset_id && t.params?.kind === 'task_object')
      if (doneTask) setTaskRef(doneTask.asset_id)
      const doneProp = tasks.find((t: any) => t.state === 'succeeded' && t.asset_id && (t.params?.kind === 'static_prop' || t.params?.kind === 'robot_visual'))
      if (doneProp && !nodesTouched) {
        setAssetNodesText(JSON.stringify([{
          node_id: useAsGoalTray ? 'goal_tray' : 'generated_prop',
          asset_ref: doneProp.asset_id,
          role: useAsGoalTray ? 'static' : (doneProp.params?.role === 'visual_only' ? 'visual_only' : 'static'),
          position: useAsGoalTray ? [0.133, 0.050, supportZ] : [0.25, -0.15, supportZ],
          quaternion_xyzw: [0, 0, 0, 1], uniform_scale: 1,
        }], null, 1))
      }
      const doneEnv = tasks.find((t: any) => t.state === 'succeeded' && t.asset_id && t.params?.kind === 'environment_mesh')
      if (doneEnv && !bgRef) setBgRef(doneEnv.asset_id)
    }).catch(() => {})
  }, [s.session_id, s.job_ids?.length, taskRev, nodesTouched, useAsGoalTray, supportZ, bgRef])

  const preset = () => { setCollidersText(JSON.stringify([WORKBENCH, TOOLBOX], null, 1)); setUseTable(false) }

  const create = async () => {
    setBusy(true); setMsg(''); setApplied(null)
    try {
      const colliders = JSON.parse(collidersText || '[]')
      const asset_nodes = JSON.parse(assetNodesText || '[]')
      const trayNodeId = useAsGoalTray ? (asset_nodes.find((n: any) => n.role === 'static')?.node_id || 'goal_tray') : null
      const r = await api.recipeFromSession(s.session_id, {
        scene_mode: mode, task_asset_ref: taskRef || null,
        background_ref: (mode === 'background_scene' || bgRef) ? (bgRef || null) : null,
        sensor_renderer: renderer, use_default_table: useTable, support_plane_z: supportZ,
        colliders, asset_nodes,
        use_as_goal_tray: useAsGoalTray,
        goal_tray_node_id: trayNodeId,
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
          <Field label="桌面支持面高度 support_plane_z (m)">
            <input type="number" step="0.005" value={supportZ} onChange={(e) => setSupportZ(Number(e.target.value) || 0)} />
          </Field>
          {s.asset_spec?.kind === 'static_prop' && (
            <label className="flex items-center gap-2 text-xs text-accent">
              <input type="checkbox" className="!w-auto" checked={useAsGoalTray} onChange={(e) => { setUseAsGoalTray(e.target.checked); setNodesTouched(false) }} />
              将该静态道具设为抓放目标托盘（替代内置绿盒）
            </label>
          )}
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
          <textarea className="w-full h-28 font-mono text-[11px]" value={collidersText} onChange={(e) => setCollidersText(e.target.value)} />
          <div className="text-accent mt-2 mb-1">多实例资产节点（asset_nodes JSON）</div>
          <textarea className="w-full h-24 font-mono text-[11px]" value={assetNodesText}
            onChange={(e) => { setNodesTouched(true); setAssetNodesText(e.target.value) }} />
          <div className="text-[10px] text-muted mt-1">static 节点会装载网格并自动/关联落地碰撞代理；visual_only 仅装载视觉；支持面 z=0。</div>
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
          goal_tray: report.checks?.goal_tray,
          asset_nodes: report.checks?.asset_nodes,
          background: report.checks?.background, dependency_graph: report.checks?.dependency_graph,
        }, null, 1)}</pre>
      </div>}
      {applied && <div className="mt-2 text-xs border border-edge rounded p-2 bg-panel2">
        <div className="text-accent">已应用：{applied.project_id} v{applied.version}（{applied.validation_status}）· 任务对象 {applied.task_asset}</div>
        <div className="text-[11px] text-muted mt-1">读回：nodes [{(applied.read_back?.asset_nodes || []).map((n: any) => `${n.node_id}:${n.asset_ref}`).join(', ')}] · colliders [{(applied.read_back?.colliders || []).map((c: any) => c.name).join(', ')}] · use_default_table={String(applied.read_back?.use_default_table)} · renderer={applied.read_back?.sensor_renderer} · tray={applied.read_back?.task?.goal_tray_collider_ref || (applied.read_back?.task?.show_builtin_tray === false ? 'custom' : 'builtin')}</div>
        <div className="flex gap-3 mt-1">
          <Link className="text-accent text-[11px]" to={`/projects/${applied.project_id}/sim`}>进入仿真 →</Link>
          <span className="text-[10px] text-muted">远程/ mock 成功≠可训练；此处以真实 MuJoCo 校验为准。</span>
        </div>
      </div>}
    </Section>
  )
}