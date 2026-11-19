import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api'

type Target = 'object' | 'chassis'
type Receipt = { requestId: string; target: Target; prompt: string; name: string; sessionId?: string;
  patched?: boolean; jobId?: string; state: string; node?: string; assetId?: string; error?: string }
const key = 'roboforge.duck-generation.v2'
const statuses: Record<string, string> = { preparing: '正在准备规格', submitting: '正在提交',
  remote_processing: 'Aholo 正在生成', downloading: '下载模型', importing: '导入模型',
  succeeded: '生成并导入完成', failed: '生成未完成', submit_unknown: '提交结果待确认' }
const initialPrompts = {
  object: '一只黄色塑料小鸭，圆润的身体、橙色短嘴、黑色眼睛，表面光滑，平整底面，单个完整刚体。',
  chassis: '紧凑的四轮抓取机器人底盘，白色与深绿色外壳、四个清晰独立的黑色轮胎，顶部平整安装台，不带机械臂和夹爪。',
}

export default function ModelGeneration({ locked, onUse }: { locked: boolean; onUse: (id: string, target: Target) => Promise<void> }) {
  const [target, setTarget] = useState<Target>('object')
  const [name, setName] = useState('custom-duck')
  const [prompt, setPrompt] = useState(initialPrompts.object)
  const [receipt, setReceipt] = useState<Receipt | null>(() => {
    try { return JSON.parse(localStorage.getItem(key) ?? 'null') } catch { return null }
  })
  const [configured, setConfigured] = useState<boolean | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const inFlight = useRef(false)
  const active = !!receipt && !['succeeded', 'failed'].includes(receipt.state)
  const disabled = locked || busy || active
  function save(next: Receipt) { localStorage.setItem(key, JSON.stringify(next)); setReceipt({ ...next }) }
  useEffect(() => { api.lux3dStatus().then(s => setConfigured(s.configured)).catch(e => setError(e.message)) }, [])

  async function submit(resume = false) {
    if (inFlight.current) return
    inFlight.current = true; setBusy(true); setError('')
    let next: Receipt = resume && receipt ? { ...receipt } : {
      requestId: crypto.randomUUID(), target, prompt: prompt.trim(), name: name.trim(), state: 'preparing',
    }
    save(next)
    try {
      if (!next.sessionId) {
        const session = await api.genCreateSession({ initial_text: next.prompt })
        next.sessionId = session.session_id; save({ ...next })
      }
      if (!next.patched) {
        const session = await api.genSession(next.sessionId!)
        const isObject = next.target === 'object'
        await api.genPatchSpec(next.sessionId!, { expected_revision: session.revision,
          name: next.name, kind: isObject ? 'task_object' : 'robot_visual', role: isObject ? 'dynamic' : 'visual_only',
          appearance: next.prompt, user_words: next.prompt,
          height_m: isObject ? .06 : .065, radius_m: isObject ? .024 : .09,
          width_m: isObject ? .048 : .18, depth_m: isObject ? .062 : .20,
          mass_kg: isObject ? .035 : .9,
        })
        next.patched = true; save({ ...next })
      }
      // A saved request ID makes recovery reuse the same backend submission.
      next.state = 'submitting'; save({ ...next })
      const result = await api.genConfirm(next.sessionId!, { provider: 'lux3d', client_request_id: next.requestId })
      next.jobId = result.job.id; next.state = result.job.state.toLowerCase()
      next.assetId = result.job.asset_id; next.node = result.job.node; next.error = result.job.error
      save({ ...next })
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e)
      const state = next.state === 'submitting' && !message.startsWith('400:') ? 'submit_unknown' : 'failed'
      save({ ...next, state, error: message }); setError(message)
    } finally { setBusy(false); inFlight.current = false }
  }

  useEffect(() => {
    if (!receipt?.sessionId || !receipt.jobId || ['succeeded', 'failed'].includes(receipt.state)) return
    let cancelled = false
    let timer: ReturnType<typeof setTimeout>
    const poll = async () => {
      try {
        const job = await api.genPoll(receipt.sessionId!, receipt.jobId!)
        if (cancelled) return
        const state = String(job.state).toLowerCase()
        save({ ...receipt, state, node: job.node, assetId: job.asset_id, error: job.error })
        setError('')
        if (!['succeeded', 'failed'].includes(state)) timer = setTimeout(poll, 4000)
      } catch (e) {
        if (!cancelled) { setError(e instanceof Error ? e.message : String(e)); timer = setTimeout(poll, 6000) }
      }
    }
    void poll()
    return () => { cancelled = true; clearTimeout(timer) }
  }, [receipt?.sessionId, receipt?.jobId, receipt?.state])

  return <details className="rf-generation" open>
    <summary><strong>描述你想要的 3D 模型</strong><span>直接调用 Aholo · 生成结果可用于当前场景</span></summary>
    <div className="rf-generation-body">
      <div className="rf-generation-form">
        <div className="rf-generation-fields">
          <label className="rf-field">模型用途<select aria-label="生成模型用途" value={target} disabled={disabled} onChange={e => {
            const t = e.target.value as Target; setTarget(t); setPrompt(initialPrompts[t]); setName(t === 'object' ? 'custom-duck' : 'custom-chassis')
          }}><option value="object">传送带上的抓取物体</option><option value="chassis">四轮机器人底盘外观</option></select></label>
          <label className="rf-field">模型名称<input aria-label="模型名称" maxLength={60} value={name} disabled={disabled} onChange={e => setName(e.target.value)} /></label>
        </div>
        <label className="rf-field">外形要求<textarea aria-label="3D 模型要求" maxLength={1500} value={prompt} disabled={disabled} onChange={e => setPrompt(e.target.value)} /></label>
        <button className="rf-generate-button" disabled={disabled || !configured || prompt.trim().length < 4 || !name.trim()} onClick={() => void submit()}>{busy ? '正在提交…' : '提交 Aholo 生成'}</button>
        <small>点击会提交一次在线生成，使用后端配置的 Aholo 额度。模型生成需要等待，刷新可恢复进度。</small>
      </div>
      <div className="rf-generation-result" aria-live="polite">
        <span className="rf-eyebrow">MODEL GENERATION</span>
        <strong>{receipt ? statuses[receipt.state] ?? '处理模型中' : '从你的描述开始'}</strong>
        <p>{receipt ? receipt.prompt : '描述物体的造型、颜色和材质；或生成一款新的四轮底盘。生成后点击采用，场景预览就会使用它。'}</p>
        {receipt?.sessionId && <Link to={`/generation/${receipt.sessionId}`}>查看完整生成任务 ↗</Link>}
        {receipt?.assetId && receipt.state === 'succeeded' && <button disabled={locked || busy} onClick={async () => {
          setBusy(true); setError(''); try { await onUse(receipt.assetId!, receipt.target) } catch (e) { setError(e instanceof Error ? e.message : String(e)) } finally { setBusy(false) }
        }}>用于当前场景 →</button>}
        {receipt && !receipt.jobId && !busy && <button disabled={locked} onClick={() => void submit(true)}>恢复这次提交</button>}
        {receipt && !active && <button className="rf-generation-reset" disabled={busy || locked} onClick={() => { localStorage.removeItem(key); setReceipt(null); setError('') }}>准备新的模型要求</button>}
        {(error || receipt?.error) && <p className="rf-generation-error" role="alert">{error || receipt?.error}</p>}
        {configured === false && <p className="rf-generation-error">后端尚未配置 Aholo，生成暂不可用。</p>}
        <small>当前物体按高 6 cm、35 g 导入；底盘按高 6.5 cm 导入。尺寸与质量为演示设定。底盘生成外观，关节臂沿用现有机构；新物体需要通过夹爪尺寸校验。</small>
      </div>
    </div>
  </details>
}
