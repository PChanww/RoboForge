import { useEffect, useState, useCallback } from 'react'
import { Link, useNavigate, useParams, useSearchParams } from 'react-router-dom'
import { api } from '../api'
import { Button, Section } from './ui'
import SceneCanvas from './SceneCanvas'

function useFetch<T>(fn: () => Promise<T>, deps: any[] = []) {
  const [data, setData] = useState<T | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const reload = useCallback(() => { fn().then(setData).catch((e) => setErr(e.message)) }, deps)
  useEffect(() => { reload() }, [reload])
  return { data, err, reload }
}

function JobStatus({ s }: { s: string }) {
  const color = s === 'succeeded' ? 'text-accent' : s === 'failed' ? 'text-red-400'
    : s === 'running' ? 'text-yellow-300' : 'text-muted'
  return <span className={color}>{s}</span>
}

export function JobsTable({ filter }: { filter?: string }) {
  const { data, reload } = useFetch(() => api.jobs(), [])
  useEffect(() => { const t = setInterval(reload, 2500); return () => clearInterval(t) }, [reload])
  const jobs = (data?.jobs || []).filter((j) => !filter || j.kind === filter || j.kind.startsWith(filter + '_'))
  return (
    <table className="w-full text-xs">
      <thead><tr className="text-muted text-left"><th className="py-1">ID</th><th>类型</th><th>状态</th><th>创建</th><th></th></tr></thead>
      <tbody>
        {jobs.map((j) => (
          <tr key={j.id} className="border-t border-edge">
            <td className="py-1 font-mono">{j.id}</td><td>{j.kind}</td><td><JobStatus s={j.status} /></td>
            <td className="text-muted">{j.created?.slice(11, 19)}</td>
            <td className="space-x-2">
              <Link className="text-accent" to={`/training/${j.id}`}>日志</Link>
              {j.status === 'running' && <button className="text-red-400" onClick={() => api.stopJob(j.id).then(reload)}>停止</button>}
            </td>
          </tr>
        ))}
        {!jobs.length && <tr><td colSpan={5} className="text-muted py-2">暂无任务</td></tr>}
      </tbody>
    </table>
  )
}

export function ProjectsPage() {
  const nav = useNavigate()
  const { data, reload } = useFetch(() => api.projects(), [])
  const [busy, setBusy] = useState(false)
  const [demoMsg, setDemoMsg] = useState('')
  const launchDemo = async (withWorld: boolean) => {
    setBusy(true); setDemoMsg('')
    try {
      const r = await api.setupHackathonDemo({ project_id: 'hackathon_demo', with_world_background: withWorld })
      reload()
      nav(`/projects/${r.project_id}/sim`)
    } catch (e: any) {
      setDemoMsg(e.message || 'Demo 加载失败')
    } finally {
      setBusy(false)
    }
  }
  return (
    <div className="p-4">
      <h2 className="text-lg text-ink mb-3">项目中心</h2>
      <div className="border border-accent/40 rounded p-3 bg-panel2 mb-4">
        <div className="text-sm text-accent font-medium mb-1">Aholo x RoboForge 黑客松旗舰演示</div>
        <div className="text-xs text-muted mb-2">
          一键装配（全链路 Aholo Lux3D 生成资产）：体素凹面分解托盘（真实内腔底面 + 凹面四壁碰撞）+ 多动态自由刚体（Lux3D 陶瓷杯 + Lux3D 伴随木块）+ 参数化支持面 + 标准 URDF 导出。
        </div>
        <div className="flex gap-2 flex-wrap">
          <Button size="sm" disabled={busy} onClick={() => launchDemo(false)}>
            {busy ? '装配并校验中…' : '一键进入物理复合 Demo (hackathon_demo)'}
          </Button>
          <Button size="sm" variant="outline" disabled={busy} onClick={() => launchDemo(true)}>
            加载 Aholo World 高斯融合版
          </Button>
        </div>
        {demoMsg && <div className="text-xs text-red-400 mt-1">{demoMsg}</div>}
      </div>
      <div className="grid grid-cols-2 gap-3">
        {(data?.projects || []).map((p) => (
          <div key={p.id} className="border border-edge rounded p-3 bg-panel2">
            <div className="text-accent">{p.id} <span className="text-muted text-xs">v{p.version}</span></div>
            <div className="text-xs text-muted my-1">机器人 {p.robots?.join(', ')} · 任务 {p.task}</div>
            <div className="space-x-2 text-xs">
              {(p.robots || []).map((r: string) => (
                <Link key={r} className="text-accent" to={`/robots/${r}/edit`}>机器人 {r}</Link>
              ))}
              <Link className="text-accent" to={`/projects/${p.id}/sim`}>仿真</Link>
              <Link className="text-accent" to="/datasets">数据集</Link>
              <Link className="text-accent" to="/training">训练</Link>
              <Link className="text-accent" to="/evaluations">评测</Link>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export function LibraryPage() {
  const { data } = useFetch(() => api.assetLibrary(), [])
  const scenes = useFetch(() => api.sceneAssets(), [])
  const gen = useFetch(() => api.generationStatus(), [])
  const [msg, setMsg] = useState('')
  const [prompt, setPrompt] = useState('An empty robotics laboratory with clear central workspace, no robots or people.')
  const [upAxis, setUpAxis] = useState('Y')
  const [unitScale, setUnitScale] = useState(1)
  const [calibrated, setCalibrated] = useState(false)
  const [importId, setImportId] = useState('')
  const [generating, setGenerating] = useState(false)
  const [singleName, setSingleName] = useState('cup-web')
  const [singlePrompt, setSinglePrompt] = useState('A single upright handleless ceramic tumbler, 9 cm tall')
  const lux = useFetch(() => api.lux3dStatus(), [])
  const generateObject = async () => {
    try {
      const r = await api.lux3dGenerate({ name: singleName, prompt: singlePrompt })
      setMsg(`已提交单体生成任务 ${r.id}（provider lux3d）`)
    } catch (e: any) { setMsg(`单体生成不可用: ${e.message}`) }
  }
  const generate = async () => {
    setGenerating(true)
    try { await api.generateScene({prompt}); gen.reload(); setMsg('已提交场景任务；查询状态会下载并登记完成的产物。') }
    catch (e: any) { setMsg(e.message) }
    finally { setGenerating(false) }
  }
  const onImport = async (file: File) => {
    if (file.size > 256 * 1024 * 1024) { setMsg('场景不能超过 256 MiB'); return }
    const buf = await file.arrayBuffer()
    let bin = ''
    const bytes = new Uint8Array(buf)
    for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i])
    const slug = importId || file.name.replace(/\.(glb|ply)$/i, '').replace(/[^a-zA-Z0-9_-]/g, '-').replace(/^-+|-+$/g, '') || ('scene_' + Date.now())
    try {
      await api.sceneImport({ filename: file.name, data_b64: btoa(bin), asset_id: slug, up_axis: upAxis, unit_scale: unitScale, calibrated })
      setMsg(`已导入场景 ${slug}`)
      scenes.reload()
    } catch (e: any) { setMsg(`导入失败: ${e.message}`) }
  }
  return (
    <div className="p-4">
      <h2 className="text-lg text-ink mb-2">资产中心</h2>
      <div className="text-xs text-muted mb-3">World：{gen.data?.configured ? '凭据已配置' : '请在后端配置 AHOLO_API_KEY'}；高斯场景可生成和登记，训练渲染仍未支持。</div>
      <div className="flex gap-2 mb-3 items-center text-xs">
        <input aria-label="场景提示词" value={prompt} onChange={e => setPrompt(e.target.value)} />
        <Button disabled={generating || !gen.data?.configured} onClick={generate}>生成场景</Button>
      </div>
      {(gen.data?.jobs || []).map((j: any) => <div key={j.id} className="text-xs mb-2 flex gap-2 items-center">
        <span>{j.id} · {j.state}{j.asset_status ? ` · ${j.asset_status}` : ''} {j.error}</span>
        {j.remote_id && !['succeeded','failed'].includes(j.state) && <Button size="sm" onClick={async () => {
          try { await api.pollGeneration(j.id); gen.reload(); scenes.reload() } catch (e: any) { setMsg(e.message) }
        }}>查询 / 重试下载</Button>}
      </div>)}
      <div className="flex gap-2 mb-3 text-xs items-center">
        <label>导入资产 ID（可选）<input value={importId} onChange={e => setImportId(e.target.value)} placeholder="已导入资产请使用新 ID" /></label>
        <label>原始上轴 <select value={upAxis} onChange={e => setUpAxis(e.target.value)}><option>Y</option><option>Z</option></select></label>
        <label>米/原单位 <input type="number" min="0.000001" step="any" value={unitScale} onChange={e => setUnitScale(Number(e.target.value))} /></label>
        <label><input type="checkbox" checked={calibrated} onChange={e => setCalibrated(e.target.checked)} />已确认实际尺寸</label>
      </div>
      <div className="flex gap-2 mb-2 items-center text-xs flex-wrap border-t border-edge pt-2">
        <span className="text-muted">单体资产（Lux3D）：{lux.data?.configured ? '已配置' : '等待配置 AHOLO_API_KEY'}</span>
        <input className="!w-28" placeholder="资产名" value={singleName} onChange={e => setSingleName(e.target.value)} />
        <input className="!w-80" placeholder="提示词" value={singlePrompt} onChange={e => setSinglePrompt(e.target.value)} />
        <Button size="sm" disabled={!singleName.trim()} onClick={generateObject}>生成单体资产</Button>
      </div>
      <div className="flex gap-2 mb-3 items-center">
        <label className="inline-flex items-center px-3 py-1.5 rounded-md border border-edge text-xs text-muted hover:text-ink cursor-pointer">
          导入场景(.glb/.ply)
          <input type="file" accept=".glb,.ply" className="hidden"
            onChange={(e) => { const f = e.target.files?.[0]; if (f) onImport(f) }} />
        </label>
      </div>
      {msg && <div className="text-xs text-yellow-300 mb-2">{msg}</div>}
      <div className="grid grid-cols-3 gap-3">
        {(data?.assets || []).map((a) => (
          <Link key={a.id} to={`/library/${encodeURIComponent(a.id)}`} className="border border-edge rounded p-3 bg-panel2 hover:border-accent">
            <div className="text-accent text-sm">{a.name}</div>
            <div className="text-xs text-muted">{a.category} · {a.source}{a.generation_id ? ` · #${a.generation_id}` : ''}</div>
          </Link>
        ))}
      </div>
      <h3 className="text-sm text-ink mt-4 mb-2">场景</h3>
      <div className="grid grid-cols-3 gap-3">
        {(scenes.data?.scenes || []).map((s: any, i: number) => (
          <Link key={s.asset_id + s.revision + i} to={`/scenes/${s.asset_id}/edit?revision=${s.revision}`}
            className="border border-edge rounded p-3 bg-panel2 hover:border-accent">
            <div className="text-accent text-sm">{s.display_name} <span className="text-[10px] text-muted">rev{s.revision}</span></div>
            <div className="text-xs text-muted">{s.format} · {s.status}{s.calibrated ? ' · 已标定' : ' · 未标定'}</div>
          </Link>
        ))}
        {!(scenes.data?.scenes || []).length && <div className="text-xs text-muted">暂无场景，使用上方导入或生成。</div>}
      </div>
    </div>
  )
}

export function SceneEditPage() {
  const { sceneId } = useParams()
  const [query] = useSearchParams()
  const revision = Number(query.get('revision')) || undefined
  const { data } = useFetch(() => api.sceneAsset(sceneId || '', revision), [sceneId, revision])
  const [project, setProject] = useState('scene_demo')
  const [defaultTable, setDefaultTable] = useState(true)
  const [msg, setMsg] = useState('')
  const bind = async () => {
    try {
      await api.bindScene(project, { asset_id: sceneId, revision: data?.revision ?? 1, use_default_table: defaultTable })
      setMsg(`已绑定到项目 ${project}（默认桌面${defaultTable ? '启用' : '关闭'}）`)
    } catch (e: any) { setMsg(`绑定失败: ${e.message}`) }
  }
  const validate = async () => {
    try { const r = await api.sceneValidate(sceneId || '', revision); setMsg(r.valid ? '校验通过' : '校验问题: ' + r.issues.join('; ')) }
    catch (e: any) { setMsg(e.message) }
  }
  return (
    <div className="p-4">
      <Link to="/library" className="text-xs text-accent">← 资产中心</Link>
      <h2 className="text-lg text-ink my-2">场景编辑 · {data?.display_name ?? sceneId}</h2>
      {data?.preview_asset && <img alt="场景全景预览（非训练相机）" className="w-full max-h-80 object-contain bg-panel2 mb-3"
        src={`/api/v1/scene-assets/${encodeURIComponent(sceneId || '')}/revisions/${data.revision}/preview`} />}
      <div className="text-xs text-muted space-y-1">
        <div>来源 {data?.provider}/{data?.source_kind} · 格式 {data?.format} · 状态 {data?.status}</div>
        <div>revision {data?.revision} · up_axis {data?.up_axis} · unit_scale {data?.unit_scale} · 已标定 {String(data?.calibrated)}</div>
        {data?.bbox && <div>包围盒 {JSON.stringify(data.bbox)}</div>}
      </div>
      <div className="flex gap-2 items-center mt-3 text-xs">
        <input className="!w-40" value={project} onChange={(e) => setProject(e.target.value)} />
        <label><input type="checkbox" checked={defaultTable} onChange={e => setDefaultTable(e.target.checked)} />使用默认桌面；已有碰撞代理时可关闭</label>
        <Button size="sm" disabled={!data?.calibrated || !data?.visual_asset || data?.status !== 'ready'} onClick={bind}>用于项目（绑定场景）</Button>
        <Button size="sm" variant="outline" onClick={validate}>校验</Button>
      </div>
      {msg && <div className="text-xs text-yellow-300 mt-2">{msg}</div>}
      <Link className="text-accent text-xs" to={`/projects/${encodeURIComponent(project)}/sim`}>打开该项目仿真</Link>
      <SceneCanvasPanel project={project} assetId={sceneId} revision={revision} asset={data} onMsg={setMsg} />
      <div className="text-xs text-muted mt-3">三维画布拖拽编辑（TransformControls/测量/撤销重做）为后续增强；当前提供数值标定与碰撞代理编辑，保存即写入项目配置版本，刷新/重开一致。</div>
    </div>
  )
}

function SceneCanvasPanel({ project, assetId, revision, asset, onMsg }:
  { project: string; assetId?: string; revision?: number; asset: any; onMsg: (s: string) => void }) {
  const { data, reload } = useFetch<any>(() => api.getScene(project), [project])
  const [inst, setInst] = useState<any>({ position: [0, 0, 0], quaternion_xyzw: [0, 0, 0, 1], uniform_scale: 1 })
  const [cols, setCols] = useState<any[]>([])
  const [nodes, setNodes] = useState<any[]>([])
  const [sel, setSel] = useState<any>(null)
  const [grid, setGrid] = useState(true)
  const [dirty, setDirty] = useState(false)
  const [undo, setUndo] = useState<any[]>([])
  const [redo, setRedo] = useState<any[]>([])
  const [ver, setVer] = useState<number>(0)
  const [p1, setP1] = useState<number[]>([0, 0, 0])
  const [p2, setP2] = useState<number[]>([0.1, 0, 0])
  const [known, setKnown] = useState(0.1)
  useEffect(() => {
    if (data) {
      const e = data.environment
      setInst({ position: e?.position ?? [0, 0, 0], quaternion_xyzw: e?.quaternion_xyzw ?? [0, 0, 0, 1], uniform_scale: e?.uniform_scale ?? 1 })
      setCols(data.colliders || []); setNodes(data.asset_nodes || []); setVer(data.version ?? 0); setDirty(false); setUndo([]); setRedo([])
    }
  }, [data])
  const snapshot = () => { setUndo((u) => [...u.slice(-49), { inst, cols }]); setRedo([]) }
  const doUndo = () => { if (!undo.length) return; const prev = undo[undo.length - 1]; setRedo((r) => [...r, { inst, cols }]); setUndo(undo.slice(0, -1)); setInst(prev.inst); setCols(prev.cols); setDirty(true) }
  const doRedo = () => { if (!redo.length) return; const nx = redo[redo.length - 1]; setUndo((u) => [...u, { inst, cols }]); setRedo(redo.slice(0, -1)); setInst(nx.inst); setCols(nx.cols); setDirty(true) }
  const moveCol = (i: number, p: number[], q: number[]) => { snapshot(); setCols(cols.map((c, k) => k === i ? { ...c, position: p.map((v) => Math.round(v * 1000) / 1000), quaternion_xyzw: q.map((v) => Math.round(v * 1e6) / 1e6) } : c)); setDirty(true) }
  const moveInst = (p: number[], q: number[]) => { snapshot(); setInst({ ...inst, position: p.map((v) => Math.round(v * 1000) / 1000), quaternion_xyzw: q.map((v) => Math.round(v * 1e6) / 1e6) }); setDirty(true) }
  const setCol = (i: number, patch: any) => { setCols(cols.map((c, k) => k === i ? { ...c, ...patch } : c)); setDirty(true) }
  const addCol = () => { snapshot(); setCols([...cols, { name: `proxy_${cols.length + 1}`, shape: 'box', full_size_m: [0.2, 0.2, 0.05], position: [0, 0, 0], quaternion_xyzw: [0, 0, 0, 1] }]); setDirty(true) }
  const delCol = (i: number) => { snapshot(); setCols(cols.filter((_, k) => k !== i)); setDirty(true) }
  const measured = Math.hypot(p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])
  const applyScale = () => {
    if (!(measured > 1e-6) || !(known > 0)) return
    snapshot(); const f = known / measured
    setInst({ ...inst, uniform_scale: Math.round(inst.uniform_scale * f * 1e6) / 1e6 }); setDirty(true)
    onMsg(`按已知距离缩放 ×${f.toFixed(3)}（测量 ${measured.toFixed(4)} → ${known}）`)
  }
  const saveAll = async () => {
    if (!data?.environment && !cols.length && !nodes.length) return
    try {
      const r = await api.commitScene(project, {
        environment: data?.environment ? { ...data.environment, position: inst.position, quaternion_xyzw: inst.quaternion_xyzw, uniform_scale: inst.uniform_scale } : null,
        colliders: cols, asset_nodes: nodes, use_default_table: data?.use_default_table, work_surface_z: data?.work_surface_z,
        expected_version: ver,
      })
      setVer(r.version); onMsg(`已原子保存场景实例、${(r.asset_nodes || []).length} 个资产节点与 ${r.colliders.length} 个碰撞代理 → v${r.version}`); setDirty(false); reload()
    } catch (e: any) {
      onMsg(`保存失败: ${e.message}${String(e.message).includes('409') ? '（其他页面已更新，请放弃本地修改后重试）' : ''}`)
    }
  }
  const dirtyRef = () => { if (dirty) { if (!confirm('有未保存修改，放弃？')) return; } reload() }
  const assetUrl = asset?.visual_asset && assetId
    ? `/api/v1/scene-assets/${encodeURIComponent(assetId)}/revisions/${revision ?? asset?.revision ?? 1}/visual.obj` : null
  return (
    <div className="mt-4 border-t border-edge pt-3">
      <div className="flex items-center gap-2 mb-2 text-xs">
        <span className="text-accent">三维场景编辑（{project} · v{ver}）{dirty && <span className="text-yellow-400">● 未保存</span>}</span>
        <label className="ml-auto flex items-center gap-1 text-muted"><input type="checkbox" checked={grid} onChange={(e) => setGrid(e.target.checked)} />网格</label>
        <Button size="sm" variant="outline" onClick={doUndo} disabled={!undo.length}>撤销</Button>
        <Button size="sm" variant="outline" onClick={doRedo} disabled={!redo.length}>重做</Button>
        <Button size="sm" variant="outline" onClick={dirtyRef}>放弃修改</Button>
        <Button size="sm" onClick={saveAll} disabled={!data?.environment && !cols.length && !nodes.length}>保存</Button>
      </div>
      {!data?.environment && !nodes.length && <div className="text-xs text-muted mb-2">该项目尚未绑定场景；先在下方绑定或应用配方。</div>}
      {asset && !asset.visual_asset && <div className="text-xs text-yellow-300 mb-2">
        该资产为高斯/点云（无 MJCF 网格视觉）：画布仅显示全景预览；渲染由 hybrid 管线负责。</div>}
      {(assetUrl || nodes.length > 0) ? <SceneCanvas assetUrl={assetUrl} inst={inst} cols={cols} assetNodes={nodes} selected={sel}
        onSelect={setSel} onMove={moveCol} onInstanceMove={moveInst} showGrid={grid} /> :
        asset?.preview_asset ? <img alt="场景全景预览" className="w-full max-h-72 object-contain bg-panel2 rounded" src={`/api/v1/scene-assets/${encodeURIComponent(assetId || '')}/revisions/${revision ?? asset?.revision ?? 1}/preview`} /> : null}
      <div className="text-[11px] text-muted mt-1">点击物体/线框选中；拖动 gizmo 平移并回写数值；碰撞为线框，视觉与同一坐标。</div>
      <div className="flex flex-wrap gap-2 items-end text-xs mt-2">
        {['x', 'y', 'z'].map((ax, i) => (
          <label key={ax} className="flex flex-col"><span className="text-muted">实例 {ax}</span>
            <input className="!w-20" type="number" step="0.01" value={inst.position[i] ?? 0}
              onChange={(e) => { setInst({ ...inst, position: inst.position.map((v: number, k: number) => k === i ? parseFloat(e.target.value) || 0 : v) }); setDirty(true) }} /></label>))}
        <label className="flex flex-col"><span className="text-muted">统一缩放</span>
          <input className="!w-20" type="number" step="0.01" value={inst.uniform_scale}
            onChange={(e) => { setInst({ ...inst, uniform_scale: parseFloat(e.target.value) || 1 }); setDirty(true) }} /></label>
        {['qx', 'qy', 'qz', 'qw'].map((ax, i) => (
          <label key={ax} className="flex flex-col"><span className="text-muted">实例 {ax}</span>
            <input className="!w-20" type="number" step="0.01" value={inst.quaternion_xyzw[i] ?? (i === 3 ? 1 : 0)}
              onChange={(e) => { const q = [...inst.quaternion_xyzw]; q[i] = parseFloat(e.target.value) || 0
                setInst({ ...inst, quaternion_xyzw: q }); setDirty(true) }} /></label>))}
      </div>
      <div className="flex flex-wrap gap-2 items-end text-xs mt-2 border-t border-edge pt-2">
        <span className="text-muted">测量：</span>
        {p1.map((v, i) => <input key={'a' + i} className="!w-16" type="number" step="0.01" value={v}
          onChange={(e) => setP1(p1.map((x, j) => j === i ? parseFloat(e.target.value) || 0 : x))} />)}
        <span className="text-muted">→</span>
        {p2.map((v, i) => <input key={'b' + i} className="!w-16" type="number" step="0.01" value={v}
          onChange={(e) => setP2(p2.map((x, j) => j === i ? parseFloat(e.target.value) || 0 : x))} />)}
        <span className="text-muted">= {measured.toFixed(4)}</span>
        <span className="text-muted">已知(m)</span><input className="!w-20" type="number" step="0.01" value={known} onChange={(e) => setKnown(parseFloat(e.target.value) || 0)} />
        <Button size="sm" variant="outline" onClick={applyScale}>按已知距离缩放</Button>
      </div>
      <div className="text-xs text-accent mt-3 mb-1">碰撞代理</div>
      <div className="space-y-1 text-xs">
        {cols.map((c, i) => (
          <div key={i} onClick={() => setSel({ kind: 'collider', index: i })}
            className={`flex flex-wrap gap-1 items-center p-1 rounded ${sel?.kind === 'collider' && sel.index === i ? 'bg-panel' : ''}`}>
            <input className="!w-24" value={c.name} onChange={(e) => setCol(i, { name: e.target.value })} />
            <select className="!w-24" value={c.shape} onChange={(e) => setCol(i, { shape: e.target.value })}>
              <option value="box">box</option><option value="plane">plane</option><option value="cylinder">cylinder</option>
            </select>
            {c.full_size_m.map((v: number, k: number) => (
              <input key={k} className="!w-16" type="number" step="0.01" value={v}
                onChange={(e) => setCol(i, { full_size_m: c.full_size_m.map((x: number, j: number) => j === k ? parseFloat(e.target.value) || 0 : x) })} />
            ))}
            {c.position.map((v: number, k: number) => (
              <input key={'p' + k} className="!w-16" type="number" step="0.01" value={v}
                onChange={(e) => setCol(i, { position: c.position.map((x: number, j: number) => j === k ? parseFloat(e.target.value) || 0 : x) })} />
            ))}
            {(c.quaternion_xyzw || [0, 0, 0, 1]).map((v: number, k: number) => (
              <input key={'q' + k} className="!w-14" type="number" step="0.01" value={v}
                onChange={(e) => { const q = [...(c.quaternion_xyzw || [0, 0, 0, 1])]; q[k] = parseFloat(e.target.value) || 0
                  setCol(i, { quaternion_xyzw: q }) }} />
            ))}
            <button className="text-red-400" onClick={(e) => { e.stopPropagation(); delCol(i) }}>删除</button>
          </div>))}
      </div>
      <div className="flex gap-2 mt-2">
        <Button size="sm" variant="outline" onClick={addCol}>添加代理</Button>
      </div>
    </div>
  )
}

export function LibraryDetail() {
  const { assetId } = useParams()
  const { data } = useFetch(() => api.assetLibrary(), [])
  const a = (data?.assets || []).find((x) => x.id === decodeURIComponent(assetId || ''))
  return (
    <div className="p-4">
      <Link to="/library" className="text-xs text-accent">← 资产中心</Link>
      <h2 className="text-lg text-ink my-2">{a?.name ?? assetId}</h2>
      <div className="text-xs text-muted space-y-1">
        <div>类别 {a?.category} · 来源 {a?.source}</div>
        {a?.generation_id && <div>生成任务 #{a.generation_id}</div>}
        {a?.height_m && <div>校准高度 {a.height_m} m</div>}
      </div>
      <div className="mt-3 space-x-2">
        <Link to="/projects/default/sim"><Button>在仿真中查看</Button></Link>
      </div>
      <div className="text-xs text-muted mt-3">三维预览与绑定到连杆请在仿真工作台选择该资产/相机；机器人外观绑定见机器人编辑。</div>
    </div>
  )
}

function Num({ value, onChange, step = 0.01, w = 20 }: { value: number; onChange: (v: number) => void; step?: number; w?: number }) {
  return <input type="number" step={step} value={value} style={{ width: `${w * 4}px` }}
    onChange={(e) => onChange(Number(e.target.value))} />
}

export function RobotEditPage() {
  const { robotId: routeRobot } = useParams()
  const projects = useFetch(() => api.projects(), [])
  const [project, setProject] = useState('default')
  const cfgF = useFetch(() => api.getConfig(project), [project])
  const [draft, setDraft] = useState<any>(null)
  const [robot, setRobot] = useState(routeRobot || '')
  const [tab, setTab] = useState<'joints' | 'links' | 'actuators' | 'gripper' | 'sensors'>('joints')
  const [msg, setMsg] = useState('')
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    if (!cfgF.data) return
    setDraft(JSON.parse(JSON.stringify(cfgF.data)))
    const ids = (cfgF.data.robots || []).map((r: any) => r.id)
    setRobot(routeRobot && ids.includes(routeRobot) ? routeRobot : ids[0] || '')
  }, [cfgF.data, routeRobot])

  if (!draft) return <div className="p-4 text-xs text-muted">{cfgF.err || '加载配置…'}</div>
  const robots = draft.robots || []
  const ri = robots.findIndex((r: any) => r.id === robot)
  const r = robots[ri]
  const patchLink = (i: number, k: string, v: any) => {
    const d = { ...draft }; d.robots[ri].links[i] = { ...r.links[i], [k]: v }; setDraft(d)
  }
  const patchVec = (arr: string, i: number, k: string, j: number, v: number) => {
    const d = { ...draft }; const vec = [...(r[arr][i][k] || [0, 0, 0])]; vec[j] = v
    d.robots[ri][arr][i] = { ...r[arr][i], [k]: vec }; setDraft(d)
  }
  const patchJoint = (i: number, k: string, v: any) => {
    const d = { ...draft }; d.robots[ri].joints[i] = { ...r.joints[i], [k]: v }; setDraft(d)
  }
  const patchAct = (i: number, k: string, v: any) => {
    const d = { ...draft }; d.robots[ri].actuators[i] = { ...r.actuators[i], [k]: v }; setDraft(d)
  }
  const patchSensor = (i: number, path: string, v: any) => {
    const d = { ...draft }; const s = { ...d.sensors[i] }
    if (path.startsWith('intrinsics.')) s.intrinsics = { ...s.intrinsics, [path.split('.')[1]]: v }
    else s[path] = v
    d.sensors[i] = s; setDraft(d)
  }
  const validate = async () => {
    try {
      const res = await api.validateConfig(draft)
      setMsg(res.valid ? '校验通过' : `校验失败：${res.issues.join('；')}`)
      return res.valid
    } catch (e: any) { setMsg(`校验失败：${e.message}`); return false }
  }
  const save = async () => {
    setBusy(true); setMsg('')
    try {
      const cfg = await api.saveConfig(draft, project)
      setDraft(JSON.parse(JSON.stringify(cfg)))
      setMsg(`已保存为 ${project} v${cfg.version}`)
    } catch (e: any) { setMsg(`保存失败：${e.message}`) }
    finally { setBusy(false) }
  }
  if (!r) return <div className="p-4 text-xs text-yellow-300">
    机器人 “{routeRobot}” 不在项目 {project} 中。 <Link to="/projects" className="text-accent">返回项目</Link>
  </div>
  const tabs: [typeof tab, string][] = [['joints', '关节'], ['links', '连杆'], ['actuators', '执行器'],
    ['gripper', '夹爪'], ['sensors', '相机']]
  return (
    <div className="p-4">
      <div className="flex items-center gap-2 mb-2">
        <Link to="/projects" className="text-xs text-accent">← 项目</Link>
        <h2 className="text-lg text-ink">机器人编辑 · {r.id}</h2>
        <span className="text-xs text-muted">配置 v{draft.version}</span>
        <span className="flex-1" />
        <label className="text-xs text-muted">项目
          <select value={project} onChange={(e) => setProject(e.target.value)}>
            {(projects.data?.projects || []).map((p: any) => <option key={p.id}>{p.id}</option>)}
          </select>
        </label>
        <Button size="sm" variant="outline" onClick={validate}>校验</Button>
        <Button size="sm" disabled={busy} onClick={save}>保存新版本</Button>
        <a className="text-xs text-accent border border-edge rounded px-2 py-1 hover:bg-panel"
          href={api.robotUrdfUrl(project, robot)} download={`${project}-${robot}.urdf`}>导出 URDF</a>
      </div>
      <div className="flex gap-2 mb-2 text-xs">
        {robots.map((x: any) => (
          <button key={x.id} onClick={() => { setRobot(x.id); setMsg('') }}
            className={`px-2 py-1 rounded border ${x.id === robot ? 'border-accent text-accent' : 'border-edge text-muted'}`}>
            {x.id}
          </button>
        ))}
        <Link className="ml-auto text-accent" to={`/projects/${project}/sim`}>在仿真中验证 →</Link>
      </div>
      <div className="flex gap-1 mb-2 text-xs">
        {tabs.map(([k, label]) => (
          <button key={k} onClick={() => setTab(k)}
            className={`px-2 py-1 rounded ${tab === k ? 'bg-panel text-accent' : 'text-muted hover:text-ink'}`}>{label}</button>
        ))}
      </div>
      {msg && <div className="text-xs text-yellow-300 mb-2">{msg}</div>}
      {tab === 'joints' && <Section title="关节（initial / damping / armature / 限位弧度）">
        <table className="w-full text-xs"><thead><tr className="text-muted text-left">
          <th>名称</th><th>类型</th><th>初始</th><th>阻尼</th><th>惯量</th><th>下限</th><th>上限</th>
        </tr></thead><tbody>
          {r.joints.map((j: any, i: number) => <tr key={j.name} className="border-t border-edge">
            <td className="py-1">{j.name}</td><td>{j.type}</td>
            <td><Num value={j.initial ?? 0} onChange={(v) => patchJoint(i, 'initial', v)} /></td>
            <td><Num value={j.damping} onChange={(v) => patchJoint(i, 'damping', v)} /></td>
            <td><Num value={j.armature} onChange={(v) => patchJoint(i, 'armature', v)} step={0.001} /></td>
            <td>{j.range ? <Num value={j.range[0]} onChange={(v) => { const d = { ...draft }; d.robots[ri].joints[i] = { ...j, range: [v, j.range[1]] }; setDraft(d) }} /> : '—'}</td>
            <td>{j.range ? <Num value={j.range[1]} onChange={(v) => { const d = { ...draft }; d.robots[ri].joints[i] = { ...j, range: [j.range[0], v] }; setDraft(d) }} /> : '—'}</td>
          </tr>)}
        </tbody></table>
      </Section>}
      {tab === 'links' && <Section title="连杆（父子原点 xyz / 质心 / 重力补偿）">
        <table className="w-full text-xs"><thead><tr className="text-muted text-left">
          <th>名称</th><th>父</th><th>原点</th><th>质心</th><th>补偿</th>
        </tr></thead><tbody>
          {r.links.map((l: any, i: number) => <tr key={l.name} className="border-t border-edge">
            <td className="py-1">{l.name}</td><td>{l.parent ?? 'world'}</td>
            <td>{[0, 1, 2].map((j) => <Num key={j} w={14} value={l.origin[j]} step={0.001}
              onChange={(v) => patchVec('links', i, 'origin', j, v)} />)}</td>
            <td>{l.com ? [0, 1, 2].map((j) => <Num key={j} w={12} value={l.com[j]} step={0.001}
              onChange={(v) => patchVec('links', i, 'com', j, v)} />) : '—'}</td>
            <td><input type="checkbox" checked={!!l.gravcomp}
              onChange={(e) => patchLink(i, 'gravcomp', e.target.checked)} /></td>
          </tr>)}
        </tbody></table>
        <div className="text-xs mt-2 flex gap-3 items-center">工具偏移
          {[0, 1, 2].map((j) => <Num key={j} value={r.tool_offset_m?.[j] ?? 0} step={0.001}
            onChange={(v) => { const d = { ...draft }; const t = [...(r.tool_offset_m || [0, 0, 0])]; t[j] = v; d.robots[ri].tool_offset_m = t; setDraft(d) }} />)}
        </div>
      </Section>}
      {tab === 'actuators' && <Section title="执行器（位置伺服 kp / kv / 力矩上限）">
        <table className="w-full text-xs"><thead><tr className="text-muted text-left">
          <th>关节</th><th>类型</th><th>kp</th><th>kv</th><th>力矩上限</th>
        </tr></thead><tbody>
          {r.actuators.map((a: any, i: number) => <tr key={a.joint} className="border-t border-edge">
            <td className="py-1">{a.joint}</td><td>{a.kind}</td>
            <td><Num value={a.kp} onChange={(v) => patchAct(i, 'kp', v)} step={1} /></td>
            <td><Num value={a.kv} onChange={(v) => patchAct(i, 'kv', v)} step={0.1} /></td>
            <td><Num value={a.force_limit} onChange={(v) => patchAct(i, 'force_limit', v)} step={0.5} /></td>
          </tr>)}
        </tbody></table>
      </Section>}
      {tab === 'gripper' && <Section title="夹爪">
        {r.gripper ? <div className="text-xs flex gap-4 items-center">
          <span>左关节 {r.gripper.left_joint}</span><span>右关节 {r.gripper.right_joint}</span>
          <label>张开 m <Num value={r.gripper.open_m} step={0.001} onChange={(v) => {
            const d = { ...draft }; d.robots[ri].gripper = { ...r.gripper, open_m: v }; setDraft(d) }} /></label>
          <label>闭合 m <Num value={r.gripper.closed_m} step={0.001} onChange={(v) => {
            const d = { ...draft }; d.robots[ri].gripper = { ...r.gripper, closed_m: v }; setDraft(d) }} /></label>
        </div> : <div className="text-xs text-muted">该机器人没有夹爪配置。</div>}
      </Section>}
      {tab === 'sensors' && <Section title="相机（内参像素主点 / 分辨率 / 帧率 / 延迟）">
        <table className="w-full text-xs"><thead><tr className="text-muted text-left">
          <th>id</th><th>父</th><th>fx</th><th>fy</th><th>cx</th><th>cy</th><th>宽</th><th>高</th><th>fps</th><th>延迟 s</th>
        </tr></thead><tbody>
          {(draft.sensors || []).map((s: any, i: number) => <tr key={s.id} className="border-t border-edge">
            <td className="py-1">{s.id}</td><td>{s.parent}</td>
            <td><Num w={12} step={1} value={s.intrinsics.fx} onChange={(v) => patchSensor(i, 'intrinsics.fx', v)} /></td>
            <td><Num w={12} step={1} value={s.intrinsics.fy} onChange={(v) => patchSensor(i, 'intrinsics.fy', v)} /></td>
            <td><Num w={10} step={1} value={s.intrinsics.cx} onChange={(v) => patchSensor(i, 'intrinsics.cx', v)} /></td>
            <td><Num w={10} step={1} value={s.intrinsics.cy} onChange={(v) => patchSensor(i, 'intrinsics.cy', v)} /></td>
            <td><Num w={10} step={1} value={s.resolution[0]} onChange={(v) => patchSensor(i, 'resolution', [v, s.resolution[1]])} /></td>
            <td><Num w={10} step={1} value={s.resolution[1]} onChange={(v) => patchSensor(i, 'resolution', [s.resolution[0], v])} /></td>
            <td><Num w={10} step={1} value={s.fps} onChange={(v) => patchSensor(i, 'fps', v)} /></td>
            <td><Num w={10} step={0.005} value={s.latency_s} onChange={(v) => patchSensor(i, 'latency_s', v)} /></td>
          </tr>)}
        </tbody></table>
        <div className="text-xs text-muted mt-2">主点为角点像素坐标；导出 MJCF 时自动转换为 MuJoCo 的中心相对偏移。</div>
      </Section>}
    </div>
  )
}

export function DatasetsPage() {
  const { data, reload } = useFetch(() => api.datasets(), [])
  const projects = useFetch(() => api.projects(), [])
  const [project, setProject] = useState('default')
  const [collection, setCollection] = useState('capture_' + Date.now())
  const [dataset, setDataset] = useState('scene_data_' + Date.now())
  const [msg, setMsg] = useState('')
  const start = async (kind: string) => {
    try {
      const params = kind === 'collect' ? {project_id: project, dataset_id: collection} : {collection_id: collection, dataset_id: dataset}
      const job = await api.createJob(kind, params); setMsg(`任务 ${job.id} 已创建`); reload()
    } catch (e: any) { setMsg(e.message) }
  }
  return (
    <div className="p-4">
      <h2 className="text-lg text-ink mb-2">数据集</h2>
      <div className="flex gap-2 text-xs mb-3">
        <label>项目 <select value={project} onChange={e => setProject(e.target.value)}>{projects.data?.projects.map(p => <option key={p.id}>{p.id}</option>)}</select></label>
        <label>采集 ID <input value={collection} onChange={e => setCollection(e.target.value)} /></label>
        <label>转换后数据集 ID <input value={dataset} onChange={e => setDataset(e.target.value)} /></label>
      </div>
      <div className="flex gap-2 mb-3">
        <Button onClick={() => start('collect')}>采集（伺服专家）</Button>
        <Button variant="outline" onClick={() => start('convert')}>转换 LeRobot</Button>
        <span className="text-xs text-muted">转换包含实际回读验证；使用新 ID，避免覆盖已有数据。</span>
      </div>
      {msg && <div className="text-xs mb-3">{msg}</div>}
      <div className="grid grid-cols-2 gap-3 mb-4">
        {(data?.datasets || []).map((d) => (
          <Link key={d.id} to={`/datasets/${d.id}`} className="border border-edge rounded p-3 bg-panel2 hover:border-accent">
            <div className="text-accent text-sm">{d.id} <span className="text-muted text-xs">{d.kind}</span></div>
            <div className="text-xs text-muted">{d.episodes} episodes · {d.frames} 帧 · 成功 {d.successes ?? '--'}</div>
          </Link>
        ))}
      </div>
      <Section title="采集/转换任务"><JobsTable /></Section>
    </div>
  )
}

export function DatasetDetail() {
  const { datasetId } = useParams()
  const { data } = useFetch(() => api.dataset(datasetId || 'staging'), [datasetId])
  return (
    <div className="p-4">
      <Link to="/datasets" className="text-xs text-accent">← 数据集</Link>
      <h2 className="text-lg text-ink my-2">{datasetId}</h2>
      <pre className="text-[11px] text-muted bg-panel2 border border-edge rounded p-3 overflow-auto max-h-[60vh]">
        {JSON.stringify(data, null, 1)}
      </pre>
    </div>
  )
}

export function TrainingPage() {
  const { reload } = useFetch(() => api.jobs(), [])
  const datasets = useFetch(() => api.datasets(), [])
  const [dataset, setDataset] = useState('')
  const [model, setModel] = useState('repaired_visual')
  const [msg, setMsg] = useState('')
  const available = (datasets.data?.datasets || []).filter(d => d.kind === 'lerobot')
  const start = (priv: boolean) => api.createJob('train', { priv, epochs: 60 }).then(reload)
  return (
    <div className="p-4">
      <h2 className="text-lg text-ink mb-2">训练中心</h2>
      <div className="flex gap-2 mb-3 text-xs">
        <label>视觉数据集 <select value={dataset || available[0]?.id || ''} onChange={e => setDataset(e.target.value)}>{available.map(d => <option key={d.id}>{d.id}</option>)}</select></label>
        <label>输出模型 ID <input value={model} onChange={e => setModel(e.target.value)} /></label>
      </div>
      <div className="flex gap-2 mb-3">
        <Button onClick={() => start(false)}>训练状态 BC</Button>
        <Button variant="outline" onClick={() => start(true)}>训练特权 BC</Button>
        <Button variant="outline" disabled={!available.length} onClick={async () => {
          try { await api.createJob('train_visual', {dataset_id: dataset || available[0].id, model_id: model}); reload(); setMsg('视觉训练任务已创建') }
          catch (e: any) { setMsg(e.message) }
        }}>训练视觉 BC</Button>
      </div>
      {msg && <div className="text-xs mb-3">{msg}</div>}
      <div className="text-xs text-muted">视觉训练需要新转换数据集中的 simulation.json；状态/特权 BC 仍使用原固定基线。</div>
      <div className="text-xs text-muted mb-3">训练在独立后台进程（WSL 学习环境的 PyTorch）运行，与 Web 请求生命周期解耦；产物写入 datasets/models 与 reports。</div>
      <JobsTable filter="train" />
    </div>
  )
}

export function JobDetail() {
  const { jobId } = useParams()
  const { data: job } = useFetch(() => api.job(jobId || ''), [jobId])
  const { data: logs, reload } = useFetch(() => api.jobLogs(jobId || ''), [jobId])
  useEffect(() => { const t = setInterval(reload, 2000); return () => clearInterval(t) }, [reload])
  return (
    <div className="p-4">
      <Link to="/training" className="text-xs text-accent">← 任务列表</Link>
      <h2 className="text-lg text-ink my-2">任务 {jobId} · {job?.kind} · <JobStatus s={job?.status ?? '…'} /></h2>
      <pre className="text-[11px] text-muted bg-panel2 border border-edge rounded p-3 overflow-auto max-h-[65vh]">
        {(logs?.lines || []).join('\n')}
      </pre>
    </div>
  )
}

export function EvaluationsPage() {
  const { reload } = useFetch(() => api.jobs(), [])
  const projects = useFetch(() => api.projects(), [])
  const [project, setProject] = useState('default')
  const [model, setModel] = useState('repaired_visual')
  const [msg, setMsg] = useState('')
  const start = (priv: boolean) => api.createJob('eval', { priv, group: 'all' }).then(reload)
  return (
    <div className="p-4">
      <h2 className="text-lg text-ink mb-2">评测中心</h2>
      <div className="flex gap-2 mb-3 text-xs">
        <label>视觉评估项目 <select value={project} onChange={e => setProject(e.target.value)}>{projects.data?.projects.map(p => <option key={p.id}>{p.id}</option>)}</select></label>
        <label>模型 ID <input value={model} onChange={e => setModel(e.target.value)} /></label>
      </div>
      <div className="flex gap-2 mb-3">
        <Button onClick={() => start(false)}>评测状态 BC</Button>
        <Button variant="outline" onClick={() => start(true)}>评测特权 BC</Button>
        <Button variant="outline" onClick={async () => {
          try { await api.createJob('eval_visual', {project_id: project, model_id: model}); reload(); setMsg('评估任务已创建；结果路径见作业参数') }
          catch (e: any) { setMsg(e.message) }
        }}>评测视觉 BC</Button>
      </div>
      {msg && <div className="text-xs mb-3">{msg}</div>}
      <div className="text-xs text-muted mb-3">独立 rollout，按训练内/留出摆位/留出资产分组报告；结果写入 datasets/reports/eval_policy.json / eval_priv.json。特权成绩标明特权观测。</div>
      <JobsTable filter="eval" />
    </div>
  )
}

function CompatChecker() {
  const [robot, setRobot] = useState('lux3d_5r')
  const [model, setModel] = useState('state')
  const [res, setRes] = useState<any>(null)
  return (
    <div className="border border-edge rounded p-3 bg-panel2 mt-3">
      <div className="text-xs text-accent mb-2">模型 / 机器人兼容性检查</div>
      <div className="flex gap-2 items-center text-xs">
        <select className="!w-auto" value={model} onChange={(e) => setModel(e.target.value)}>
          <option value="state">状态 BC</option><option value="priv">特权 BC</option><option value="visual">视觉 BC</option>
        </select>
        <select className="!w-auto" value={robot} onChange={(e) => setRobot(e.target.value)}>
          <option value="lux3d_5r">lux3d_5r</option><option value="cartesian_3dof">cartesian_3dof</option>
        </select>
        <Button size="sm" onClick={() => api.compat(robot, model).then(setRes)}>检查</Button>
      </div>
      {res && (
        <div className="text-xs mt-2">
          <span className={res.compatible ? 'text-accent' : 'text-red-400'}>
            {res.compatible ? '兼容，可运行' : '不兼容，阻止运行'}
          </span>
          <ul className="text-muted list-disc pl-4 mt-1">{res.mismatches?.map((m: string, i: number) => <li key={i}>{m}</li>)}</ul>
        </div>
      )}
    </div>
  )
}

export function DeploymentsPage() {
  const { data, reload } = useFetch(() => api.deployments(), [])
  const [sel, setSel] = useState<string | null>(null)
  const [detail, setDetail] = useState<any>(null)
  const [robot, setRobot] = useState('lux3d_5r')
  const [compat, setCompat] = useState<any>(null)
  const pick = async (id: string) => {
    setSel(id); setCompat(null)
    try { setDetail(await api.deploymentDetail(id)) } catch (e: any) { setDetail({ error: e.message }) }
  }
  const check = async () => { if (sel) setCompat(await api.deploymentCompat(sel, robot, 'hybrid_engine')) }
  return (
    <div className="p-4">
      <h2 className="text-lg text-ink mb-2">部署中心</h2>
      <div className="flex gap-2 mb-2">
        <Button onClick={() => api.createJob('export').then(reload)}>构建基线包</Button>
        <Button variant="outline" onClick={() => api.createJob('export_visual', { model: 'visual_hybrid', project: 'hybrid_engine' }).then(reload)}>构建视觉包</Button>
      </div>
      <div className="grid grid-cols-2 gap-3 mt-2">
        {(data?.deployments || []).map((d) => (
          <div key={d.id} onClick={() => pick(d.id)}
            className={`border rounded p-3 bg-panel2 cursor-pointer ${sel === d.id ? 'border-accent' : 'border-edge'}`}>
            <div className="text-accent text-sm">{d.id} <span className="text-[10px] text-muted">{d.kind}</span></div>
            <div className="text-xs text-muted">{d.robot ?? '—'} · {d.created?.slice(0, 19) ?? '—'}</div>
          </div>
        ))}
      </div>
      {sel && (
        <div className="border border-edge rounded p-3 bg-panel2 mt-3 text-xs">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-accent">包 {sel} 兼容检查</span>
            <select className="!w-auto" value={robot} onChange={(e) => setRobot(e.target.value)}>
              <option value="lux3d_5r">lux3d_5r</option><option value="cartesian_3dof">cartesian_3dof</option>
            </select>
            <Button size="sm" onClick={check}>检查</Button>
          </div>
          {compat && <div className={compat.compatible ? 'text-accent' : 'text-red-400'}>
            {compat.compatible === true ? '兼容' : compat.compatible === null ? '无契约（基线包）' : '不兼容'}
            {compat.mismatches?.length > 0 && <ul className="text-muted list-disc pl-4">{compat.mismatches.map((m: string, i: number) => <li key={i}>{m}</li>)}</ul>}
          </div>}
          <div className="text-muted mt-2">权重 {detail?.weights_sha256?.slice(0, 16) ?? '—'} · 架构 {detail?.architecture ?? detail?.kind ?? '—'}</div>
          {detail?.contract && <pre className="text-[10px] text-muted bg-panel border border-edge rounded p-2 mt-1 overflow-auto max-h-40">{JSON.stringify(detail.contract, null, 1)}</pre>}
          {detail?.reload_check && <div className="text-muted mt-1">重载一致性最大差 {detail.reload_check.max_abs_diff_two_loads}</div>}
          <div className="text-muted mt-1">载入验证：datasets/reports/deploy_check.json（基线）/ 包内 run_http.py（视觉）</div>
        </div>
      )}
      <Section title="构建任务"><JobsTable filter="export" /></Section>
    </div>
  )
}

export function SettingsPage() {
  const { data } = useFetch(() => api.generationStatus(), [])
  const { data: cap } = useFetch(() => api.capabilities(), [])
  return (
    <div className="p-4">
      <h2 className="text-lg text-ink mb-2">设置</h2>
      <Section title="运行环境">
        <div className="text-xs text-muted space-y-1">
          <div>物理后端 {cap?.physics_backend} · MuJoCo {cap?.mujoco_version} · Gymnasium {cap?.gymnasium_version}</div>
          <div>schema {cap?.schema_version}</div>
        </div>
      </Section>
      <Section title="生成 / 辅助模型服务">
        <div className="text-xs text-muted space-y-1">
          <div>辅助模型状态：{data?.aux?.state}</div>
          <div>base_url：{data?.aux?.base_url ?? '未配置'}（凭据仅存于后端环境，不发送到浏览器）</div>
          <div>{data?.note}</div>
        </div>
      </Section>
    </div>
  )
}
