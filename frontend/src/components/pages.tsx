import { useEffect, useState, useCallback } from 'react'
import { Link, useParams } from 'react-router-dom'
import { api } from '../api'
import { Button, Section } from './ui'

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
  const jobs = (data?.jobs || []).filter((j) => !filter || j.kind === filter)
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
  const { data } = useFetch(() => api.projects(), [])
  return (
    <div className="p-4">
      <h2 className="text-lg text-ink mb-3">项目中心</h2>
      <div className="grid grid-cols-2 gap-3">
        {(data?.projects || []).map((p) => (
          <div key={p.id} className="border border-edge rounded p-3 bg-panel2">
            <div className="text-accent">{p.id} <span className="text-muted text-xs">v{p.version}</span></div>
            <div className="text-xs text-muted my-1">机器人 {p.robots?.join(', ')} · 任务 {p.task}</div>
            <div className="space-x-2 text-xs">
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
  const gen = useFetch(() => api.generationStatus(), [])
  const [msg, setMsg] = useState('')
  return (
    <div className="p-4">
      <h2 className="text-lg text-ink mb-2">资产中心</h2>
      <div className="text-xs text-muted mb-3">生成服务：{gen.data?.aux?.state ?? '…'}（凭据由后端环境配置；无凭据时可本地导入与复用内置资产）</div>
      <div className="flex gap-2 mb-3">
        <Button onClick={() => setMsg('生成需后端配置 AHOLO_API_KEY；当前为等待配置状态。')}>生成资产</Button>
        <Button variant="outline" onClick={() => setMsg('本地导入复用 asset_pipeline/import_asset.py（GLB/ZIP），见 README。')}>导入资产</Button>
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

export function RobotEditPage() {
  const { robotId } = useParams()
  return (
    <div className="p-4">
      <h2 className="text-lg text-ink mb-2">机器人编辑 · {robotId}</h2>
      <div className="text-xs text-muted mb-3">连杆/关节/相机/执行器编辑在仿真工作台右侧属性面板进行（保存产生配置版本）。</div>
      <Link to={`/projects/default/sim`}><Button>打开编辑/验证</Button></Link>
    </div>
  )
}

export function DatasetsPage() {
  const { data, reload } = useFetch(() => api.datasets(), [])
  const start = (kind: string) => api.createJob(kind).then(reload)
  return (
    <div className="p-4">
      <h2 className="text-lg text-ink mb-2">数据集</h2>
      <div className="flex gap-2 mb-3">
        <Button onClick={() => start('collect')}>采集（伺服专家）</Button>
        <Button variant="outline" onClick={() => start('convert')}>转换 LeRobot</Button>
        <Button variant="outline" onClick={() => start('checks')}>质量检查</Button>
      </div>
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
  const start = (priv: boolean) => api.createJob('train', { priv, epochs: 60 }).then(reload)
  return (
    <div className="p-4">
      <h2 className="text-lg text-ink mb-2">训练中心</h2>
      <div className="flex gap-2 mb-3">
        <Button onClick={() => start(false)}>训练状态 BC</Button>
        <Button variant="outline" onClick={() => start(true)}>训练特权 BC</Button>
      </div>
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
  const start = (priv: boolean) => api.createJob('eval', { priv, group: 'all' }).then(reload)
  return (
    <div className="p-4">
      <h2 className="text-lg text-ink mb-2">评测中心</h2>
      <div className="flex gap-2 mb-3">
        <Button onClick={() => start(false)}>评测状态 BC</Button>
        <Button variant="outline" onClick={() => start(true)}>评测特权 BC</Button>
      </div>
      <div className="text-xs text-muted mb-3">独立 rollout，按训练内/留出摆位/留出资产分组报告；结果写入 datasets/reports/eval_policy.json / eval_priv.json。特权成绩标明特权观测。</div>
      <JobsTable filter="eval" />
    </div>
  )
}

export function DeploymentsPage() {
  const { data, reload } = useFetch(() => api.deployments(), [])
  return (
    <div className="p-4">
      <h2 className="text-lg text-ink mb-2">部署中心</h2>
      <Button onClick={() => api.createJob('export').then(reload)}>构建部署包</Button>
      <div className="grid grid-cols-2 gap-3 mt-3">
        {(data?.deployments || []).map((d) => (
          <div key={d.id} className="border border-edge rounded p-3 bg-panel2">
            <div className="text-accent text-sm">{d.id}</div>
            <div className="text-xs text-muted">{d.robot} · {d.created?.slice(0, 19)}</div>
            <div className="text-xs text-muted mt-1">独立进程载入验证见 datasets/reports/deploy_check.json</div>
          </div>
        ))}
      </div>
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