import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api'
import transportImage from '../transport/transport-workcell.jpg'

type Project = { id: string; version: number | null; robots: string[]; task: string | null }
type TransportRun = {
  id: string
  experiment_name?: string
  status: string
  created: string | null
  task_kind: string
  policy_identity?: { label?: string; learned_policy?: boolean }
}

const TASK_LABELS: Record<string, string> = {
  deliver: '传送带抓取并入箱',
  pick_return: '抓取后放回',
  bin_retrieve: '入箱后取出',
  bin_return: '取出后返程',
  cup_pick_place: '杯子抓取放置',
  duck_conveyor: '传送带小鸭分拣',
  tabletop_push: '桌面推物',
}
const PAGE_SIZE = 5
const projectDate = new Intl.DateTimeFormat('zh-CN', { timeZone: 'Asia/Shanghai', year: 'numeric', month: '2-digit', day: '2-digit' })

async function transportHistory(): Promise<{ runs: TransportRun[] }> {
  const response = await fetch('/api/v1/transport/history')
  if (!response.ok) throw new Error('无法读取实验记录')
  return response.json()
}

function useWorkspaceData<T>(load: () => Promise<T>) {
  const [data, setData] = useState<T | null>(null)
  const [failed, setFailed] = useState(false)
  const [revision, setRevision] = useState(0)
  useEffect(() => {
    let active = true
    load().then(value => {
      if (active) setData(value)
    }).catch(() => {
      if (active) setFailed(true)
    })
    return () => { active = false }
  }, [load, revision])
  return { data, failed, retry: () => { setFailed(false); setRevision(value => value + 1) } }
}

function DataNotice({ children, retry }: { children: string; retry?: () => void }) {
  return <div className="workspace-notice" role="status">{children}{retry && <button type="button" onClick={retry}>重试</button>}</div>
}

function projectName(project: Project) {
  return project.id === 'default' ? '默认机械臂项目' : project.id
}

export default function ProjectsPage() {
  const projects = useWorkspaceData<{ projects: Project[] }>(api.projects)
  const training = useWorkspaceData(api.transportTraining)
  const history = useWorkspaceData(transportHistory)
  const [search, setSearch] = useState('')
  const [page, setPage] = useState(0)

  const query = search.trim().toLowerCase()
  const allProjects = projects.data?.projects ?? []
  const defaultProject = allProjects.find(project => project.id === 'default')
  const orderedProjects = defaultProject ? [defaultProject, ...allProjects.filter(project => project !== defaultProject)] : allProjects
  const filtered = orderedProjects.filter(project => {
    const task = project.task ?? ''
    return [projectName(project), project.id, task, TASK_LABELS[task] ?? '', ...project.robots]
      .some(value => value.toLowerCase().includes(query))
  })
  const lastPage = Math.max(0, Math.ceil(filtered.length / PAGE_SIZE) - 1)
  const currentPage = Math.min(page, lastPage)
  const visible = filtered.slice(currentPage * PAGE_SIZE, (currentPage + 1) * PAGE_SIZE)
  const evaluated = (history.data?.runs ?? []).filter(run => run.policy_identity?.learned_policy
    && (run.status === 'succeeded' || run.status === 'failed'))

  const workflow = [
    { icon: '◇', title: '搬运仿真', detail: '设置任务、物体与执行策略', to: '/transport?view=task' },
    { icon: '▤', title: '实验数据', detail: training.data ? `${training.data.datasets.length} 个可训练示范记录` : '查看与导出采集记录', to: '/transport?view=datasets' },
    { icon: '⟁', title: 'GPU 训练', detail: training.data ? `${training.data.checkpoints.length} 个移动检查点` : '从专家示范训练移动策略', to: '/training' },
    { icon: '◎', title: '闭环评测', detail: '查看物理结果与失败阶段', to: '/evaluations' },
  ]

  return (
    <div className="workspace-page projects-workspace">
      <header className="workspace-page-heading"><h1>项目工作台</h1></header>

      <section className="projects-hero" aria-labelledby="transport-heading">
        <div className="projects-hero__copy">
          <h2 id="transport-heading">移动搬运仿真</h2>
          <p>配置传送带抓取任务，用专家或学习策略执行，保留双相机和动作数据。</p>
          <div className="projects-hero__actions">
            <Link className="workspace-cta workspace-cta--primary" to="/transport?view=task">进入搬运仿真 <span aria-hidden="true">→</span></Link>
            <Link className="workspace-cta workspace-cta--quiet" to="/evaluations">查看评测</Link>
          </div>
        </div>
        <figure className="projects-hero__image">
          <img src={transportImage} alt="MuJoCo 前视相机中的移动机械臂、传送带和收纳箱" fetchPriority="high" />
          <figcaption>MuJoCo · 前视相机</figcaption>
        </figure>
      </section>

      <section className="projects-flow" aria-labelledby="flow-heading">
        <h2 id="flow-heading">实验流程</h2>
        <div className="projects-flow__grid">
          {workflow.map(step => <Link to={step.to} className="projects-flow__item" key={step.title}>
            <span className="projects-flow__icon" aria-hidden="true">{step.icon}</span>
            <span className="projects-flow__copy"><strong>{step.title}</strong><span>{step.detail}</span></span>
            <span className="projects-flow__arrow" aria-hidden="true">→</span>
          </Link>)}
        </div>
      </section>

      <div className="projects-overview">
        <section className="projects-panel" aria-labelledby="local-projects-heading">
          <div className="projects-panel__head">
            <div><h2 id="local-projects-heading">已有项目</h2>{projects.data && <p>{query ? `${filtered.length} 个匹配项目` : `${allProjects.length} 个项目`}</p>}</div>
            <input aria-label="搜索项目" placeholder="搜索项目、机器人或任务" className="projects-search" type="search" value={search}
              onChange={event => { setSearch(event.target.value); setPage(0) }} />
          </div>
          {projects.failed ? <DataNotice retry={projects.retry}>项目暂时无法加载。</DataNotice>
            : !projects.data ? <DataNotice>正在读取项目…</DataNotice>
            : !visible.length ? <DataNotice>{query ? '没有匹配的项目。' : '暂无项目。'}</DataNotice>
            : <ul className="projects-list">
              {visible.map(project => <li key={project.id}>
                <div className="projects-list__description">
                  <strong title={project.id}>{projectName(project)}</strong>
                  <span className="projects-list__meta">
                    {project.task && <span>{TASK_LABELS[project.task] ?? project.task}</span>}
                    {project.version != null && <span>v{project.version}</span>}
                    {project.robots.map(robot => <Link key={robot} to={`/robots/${encodeURIComponent(robot)}/edit`} aria-label={`编辑机器人 ${robot}`}>{robot}</Link>)}
                  </span>
                </div>
                <Link className="workspace-cta workspace-cta--quiet" to={`/projects/${encodeURIComponent(project.id)}/sim`} aria-label={`打开 ${projectName(project)} 的仿真`}>打开仿真 <span aria-hidden="true">→</span></Link>
              </li>)}
            </ul>}
          {lastPage > 0 && <div className="projects-pagination" aria-label="项目分页">
            <button type="button" onClick={() => setPage(currentPage - 1)} disabled={currentPage === 0}>上一页</button>
            <span aria-live="polite">{currentPage + 1} / {lastPage + 1}</span>
            <button type="button" onClick={() => setPage(currentPage + 1)} disabled={currentPage >= lastPage}>下一页</button>
          </div>}
        </section>

        <section className="projects-panel" aria-labelledby="recent-evaluations-heading">
          <div className="projects-panel__head">
            <h2 id="recent-evaluations-heading">最近策略评测</h2>
            <Link className="projects-text-link" to="/evaluations">查看全部 <span aria-hidden="true">→</span></Link>
          </div>
          {history.failed ? <DataNotice retry={history.retry}>评测记录暂时无法加载。</DataNotice>
            : !history.data ? <DataNotice>正在读取评测记录…</DataNotice>
            : evaluated.length === 0 ? <DataNotice>暂无已结束的策略评测。</DataNotice>
            : <ul className="projects-run-list">
              {evaluated.slice(0, 4).map(run => <li key={run.id}>
                <Link to={`/evaluations?run=${encodeURIComponent(run.id)}`}>
                  <div className="projects-run-list__title">
                    <strong>{run.experiment_name || run.policy_identity?.label || '学习策略'}</strong>
                    <span className={`projects-result ${run.status === 'succeeded' ? 'is-passed' : 'is-failed'}`}>{run.status === 'succeeded' ? '通过' : '未通过'}</span>
                  </div>
                  <p>{TASK_LABELS[run.task_kind] ?? run.task_kind}
                    {run.created && Number.isFinite(Date.parse(run.created)) && <><span aria-hidden="true"> · </span><time dateTime={run.created}>{projectDate.format(new Date(run.created))}</time></>}
                  </p>
                </Link>
              </li>)}
            </ul>}
        </section>
      </div>
    </div>
  )
}
