import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api'
import { useI18n } from '../i18n'
import type { ProjectListItem } from '../types'

const TASK_LABELS: Record<string, string> = {
  deliver: '传送带抓取并入箱',
  pick_return: '抓取后放回',
  bin_retrieve: '入箱后取出',
  bin_return: '取出后返程',
  cup_pick_place: '杯子抓取放置',
  duck_conveyor: '传送带小鸭分拣',
  tabletop_push: '桌面推物',
}
const PAGE_SIZE = 8
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

function projectName(project: ProjectListItem) {
  if (project.id === 'hackathon_demo') return '多物体 VLA 搬运'
  return project.id === 'default' ? '默认机械臂项目' : project.id
}

function projectListPriority(id: string) {
  if (id === 'default') return 0
  return /^(?:aholo_demo|eval|qa|test)_/.test(id) ? 2 : 1
}

function ProjectCard({ project }: { project: ProjectListItem }) {
  const { t } = useI18n()
  const name = t(projectName(project))
  const [loadedSummary, setLoadedSummary] = useState<ProjectListItem['summary']>()
  useEffect(() => {
    if (project.summary) return
    let active = true
    api.getConfig(project.id).then(({ task, sensors, scene }) => {
      if (!active) return
      setLoadedSummary({
        instruction: task.instruction,
        task_asset: task.asset,
        sensor_ids: sensors.map(sensor => sensor.id),
        sensor_renderer: scene.sensor_renderer ?? 'mujoco',
        environment_id: scene.environment?.asset_id ?? null,
        scene_asset_count: scene.asset_nodes?.length ?? 0,
      })
    }).catch(() => {
      if (active) setLoadedSummary(undefined)
    })
    return () => { active = false }
  }, [project.id, project.version, project.summary])

  const summary = project.summary ?? loadedSummary
  const instruction = summary?.instruction?.trim()
  const task = project.id === 'hackathon_demo' ? t('蓝杯和橙色方块同台，按指令每轮搬运一个。')
    : project.task ? t(TASK_LABELS[project.task] ?? project.task) : null
  const renderer = summary?.sensor_renderer === 'mujoco' ? 'MuJoCo'
    : summary?.sensor_renderer === 'hybrid' ? t('固定相机混合渲染') : summary?.sensor_renderer

  return (
    <li className="workspace-card">
      <header className="project-card__heading">
        <h2 title={project.id}>{name}</h2>
        {project.version != null && <span>v{project.version}</span>}
      </header>
      <dl className="project-card__details">
        {(task || summary?.task_asset || instruction) && <>
          <dt>{t('任务')}</dt>
          <dd>
            {task}
            {summary?.task_asset && <span className="project-card__asset" title={t('任务资产')}>{summary.task_asset}</span>}
            {instruction && <p className="project-card__instruction" title={instruction}>{instruction}</p>}
          </dd>
        </>}
        {project.robots.length > 0 && <>
          <dt>{t('机器人')}</dt>
          <dd className="project-card__robots">
            {project.robots.map(robot => (
              <Link key={robot} to={`/robots/${encodeURIComponent(robot)}/edit`}
                aria-label={`${t('编辑机器人')} ${robot}`}>{robot}</Link>
            ))}
          </dd>
        </>}
        {summary && <>
          <dt>{t('传感器')}</dt>
          <dd>{summary.sensor_ids.length ? summary.sensor_ids.join(' · ') : t('未配置')}</dd>
          <dt>{t('场景')}</dt>
          <dd>
            {renderer}
            {summary.scene_asset_count > 0 && <> · {t('场景资产')} {summary.scene_asset_count}</>}
            {summary.environment_id && <span className="project-card__environment">{t('环境')} {summary.environment_id}</span>}
          </dd>
        </>}
      </dl>
      <Link className="workspace-cta workspace-cta--quiet project-card__open"
        to={`/projects/${encodeURIComponent(project.id)}/sim`}
        aria-label={`${t('打开仿真')} · ${name}`}>{t('打开仿真')}</Link>
    </li>
  )
}

export default function ProjectsPage() {
  const projects = useWorkspaceData(api.projects)
  const { t } = useI18n()
  const [search, setSearch] = useState('')
  const [page, setPage] = useState(0)

  const query = search.trim().toLowerCase()
  const allProjects = projects.data?.projects ?? []
  const defaultProject = allProjects.find(project => project.id === 'default')
  const orderedProjects = [...allProjects].sort((a, b) => projectListPriority(a.id) - projectListPriority(b.id))
  const filtered = orderedProjects.filter(project => {
    const task = project.task ?? ''
    const { summary } = project
    return [projectName(project), t(projectName(project)), project.id, task, TASK_LABELS[task],
      summary?.instruction, summary?.task_asset, summary?.environment_id, ...project.robots, ...(summary?.sensor_ids ?? [])]
      .some(value => value?.toLowerCase().includes(query))
  })
  const lastPage = Math.max(0, Math.ceil(filtered.length / PAGE_SIZE) - 1)
  const currentPage = Math.min(page, lastPage)
  const visible = filtered.slice(currentPage * PAGE_SIZE, (currentPage + 1) * PAGE_SIZE)
  return (
    <section className="workspace-page" aria-labelledby="project-heading">
      <header className="workspace-page-heading">
        <h1 id="project-heading">{t('项目工作台')}</h1>
      </header>
      <div className="projects-toolbar">
        <div className="experiment-actions">
          {defaultProject && <Link className="workspace-cta workspace-cta--primary" to={`/projects/${encodeURIComponent(defaultProject.id)}/sim`}>
            {t('打开仿真')}
          </Link>}
          <Link className="workspace-cta workspace-cta--quiet" to="/collect">{t('开始采集')}</Link>
        </div>
        <input aria-label="搜索项目" placeholder="搜索项目、机器人或任务" className="projects-search" type="search" value={search}
          onChange={event => { setSearch(event.target.value); setPage(0) }} />
      </div>
      <div>
          {projects.failed ? <DataNotice retry={projects.retry}>项目暂时无法加载。</DataNotice>
            : !projects.data ? <DataNotice>正在读取项目…</DataNotice>
            : !visible.length ? <DataNotice>{query ? '没有匹配的项目。' : '暂无项目。'}</DataNotice>
            : <ul className="projects-list workspace-card-grid" aria-label="项目列表">
              {visible.map(project => <ProjectCard key={`${project.id}:${project.version}`} project={project} />)}
            </ul>}
          {lastPage > 0 && <div className="projects-pagination" aria-label="项目分页">
            <button type="button" onClick={() => setPage(currentPage - 1)} disabled={currentPage === 0}>上一页</button>
            <span aria-live="polite">{currentPage + 1} / {lastPage + 1}</span>
            <button type="button" onClick={() => setPage(currentPage + 1)} disabled={currentPage >= lastPage}>下一页</button>
          </div>}
      </div>
    </section>
  )
}
