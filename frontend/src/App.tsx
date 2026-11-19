import { useLayoutEffect, useRef } from 'react'
import { Link, Route, Routes, Navigate, useLocation, useParams, useSearchParams } from 'react-router-dom'
import CollectionDemo from './demo/CollectionDemo'
import TaskStudio from './studio/TaskStudio'
import ScenarioWorkbench from './scenario/ScenarioWorkbench'
import MobileDemo from './mobile/MobileDemo'
import TransportDemo from './transport/TransportDemo'
import TransportEvaluations from './transport/TransportEvaluations'
import ProjectsPage from './components/ProjectsPage'
import SimWorkspace from './SimWorkspace'
import { LanguageSwitch, useI18n } from './i18n'
import {
  LibraryPage, LibraryDetail, RobotEditPage, DatasetsPage, DatasetDetail,
  TrainingPage, JobDetail, EvaluationsPage, DeploymentsPage, SettingsPage, SceneEditPage,
} from './components/pages'
import { GenerationList, GenerationWorkbench, GenerationTaskDetail } from './components/Generation'
import './workspace.css'

// Labels stay as text keys: the English dictionary maps them, and an untranslated
// entry falls back to the Chinese source instead of showing a bare key.
const navigation = [
  { label: '工作台', links: [
    { to: '/projects', name: '项目', icon: '▦' },
    { to: '/collect', name: '数据采集', icon: '◉' },
    { to: '/datasets', name: '数据集', icon: '▤' },
  ] },
  { label: '资产与场景', links: [
    { to: '/generation', name: '资产生成', icon: '✧' },
    { to: '/library', name: '资产库', icon: '▣' },
  ] },
  { label: '实验工具', links: [
    { to: '/experiments', name: '实验示例', icon: '◇' },
    { to: '/training?tab=fixed', name: '训练', icon: '⟁' },
    { to: '/evaluations?tab=fixed', name: '评测', icon: '◎' },
    { to: '/deployments', name: '部署', icon: '⇧' },
    { to: '/settings', name: '设置', icon: '⚙' },
  ] },
]

const experiments = [
  { path: '/transport', name: '移动搬运', detail: '配置传送带抓取、底盘搬运与入箱任务。', icon: '⇄', page: <TransportDemo /> },
  { path: '/duck-sorting', name: '传送带分拣', detail: '跟随移动目标，采集抓取和放置示范。', icon: '◇', page: <CollectionDemo key="ducks" duckSorting /> },
  { path: '/drive', name: '底盘驾驶', detail: '手动驾驶四轮底盘并记录轨迹。', icon: '⌁', page: <MobileDemo /> },
  { path: '/create', name: '整场景生成', detail: '描述场景、机器人外观和任务目标。', icon: '✧', page: <ScenarioWorkbench /> },
  { path: '/studio', name: '任务编排', detail: '从桌面操作需求生成任务方案并回放。', icon: '▦', page: <TaskStudio /> },
]

function ExperimentIndex() {
  const { t } = useI18n()
  return (
    <section className="workspace-page">
      <header className="workspace-page-heading">
        <h1>{t('实验示例')}</h1>
        <p>{t('示例需要对应的仿真服务。')}</p>
      </header>
      <div className="projects-flow__grid experiment-directory">
        {experiments.map(({ path, name, detail, icon }) => (
          <Link to={path} className="projects-flow__item" key={path}>
            <span className="projects-flow__icon" aria-hidden="true">{icon}</span>
            <span className="projects-flow__copy"><strong>{t(name)}</strong><span>{t(detail)}</span></span>
            <span className="projects-flow__arrow" aria-hidden="true">→</span>
          </Link>
        ))}
      </div>
    </section>
  )
}

function EvaluationWorkspace() {
  const { t } = useI18n()
  const [params] = useSearchParams()
  const { runId } = useParams()
  if (runId?.startsWith('transport_')) return <Navigate to={`/evaluations?run=${encodeURIComponent(runId)}`} replace />
  const fixed = params.get('tab') === 'fixed' || Boolean(runId)
  return (
    <div className="workspace-page">
      <header className="workspace-page-heading"><h1>{t('评测')}</h1></header>
      <nav className="workspace-tabs" aria-label={t('评测类型')}>
        <Link to="/evaluations" aria-current={!fixed ? 'page' : undefined}>{t('移动搬运')}</Link>
        <Link to="/evaluations?tab=fixed" aria-current={fixed ? 'page' : undefined}>{t('固定机械臂')}</Link>
      </nav>
      {fixed ? <EvaluationsPage /> : <TransportEvaluations />}
    </div>
  )
}

export default function App() {
  const { pathname, search } = useLocation()
  const { t } = useI18n()
  const contentRef = useRef<HTMLDivElement>(null)
  useLayoutEffect(() => { contentRef.current?.scrollTo(0, 0) }, [pathname, search])

  const simulation = pathname.match(/^\/projects\/([^/]+)\/sim\/?$/)
  const projectSimulation = Boolean(simulation)
  const Content = projectSimulation ? 'div' : 'main'
  const experiment = experiments.find(({ path }) => pathname.replace(/\/$/, '') === path)
  const assetEditor = pathname.startsWith('/robots/') || pathname.startsWith('/scenes/')
  const sectionPath = experiment ? '/experiments' : assetEditor ? '/library' : pathname
  const current = navigation.flatMap(group => group.links)
    .find(link => sectionPath === link.to.split('?')[0] || sectionPath.startsWith(`${link.to.split('?')[0]}/`))
  const currentLabel = experiment?.name ?? (assetEditor ? '资产编辑' : current?.name)
  return (
    <div className="app-shell">
      <a className="app-shell__skip" href="#app-content">{t('跳至内容')}</a>
      <aside className="app-shell__rail">
        <Link to="/projects" className="app-shell__brand" aria-label={t('RoboForge 项目工作台')}>
          <span className="app-shell__mark" aria-hidden="true">R</span>
          <span className="app-shell__wordmark">RoboForge</span>
        </Link>
        <nav className="app-shell__navigation" aria-label={t('工作台导航')}>
          {navigation.map(group => <div className="app-shell__group" key={group.label}>
            <span className="app-shell__category">{t(group.label)}</span>
            {group.links.map(({ to, name, icon }) => <Link key={to} to={to} className="app-shell__link" title={t(name)} aria-label={t(name)} aria-current={current?.to === to ? 'page' : undefined}>
              <span className="app-shell__icon" aria-hidden="true">{icon}</span>
              <span className="app-shell__label">{t(name)}</span>
            </Link>)}
          </div>)}
        </nav>
        <LanguageSwitch className="lang-switch lang-switch--rail" />
      </aside>
      <div className="app-shell__body">
        <header className="app-shell__bar">
          <Link to={experiment ? '/experiments' : '/projects'} className={projectSimulation ? 'app-shell__back' : undefined}>
            {t(projectSimulation ? '← 返回项目' : experiment ? '实验示例' : '工作台')}
          </Link>
          {currentLabel && <><span aria-hidden="true">/</span><span>{t(projectSimulation ? '仿真' : currentLabel)}</span></>}
        </header>
        <Content ref={contentRef} id="app-content" tabIndex={-1} className="app-shell__content">
        <Routes>
          <Route path="/" element={<Navigate to="/projects" replace />} />
          <Route path="/projects" element={<ProjectsPage />} />
          <Route path="/projects/:projectId/sim" element={<SimWorkspace />} />
          <Route path="/collect" element={<CollectionDemo key="cups" />} />
          <Route path="/experiments" element={<ExperimentIndex />} />
          {experiments.map(({ path, page }) => <Route key={path} path={path} element={page} />)}
          <Route path="/generation" element={<GenerationList />} />
          <Route path="/generation/:sessionId" element={<GenerationWorkbench />} />
          <Route path="/generation/:sessionId/tasks/:jobId" element={<GenerationTaskDetail />} />
          <Route path="/library" element={<LibraryPage />} />
          <Route path="/library/:assetId" element={<LibraryDetail />} />
          <Route path="/scenes/:sceneId/edit" element={<SceneEditPage />} />
          <Route path="/robots/:robotId/edit" element={<RobotEditPage />} />
          <Route path="/datasets" element={<DatasetsPage />} />
          <Route path="/datasets/:datasetId" element={<DatasetDetail />} />
          <Route path="/training" element={<TrainingPage />} />
          <Route path="/training/:jobId" element={<JobDetail />} />
          <Route path="/evaluations" element={<EvaluationWorkspace />} />
          <Route path="/evaluations/:runId" element={<EvaluationWorkspace />} />
          <Route path="/deployments" element={<DeploymentsPage />} />
          <Route path="/deployments/:packageId" element={<DeploymentsPage />} />
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="*" element={<Navigate to="/projects" replace />} />
        </Routes>
        </Content>
      </div>
    </div>
  )
}
