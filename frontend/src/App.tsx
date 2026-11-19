import { useEffect, useState } from 'react'
import { Link, NavLink, Route, Routes, Navigate, useLocation, useParams, useSearchParams } from 'react-router-dom'
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
import { cn } from './components/ui'
import './workspace.css'

// Labels stay as text keys: the English dictionary maps them, and an untranslated
// entry falls back to the Chinese source instead of showing a bare key.
const navigation = [
  { label: '实验', links: [
    { to: '/projects', name: '项目', icon: '▦' },
    { to: '/transport?view=task', name: '搬运仿真', icon: '◇' },
    { to: '/datasets', name: '数据集', icon: '▤' },
    { to: '/training', name: '训练', icon: '⟁' },
    { to: '/evaluations', name: '评测', icon: '◎' },
  ] },
  { label: '资源', links: [
    { to: '/generation', name: '生成', icon: '✧' },
    { to: '/library', name: '资产', icon: '▣' },
    { to: '/deployments', name: '部署', icon: '⇧' },
    { to: '/settings', name: '设置', icon: '⚙' },
  ] },
]

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
  const location = useLocation()
  const { t } = useI18n()
  const [demoPath, setDemoPath] = useState(() => {
    try { return sessionStorage.getItem('roboforge.last-demo') === '/collect' ? '/collect' : '/' }
    catch { return '/' }
  })
  useEffect(() => {
    const path = location.pathname
    if (path !== '/' && path !== '/collect') return
    setDemoPath(path)
    try { sessionStorage.setItem('roboforge.last-demo', path) } catch { /* Navigation still works without storage. */ }
  }, [location.pathname])

  if (location.pathname === '/transport') return <TransportDemo />
  if (location.pathname === '/drive') return <MobileDemo />
  if (location.pathname === '/create') return <ScenarioWorkbench />
  if (location.pathname === '/studio') return <TaskStudio />
  if (location.pathname === '/') return <CollectionDemo key="ducks" duckSorting />
  if (location.pathname === '/collect') return <CollectionDemo key="cups" />
  const projectSimulation = /^\/projects\/[^/]+\/sim$/.test(location.pathname)
  const Content = projectSimulation ? 'div' : 'main'
  const current = navigation.flatMap(group => group.links)
    .find(link => location.pathname === link.to || location.pathname.startsWith(`${link.to}/`))
  const currentLabel = current?.name ?? (location.pathname.startsWith('/robots/') || location.pathname.startsWith('/scenes/') ? '资产编辑' : null)
  return (
    <div className={cn('app-shell', projectSimulation && 'app-shell--compact')}>
      <a className="app-shell__skip" href="#app-content">{t('跳至内容')}</a>
      <aside className="app-shell__rail">
        <Link to="/projects" className="app-shell__brand" aria-label={t('RoboForge 项目工作台')}>
          <span className="app-shell__mark" aria-hidden="true">R</span>
          <span className="app-shell__wordmark">RoboForge</span>
        </Link>
        <nav className="app-shell__navigation" aria-label={t('工作台导航')}>
          {navigation.map(group => <div className="app-shell__group" key={group.label}>
            <span className="app-shell__category">{t(group.label)}</span>
            {group.links.map(({ to, name, icon }) => <NavLink key={to} to={to} className="app-shell__link" title={t(name)} aria-label={t(name)}>
              <span className="app-shell__icon" aria-hidden="true">{icon}</span>
              <span className="app-shell__label">{t(name)}</span>
            </NavLink>)}
          </div>)}
        </nav>
        <LanguageSwitch className="lang-switch lang-switch--rail" />
      </aside>
      <div className="app-shell__body">
        <header className="app-shell__bar">
          <Link to="/projects" className={projectSimulation ? 'app-shell__back' : undefined}>{projectSimulation ? t('← 返回项目') : t('工作台')}</Link>
          {currentLabel && <><span aria-hidden="true">/</span><span>{projectSimulation ? t('仿真') : t(currentLabel)}</span></>}
          <Link to={demoPath} className="app-shell__back app-shell__back--demo">{t('← 返回简版')}</Link>
        </header>
        <Content id="app-content" tabIndex={-1} className="app-shell__content">
        <Routes>
          <Route path="/projects" element={<ProjectsPage />} />
          <Route path="/projects/:projectId/sim" element={<SimWorkspace />} />
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
