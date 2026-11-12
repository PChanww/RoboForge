import { NavLink, Route, Routes, Navigate } from 'react-router-dom'
import SimWorkspace from './SimWorkspace'
import {
  ProjectsPage, LibraryPage, LibraryDetail, RobotEditPage, DatasetsPage, DatasetDetail,
  TrainingPage, JobDetail, EvaluationsPage, DeploymentsPage, SettingsPage, SceneEditPage,
} from './components/pages'
import { GenerationList, GenerationWorkbench, GenerationTaskDetail } from './components/Generation'
import { cn } from './components/ui'

const nav = [
  ['/projects', '项目'], ['/generation', '生成'], ['/library', '资产'], ['/datasets', '数据集'],
  ['/training', '训练'], ['/evaluations', '评测'], ['/deployments', '部署'], ['/settings', '设置'],
]

export default function App() {
  return (
    <div className="h-full flex flex-col overflow-hidden">
      <nav className="flex items-center gap-1 px-3 h-10 shrink-0 bg-panel2 border-b border-edge">
        <span className="text-accent font-semibold text-sm mr-3">RoboForge</span>
        {nav.map(([to, label]) => (
          <NavLink key={to} to={to}
            className={({ isActive }) => cn('px-2 py-1 rounded text-xs', isActive ? 'bg-panel text-accent' : 'text-muted hover:text-ink')}>
            {label}
          </NavLink>
        ))}
      </nav>
      <div className="flex-1 min-h-0 overflow-auto">
        <Routes>
          <Route path="/" element={<Navigate to="/projects/default/sim" replace />} />
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
          <Route path="/evaluations" element={<EvaluationsPage />} />
          <Route path="/evaluations/:runId" element={<EvaluationsPage />} />
          <Route path="/deployments" element={<DeploymentsPage />} />
          <Route path="/deployments/:packageId" element={<DeploymentsPage />} />
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="*" element={<Navigate to="/projects" replace />} />
        </Routes>
      </div>
    </div>
  )
}