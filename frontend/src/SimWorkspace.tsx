import { useEffect, useRef, useState } from 'react'
import { useStore } from './store'
import Toolbar from './components/Toolbar'
import TreePanel from './components/TreePanel'
import PropertiesPanel from './components/PropertiesPanel'
import Scene from './components/Scene'
import Dock from './components/Dock'
import { cn } from './components/ui'

const LS_KEY = 'vlab-layout'

function loadLayout() {
  try { return JSON.parse(localStorage.getItem(LS_KEY) || '{}') } catch { return {} }
}

function RobotSelect() {
  const config = useStore((s) => s.config)
  const active = useStore((s) => s.activeRobotId)
  const setRobot = useStore((s) => s.setRobot)
  if (!config) return null
  return (
    <select className="!w-auto text-[11px] ml-2" value={active || config.robots[0].id}
      onChange={(e) => setRobot(e.target.value)} title="活动机器人">
      {config.robots.map((r) => <option key={r.id} value={r.id}>{r.id}</option>)}
    </select>
  )
}

export default function App() {
  const init = useStore((s) => s.init)
  const error = useStore((s) => s.error)
  const busy = useStore((s) => s.busy)
  const saved = useRef(loadLayout())
  const [leftOpen, setLeftOpen] = useState(saved.current.leftOpen ?? window.innerWidth >= 1024)
  const [rightOpen, setRightOpen] = useState(saved.current.rightOpen ?? window.innerWidth >= 1280)
  const [dockOpen, setDockOpen] = useState(saved.current.dockOpen ?? true)
  const [leftW, setLeftW] = useState(saved.current.leftW ?? 200)
  const [rightW, setRightW] = useState(saved.current.rightW ?? 280)
  const [dockH, setDockH] = useState(saved.current.dockH ?? 148)
  const drag = useRef<null | { kind: 'right' | 'dock' | 'left'; start: number; base: number }>(null)

  useEffect(() => { init() }, [init])
  useEffect(() => {
    localStorage.setItem(LS_KEY, JSON.stringify({ leftOpen, rightOpen, dockOpen, leftW, rightW, dockH }))
  }, [leftOpen, rightOpen, dockOpen, leftW, rightW, dockH])
  useEffect(() => {
    const onResize = () => {
      if (window.innerWidth < 1280) setRightOpen(false)
      if (window.innerWidth < 1024) setLeftOpen(false)
    }
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [])
  useEffect(() => {
    const move = (e: MouseEvent) => {
      const d = drag.current
      if (!d) return
      if (d.kind === 'right') setRightW(Math.min(420, Math.max(220, d.base - (e.clientX - d.start))))
      if (d.kind === 'left') setLeftW(Math.min(320, Math.max(150, d.base + (e.clientX - d.start))))
      if (d.kind === 'dock') setDockH(Math.min(320, Math.max(110, d.base - (e.clientY - d.start))))
    }
    const up = () => { drag.current = null }
    window.addEventListener('mousemove', move)
    window.addEventListener('mouseup', up)
    return () => { window.removeEventListener('mousemove', move); window.removeEventListener('mouseup', up) }
  }, [])

  return (
    <div className="h-full flex flex-col overflow-hidden">
      <Toolbar />
      {error && <div className="bg-red-900/50 text-red-200 text-xs px-3 py-1.5 border-b border-red-800">{error}</div>}
      <div className="flex items-center gap-1 px-2 py-1 bg-panel2 border-b border-edge text-xs shrink-0">
        <button className={cn('px-2 py-0.5 rounded', leftOpen ? 'text-accent bg-panel' : 'text-muted')} onClick={() => setLeftOpen(!leftOpen)}>场景树</button>
        <button className={cn('px-2 py-0.5 rounded', rightOpen ? 'text-accent bg-panel' : 'text-muted')} onClick={() => setRightOpen(!rightOpen)}>属性</button>
        <button className={cn('px-2 py-0.5 rounded', dockOpen ? 'text-accent bg-panel' : 'text-muted')} onClick={() => setDockOpen(!dockOpen)}>停靠区</button>
        <RobotSelect />
        <span className="ml-auto text-[10px] text-muted">三维主视图 · 拖拽边缘调整面板</span>
      </div>
      <div className="flex-1 flex min-h-0 min-w-0">
        {leftOpen && (
          <>
            <aside style={{ width: leftW }} className="shrink-0 border-r border-edge bg-panel2 overflow-hidden">
              <TreePanel />
            </aside>
            <div className="w-1 shrink-0 cursor-col-resize hover:bg-accent/40"
              onMouseDown={(e) => { drag.current = { kind: 'left', start: e.clientX, base: leftW } }} />
          </>
        )}
        <main className="flex-1 flex flex-col min-w-0 min-h-0">
          <div className="flex-1 relative min-h-0">
            <Scene />
            {busy && <div className="absolute inset-0 flex items-center justify-center bg-black/40 text-accent text-sm pointer-events-none">处理中…</div>}
          </div>
          {dockOpen && (
            <>
              <div className="h-1 shrink-0 cursor-row-resize hover:bg-accent/40"
                onMouseDown={(e) => { drag.current = { kind: 'dock', start: e.clientY, base: dockH } }} />
              <div style={{ height: dockH }} className="shrink-0">
                <Dock onCollapse={() => setDockOpen(false)} />
              </div>
            </>
          )}
        </main>
        {rightOpen && (
          <>
            <div className="w-1 shrink-0 cursor-col-resize hover:bg-accent/40"
              onMouseDown={(e) => { drag.current = { kind: 'right', start: e.clientX, base: rightW } }} />
            <aside style={{ width: rightW }} className="shrink-0 border-l border-edge bg-panel2 overflow-hidden">
              <PropertiesPanel />
            </aside>
          </>
        )}
      </div>
    </div>
  )
}