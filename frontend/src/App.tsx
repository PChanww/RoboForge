import { useEffect } from 'react'
import { useStore } from './store'
import Toolbar from './components/Toolbar'
import TreePanel from './components/TreePanel'
import PropertiesPanel from './components/PropertiesPanel'
import Scene from './components/Scene'
import CameraViews from './components/CameraViews'
import BottomPanel from './components/BottomPanel'

export default function App() {
  const init = useStore((s) => s.init)
  const error = useStore((s) => s.error)
  const busy = useStore((s) => s.busy)
  useEffect(() => { init() }, [init])
  return (
    <div className="h-full flex flex-col">
      <Toolbar />
      {error && <div className="bg-red-900/50 text-red-200 text-xs px-3 py-1.5 border-b border-red-800">{error}</div>}
      <div className="flex-1 flex min-h-0">
        <div className="w-56 border-r border-edge bg-panel2 overflow-hidden">
          <TreePanel />
        </div>
        <div className="flex-1 flex flex-col min-w-0">
          <div className="flex-1 relative min-h-0">
            <Scene />
            {busy && <div className="absolute inset-0 flex items-center justify-center bg-black/40 text-accent text-sm">处理中…</div>}
          </div>
          <CameraViews />
        </div>
        <div className="w-72 border-l border-edge bg-panel2 overflow-hidden">
          <PropertiesPanel />
        </div>
      </div>
      <div className="h-44 border-t border-edge">
        <BottomPanel />
      </div>
    </div>
  )
}