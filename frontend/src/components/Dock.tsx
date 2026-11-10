import { useState } from 'react'
import { useStore } from '../store'
import { cn } from './ui'
import CameraStrip from './CameraStrip'
import CurveChart from './CurveChart'

export default function Dock({ onCollapse }: { onCollapse?: () => void }) {
  const [tab, setTab] = useState<'cam' | 'log' | 'curve'>('cam')
  const logs = useStore((s) => s.logs)
  return (
    <div className="h-full flex flex-col bg-panel2 border-t border-edge">
      <div className="flex items-center gap-1 px-2 pt-1 border-b border-edge shrink-0">
        {([['cam', '相机'], ['log', '日志'], ['curve', '曲线']] as const).map(([k, label]) => (
          <button key={k} onClick={() => setTab(k)}
            className={cn('px-3 py-1 text-xs rounded-t', tab === k ? 'bg-panel text-accent' : 'text-muted hover:text-ink')}>
            {label}
          </button>
        ))}
        <div className="ml-auto flex items-center gap-2">
          <span className="text-[10px] text-muted">停靠区</span>
          {onCollapse && <button className="text-muted hover:text-ink text-xs px-1" onClick={onCollapse} title="收起">▾</button>}
        </div>
      </div>
      <div className="flex-1 min-h-0">
        {tab === 'cam' && <CameraStrip />}
        {tab === 'log' && (
          <div className="h-full overflow-y-auto p-2 font-mono text-[11px] space-y-0.5">
            {logs.map((l, i) => (
              <div key={i} className={l.level === 'error' ? 'text-red-400' : 'text-muted'}>
                <span className="text-edge">{l.t}</span> {l.msg}
              </div>
            ))}
            {!logs.length && <div className="text-muted">暂无日志</div>}
          </div>
        )}
        {tab === 'curve' && <CurveChart />}
      </div>
    </div>
  )
}