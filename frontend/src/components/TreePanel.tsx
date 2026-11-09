import { useStore } from '../store'
import { cn } from './ui'
import type { Selection } from '../types'

function Item({ label, sel, depth = 0, tag }:
  { label: string; sel: Selection; depth?: number; tag?: string }) {
  const selection = useStore((s) => s.selection)
  const select = useStore((s) => s.select)
  const active = JSON.stringify(selection) === JSON.stringify(sel)
  return (
    <div onClick={() => select(sel)}
      className={cn('cursor-pointer px-2 py-1 rounded text-xs flex items-center gap-1 hover:bg-panel',
        active && 'bg-edge text-accent')}
      style={{ paddingLeft: 8 + depth * 14 }}>
      <span className="truncate">{label}</span>
      {tag && <span className="ml-auto text-[10px] text-muted">{tag}</span>}
    </div>
  )
}

export default function TreePanel() {
  const config = useStore((s) => s.config)
  if (!config) return <div className="p-3 text-xs text-muted">加载配置中…</div>
  const robot = config.robots[0]
  const jointByChild = Object.fromEntries(robot.joints.map((j) => [j.child, j]))
  return (
    <div className="p-2 overflow-y-auto h-full">
      <div className="text-[11px] uppercase tracking-wide text-accent px-2 mb-1">场景树</div>
      <Item label={`机器人 · ${robot.id}`} sel={{ kind: 'robot', id: robot.id }} />
      {robot.links.map((l) => {
        const j = jointByChild[l.name]
        return (
          <div key={l.name}>
            <Item label={l.name} sel={{ kind: 'link', id: l.name }} depth={1}
              tag={l.parent ?? 'root'} />
            {j && j.type !== 'fixed' && (
              <Item label={`关节 ${j.name}`} sel={{ kind: 'joint', id: j.name }} depth={2}
                tag={j.type === 'revolute' ? '旋转' : '移动'} />
            )}
          </div>
        )
      })}
      <div className="text-[11px] uppercase tracking-wide text-accent px-2 mt-3 mb-1">相机</div>
      {config.sensors.map((s) => (
        <Item key={s.id} label={`相机 ${s.id}`} sel={{ kind: 'sensor', id: s.id }}
          tag={s.parent === 'world' ? '世界' : s.parent} />
      ))}
      <div className="text-[11px] uppercase tracking-wide text-accent px-2 mt-3 mb-1">任务</div>
      <Item label={`抓放任务 · ${config.task.asset}`} sel={{ kind: 'task' }} />
    </div>
  )
}