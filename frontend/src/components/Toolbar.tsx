import { useState } from 'react'
import { useStore } from '../store'
import { Button } from './ui'

export default function Toolbar() {
  const { config, dirty, running, busy, save, reset, step, toggleRun, observation, metrics } = useStore()
  const [seed, setSeed] = useState(0)
  return (
    <div className="flex items-center gap-2 px-3 py-2 bg-panel2 border-b border-edge flex-wrap">
      <span className="text-sm font-semibold text-accent mr-2">VLA Lab</span>
      <Button onClick={save} disabled={!dirty || busy}>保存并应用</Button>
      <Button variant="outline" onClick={() => reset(seed)} disabled={busy}>重置</Button>
      <input type="number" className="!w-16" value={seed} onChange={(e) => setSeed(parseInt(e.target.value) || 0)} />
      <Button variant="outline" onClick={step} disabled={running || busy}>单步</Button>
      <Button variant={running ? 'danger' : 'default'} onClick={toggleRun} disabled={busy}>
        {running ? '暂停' : '运行专家'}
      </Button>
      <div className="ml-auto flex items-center gap-3 text-[11px] text-muted">
        {config && <span>配置 v{config.version}</span>}
        {observation && <span>tick {observation.tick} · {observation.sim_time.toFixed(2)}s</span>}
        {metrics && (
          <span className={metrics.success ? 'text-accent' : ''}>
            {metrics.success ? '✓ 抓放成功' : metrics.lifted ? '已抬起' : metrics.grasped ? '双指接触' : '运行中'}
          </span>
        )}
      </div>
    </div>
  )
}