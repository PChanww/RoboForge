import { useEffect, useState } from 'react'
import { api } from '../api'
import { useStore } from '../store'
import { Button } from './ui'

export default function Toolbar() {
  const { config, dirty, running, runMode, policyModel, setPolicyModel, busy, save, reset, step, toggleRun, togglePolicyRun, observation, metrics } = useStore()
  const [seed, setSeed] = useState(0)
  const [models, setModels] = useState<any[]>([])
  useEffect(() => {
    api.models().then((r) => {
      const list = (r.models || []).filter((m: any) => m.runnable)
      setModels(list)
      if (list.length && !list.some((m: any) => m.id === policyModel)) {
        setPolicyModel(list[0].id)
      }
    }).catch(() => {})
  }, [config?.project_id])
  return (
    <div className="flex items-center gap-2 px-3 py-2 bg-panel2 border-b border-edge flex-wrap">
      <span className="text-sm font-semibold text-accent mr-2">VLA Lab</span>
      <Button onClick={save} disabled={!dirty || busy}>保存并应用</Button>
      <Button variant="outline" onClick={() => reset(seed)} disabled={busy}>重置</Button>
      <input type="number" className="!w-16" value={seed} onChange={(e) => setSeed(parseInt(e.target.value) || 0)} />
      <Button variant="outline" onClick={step} disabled={running || busy}>单步</Button>
      <Button variant={running && runMode === 'expert' ? 'danger' : 'default'} onClick={toggleRun} disabled={busy || (running && runMode === 'policy')}>
        {running && runMode === 'expert' ? '暂停' : '运行专家'}
      </Button>
      {models.length > 0 && (
        <div className="flex items-center gap-1 border-l border-edge pl-2">
          <select className="!w-auto text-xs" value={policyModel} onChange={(e) => setPolicyModel(e.target.value)} disabled={running || busy}>
            {models.map((m) => (
              <option key={m.id} value={m.id}>{m.id} ({(m.camera_keys || ['front']).join('+')})</option>
            ))}
          </select>
          <Button size="sm" variant={running && runMode === 'policy' ? 'danger' : 'outline'}
            onClick={togglePolicyRun} disabled={busy || (running && runMode === 'expert')}>
            {running && runMode === 'policy' ? '暂停策略' : '运行视觉策略'}
          </Button>
        </div>
      )}
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