import { useEffect, useState } from 'react'
import { api } from '../api'
import { useStore } from '../store'
import { Button } from './ui'

export default function Toolbar() {
  const { config, dirty, running, runMode, policyModel, setPolicyModel, policyInstruction, setPolicyInstruction, busy, save, reset, step, toggleRun, togglePolicyRun, observation, metrics } = useStore()
  const [seed, setSeed] = useState(0)
  const [models, setModels] = useState<any[]>([])
  useEffect(() => {
    api.models().then((r) => {
      const list = (r.models || []).filter((m: any) => m.runnable)
      setModels(list)
      if (list.length && !list.some((m: any) => m.id === policyModel)) {
        const first = list[0]
        setPolicyModel(first.id)
        if (first.language) setPolicyInstruction(first.default_instruction || '')
      }
    }).catch(() => {})
  }, [config?.project_id])
  const cur = models.find((m: any) => m.id === policyModel)
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
          <select className="!w-auto text-xs" value={policyModel} disabled={running || busy}
            onChange={(e) => { const id = e.target.value; setPolicyModel(id); const m = models.find((x: any) => x.id === id); setPolicyInstruction(m?.language ? (m.default_instruction || '') : '') }}>
            {models.map((m) => (
              <option key={m.id} value={m.id}>{m.id} ({(m.camera_keys || ['front']).join('+')}){m.language ? ' · 语言' : ''}</option>
            ))}
          </select>
          <input className="!w-64 text-xs" list="vla-instructions" placeholder={cur?.language ? '语言指令（VLA，可输入或选已有）' : '固定动作模型：忽略指令'}
            value={policyInstruction} disabled={!cur?.language || running || busy}
            onChange={(e) => setPolicyInstruction(e.target.value)}
            title="VLA 使用该指令；固定动作模型（VisualBC/ACT）忽略它，二者同契约可互换" />
          {(cur?.instructions?.length || 0) > 0 && (
            <datalist id="vla-instructions">
              {cur.instructions.map((s: string) => <option key={s} value={s} />)}
            </datalist>
          )}
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