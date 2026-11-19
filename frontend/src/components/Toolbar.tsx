import { useEffect, useState } from 'react'
import { api } from '../api'
import { OMNI_VLA, useStore } from '../store'
import { Button } from './ui'

export default function Toolbar() {
  const { config, dirty, running, busy, save, reset, step, toggleRun, observation, metrics, runVla, vlaStatus } = useStore()
  const [seed, setSeed] = useState(0)
  const [model, setModel] = useState<Awaited<ReturnType<typeof api.models>>['models'][number]>()
  const [showVla, setShowVla] = useState(false)
  const [instruction, setInstruction] = useState('')
  useEffect(() => {
    let mounted = true
    api.models().then(({ models }) => {
      if (mounted) setModel(models.find((entry) => entry.id === OMNI_VLA && entry.runnable))
    }).catch(() => {})
    return () => { mounted = false }
  }, [])
  useEffect(() => { setInstruction(''); setShowVla(false) }, [config?.project_id])
  const hasCube = config?.scene.asset_nodes?.some((node) => node.node_id === 'companion_cube' && node.role === 'dynamic')
  const instructions = model?.instructions ?? []
  const blueCupPreset = instructions.find((text) => /\bblue cup\b/i.test(text))
  const orangeCubePreset = instructions.find((text) => /\borange cube\b/i.test(text))
  const compatible = observation?.control_mode === 'cartesian_delta_ik'
  const camerasReady = Boolean(observation?.images.front?.jpeg_b64 && observation?.images.wrist?.jpeg_b64)
  return (
    <div className="flex items-center gap-2 px-3 py-2 bg-panel2 border-b border-edge flex-wrap shrink-0">
      <Button onClick={save} disabled={!dirty || busy || running}>保存并应用</Button>
      <Button variant="outline" onClick={() => reset(seed)} disabled={busy || running}>重置</Button>
      <input aria-label="随机种子" type="number" className="!w-16" value={seed} onChange={(e) => setSeed(parseInt(e.target.value) || 0)} />
      <Button variant="outline" onClick={step} disabled={running || busy}>单步</Button>
      <Button variant={running ? 'danger' : 'default'} onClick={toggleRun} disabled={busy}>
        {running ? '暂停' : '运行专家'}
      </Button>
      {model && <Button variant="outline" aria-expanded={showVla} aria-controls="vla-test-controls"
        onClick={() => setShowVla(!showVla)}>自训 VLA 测试</Button>}
      <div className="ml-auto flex items-center gap-3 text-[11px] text-muted">
        {config && <span>配置 v{config.version}</span>}
        {observation && <span>tick {observation.tick} · {observation.sim_time.toFixed(2)}s</span>}
        {metrics && (
          <span className={metrics.success ? 'text-accent' : ''}>
            {metrics.success ? '✓ 抓放成功' : metrics.lifted ? '已抬起' : metrics.grasped ? '双指接触' : '运行中'}
          </span>
        )}
      </div>
      {showVla && model && (
        <div id="vla-test-controls" className="flex items-center gap-2 flex-wrap w-full border-t border-edge pt-2">
          <span className="text-[11px] text-muted">{hasCube ? '每轮搬运一个目标' : model.id}</span>
          {hasCube && blueCupPreset && <Button variant="outline" size="sm" onClick={() => setInstruction(blueCupPreset)} disabled={running || busy}>蓝杯入托盘</Button>}
          {hasCube && orangeCubePreset && <Button variant="outline" size="sm" onClick={() => setInstruction(orangeCubePreset)} disabled={running || busy}>橙色方块入托盘</Button>}
          <input aria-label="VLA 任务指令" type="text" placeholder="输入搬运指令" value={instruction}
            className="!w-auto flex-1 min-w-48 text-xs" maxLength={400} disabled={running || busy}
            onChange={(event) => setInstruction(event.target.value)} />
          <Button disabled={!compatible || !camerasReady || running || busy || dirty || !instruction.trim()}
            onClick={() => void runVla(instruction)}>运行 VLA</Button>
          {!compatible && <span className="text-[11px] text-muted">请选择支持四维笛卡尔控制的机械臂。</span>}
          {compatible && !camerasReady && <span className="text-[11px] text-muted">VLA 需要前视和腕部相机画面。</span>}
          {dirty && <span className="text-[11px] text-muted">请先保存场景改动。</span>}
          {vlaStatus && <span role="status" className="text-xs">{vlaStatus}</span>}
        </div>
      )}
    </div>
  )
}
