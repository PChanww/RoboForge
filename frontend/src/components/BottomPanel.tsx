import { useEffect, useRef, useState } from 'react'
import ReactECharts from 'echarts-for-react'
import { useStore } from '../store'
import { cn } from './ui'

export default function BottomPanel() {
  const [tab, setTab] = useState<'logs' | 'timing' | 'curve'>('logs')
  const logs = useStore((s) => s.logs)
  const obs = useStore((s) => s.observation)
  const history = useRef<{ tick: number; z: number; grip: number }[]>([])
  const [, force] = useState(0)

  useEffect(() => {
    if (obs) {
      const h = history.current
      if (!h.length || h[h.length - 1].tick !== obs.tick) {
        h.push({ tick: obs.tick, z: obs.state.eef_position[2], grip: obs.state.gripper ?? 0 })
        if (h.length > 300) h.shift()
        force((x) => x + 1)
      }
    }
  }, [obs])

  const option = {
    backgroundColor: 'transparent',
    grid: { left: 40, right: 20, top: 20, bottom: 30 },
    tooltip: { trigger: 'axis' },
    legend: { data: ['末端高度', '夹爪开度'], textStyle: { color: '#b1c0c3', fontSize: 10 }, top: 0 },
    xAxis: { type: 'category', data: history.current.map((h) => h.tick), axisLabel: { color: '#b1c0c3' } },
    yAxis: { type: 'value', axisLabel: { color: '#b1c0c3' }, splitLine: { lineStyle: { color: '#26333a' } } },
    series: [
      { name: '末端高度', type: 'line', showSymbol: false, data: history.current.map((h) => h.z.toFixed(3)), lineStyle: { color: '#81d2bd' } },
      { name: '夹爪开度', type: 'line', showSymbol: false, data: history.current.map((h) => h.grip.toFixed(3)), lineStyle: { color: '#e07a5f' } },
    ],
  }

  return (
    <div className="h-full flex flex-col bg-panel2">
      <div className="flex gap-1 px-2 pt-1 border-b border-edge">
        {([['logs', '日志'], ['timing', '时间调度'], ['curve', '曲线']] as const).map(([k, label]) => (
          <button key={k} onClick={() => setTab(k)}
            className={cn('px-3 py-1 text-xs rounded-t', tab === k ? 'bg-panel text-accent' : 'text-muted hover:text-ink')}>
            {label}
          </button>
        ))}
      </div>
      <div className="flex-1 overflow-y-auto p-2">
        {tab === 'logs' && (
          <div className="font-mono text-[11px] space-y-0.5">
            {logs.map((l, i) => (
              <div key={i} className={l.level === 'error' ? 'text-red-400' : 'text-muted'}>
                <span className="text-edge">{l.t}</span> {l.msg}
              </div>
            ))}
            {!logs.length && <div className="text-muted">暂无日志</div>}
          </div>
        )}
        {tab === 'timing' && obs && (
          <div className="text-[11px] text-muted space-y-1">
            <div>控制周期：{obs.timing.control_period_s}s · 物理步长：0.002s</div>
            {Object.entries(obs.timing.camera_period_s).map(([cid, p]) => (
              <div key={cid}>相机 {cid}：周期 {p.toFixed(4)}s · 已采 {obs.timing.capture_count[cid]} 帧 ·
                最近间隔 {obs.timing.actual_capture_intervals_s[cid]?.map((v) => v.toFixed(3)).join(', ')}</div>
            ))}
            <div>图像年龄：{Object.entries(obs.timing.image_age_s).map(([c, a]) => `${c} ${(a * 1000).toFixed(0)}ms`).join(' · ')}</div>
          </div>
        )}
        {tab === 'curve' && <ReactECharts option={option} style={{ height: '100%', minHeight: 120 }} />}
      </div>
    </div>
  )
}