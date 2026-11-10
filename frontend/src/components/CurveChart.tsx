import { useEffect, useRef, useState } from 'react'
import ReactECharts from 'echarts-for-react'
import { useStore } from '../store'

export default function CurveChart() {
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
    grid: { left: 40, right: 16, top: 24, bottom: 24 },
    tooltip: { trigger: 'axis' },
    legend: { data: ['末端高度', '夹爪开度'], textStyle: { color: '#b1c0c3', fontSize: 10 }, top: 0 },
    xAxis: { type: 'category', data: history.current.map((h) => h.tick), axisLabel: { color: '#b1c0c3' } },
    yAxis: { type: 'value', axisLabel: { color: '#b1c0c3' }, splitLine: { lineStyle: { color: '#26333a' } } },
    series: [
      { name: '末端高度', type: 'line', showSymbol: false, data: history.current.map((h) => h.z.toFixed(3)), lineStyle: { color: '#81d2bd' } },
      { name: '夹爪开度', type: 'line', showSymbol: false, data: history.current.map((h) => h.grip.toFixed(3)), lineStyle: { color: '#e07a5f' } },
    ],
  }
  return <ReactECharts option={option} style={{ height: '100%', width: '100%' }} />
}