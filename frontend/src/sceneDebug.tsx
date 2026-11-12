import { useState } from 'react'
import { createRoot } from 'react-dom/client'
import SceneCanvas, { Col, Inst } from './components/SceneCanvas'

function App() {
  const [inst, setInst] = useState<Inst>({ position: [1, 2, 3], quaternion_xyzw: [0, 0, 0, 1], uniform_scale: 1 })
  const [cols, setCols] = useState<Col[]>([
    { name: 'cyl', shape: 'cylinder', full_size_m: [0.2, 0.2, 0.5], position: [0.5, 0, 0.25], quaternion_xyzw: [0, 0, 0, 1] },
  ])
  const [sel, setSel] = useState<any>(null)
  ;(window as any).__lastInstance = inst
  ;(window as any).__lastCollider = cols[0]
  return (
    <div style={{ width: 640, height: 480 }}>
      <SceneCanvas assetUrl={null} inst={inst} cols={cols} selected={sel} onSelect={setSel}
        onMove={(i, p, q) => { const next = cols.map((c, k) => k === i ? { ...c, position: p, quaternion_xyzw: q } : c); setCols(next); (window as any).__lastCollider = next[i] }}
        onInstanceMove={(p, q) => { const next = { ...inst, position: p, quaternion_xyzw: q }; setInst(next); (window as any).__lastInstance = next }}
        showGrid={false} />
    </div>
  )
}
createRoot(document.getElementById('root')!).render(<App />)
