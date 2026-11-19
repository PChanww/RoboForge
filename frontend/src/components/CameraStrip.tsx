import { useEffect, useState } from 'react'
import { useStore } from '../store'
import { cn } from './ui'

function CameraImage({ jpeg, label, className }: { jpeg: string; label: string; className: string }) {
  const src = `data:image/jpeg;base64,${jpeg}`
  const [decoded, setDecoded] = useState(src)
  useEffect(() => {
    let active = true
    const incoming = new Image()
    incoming.src = src
    void incoming.decode().then(() => { if (active) setDecoded(src) }).catch(() => {})
    return () => { active = false }
  }, [src])
  return <img src={decoded} alt={label} decoding="async" className={className} />
}

export default function CameraStrip() {
  const obs = useStore((s) => s.observation)
  const config = useStore((s) => s.config)
  const select = useStore((s) => s.select)
  const selection = useStore((s) => s.selection)
  const [enlarged, setEnlarged] = useState<string | null>(null)
  if (!config) return null
  const big = enlarged ? obs?.images[enlarged] : null
  return (
    <div className="h-full flex gap-2 overflow-x-auto overflow-y-hidden px-2 py-1">
      {config.sensors.map((s) => {
        const f = obs?.images[s.id]
        const st = obs?.camera_status?.[s.id]
        const active = selection?.kind === 'sensor' && selection.id === s.id
        return (
          <div key={s.id}
            className={cn('shrink-0 w-[180px] h-full flex flex-col rounded border bg-panel2 cursor-pointer',
              active ? 'border-accent' : 'border-edge')}
            onClick={() => select({ kind: 'sensor', id: s.id })}>
            <div className="flex items-center justify-between px-1.5 pt-1 text-[11px]">
              <span className="text-accent truncate">{s.id}</span>
              <button className="text-muted hover:text-ink px-1" title="放大"
                onClick={(e) => { e.stopPropagation(); setEnlarged(s.id) }}>⤢</button>
            </div>
            <div className="flex-1 min-h-0 mx-1.5 my-1 bg-black rounded overflow-hidden flex items-center justify-center">
              {f ? <CameraImage jpeg={f.jpeg_b64} label={s.id} className="max-w-full max-h-full object-contain" />
                : <span className="text-[10px] text-muted">{st?.state === 'waiting' ? '等待首帧…' : '无画面'}</span>}
            </div>
            <div className="px-1.5 pb-1 text-[10px] text-muted leading-tight">
              <div>{s.resolution[0]}×{s.resolution[1]} · 配置{s.fps} / 仿真{st?.actual_capture_fps ?? '--'} FPS</div>
              <div>年龄 {st?.age_s != null ? `${(st.age_s * 1000).toFixed(0)}ms` : '--'} · {st?.state ?? '--'}</div>
            </div>
          </div>
        )
      })}
      {enlarged && (
        <div className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center" onClick={() => setEnlarged(null)}>
          <div className="max-w-[80vw] max-h-[80vh] flex flex-col" onClick={(e) => e.stopPropagation()}>
            <div className="flex justify-between text-ink text-sm mb-2">
              <span>相机 {enlarged}</span>
              <button className="text-muted hover:text-ink" onClick={() => setEnlarged(null)}>关闭 ✕</button>
            </div>
            {big ? <CameraImage key={enlarged} jpeg={big.jpeg_b64} label={enlarged} className="max-w-full max-h-[70vh] object-contain rounded" />
              : <div className="text-muted">无画面</div>}
            <div className="text-[11px] text-muted mt-1">capture {big?.capture_time ?? '--'}s · frame {big?.frame_id ?? '--'}</div>
          </div>
        </div>
      )}
    </div>
  )
}
