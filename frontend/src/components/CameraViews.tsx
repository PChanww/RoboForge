import { useStore } from '../store'

export default function CameraViews() {
  const obs = useStore((s) => s.observation)
  const config = useStore((s) => s.config)
  if (!config) return null
  return (
    <div className="flex gap-2 p-2 bg-panel2 border-t border-edge">
      {config.sensors.map((s) => {
        const f = obs?.images[s.id]
        const age = obs?.timing.image_age_s[s.id]
        return (
          <div key={s.id} className="flex-1 min-w-0">
            <div className="text-[11px] text-muted mb-1 flex justify-between">
              <span className="text-accent">{s.id}</span>
              <span>{s.resolution[0]}×{s.resolution[1]} @ {s.fps}FPS
                {age !== undefined && ` · ${(age * 1000).toFixed(0)}ms`}</span>
            </div>
            <div className="bg-black rounded overflow-hidden aspect-square flex items-center justify-center">
              {f ? <img src={`data:image/jpeg;base64,${f.jpeg_b64}`} className="w-full h-full object-contain" />
                : <span className="text-[11px] text-muted">无画面</span>}
            </div>
            {f && <div className="text-[10px] text-muted mt-0.5">capture {f.capture_time.toFixed(3)}s</div>}
          </div>
        )
      })}
    </div>
  )
}