import { useStore, activeRobot } from '../store'
import { Field, NumberInput, Vec3Input, Section, Slider } from './ui'

function JointEditor({ id }: { id: string }) {
  const config = useStore((s) => s.config)!
  const edit = useStore((s) => s.edit)
  const obs = useStore((s) => s.observation)
  const robot = useStore(activeRobot)!
  const j = robot.joints.find((x) => x.name === id)!
  const live = obs?.state.joint_pos[id]
  const upd = (patch: Partial<typeof j>) =>
    edit((c) => { Object.assign(c.robots.find((r) => r.id === robot.id)!.joints.find((x) => x.name === id)!, patch) })
  return (
    <div>
      <Section title={`关节 · ${j.name}`}>
        <Field label="类型">
          <select value={j.type} onChange={(e) => upd({ type: e.target.value as any })}>
            <option value="revolute">旋转 revolute</option>
            <option value="prismatic">移动 prismatic</option>
            <option value="fixed">固定 fixed</option>
          </select>
        </Field>
        <Field label="轴向 (局部坐标)">
          <Vec3Input value={j.axis} step={0.1} onChange={(v) => upd({ axis: v })} />
        </Field>
        {j.type !== 'fixed' && (
          <>
            <Field label={`下限 (${j.type === 'revolute' ? 'rad' : 'm'})`}>
              <NumberInput value={j.range?.[0] ?? 0} onChange={(v) => upd({ range: [v, j.range?.[1] ?? 0] })} />
            </Field>
            <Field label={`上限 (${j.type === 'revolute' ? 'rad' : 'm'})`}>
              <NumberInput value={j.range?.[1] ?? 0} onChange={(v) => upd({ range: [j.range?.[0] ?? 0, v] })} />
            </Field>
            <Field label="初始角/位置">
              <NumberInput value={j.initial} onChange={(v) => upd({ initial: v })} />
            </Field>
            <Field label="阻尼"><NumberInput value={j.damping} onChange={(v) => upd({ damping: v })} /></Field>
            <Field label="电枢惯量"><NumberInput value={j.armature} onChange={(v) => upd({ armature: v })} /></Field>
            {live !== undefined && (
              <div className="text-[11px] text-muted mt-2">当前实时值：<span className="text-accent">{live.toFixed(4)}</span></div>
            )}
          </>
        )}
        <div className="text-[11px] text-muted mt-1">子连杆：{j.child}</div>
      </Section>
    </div>
  )
}

function LinkEditor({ id }: { id: string }) {
  const config = useStore((s) => s.config)!
  const edit = useStore((s) => s.edit)
  const robot = useStore(activeRobot)!
  const l = robot.links.find((x) => x.name === id)!
  const mass = l.collisions.reduce((a, g) => a + g.mass_kg, 0)
  const upd = (patch: any) => edit((c) => { Object.assign(c.robots.find((r) => r.id === robot.id)!.links.find((x) => x.name === id)!, patch) })
  return (
    <Section title={`连杆 · ${l.name}`}>
      <Field label="原点 (父坐标系, m)"><Vec3Input value={l.origin} onChange={(v) => upd({ origin: v })} /></Field>
      <div className="text-[11px] text-muted mb-2">父连杆：{l.parent ?? '世界（根）'}</div>
      <div className="text-[11px] text-muted mb-2">碰撞体总质量：<span className="text-accent">{mass.toFixed(3)} kg</span></div>
      {l.collisions.map((g, i) => (
        <Field key={i} label={`碰撞体 ${i} 质量 (kg)`}>
          <NumberInput value={g.mass_kg} onChange={(v) =>
            edit((c) => { c.robots.find((r) => r.id === robot.id)!.links.find((x) => x.name === id)!.collisions[i].mass_kg = v })} />
        </Field>
      ))}
      <label className="flex items-center gap-2 text-xs text-muted">
        <input type="checkbox" className="w-auto" checked={l.gravcomp}
          onChange={(e) => upd({ gravcomp: e.target.checked })} /> 重力补偿
      </label>
      <div className="text-[11px] text-muted mt-2">视觉网格：{l.visuals.map((v) => v.mesh).join(', ') || '无'}</div>
    </Section>
  )
}

function SensorEditor({ id }: { id: string }) {
  const config = useStore((s) => s.config)!
  const edit = useStore((s) => s.edit)
  const obs = useStore((s) => s.observation)
  const s = config.sensors.find((x) => x.id === id)!
  const robot = useStore(activeRobot)!
  const bodies = ['world', ...robot.links.map((l) => l.name)]
  const upd = (patch: any) => edit((c) => { Object.assign(c.sensors.find((x) => x.id === id)!, patch) })
  const updK = (patch: any) => edit((c) => { Object.assign(c.sensors.find((x) => x.id === id)!.intrinsics, patch) })
  const age = obs?.timing.image_age_s[id]
  return (
    <div>
      <Section title={`相机 · ${s.id}`}>
        <Field label="安装父坐标系">
          <select value={s.parent} onChange={(e) => upd({ parent: e.target.value })}>
            {bodies.map((b) => <option key={b} value={b}>{b === 'world' ? '世界（固定）' : b}</option>)}
          </select>
        </Field>
        <Field label="位置 (m)"><Vec3Input value={s.position} onChange={(v) => upd({ position: v })} /></Field>
        <Field label="朝向 xyaxes (x轴, y轴)">
          <div className="grid grid-cols-3 gap-1">
            {s.xyaxes.map((v, i) => (
              <NumberInput key={i} step={0.01} value={v}
                onChange={(nv) => { const a = [...s.xyaxes]; a[i] = nv; upd({ xyaxes: a }) }} />
            ))}
          </div>
        </Field>
      </Section>
      <Section title="内参 (像素)">
        <div className="grid grid-cols-2 gap-2">
          <Field label="fx"><NumberInput step={1} value={s.intrinsics.fx} onChange={(v) => updK({ fx: v })} /></Field>
          <Field label="fy"><NumberInput step={1} value={s.intrinsics.fy} onChange={(v) => updK({ fy: v })} /></Field>
          <Field label="cx"><NumberInput step={1} value={s.intrinsics.cx} onChange={(v) => updK({ cx: v })} /></Field>
          <Field label="cy"><NumberInput step={1} value={s.intrinsics.cy} onChange={(v) => updK({ cy: v })} /></Field>
        </div>
      </Section>
      <Section title="采样">
        <div className="grid grid-cols-2 gap-2">
          <Field label="宽 (px)"><NumberInput step={1} value={s.resolution[0]}
            onChange={(v) => upd({ resolution: [v, s.resolution[1]] })} /></Field>
          <Field label="高 (px)"><NumberInput step={1} value={s.resolution[1]}
            onChange={(v) => upd({ resolution: [s.resolution[0], v] })} /></Field>
        </div>
        <Field label={`帧率 ${s.fps.toFixed(1)} FPS`}>
          <Slider value={s.fps} min={1} max={60} step={1} onChange={(v) => upd({ fps: v })} />
        </Field>
        <Field label="延迟 (s)"><NumberInput value={s.latency_s} onChange={(v) => upd({ latency_s: v })} /></Field>
        {age !== undefined && <div className="text-[11px] text-muted">当前图像年龄：<span className="text-accent">{(age * 1000).toFixed(0)} ms</span></div>}
      </Section>
    </div>
  )
}

function TaskEditor() {
  const config = useStore((s) => s.config)!
  const edit = useStore((s) => s.edit)
  const t = config.task
  const upd = (patch: any) => edit((c) => { Object.assign(c.task, patch) })
  return (
    <Section title="抓放任务">
      <Field label="资产">
        <select value={t.asset} onChange={(e) => upd({ asset: e.target.value })}>
          <option value="standard">标准杯</option>
          <option value="tall">高杯</option>
          <option value="wide">宽口杯</option>
          <option value="imported/lux3d-ivory">Lux3D 杯</option>
        </select>
      </Field>
      <Field label="语言指令"><input type="text" value={t.instruction} onChange={(e) => upd({ instruction: e.target.value })} /></Field>
      <Field label="目标位置 XY (m)">
        <div className="grid grid-cols-2 gap-1">
          <NumberInput value={t.goal_xy[0]} onChange={(v) => upd({ goal_xy: [v, t.goal_xy[1]] })} />
          <NumberInput value={t.goal_xy[1]} onChange={(v) => upd({ goal_xy: [t.goal_xy[0], v] })} />
        </div>
      </Field>
      <Field label="时间上限 (s)"><NumberInput step={0.5} value={t.time_limit_s} onChange={(v) => upd({ time_limit_s: v })} /></Field>
    </Section>
  )
}

function RobotEditor({ id }: { id: string }) {
  const config = useStore((s) => s.config)!
  const edit = useStore((s) => s.edit)
  const r = config.robots.find((x) => x.id === id)!
  return (
    <Section title={`机器人 · ${r.id}`}>
      <Field label="末端连杆"><input type="text" value={r.end_effector}
        onChange={(e) => edit((c) => { c.robots.find((x) => x.id === id)!.end_effector = e.target.value })} /></Field>
      <Field label="工具偏移 (m)"><NumberInput value={r.tool_offset_m}
        onChange={(v) => edit((c) => { c.robots.find((x) => x.id === id)!.tool_offset_m = v })} /></Field>
      <div className="text-[11px] text-muted">连杆 {r.links.length} · 关节 {r.joints.filter((j) => j.type !== 'fixed').length} · 执行器 {r.actuators.length}</div>
    </Section>
  )
}

export default function PropertiesPanel() {
  const selection = useStore((s) => s.selection)
  const dirty = useStore((s) => s.dirty)
  return (
    <div className="p-3 overflow-y-auto h-full">
      <div className="text-[11px] uppercase tracking-wide text-accent mb-2 flex items-center">
        属性 {dirty && <span className="ml-2 text-[10px] text-yellow-400">● 未保存</span>}
      </div>
      {!selection && <div className="text-xs text-muted">在场景树或三维视图中选择一个对象以编辑其参数。修改后点击「保存并应用」。</div>}
      {selection?.kind === 'joint' && <JointEditor id={selection.id} />}
      {selection?.kind === 'link' && <LinkEditor id={selection.id} />}
      {selection?.kind === 'sensor' && <SensorEditor id={selection.id} />}
      {selection?.kind === 'robot' && <RobotEditor id={selection.id} />}
      {selection?.kind === 'task' && <TaskEditor />}
    </div>
  )
}