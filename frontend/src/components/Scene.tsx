import { Suspense, useEffect, useMemo, useRef } from 'react'
import { Canvas, useThree, useLoader } from '@react-three/fiber'
import { OrbitControls, Grid, Line, TransformControls } from '@react-three/drei'
import * as THREE from 'three'
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js'
import { create } from 'zustand'
import { useStore, activeRobot } from '../store'
import type { Link } from '../types'
import { cn } from './ui'

export const useView = create<{
  grid: boolean; axes: boolean; frustums: boolean; colliders: boolean;
  cmd: { type: string; seq: number };
  toggle: (k: 'grid' | 'axes' | 'frustums' | 'colliders') => void;
  send: (type: string) => void;
}>((set) => ({
  grid: true, axes: true, frustums: true, colliders: false,
  cmd: { type: '', seq: 0 },
  toggle: (k) => set((s) => ({ [k]: !s[k] } as any)),
  send: (type) => set((s) => ({ cmd: { type, seq: s.cmd.seq + 1 } })),
}))

function poseMatrix(pose?: { position: number[]; matrix: number[][] }, offset?: number[]) {
  const m = new THREE.Matrix4()
  if (!pose) return m
  const r = pose.matrix, p = pose.position
  m.set(r[0][0], r[0][1], r[0][2], p[0], r[1][0], r[1][1], r[1][2], p[1],
        r[2][0], r[2][1], r[2][2], p[2], 0, 0, 0, 1)
  if (offset) m.multiply(new THREE.Matrix4().makeTranslation(offset[0], offset[1], offset[2]))
  return m
}

function MeshPart({ url, pose, color, offset }: { url: string; pose?: any; color: string; offset?: number[] }) {
  const obj = useLoader(OBJLoader, url)
  const cloned = useMemo(() => {
    const c = obj.clone(true)
    c.traverse((n: any) => { if (n.isMesh) { n.material = new THREE.MeshStandardMaterial({ color, roughness: .55, metalness: .1 }); n.castShadow = true } })
    return c
  }, [obj, color])
  cloned.matrixAutoUpdate = false
  cloned.matrix.copy(poseMatrix(pose, offset))
  return <primitive object={cloned} />
}

function Robot() {
  const config = useStore((s) => s.config)
  const obs = useStore((s) => s.observation)
  const selection = useStore((s) => s.selection)
  const select = useStore((s) => s.select)
  const showAxes = useView((s) => s.axes)
  const showColliders = useView((s) => s.colliders)
  const robot = useStore(activeRobot)
  if (!config || !robot) return null
  const poses = obs?.state.body_poses
  return (
    <group>
      {robot.links.map((link: Link) => link.visuals.map((v, i) => (
        <group key={link.name + i} onClick={(e) => { e.stopPropagation(); select({ kind: 'link', id: link.name }) }}>
          <MeshPart url={`/${robot.mesh_dir}/${v.mesh}.obj`} pose={poses?.[link.name]}
            color={selection?.kind === 'link' && selection.id === link.name ? '#81d2bd' : '#d6c39a'} />
        </group>
      )))}
      {showColliders && robot.links.map((link: Link) => link.collisions.filter((c) => c.contype !== 0).map((c, i) => {
        const pose = poses?.[link.name]
        if (!pose) return null
        const m = poseMatrix(pose).multiply(new THREE.Matrix4().makeTranslation(...(c.pos as [number, number, number])))
        return (
          <mesh key={link.name + 'c' + i} matrixAutoUpdate={false} matrix={m}>
            {c.type === 'box' ? <boxGeometry args={[c.size[0] * 2, c.size[1] * 2, c.size[2] * 2]} />
              : c.type === 'sphere' ? <sphereGeometry args={[c.size[0], 12, 12]} />
              : <capsuleGeometry args={[c.size[0], Math.max(0.01, (c.fromto ? c.fromto[5] - c.fromto[2] : c.size[1] * 2)), 4, 8]} />}
            <meshBasicMaterial color="#ff5577" wireframe transparent opacity={0.35} />
          </mesh>
        )
      }))}
      {showAxes && robot.joints.filter((j) => j.type !== 'fixed').map((j) => {
        const pose = poses?.[j.child]
        if (!pose) return null
        const sel = selection?.kind === 'joint' && selection.id === j.name
        const m = pose.matrix
        const rot = new THREE.Matrix3().set(m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2])
        const axis = new THREE.Vector3(...j.axis).applyMatrix3(rot).normalize()
        const p = new THREE.Vector3(...pose.position)
        return (
          <group key={j.name} onClick={(e) => { e.stopPropagation(); select({ kind: 'joint', id: j.name }) }}>
            <Line points={[p.clone().addScaledVector(axis, -.05).toArray(), p.clone().addScaledVector(axis, .05).toArray()]}
              color={sel ? '#ffcc66' : '#88aabb'} lineWidth={sel ? 3 : 1.5} />
            <mesh position={p.toArray()}><sphereGeometry args={[sel ? .012 : .008, 12, 12]} /><meshBasicMaterial color={sel ? '#ffcc66' : '#88aabb'} /></mesh>
          </group>
        )
      })}
    </group>
  )
}

function Cup() {
  const config = useStore((s) => s.config)
  const obs = useStore((s) => s.observation)
  if (!config || !obs) return null
  const pose = obs.state.object_pose
  return <MeshPart url={`/api/v1/cup.obj?asset=${encodeURIComponent(config.task.asset)}`} pose={pose} offset={(pose as any).visual_offset} color="#3a8f9c" />
}

function Cameras() {
  const config = useStore((s) => s.config)
  const obs = useStore((s) => s.observation)
  const select = useStore((s) => s.select)
  const selection = useStore((s) => s.selection)
  const show = useView((s) => s.frustums)
  if (!config || !show) return null
  return (
    <group>
      {config.sensors.map((s) => {
        const parentPose = s.parent === 'world' ? undefined : obs?.state.body_poses?.[s.parent]
        const world = poseMatrix(parentPose).multiply(new THREE.Matrix4().makeTranslation(...(s.position as [number, number, number])))
        const pos = new THREE.Vector3().setFromMatrixPosition(world)
        const sel = selection?.kind === 'sensor' && selection.id === s.id
        // frustum from intrinsics: half-angle from fy and resolution height
        const fy = s.intrinsics.fy, h = s.resolution[1], w = s.resolution[0]
        const depth = 0.12
        const hy = depth * (h / 2) / fy, hx = depth * (w / 2) / fy
        const rot = new THREE.Matrix4().extractRotation(world)
        const corners = [[-hx, -hy, -depth], [hx, -hy, -depth], [hx, hy, -depth], [-hx, hy, -depth]]
          .map((c) => new THREE.Vector3(...c).applyMatrix4(rot).add(pos))
        return (
          <group key={s.id} onClick={(e) => { e.stopPropagation(); select({ kind: 'sensor', id: s.id }) }}>
            <mesh position={pos.toArray()}><boxGeometry args={[.02, .02, .03]} /><meshBasicMaterial color={sel ? '#ffcc66' : '#e07a5f'} /></mesh>
            {corners.map((c, i) => <Line key={i} points={[pos.toArray(), c.toArray()]} color={sel ? '#ffcc66' : '#e07a5f'} lineWidth={1} />)}
            <Line points={[...corners.map((c) => c.toArray()), corners[0].toArray()]} color={sel ? '#ffcc66' : '#e07a5f'} lineWidth={1} />
          </group>
        )
      })}
    </group>
  )
}

function StaticScene() {
  const config = useStore((s) => s.config)
  const obs = useStore((s) => s.observation)
  const showColliders = useView(s => s.colliders)
  const goal = config?.task.goal_xy ?? [0.133, 0.05]
  const goalSurface = config?.task.goal_surface_m ?? 0.009
  const goalHalf = config?.task.goal_half_extents ?? [0.074, 0.069]
  const showBuiltinTray = config?.task.show_builtin_tray !== false
  const wsZ = config?.scene.work_surface_z ?? 0
  const trayThickness = Math.max(0.004, goalSurface - wsZ)
  const environment = config?.scene.environment
  const assetNodes = config?.scene.asset_nodes || []
  const nodeColliderRefs = new Set(assetNodes.map((n) => n.collider_ref).filter(Boolean) as string[])
  const nodeColliderPrefixes = Array.from(nodeColliderRefs).map((r) => (r.endsWith('_part_0') ? r.slice(0, -7) + '_part_' : r + '_part_'))
  const isNodeCollider = (name: string) => nodeColliderRefs.has(name) || nodeColliderPrefixes.some((p) => name.startsWith(p))
  return (
    <group>
      <mesh position={[0, 0, -0.061]}><planeGeometry args={[4, 4]} /><meshStandardMaterial color="#303b40" side={THREE.DoubleSide} /></mesh>
      {config?.scene.use_default_table !== false && <mesh position={[0, 0, wsZ - 0.028]} receiveShadow><boxGeometry args={[0.64, 0.43, 0.056]} /><meshStandardMaterial color="#4a5c60" roughness={.8} /></mesh>}
      {environment && <group position={environment.position as [number, number, number]}
        quaternion={environment.quaternion_xyzw as [number, number, number, number]} scale={environment.uniform_scale}>
        <MeshPart url={`/api/v1/scene-assets/${encodeURIComponent(environment.asset_id)}/revisions/${environment.revision}/visual.obj`} color="#9ea8b2" />
      </group>}
      {assetNodes.map((n) => {
        const dynPose = n.role === 'dynamic' ? obs?.state?.dynamic_objects?.[n.node_id] : undefined
        const url = `/api/v1/asset-mesh?ref=${encodeURIComponent(n.asset_ref)}`
        const tint = n.role === 'dynamic' ? '#e08a4e' : '#9ea8b2'
        if (dynPose) {
          return <MeshPart key={n.node_id} url={url} pose={dynPose} color={tint} />
        }
        return (
          <group key={n.node_id} position={n.position as [number, number, number]}
            quaternion={n.quaternion_xyzw as [number, number, number, number]} scale={n.uniform_scale}>
            <MeshPart url={url} color={tint} />
          </group>
        )
      })}
      {(config?.scene.colliders || []).filter(c => c.enabled && ((!environment && !isNodeCollider(c.name)) || showColliders)).map(c => (
        <group key={c.name} position={c.position as [number, number, number]} quaternion={c.quaternion_xyzw as [number, number, number, number]}>
          <mesh rotation={c.shape === 'cylinder' ? [Math.PI / 2, 0, 0] : [0, 0, 0]}>
            {c.shape === 'box' ? <boxGeometry args={c.full_size_m as [number, number, number]} /> :
              c.shape === 'cylinder' ? <cylinderGeometry args={[c.full_size_m[0]/2, c.full_size_m[0]/2, c.full_size_m[2], 32]} /> :
              <planeGeometry args={[c.full_size_m[0], c.full_size_m[1]]} />}
            <meshStandardMaterial color={showColliders ? '#ff5577' : '#80909b'} wireframe={showColliders} transparent={showColliders} depthTest={!showColliders} opacity={showColliders ? .6 : 1} side={THREE.DoubleSide} />
          </mesh>
        </group>
      ))}
      {showBuiltinTray ? (
        <mesh position={[goal[0], goal[1], wsZ + trayThickness / 2]}>
          <boxGeometry args={[goalHalf[0] * 2, goalHalf[1] * 2, trayThickness]} />
          <meshStandardMaterial color="#a1b0a7" roughness={.7} />
        </mesh>
      ) : (
        <mesh position={[goal[0], goal[1], goalSurface]}>
          <boxGeometry args={[goalHalf[0] * 2, goalHalf[1] * 2, 0.002]} />
          <meshStandardMaterial color="#4ade80" wireframe transparent opacity={.75} />
        </mesh>
      )}
    </group>
  )
}

function ViewRig() {
  const { camera, controls, gl } = useThree() as any
  const cmd = useView((s) => s.cmd)
  const grid = useView((s) => s.grid)
  const orbit = useRef<any>(null)
  const transform = useRef<any>(null)
  const selection = useStore((s) => s.selection)
  const config = useStore((s) => s.config)
  const obs = useStore((s) => s.observation)
  const edit = useStore((s) => s.edit)
  const proxy = useRef<THREE.Group>(null)

  useEffect(() => {
    const c = orbit.current
    if (!c) return
    const target = new THREE.Vector3(0, 0.13, -0.02)
    const set = (pos: [number, number, number], tgt = target) => {
      camera.position.set(...pos); c.target.copy(tgt); c.update()
    }
    switch (cmd.type) {
      case 'reset': set([0.72, 0.52, 0.78]); break
      case 'focus': set([0.4, 0.35, 0.5], new THREE.Vector3(0, 0.15, -0.1)); break
      case 'top': set([0, 0.9, -0.02], target); break
      case 'front': set([0, 0.15, 0.9], target); break
      case 'side': set([0.9, 0.15, -0.02], target); break
      case 'fullscreen':
        if (!document.fullscreenElement) gl.domElement.parentElement?.requestFullscreen?.()
        else document.exitFullscreen?.()
        break
    }
  }, [cmd])

  // Transform proxy for selected camera / link (translate), synced back to config.
  const selPose = useMemo(() => {
    if (!config || !obs) return null
    if (selection?.kind === 'sensor') {
      const s = config.sensors.find((x) => x.id === selection.id)
      if (!s) return null
      const parent = s.parent === 'world' ? undefined : obs.state.body_poses?.[s.parent]
      return { pos: new THREE.Vector3(...s.position).applyMatrix4(poseMatrix(parent)), kind: 'sensor' as const, id: s.id }
    }
    if (selection?.kind === 'link') {
      const p = obs.state.body_poses?.[selection.id]
      return p ? { pos: new THREE.Vector3(...p.position), kind: 'link' as const, id: selection.id } : null
    }
    return null
  }, [selection, config, obs])

  return (
    <>
      <OrbitControls ref={orbit} makeDefault target={[0, 0.13, -0.02]} maxPolarAngle={Math.PI / 2} />
      {grid && <Grid args={[2, 2]} position={[0, -0.06, 0]} cellColor="#26333a" sectionColor="#30414b" fadeDistance={3} infiniteGrid />}
      {selPose && (
        <TransformControls ref={transform} mode="translate" size={0.6}
          onMouseDown={() => { if (orbit.current) orbit.current.enabled = false }}
          onMouseUp={() => { if (orbit.current) orbit.current.enabled = true }}
          onObjectChange={() => {
            if (!proxy.current || !selPose) return
            const p = proxy.current.position
            if (selPose.kind === 'sensor') {
              edit((c) => { const s = c.sensors.find((x) => x.id === selPose.id); if (s) s.position = [p.x, p.y, p.z] })
            }
          }}>
          <group ref={proxy} position={selPose.pos.toArray()} />
        </TransformControls>
      )}
    </>
  )
}

function ViewToolbar() {
  const v = useView()
  const btn = 'px-2 py-0.5 rounded bg-panel/80 border border-edge text-[11px] text-muted hover:text-ink'
  return (
    <div className="absolute top-2 left-2 flex flex-wrap gap-1 items-center pointer-events-auto">
      <button className={btn} onClick={() => v.send('focus')}>聚焦</button>
      <button className={btn} onClick={() => v.send('reset')}>复位</button>
      <button className={btn} onClick={() => v.send('top')}>俯</button>
      <button className={btn} onClick={() => v.send('front')}>正</button>
      <button className={btn} onClick={() => v.send('side')}>侧</button>
      <button className={btn} onClick={() => v.send('fullscreen')}>全屏</button>
      {(['grid', 'axes', 'frustums', 'colliders'] as const).map((k) => (
        <button key={k} className={cn(btn, v[k] && 'text-accent border-accent')} onClick={() => v.toggle(k)}>
          {{ grid: '网格', axes: '关节轴', frustums: '视锥', colliders: '碰撞' }[k]}
        </button>
      ))}
    </div>
  )
}

export default function Scene() {
  return (
    <div className="absolute inset-0">
      <Canvas shadows gl={{ preserveDrawingBuffer: true }} camera={{ position: [0.72, 0.52, 0.78], fov: 45, up: [0, 1, 0] }}
        onPointerMissed={() => useStore.getState().select(null)}>
        <color attach="background" args={['#101a20']} />
        <ambientLight intensity={0.5} />
        <directionalLight position={[0.5, 1, 0.6]} intensity={1.1} castShadow />
        <directionalLight position={[-0.5, 0.6, -0.4]} intensity={0.4} />
        <group rotation={[-Math.PI / 2, 0, 0]}>
          <Suspense fallback={null}>
            <StaticScene />
            <Robot />
            <Cup />
            <Cameras />
          </Suspense>
        </group>
        <ViewRig />
      </Canvas>
      <ViewToolbar />
    </div>
  )
}
