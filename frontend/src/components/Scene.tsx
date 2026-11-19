import { memo, Suspense, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { Canvas, useFrame, useThree, useLoader } from '@react-three/fiber'
import { AdaptiveDpr, OrbitControls, Grid, Line, TransformControls } from '@react-three/drei'
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib'
import * as THREE from 'three'
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js'
import { create } from 'zustand'
import { api } from '../api'
import { useStore, activeRobot } from '../store'
import type { Collision, Link, Observation, Robot as RobotConfig } from '../types'
import { cn } from './ui'

export const useView = create<{
  grid: boolean; axes: boolean; frustums: boolean; colliders: boolean; skybox: boolean; splatCloud: boolean;
  cmd: { type: string; seq: number };
  toggle: (k: 'grid' | 'axes' | 'frustums' | 'colliders' | 'skybox' | 'splatCloud') => void;
  send: (type: string) => void;
}>((set) => ({
  grid: true, axes: true, frustums: true, colliders: false, skybox: true, splatCloud: false,
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

type BodyPose = Observation['state']['body_poses'][string]

function PoseGroup({ pose, name, children }: { pose: BodyPose; name?: string; children: ReactNode }) {
  const group = useRef<THREE.Group>(null)
  const invalidate = useThree((state) => state.invalidate)
  const motion = useMemo(() => ({
    initialized: false, elapsed: 0, duration: 0, sampledAt: 0,
    fromPosition: new THREE.Vector3(), toPosition: new THREE.Vector3(),
    fromRotation: new THREE.Quaternion(), toRotation: new THREE.Quaternion(),
    scale: new THREE.Vector3(),
  }), [])
  useLayoutEffect(() => {
    const body = group.current!
    poseMatrix(pose).decompose(motion.toPosition, motion.toRotation, motion.scale)
    motion.fromPosition.copy(body.position)
    motion.fromRotation.copy(body.quaternion)
    const moved = body.position.distanceToSquared(motion.toPosition) > 1e-10
      || body.quaternion.angleTo(motion.toRotation) > 1e-4
    const now = performance.now()
    const { running, vlaStatus } = useStore.getState()
    if (moved && motion.initialized && running) {
      motion.elapsed = 0
      motion.duration = THREE.MathUtils.clamp((now - motion.sampledAt) / 1000, 0.07, vlaStatus ? 0.5 : 0.12)
    } else {
      motion.duration = 0
      body.position.copy(motion.toPosition)
      body.quaternion.copy(motion.toRotation)
    }
    motion.initialized = true
    motion.sampledAt = now
    invalidate()
  }, [pose, invalidate, motion])
  useFrame((_, delta) => {
    if (!group.current || motion.elapsed >= motion.duration) return
    motion.elapsed = Math.min(motion.duration, motion.elapsed + Math.min(delta, 1 / 30))
    const t = motion.elapsed / motion.duration
    group.current.position.lerpVectors(motion.fromPosition, motion.toPosition, t)
    group.current.quaternion.slerpQuaternions(motion.fromRotation, motion.toRotation, t)
    if (t < 1) invalidate()
  })
  return <group name={name} ref={group}>{children}</group>
}

function MeshPart({ url, color }: { url: string; color: THREE.ColorRepresentation }) {
  const obj = useLoader(OBJLoader, url)
  const cloned = useMemo(() => {
    const c = obj.clone(true)
    c.traverse((n: any) => { if (n.isMesh) { n.material = new THREE.MeshStandardMaterial({ color, roughness: .55, metalness: .1 }); n.castShadow = true } })
    return c
  }, [obj, color])
  return <primitive object={cloned} />
}

function LinkShape({ geom, meshDir, selected, wireframe }: {
  geom: Collision; meshDir: string; selected: boolean; wireframe: boolean
}) {
  const position = new THREE.Vector3(...geom.pos)
  const rotation = new THREE.Quaternion()
  let length = (geom.size[1] ?? 0) * 2
  if (geom.fromto?.length === 6 && (geom.type === 'capsule' || geom.type === 'cylinder')) {
    const start = new THREE.Vector3().fromArray(geom.fromto)
    const end = new THREE.Vector3().fromArray(geom.fromto, 3)
    const axis = end.sub(start)
    length = axis.length()
    position.copy(start).addScaledVector(axis, 0.5)
    if (length > 0) rotation.setFromUnitVectors(new THREE.Vector3(0, 1, 0), axis.normalize())
  } else if (geom.type === 'capsule' || geom.type === 'cylinder') {
    rotation.setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI / 2)
  }
  const [r, g, b, alpha] = geom.rgba
  const color = selected ? '#81d2bd' : new THREE.Color(r, g, b)
  const [x, y, z] = geom.size
  if (geom.type === 'mesh') {
    return geom.mesh && !wireframe
      ? <group position={position}><MeshPart url={`/${meshDir}/${geom.mesh}.obj`} color={color} /></group>
      : null
  }
  return (
    <mesh name={geom.name ?? undefined} position={position} quaternion={rotation} castShadow={!wireframe}>
      {geom.type === 'box' ? <boxGeometry args={[2 * x, 2 * y, 2 * z]} />
        : geom.type === 'sphere' ? <sphereGeometry args={[x, 12, 12]} />
        : geom.type === 'cylinder' ? <cylinderGeometry args={[x, x, length, 16]} />
        : <capsuleGeometry args={[x, length, 4, 8]} />}
      {wireframe ? <meshBasicMaterial color="#ff5577" wireframe transparent opacity={0.35} />
        : <meshStandardMaterial color={color} opacity={alpha} transparent={alpha < 1} roughness={0.55} />}
    </mesh>
  )
}

const GripperRail = memo(function GripperRail({ robot, palm, selected }: {
  robot: RobotConfig; palm: Link; selected: boolean
}) {
  if (!robot.gripper || palm.visuals.length > 0) return null
  const mount = palm.collisions.find((geom) => geom.type === 'box' && geom.rgba[3] > 0)
  if (!mount) return null
  const joints = [robot.gripper.left_joint, robot.gripper.right_joint]
    .map((name) => robot.joints.find((joint) => joint.name === name))
  const [left, right] = joints
  if (left?.type !== 'prismatic' || right?.type !== 'prismatic') return null
  const leftJaw = robot.links.find((link) => link.name === left.child && link.parent === palm.name)
  const rightJaw = robot.links.find((link) => link.name === right.child && link.parent === palm.name)
  if (!leftJaw || !rightJaw || leftJaw.visuals.length || rightJaw.visuals.length) return null
  const axis = new THREE.Vector3().fromArray(left.axis).normalize()
  const rightAxis = new THREE.Vector3().fromArray(right.axis).normalize()
  if (axis.dot(rightAxis) > -0.99) return null

  const center = new THREE.Vector3().fromArray(mount.pos)
  const towardJaws = new THREE.Vector3().fromArray(leftJaw.origin)
    .add(new THREE.Vector3().fromArray(rightJaw.origin)).multiplyScalar(0.5).sub(center)
  towardJaws.addScaledVector(axis, -towardJaws.dot(axis))
  if (towardJaws.lengthSq() < 1e-8) return null
  towardJaws.normalize()
  const surface = mount.size.reduce((reach, half, i) => reach + half * Math.abs(towardJaws.getComponent(i)), 0)
  const openRange = [robot.gripper.closed_m, robot.gripper.open_m]
  const leftOrigin = axis.dot(new THREE.Vector3().fromArray(leftJaw.origin))
  const rightOrigin = axis.dot(new THREE.Vector3().fromArray(rightJaw.origin))
  const reach = [...(left.range ?? openRange).map((value) => leftOrigin + value),
    ...(right.range ?? openRange).map((value) => rightOrigin - value)]
  const lo = Math.min(...reach), hi = Math.max(...reach)
  center.addScaledVector(axis, (lo + hi) / 2 - center.dot(axis))
  center.addScaledVector(towardJaws, surface)
  const rotation = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(1, 0, 0), axis)
  return (
    <mesh name="gripper-slide-rail" position={center} quaternion={rotation} castShadow>
      <boxGeometry args={[hi - lo + 0.010, 0.011, 0.010]} />
      <meshStandardMaterial color={selected ? '#81d2bd' : '#77858d'} metalness={0.45} roughness={0.48} />
    </mesh>
  )
})

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
      {robot.links.map((link) => {
        const pose = poses?.[link.name]
        if (!pose) return null
        const selected = selection?.kind === 'link' && selection.id === link.name
        return (
          <PoseGroup key={`${robot.id}:${link.name}`} name={`link:${link.name}`} pose={pose}>
            <group onClick={(e) => { e.stopPropagation(); select({ kind: 'link', id: link.name }) }}>
              {link.visuals.map((v, i) => (
                <group key={i} position={v.pos as [number, number, number]} scale={v.scale as [number, number, number]}
                  quaternion={[v.quat[1], v.quat[2], v.quat[3], v.quat[0]]}>
                  <MeshPart url={`/${robot.mesh_dir}/${v.mesh}.obj`} color={selected ? '#81d2bd' : '#d6c39a'} />
                </group>
              ))}
              {!link.visuals.length && link.collisions.map((geom, i) => geom.group < 3 && geom.rgba[3] > 0
                ? <LinkShape key={i} geom={geom} meshDir={robot.mesh_dir} selected={selected} wireframe={false} />
                : null)}
              {link.name === robot.end_effector && <GripperRail robot={robot} palm={link} selected={selected} />}
            </group>
            {showColliders && link.collisions.map((geom, i) => geom.contype !== 0
              ? <LinkShape key={i} geom={geom} meshDir={robot.mesh_dir} selected={selected} wireframe />
              : null)}
          </PoseGroup>
        )
      })}
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
  return (
    <PoseGroup pose={pose} name="cup">
      <group position={pose.visual_offset as [number, number, number] | undefined}>
        <MeshPart url={`/api/v1/cup.obj?asset=${encodeURIComponent(config.task.asset)}`} color="#3a8f9c" />
      </group>
    </PoseGroup>
  )
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

function WorldPanoramaDome({ sceneId, revision, onReady }: { sceneId: string; revision: number; onReady?: (ok: boolean) => void }) {
  const [tex, setTex] = useState<THREE.Texture | null>(null)
  useEffect(() => {
    let alive = true
    let loaded: THREE.Texture | null = null
    setTex(null)
    onReady?.(false)
    const url = `/api/v1/scene-assets/${encodeURIComponent(sceneId)}/revisions/${revision}/preview`
    api.sceneAsset(sceneId, revision).then((m) => {
      if (!alive || !m?.preview_asset) return
      new THREE.TextureLoader().load(
        url,
        (t) => {
          if (!alive) { t.dispose(); return }
          t.colorSpace = THREE.SRGBColorSpace
          t.needsUpdate = true
          loaded = t
          setTex(t)
          onReady?.(true)
        },
        undefined,
        () => {
          if (alive) { setTex(null); onReady?.(false) }
        },
      )
    }).catch(() => {
      if (alive) { setTex(null); onReady?.(false) }
    })
    return () => {
      alive = false
      if (loaded) loaded.dispose()
      onReady?.(false)
    }
  }, [sceneId, revision])
  if (!tex) return null
  return (
    <mesh position={[0, 0, 0.35]} rotation={[Math.PI / 2, 0, 0]} renderOrder={-10}>
      <sphereGeometry args={[6.0, 48, 32]} />
      <meshBasicMaterial map={tex} side={THREE.BackSide} depthWrite={false} />
    </mesh>
  )
}

function b64ToBytes(b64: string): Uint8Array {
  const bin = atob(b64)
  const out = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i)
  return out
}

function WorldGaussianCloud({ sceneId, revision, inst }:
  { sceneId: string; revision: number; inst?: { position: number[]; quaternion_xyzw: number[]; uniform_scale: number } | null }) {
  const [cloud, setCloud] = useState<{ geo: THREE.BufferGeometry; floorZ: number } | null>(null)
  useEffect(() => {
    let alive = true
    let built: THREE.BufferGeometry | null = null
    api.scenePointCloud(sceneId, revision, 24000).then((data) => {
      if (!alive || !data?.positions_f32_b64 || !data?.colors_u8_b64) return
      const posBytes = b64ToBytes(data.positions_f32_b64)
      const colBytes = b64ToBytes(data.colors_u8_b64)
      const positions = new Float32Array(posBytes.buffer, posBytes.byteOffset, Math.floor(posBytes.byteLength / 4))
      const colors = new Float32Array(colBytes.length)
      for (let i = 0; i < colBytes.length; i++) colors[i] = colBytes[i] / 255
      const geo = new THREE.BufferGeometry()
      geo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
      geo.setAttribute('color', new THREE.BufferAttribute(colors, 3))
      built = geo
      setCloud({ geo, floorZ: Number(data.auto_floor_offset_z || 0) })
    }).catch(() => {
      if (alive) setCloud(null)
    })
    return () => {
      alive = false
      if (built) built.dispose()
    }
  }, [sceneId, revision])
  if (!cloud) return null
  const px = inst ? inst.position[0] : 1.5
  const py = inst ? inst.position[1] : -1.0
  const pz = (inst ? inst.position[2] : 0) + cloud.floorZ
  const q = inst?.quaternion_xyzw ? (inst.quaternion_xyzw as [number, number, number, number]) : ([0, 0, 0, 1] as [number, number, number, number])
  const sc = inst?.uniform_scale || 1
  return (
    <points geometry={cloud.geo} position={[px, py, pz]} quaternion={q} scale={sc}>
      <pointsMaterial size={0.015} vertexColors transparent opacity={0.72} sizeAttenuation />
    </points>
  )
}

function StaticScene() {
  const config = useStore((s) => s.config)
  const obs = useStore((s) => s.observation)
  const showColliders = useView(s => s.colliders)
  const showSkybox = useView(s => s.skybox)
  const showSplatCloud = useView(s => s.splatCloud)
  const [panoReady, setPanoReady] = useState(false)
  const [environmentInfo, setEnvironmentInfo] = useState<{ key: string; manifest: { ply_kind?: string; visual_asset?: string; preview_asset?: string } } | null>(null)
  const goal = config?.task.goal_xy ?? [0.133, 0.05]
  const goalSurface = config?.task.goal_surface_m ?? 0.009
  const goalHalf = config?.task.goal_half_extents ?? [0.074, 0.069]
  const showBuiltinTray = config?.task.show_builtin_tray !== false
  const wsZ = config?.scene.work_surface_z ?? 0
  const trayThickness = Math.max(0.004, goalSurface - wsZ)
  const environment = config?.scene.environment
  useEffect(() => {
    useView.setState({ splatCloud: false })
  }, [config?.project_id])
  useEffect(() => {
    let alive = true
    if (!environment) return
    const key = environment.asset_id + ':' + environment.revision
    api.sceneAsset(environment.asset_id, environment.revision)
      .then((manifest) => { if (alive) setEnvironmentInfo({ key, manifest }) })
      .catch(() => { if (alive) setEnvironmentInfo(null) })
    return () => { alive = false }
  }, [environment?.asset_id, environment?.revision])
  const assetNodes = config?.scene.asset_nodes || []
  const hasWorkbench = assetNodes.some((n) => n.node_id === 'demo_workbench')
  const environmentKey = environment ? environment.asset_id + ':' + environment.revision : null
  const envMeta = environmentInfo?.key === environmentKey ? environmentInfo?.manifest : null
  const hasGaussianEnv = environment && envMeta?.ply_kind === 'gaussian'
  const nodeColliderRefs = new Set(assetNodes.map((n) => n.collider_ref).filter(Boolean) as string[])
  const nodeColliderPrefixes = Array.from(nodeColliderRefs).map((r) => (r.endsWith('_part_0') ? r.slice(0, -7) + '_part_' : r + '_part_'))
  const isNodeCollider = (name: string) => nodeColliderRefs.has(name) || nodeColliderPrefixes.some((p) => name.startsWith(p))
  return (
    <group>
      {showSkybox && environment && hasGaussianEnv && envMeta?.preview_asset && (
        <WorldPanoramaDome key={environmentKey}
          sceneId={environment.asset_id} revision={environment.revision} onReady={setPanoReady} />
      )}
      {showSplatCloud && environment && hasGaussianEnv && (
        <WorldGaussianCloud key={environmentKey}
          sceneId={environment.asset_id} revision={environment.revision} inst={environment} />
      )}
      {(!showSkybox || !panoReady) && (
        <mesh position={[0, 0, -0.061]}><planeGeometry args={[4, 4]} /><meshStandardMaterial color="#303b40" side={THREE.DoubleSide} /></mesh>
      )}
      {config?.scene.use_default_table !== false && !hasWorkbench && (
        <mesh position={[0, 0, wsZ - 0.028]} receiveShadow><boxGeometry args={[0.64, 0.43, 0.056]} /><meshStandardMaterial color="#4a5c60" roughness={.8} /></mesh>
      )}
      {environment && envMeta?.ply_kind === 'mesh' && envMeta?.visual_asset && <group position={environment.position as [number, number, number]}
        quaternion={environment.quaternion_xyzw as [number, number, number, number]} scale={environment.uniform_scale}>
        <MeshPart url={`/api/v1/scene-assets/${encodeURIComponent(environment.asset_id)}/revisions/${environment.revision}/visual.obj`} color="#9ea8b2" />
      </group>}
      {assetNodes.map((n) => {
        const dynPose = n.role === 'dynamic' ? obs?.state?.dynamic_objects?.[n.node_id] : undefined
        const url = `/api/v1/asset-mesh?ref=${encodeURIComponent(n.asset_ref)}`
        const tint = n.role === 'dynamic' ? '#e08a4e' : (n.node_id === 'demo_workbench' ? '#58676e' : '#9ea8b2')
        if (dynPose) {
          return (
            <PoseGroup key={n.node_id} name={n.node_id} pose={dynPose}>
              <MeshPart url={url} color={tint} />
              {showColliders && dynPose.colliders?.map((col, ci) => (
                <group key={`${n.node_id}_dc_${ci}`} position={col.position as [number, number, number]}>
                  <mesh rotation={col.shape === 'cylinder' ? [Math.PI / 2, 0, 0] : [0, 0, 0]}>
                    {col.shape === 'cylinder'
                      ? <cylinderGeometry args={[col.size_m[0] / 2, col.size_m[0] / 2, col.size_m[2], 20]} />
                      : <boxGeometry args={col.size_m as [number, number, number]} />}
                    <meshBasicMaterial color="#fb923c" wireframe transparent opacity={0.9} depthTest={false} />
                  </mesh>
                </group>
              ))}
            </PoseGroup>
          )
        }
        return (
          <group key={n.node_id} position={n.position as [number, number, number]}
            quaternion={n.quaternion_xyzw as [number, number, number, number]} scale={n.uniform_scale}>
            <MeshPart url={url} color={tint} />
          </group>
        )
      })}
      {(config?.scene.colliders || []).filter(c => c.enabled && ((!environment && !isNodeCollider(c.name)) || showColliders)).map(c => {
        const colColor = showColliders
          ? (c.name.endsWith('_part_0') ? '#38bdf8' : c.name.includes('_part_') ? '#f59e0b' : '#ff5577')
          : '#80909b'
        return (
          <group key={c.name} position={c.position as [number, number, number]} quaternion={c.quaternion_xyzw as [number, number, number, number]}>
            <mesh rotation={c.shape === 'cylinder' ? [Math.PI / 2, 0, 0] : [0, 0, 0]}>
              {c.shape === 'box' ? <boxGeometry args={c.full_size_m as [number, number, number]} /> :
                c.shape === 'cylinder' ? <cylinderGeometry args={[c.full_size_m[0]/2, c.full_size_m[0]/2, c.full_size_m[2], 32]} /> :
                <planeGeometry args={[c.full_size_m[0], c.full_size_m[1]]} />}
              <meshStandardMaterial color={colColor} wireframe={showColliders} transparent={showColliders} depthTest={!showColliders} opacity={showColliders ? .85 : 1} side={THREE.DoubleSide} />
            </mesh>
          </group>
        )
      })}
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

const VIEW_TARGET: [number, number, number] = [0, 0.13, -0.02]

function ViewRig() {
  const { camera, gl, invalidate } = useThree()
  const cmd = useView((s) => s.cmd)
  const grid = useView((s) => s.grid)
  const orbit = useRef<OrbitControlsImpl>(null)
  const viewMove = useRef<{
    from: THREE.Vector3; to: THREE.Vector3; fromTarget: THREE.Vector3; toTarget: THREE.Vector3; progress: number
  } | null>(null)
  const selection = useStore((s) => s.selection)
  const config = useStore((s) => s.config)
  const obs = useStore((s) => s.observation)
  const edit = useStore((s) => s.edit)
  const proxy = useRef<THREE.Group>(null)

  useEffect(() => {
    const c = orbit.current
    if (!c) return
    const set = (pos: [number, number, number], target = new THREE.Vector3(...VIEW_TARGET)) => {
      viewMove.current = {
        from: camera.position.clone(), to: new THREE.Vector3(...pos),
        fromTarget: c.target.clone(), toTarget: target, progress: 0,
      }
      invalidate()
    }
    switch (cmd.type) {
      case 'reset': set([0.72, 0.52, 0.78]); break
      case 'focus': set([0.4, 0.35, 0.5], new THREE.Vector3(0, 0.15, -0.1)); break
      case 'top': set([0, 0.9, -0.02]); break
      case 'front': set([0, 0.15, 0.9]); break
      case 'side': set([0.9, 0.15, -0.02]); break
      case 'fullscreen':
        if (!document.fullscreenElement) gl.domElement.parentElement?.requestFullscreen?.()
        else document.exitFullscreen?.()
        break
    }
  }, [camera, cmd, gl, invalidate])
  useFrame((_, delta) => {
    const motion = viewMove.current
    const c = orbit.current
    if (!motion || !c) return
    motion.progress = Math.min(1, motion.progress + Math.min(delta, 0.033) / 0.28)
    const t = motion.progress * motion.progress * (3 - 2 * motion.progress)
    camera.position.lerpVectors(motion.from, motion.to, t)
    c.target.lerpVectors(motion.fromTarget, motion.toTarget, t)
    if (motion.progress < 1) invalidate()
    else viewMove.current = null
  }, -2)

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
      <OrbitControls ref={orbit} makeDefault regress target={VIEW_TARGET} dampingFactor={0.12} maxPolarAngle={Math.PI / 2}
        onStart={() => { viewMove.current = null }} />
      {grid && <Grid args={[2, 2]} position={[0, -0.06, 0]} cellColor="#26333a" sectionColor="#30414b" fadeDistance={3} infiniteGrid />}
      {selPose && (
        <TransformControls mode="translate" size={0.6}
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
      {(['grid', 'axes', 'frustums', 'colliders', 'skybox', 'splatCloud'] as const).map((k) => (
        <button key={k} className={cn(btn, v[k] && 'text-accent border-accent')} onClick={() => v.toggle(k)}>
          {{ grid: '网格', axes: '关节轴', frustums: '视锥', colliders: '碰撞', skybox: '全景', splatCloud: '点云' }[k]}
        </button>
      ))}
    </div>
  )
}

export default function Scene() {
  return (
    <div className="absolute inset-0">
      <Canvas frameloop="demand" shadows dpr={[1, 1.35]} performance={{ min: 0.72, debounce: 200 }}
        camera={{ position: [0.72, 0.52, 0.78], fov: 45, up: [0, 1, 0] }}
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
        <AdaptiveDpr />
        <ViewRig />
      </Canvas>
      <ViewToolbar />
    </div>
  )
}
