import { Suspense, useMemo } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Grid, Line } from '@react-three/drei'
import * as THREE from 'three'
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js'
import { useLoader } from '@react-three/fiber'
import { useStore } from '../store'
import type { Link } from '../types'

function poseMatrix(pose?: { position: number[]; matrix: number[][] }, offset?: number[]) {
  const m = new THREE.Matrix4()
  if (!pose) return m
  const r = pose.matrix
  const p = pose.position
  m.set(r[0][0], r[0][1], r[0][2], p[0],
        r[1][0], r[1][1], r[1][2], p[1],
        r[2][0], r[2][1], r[2][2], p[2],
        0, 0, 0, 1)
  if (offset) m.multiply(new THREE.Matrix4().makeTranslation(offset[0], offset[1], offset[2]))
  return m
}

function MeshPart({ url, pose, color, offset }:
  { url: string; pose?: any; color: string; offset?: number[] }) {
  const obj = useLoader(OBJLoader, url)
  const cloned = useMemo(() => {
    const c = obj.clone(true)
    c.traverse((n: any) => {
      if (n.isMesh) {
        n.material = new THREE.MeshStandardMaterial({ color, roughness: 0.55, metalness: 0.1 })
        n.castShadow = true
      }
    })
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
  if (!config) return null
  const robot = config.robots[0]
  const poses = obs?.state.body_poses
  return (
    <group>
      {robot.links.map((link: Link) =>
        link.visuals.map((v, i) => (
          <group key={link.name + i}
            onClick={(e) => { e.stopPropagation(); select({ kind: 'link', id: link.name }) }}>
            <MeshPart url={`/${robot.mesh_dir}/${v.mesh}.obj`} pose={poses?.[link.name]}
              color={selection?.kind === 'link' && selection.id === link.name ? '#81d2bd' : '#d6c39a'} />
          </group>
        )),
      )}
      {robot.joints.filter((j) => j.type !== 'fixed').map((j) => {
        const pose = poses?.[j.child]
        if (!pose) return null
        const sel = selection?.kind === 'joint' && selection.id === j.name
        const rot = new THREE.Matrix3().set(
          pose.matrix[0][0], pose.matrix[0][1], pose.matrix[0][2],
          pose.matrix[1][0], pose.matrix[1][1], pose.matrix[1][2],
          pose.matrix[2][0], pose.matrix[2][1], pose.matrix[2][2])
        const axis = new THREE.Vector3(...j.axis).applyMatrix3(rot).normalize()
        const p = new THREE.Vector3(...pose.position)
        const a = p.clone().addScaledVector(axis, -0.05)
        const b = p.clone().addScaledVector(axis, 0.05)
        return (
          <group key={j.name} onClick={(e) => { e.stopPropagation(); select({ kind: 'joint', id: j.name }) }}>
            <Line points={[a.toArray(), b.toArray()]} color={sel ? '#ffcc66' : '#88aabb'} lineWidth={sel ? 3 : 1.5} />
            <mesh position={p.toArray()}>
              <sphereGeometry args={[sel ? 0.012 : 0.008, 12, 12]} />
              <meshBasicMaterial color={sel ? '#ffcc66' : '#88aabb'} />
            </mesh>
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
  return <MeshPart url={`/api/v1/cup.obj?asset=${encodeURIComponent(config.task.asset)}`}
    pose={pose} offset={(pose as any).visual_offset} color="#3a8f9c" />
}

function Cameras() {
  const config = useStore((s) => s.config)
  const obs = useStore((s) => s.observation)
  const select = useStore((s) => s.select)
  const selection = useStore((s) => s.selection)
  if (!config) return null
  return (
    <group>
      {config.sensors.map((s) => {
        const parentPose = s.parent === 'world' ? undefined : obs?.state.body_poses?.[s.parent]
        const base = poseMatrix(parentPose)
        const local = new THREE.Matrix4().makeTranslation(...(s.position as [number, number, number]))
        const world = base.multiply(local)
        const pos = new THREE.Vector3().setFromMatrixPosition(world)
        const sel = selection?.kind === 'sensor' && selection.id === s.id
        return (
          <group key={s.id} position={pos.toArray()}
            onClick={(e) => { e.stopPropagation(); select({ kind: 'sensor', id: s.id }) }}>
            <mesh>
              <boxGeometry args={[0.02, 0.02, 0.03]} />
              <meshBasicMaterial color={sel ? '#ffcc66' : '#e07a5f'} />
            </mesh>
            <Line points={[[0, 0, 0], [0.04, 0.03, 0.08]]} color="#e07a5f" lineWidth={1} />
            <Line points={[[0, 0, 0], [-0.04, 0.03, 0.08]]} color="#e07a5f" lineWidth={1} />
          </group>
        )
      })}
    </group>
  )
}

function StaticScene() {
  const config = useStore((s) => s.config)
  const goal = config?.task.goal_xy ?? [0.133, 0.05]
  return (
    <group>
      <mesh position={[0, 0, -0.028]} receiveShadow>
        <boxGeometry args={[0.64, 0.43, 0.056]} />
        <meshStandardMaterial color="#4a5c60" roughness={0.8} />
      </mesh>
      <mesh position={[goal[0], goal[1], 0.0045]}>
        <boxGeometry args={[0.148, 0.138, 0.009]} />
        <meshStandardMaterial color="#a1b0a7" roughness={0.7} />
      </mesh>
    </group>
  )
}

export default function Scene() {
  return (
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
        <Grid args={[2, 2]} position={[0, 0, -0.06]} cellColor="#26333a" sectionColor="#30414b"
          fadeDistance={3} infiniteGrid />
      </group>
      <OrbitControls makeDefault target={[0, 0.13, -0.02]} maxPolarAngle={Math.PI / 2} />
    </Canvas>
  )
}