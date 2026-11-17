import { Suspense, useEffect, useMemo, useRef, useState } from 'react'
import { Canvas, useLoader } from '@react-three/fiber'
import { OrbitControls, Grid, TransformControls } from '@react-three/drei'
import * as THREE from 'three'
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js'

export interface Inst { position: number[]; quaternion_xyzw: number[]; uniform_scale: number }
export interface Col { name: string; shape: string; full_size_m: number[]; position: number[]; quaternion_xyzw: number[]; enabled?: boolean }
export interface AssetNodeItem { node_id: string; asset_ref: string; role: string; position: number[]; quaternion_xyzw: number[]; uniform_scale: number; collider_ref?: string | null }

export function quatXYZW(q?: number[]): [number, number, number, number] {
  const v = q && q.length === 4 ? q : [0, 0, 0, 1]
  const n = Math.hypot(v[0], v[1], v[2], v[3]) || 1
  return [v[0] / n, v[1] / n, v[2] / n, v[3] / n]
}

function SceneMesh({ url }: { url: string }) {
  const obj = useLoader(OBJLoader, url)
  const cloned = useMemo(() => {
    const c = obj.clone(true)
    c.traverse((n: any) => {
      if (n.isMesh) { n.material = new THREE.MeshStandardMaterial({ color: '#8d969c', roughness: .8, side: THREE.DoubleSide }); n.castShadow = true }
    })
    return c
  }, [obj])
  return <primitive object={cloned} />
}

function ColMesh({ c, color }: { c: Col; color: string }) {
  const [fx, fy, fz] = c.full_size_m
  const geo = useMemo(() => {
    if (c.shape === 'cylinder') {
      const g = new THREE.CylinderGeometry(fx / 2, fx / 2, fz, 24, 1, true)
      g.rotateX(Math.PI / 2)  // physics cylinders use the canonical Z axis
      return g
    }
    if (c.shape === 'plane') return new THREE.PlaneGeometry(fx, fy)
    return new THREE.BoxGeometry(fx, fy, fz)
  }, [c.shape, fx, fy, fz])
  return <mesh geometry={geo}>
    <meshBasicMaterial color={color} wireframe />
  </mesh>
}

interface RigProps {
  inst: Inst; cols: Col[]; assetNodes?: AssetNodeItem[]; assetUrl: string | null; selected: any
  onSelect: (s: any) => void; onMove: (i: number, p: number[], q: number[]) => void
  onInstanceMove: (p: number[], q: number[]) => void; showGrid: boolean; gizmo: 'translate' | 'rotate'
}

function Rig({ inst, cols, assetNodes = [], assetUrl, selected, onSelect, onMove, onInstanceMove, showGrid, gizmo }: RigProps) {
  const orbit = useRef<any>(null)
  const instRef = useRef<THREE.Group>(null)
  const colRefs = useRef<Record<number, THREE.Group | null>>({})
  const [selObj, setSelObj] = useState<THREE.Object3D | null>(null)
  const selectedCol = selected?.kind === 'collider' ? selected.index : -1
  useEffect(() => {
    const o = selected?.kind === 'collider' ? (colRefs.current[selected.index] ?? null)
      : selected?.kind === 'instance' ? instRef.current : null
    setSelObj(o)
  }, [selected, cols.length, inst.position])
  const commit = () => {
    if (!selObj) return
    const p = selObj.position.toArray()
    const q = selObj.quaternion.toArray()  // THREE stores xyzw, same as the config
    if (selected?.kind === 'collider') onMove(selected.index, p, q)
    else if (selected?.kind === 'instance') onInstanceMove(p, q)
  }
  useEffect(() => {
    // Verification hook: exercises the same absolute-transform commit path as a gizmo drag.
    ;(window as any).__sceneDebug = {
      selectInstance: () => onSelect({ kind: 'instance' }),
      selectCollider: (i: number) => onSelect({ kind: 'collider', index: i }),
      nudge: (d: number[]) => {
        if (!selObj) return null
        selObj.position.set(selObj.position.x + d[0], selObj.position.y + d[1], selObj.position.z + d[2])
        commit()
        return selObj.position.toArray()
      },
      setQuat: (q: number[]) => { if (!selObj) return null; selObj.quaternion.set(q[0], q[1], q[2], q[3]).normalize(); commit(); return selObj.quaternion.toArray() },
      info: () => selObj ? {
        type: selObj.type, position: selObj.position.toArray(), quaternion: selObj.quaternion.toArray(),
        children: selObj.children.map((c: any) => {
          c.geometry?.computeBoundingBox?.()
          return { type: c.type, geometry: c.geometry?.type || null,
                   bbox: c.geometry?.boundingBox ? c.geometry.boundingBox.getSize(new THREE.Vector3()).toArray() : null }
        }),
      } : null,
    }
  }, [selObj, selected])
  const s = inst.uniform_scale || 1
  return (
    <>
      <ambientLight intensity={0.75} />
      <directionalLight position={[1, 1.4, 1]} intensity={1.0} />
      {showGrid && <Grid args={[4, 4]} position={[0, -0.02, 0]} cellColor="#26333a" sectionColor="#30414b" infiniteGrid fadeDistance={6} />}
      <group rotation={[-Math.PI / 2, 0, 0]}>
        <group ref={instRef} position={inst.position as any} quaternion={quatXYZW(inst.quaternion_xyzw)}
          scale={[s, s, s]} onClick={(e) => { e.stopPropagation(); onSelect({ kind: 'instance' }) }}>
          {assetUrl ? <Suspense fallback={null}><SceneMesh url={assetUrl} /></Suspense> : null}
        </group>
        {assetNodes.map((n) => (
          <group key={n.node_id} position={n.position as any} quaternion={quatXYZW(n.quaternion_xyzw)} scale={[n.uniform_scale || 1, n.uniform_scale || 1, n.uniform_scale || 1]}>
            <Suspense fallback={null}><SceneMesh url={`/api/v1/asset-mesh?ref=${encodeURIComponent(n.asset_ref)}`} /></Suspense>
          </group>
        ))}
        {cols.map((c, i) => c.enabled === false ? null : (
          <group key={c.name + i} ref={(g) => { colRefs.current[i] = g }}
            position={c.position as any} quaternion={quatXYZW(c.quaternion_xyzw)}
            onClick={(e) => { e.stopPropagation(); onSelect({ kind: 'collider', index: i }) }}>
            <ColMesh c={c} color={selectedCol === i ? '#ffcc66' : '#e07a5f'} />
          </group>
        ))}
      </group>
      <OrbitControls ref={orbit} makeDefault target={[0, 0.1, 0]} maxPolarAngle={Math.PI / 2} />
      {selObj && (
        <TransformControls object={selObj} mode={gizmo} size={0.7}
          onMouseDown={() => { if (orbit.current) orbit.current.enabled = false }}
          onMouseUp={() => { if (orbit.current) orbit.current.enabled = true; commit() }} />
      )}
    </>
  )
}

export default function SceneCanvas({ assetUrl, inst, cols, assetNodes = [], selected, onSelect, onMove, onInstanceMove, showGrid }:
  { assetUrl: string | null; inst: Inst; cols: Col[]; assetNodes?: AssetNodeItem[]; selected: any; onSelect: (s: any) => void
    onMove: (i: number, p: number[], q: number[]) => void; onInstanceMove: (p: number[], q: number[]) => void; showGrid: boolean }) {
  const [gizmo, setGizmo] = useState<'translate' | 'rotate'>('translate')
  return (
    <div className="relative h-[420px] rounded border border-edge overflow-hidden bg-black">
      <div className="absolute z-10 m-1 flex gap-1 text-[11px]">
        <button onClick={() => setGizmo('translate')} className={`px-2 py-0.5 rounded ${gizmo === 'translate' ? 'bg-panel text-accent' : 'bg-panel2 text-muted'}`}>平移</button>
        <button onClick={() => setGizmo('rotate')} className={`px-2 py-0.5 rounded ${gizmo === 'rotate' ? 'bg-panel text-accent' : 'bg-panel2 text-muted'}`}>旋转</button>
      </div>
      <Canvas shadows camera={{ position: [1.1, 0.9, 1.1], fov: 45, up: [0, 1, 0] }} onPointerMissed={() => onSelect(null)}>
        <color attach="background" args={['#0d151a']} />
        <Rig inst={inst} cols={cols} assetNodes={assetNodes} assetUrl={assetUrl} selected={selected} onSelect={onSelect}
          onMove={onMove} onInstanceMove={onInstanceMove} showGrid={showGrid} gizmo={gizmo} />
      </Canvas>
    </div>
  )
}
