import { Component, type ReactNode, useRef } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { Grid, OrbitControls, RoundedBox } from '@react-three/drei'
import * as THREE from 'three'

class SceneBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false }
  static getDerivedStateFromError() { return { failed: true } }
  render() {
    return this.state.failed ? <div className="rf-scene-fallback">三维预览需要支持 WebGL 的浏览器，采集设置仍可使用。</div> : this.props.children
  }
}

function Segment({ from, to }: { from: THREE.Vector3; to: THREE.Vector3 }) {
  const delta = to.clone().sub(from)
  const rotation = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), delta.clone().normalize())
  return <mesh position={from.clone().add(to).multiplyScalar(.5)} quaternion={rotation} castShadow>
    <boxGeometry args={[.13, delta.length(), .14]} />
    <meshStandardMaterial color="#e5e4df" metalness={.22} roughness={.45} />
  </mesh>
}

function Rig({ running, tall }: { running: boolean; tall: boolean }) {
  const root = useRef<THREE.Group>(null)
  const cup = useRef<THREE.Group>(null)
  const elapsed = useRef(0)
  useFrame((_, dt) => {
    elapsed.current = running ? elapsed.current + dt : 0
    const t = elapsed.current % 6
    if (root.current) root.current.rotation.y = running ? Math.sin(t / 6 * Math.PI * 2) * .45 : .1
    if (cup.current) {
      const lifted = running && t > 1.8 && t < 4.8
      cup.current.position.set(lifted ? -.32 + Math.min(1, (t - 1.8) / 2) * .72 : t >= 4.8 ? .4 : -.32,
        lifted ? .25 + Math.sin((t - 1.8) / 3 * Math.PI) * .3 : .14, .38)
    }
  })
  const p0 = new THREE.Vector3(0, .19, 0), p1 = new THREE.Vector3(.05, .7, 0), p2 = new THREE.Vector3(-.34, .86, .26)
  return <>
    <RoundedBox args={[1.9, .07, 1.3]} radius={.035} position={[0, -.045, .1]} receiveShadow><meshStandardMaterial color="#dfded5" /></RoundedBox>
    <Grid position={[0, -.005, .1]} args={[1.85, 1.25]} cellSize={.1} sectionSize={.5} cellColor="#c2c8be" sectionColor="#aeb9ac" fadeDistance={5} />
    <group position={[.14, .025, -.23]}>
      <mesh castShadow><cylinderGeometry args={[.19, .23, .08, 48]} /><meshStandardMaterial color="#344b42" metalness={.5} roughness={.4} /></mesh>
      <group ref={root}>
        <mesh position={[0, .12, 0]} castShadow><cylinderGeometry args={[.105, .14, .2, 32]} /><meshStandardMaterial color="#e5e4df" /></mesh>
        <Segment from={p0} to={p1} /><Segment from={p1} to={p2} />
        {[p0, p1, p2].map((p, i) => <mesh key={i} position={p} rotation={[Math.PI / 2, 0, 0]} castShadow><cylinderGeometry args={[.095, .095, .17, 32]} /><meshStandardMaterial color="#344b42" metalness={.5} roughness={.3} /></mesh>)}
        <mesh position={[-.34, .73, .26]} castShadow><boxGeometry args={[.1, .2, .1]} /><meshStandardMaterial color="#dedfd9" /></mesh>
        {[-1, 1].map(s => <mesh key={s} position={[-.34 + s * .055, .61, .26]} castShadow><boxGeometry args={[.025, .1, .04]} /><meshStandardMaterial color="#344b42" /></mesh>)}
      </group>
    </group>
    <RoundedBox position={[.4, .025, .38]} args={[.36, .045, .34]} radius={.025} receiveShadow><meshStandardMaterial color="#93a88c" /></RoundedBox>
    <group ref={cup} position={[-.32, .14, .38]}>
      <mesh castShadow><cylinderGeometry args={[.068, .052, tall ? .24 : .18, 40]} /><meshStandardMaterial color="#d88351" roughness={.33} /></mesh>
      <mesh position={[0, tall ? .122 : .092, 0]} rotation={[-Math.PI / 2, 0, 0]}><ringGeometry args={[.05, .069, 40]} /><meshStandardMaterial color="#e9b492" side={THREE.DoubleSide} /></mesh>
      <mesh position={[0, tall ? .119 : .089, 0]} rotation={[-Math.PI / 2, 0, 0]}><circleGeometry args={[.05, 40]} /><meshStandardMaterial color="#7b4a32" /></mesh>
    </group>
    <mesh position={[-.75, .18, -.38]} castShadow><boxGeometry args={[.03, .35, .03]} /><meshStandardMaterial color="#536258" /></mesh>
    <mesh position={[-.75, .38, -.38]} rotation={[.2, .5, 0]} castShadow><boxGeometry args={[.16, .09, .08]} /><meshStandardMaterial color="#344b42" /></mesh>
    <mesh position={[0, -.13, 0]} rotation={[-Math.PI / 2, 0, 0]} receiveShadow><planeGeometry args={[200, 200]} /><meshStandardMaterial color="#edeee7" /></mesh>
  </>
}

export default function CollectionScene({ running, tall }: { running: boolean; tall: boolean }) {
  return <SceneBoundary><Canvas shadows dpr={[1, 1.75]} camera={{ position: [2.2, 1.65, 2.4], fov: 36 }}>
    <color attach="background" args={['#edeee7']} />
    <ambientLight intensity={1.7} />
    <directionalLight position={[-2, 4, 2]} intensity={3} castShadow shadow-mapSize={[2048, 2048]} shadow-normalBias={.025} />
    <Rig running={running} tall={tall} />
    <OrbitControls target={[0, .3, .1]} minDistance={1.7} maxDistance={5} maxPolarAngle={Math.PI / 2.15} enablePan={false} />
  </Canvas></SceneBoundary>
}
