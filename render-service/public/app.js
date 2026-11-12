import * as THREE from 'three'
import { SplatMesh, SparkRenderer } from '@sparkjsdev/spark'

const W = window.__W || 256
const H = window.__H || 256
const canvas = document.getElementById('c')
canvas.width = W
canvas.height = H
const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, preserveDrawingBuffer: true, alpha: false })
renderer.setSize(W, H, false)
renderer.setPixelRatio(1)

const scene = new THREE.Scene()
const spark = new SparkRenderer({ renderer })
scene.add(spark)
scene.add(new THREE.AmbientLight(0xffffff, 1.0))

const camera = new THREE.PerspectiveCamera(50, W / H, 0.01, 100)
camera.matrixAutoUpdate = false
camera.matrixWorldAutoUpdate = false

const mesh = new SplatMesh({ url: '/scene.ply' })
scene.add(mesh)

window.__setCam = (cam) => {
  camera.projectionMatrix.fromArray(cam.P)
  camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert()
  camera.matrixWorld.fromArray(cam.M)
  camera.matrixWorldInverse.copy(camera.matrixWorld).invert()
}

window.__addBox = (b) => {
  const geo = new THREE.BoxGeometry(b.size[0], b.size[1], b.size[2])
  const mat = new THREE.MeshStandardMaterial({ color: 0xff5533, roughness: 0.6 })
  const box = new THREE.Mesh(geo, mat)
  box.matrixAutoUpdate = false
  box.matrix.copy(new THREE.Matrix4().fromArray(b.M))
  scene.add(box)
}

window.__addPlane = () => {
  const geo = new THREE.PlaneGeometry(2, 2)
  const mat = new THREE.MeshStandardMaterial({ color: 0x3366cc, side: THREE.DoubleSide })
  const p = new THREE.Mesh(geo, mat)
  p.matrixAutoUpdate = false
  const M = new THREE.Matrix4()
  M.makeTranslation(0, 0, -0.2)
  p.matrix.copy(M)
  scene.add(p)
}


window.__render = async (nframes = 12) => {
  // synchronous renders (no rAF: background tabs throttle requestAnimationFrame)
  for (let i = 0; i < nframes; i++) renderer.render(scene, camera)
  const gl = renderer.getContext()
  const px = new Uint8Array(W * H * 4)
  gl.readPixels(0, 0, W, H, gl.RGBA, gl.UNSIGNED_BYTE, px)
  const cv = document.createElement('canvas')
  cv.width = W
  cv.height = H
  const ctx = cv.getContext('2d')
  const img = ctx.createImageData(W, H)
  for (let y = 0; y < H; y++) {
    const src = (H - 1 - y) * W * 4  // GL origin is bottom-left; flip to top-left
    for (let x = 0; x < W; x++) {
      const di = (y * W + x) * 4
      const si = src + x * 4
      img.data[di] = px[si]
      img.data[di + 1] = px[si + 1]
      img.data[di + 2] = px[si + 2]
      img.data[di + 3] = 255
    }
  }
  ctx.putImageData(img, 0, 0)
  return cv.toDataURL('image/png')
}

window.__glInfo = () => {
  const gl = renderer.getContext()
  return { renderer: gl.getParameter(gl.RENDERER), version: gl.getParameter(gl.VERSION) }
}

window.__save = async (name, nframes = 12) => {
  const d = await window.__render(nframes)
  const r = await fetch('/save?name=' + encodeURIComponent(name), { method: 'POST', body: d })
  return await r.json()
}

// Look-at camera in Z-up world; bbox = {min:[3], max:[3]}
window.__autoCam = (bbox, dist = 6.0, lift = 1.2) => {
  const c = [(bbox.min[0] + bbox.max[0]) / 2, (bbox.min[1] + bbox.max[1]) / 2, (bbox.min[2] + bbox.max[2]) / 2]
  const pos = [c[0], c[1] - dist, c[2] + lift]
  const fwd = [c[0] - pos[0], c[1] - pos[1], c[2] - pos[2]]
  const fn = Math.hypot(...fwd) || 1
  const f = fwd.map((v) => v / fn)
  const upW = [0, 0, 1]
  let right = [f[1] * upW[2] - f[2] * upW[1], f[2] * upW[0] - f[0] * upW[2], f[0] * upW[1] - f[1] * upW[0]]
  const rn = Math.hypot(...right) || 1
  right = right.map((v) => v / rn)
  const up = [right[1] * f[2] - right[2] * f[1], right[2] * f[0] - right[0] * f[2], right[0] * f[1] - right[1] * f[0]]
  // camera local axes in world: X=right, Y=up, Z=-forward
  const M = [right[0], right[1], right[2], 0, up[0], up[1], up[2], 0,
    -f[0], -f[1], -f[2], 0, pos[0], pos[1], pos[2], 1]
  const fov = 50 * Math.PI / 180
  const fy = (W / 2) / Math.tan(fov / 2)
  const P = [2 * fy / W, 0, 0, 0, 0, 2 * fy / H, 0, 0, 0, 0, -1.002, -1, 0, 0, -0.02, 0]
  window.__setCam({ M, P })
  return { center: c, pos }
}

window.__loadCam = async () => {
  const c = await (await fetch('/camera.json')).json()
  window.__setCam(c)
  return { width: c.width, height: c.height }
}

window.__addCubeInFront = async () => {
  const c = await (await fetch('/camera.json')).json()
  const M = c.M
  const pos = [M[12], M[13], M[14]]
  const v = [-M[8], -M[9], -M[10]]
  const n = Math.hypot(...v) || 1
  const bp = pos.map((p, i) => p + (v[i] / n) * 0.5)
  window.__addBox({ size: [0.12, 0.12, 0.12], M: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, bp[0], bp[1], bp[2], 1] })
  return bp
}

mesh.initialized.then(() => {
  window.__ready = true
}).catch((e) => {
  window.__error = String(e)
})
