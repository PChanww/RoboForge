// Offscreen splat renderer CLI: serves the real scene.ply + a Three/Spark page, drives
// Chromium via Playwright, renders a specified camera, reads back PNGs.
import http from 'node:http'
import fs from 'node:fs'
import path from 'node:path'
import { chromium } from 'playwright'

const ROOT = path.resolve('.')
const PUB = path.join(ROOT, 'public')
const PLY = path.resolve(ROOT, '..', 'assets', 'scenes', 'world_586c79650960', '1', 'source', 'scene.ply')
const OUT = path.resolve(ROOT, '..', 'outputs', 'phase2-render-validation')
const CAM = path.join(OUT, 'camera.json')

const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.wasm': 'application/wasm',
  '.json': 'application/json', '.png': 'image/png', '.ply': 'application/octet-stream' }

function serve() {
  const server = http.createServer((req, res) => {
    const url = decodeURIComponent(req.url.split('?')[0])
    let file
    if (url === '/' || url === '/index.html') file = path.join(PUB, 'index.html')
    else if (url === '/app.js') file = path.join(PUB, 'app.js')
    else if (url === '/scene.ply') file = PLY
    else if (url.startsWith('/node_modules/')) file = path.join(ROOT, url)
    else { res.writeHead(404); res.end('not found'); return }
    if (!file.startsWith(ROOT) && file !== PLY) { res.writeHead(403); res.end('forbidden'); return }
    fs.readFile(file, (err, data) => {
      if (err) { res.writeHead(404); res.end('missing'); return }
      res.writeHead(200, { 'Content-Type': MIME[path.extname(file)] || 'application/octet-stream',
        'Access-Control-Allow-Origin': '*' })
      res.end(data)
    })
  })
  return new Promise((r) => server.listen(0, '127.0.0.1', () => r(server)))
}

function savePng(dataUrl, name) {
  const b = Buffer.from(dataUrl.split(',')[1], 'base64')
  fs.mkdirSync(OUT, { recursive: true })
  const p = path.join(OUT, name)
  fs.writeFileSync(p, b)
  return { path: p, bytes: b.length }
}

const cam = JSON.parse(fs.readFileSync(CAM, 'utf8'))
const server = await serve()
const port = server.address().port
const url = `http://127.0.0.1:${port}/`
console.log('serving', url, 'ply', fs.existsSync(PLY), fs.existsSync(PLY) ? fs.statSync(PLY).size : 0)

const browser = await chromium.launch({
  headless: true,
  args: ['--enable-unsafe-swiftshader', '--use-angle=default', '--ignore-gpu-blocklist'],
})
const page = await browser.newPage({ viewport: { width: cam.width, height: cam.height }, deviceScaleFactor: 1 })
page.on('console', (m) => console.log('PAGE', m.type(), m.text()))
page.on('pageerror', (e) => console.log('PAGEERR', e.message))
const t0 = Date.now()
await page.goto(url, { waitUntil: 'load' })
await page.evaluate((wh) => { window.__W = wh.w; window.__H = wh.h }, { w: cam.width, h: cam.height })
await page.waitForFunction(() => window.__ready || window.__error, null, { timeout: 180000 })
const perr = await page.evaluate(() => window.__error || null)
if (perr) throw new Error('spark load failed: ' + perr)
console.log('spark ready in', Date.now() - t0, 'ms')
const gl = await page.evaluate(() => window.__glInfo())
console.log('GL', JSON.stringify(gl))
await page.evaluate((c) => window.__setCam(c), { M: cam.M, P: cam.P })

const t1 = Date.now()
const splatOnly = await page.evaluate(() => window.__render(12))
const renderMs = Date.now() - t1
console.log('splat-only render', renderMs, 'ms', savePng(splatOnly, 'splat-only.png'))

// dynamic box in front of the camera (occlusion test): pos = campos + viewdir*0.5
const M = cam.M
const pos = [M[12], M[13], M[14]]
const view = [-M[8], -M[9], -M[10]]
const vn = Math.hypot(...view) || 1
const boxPos = pos.map((p, i) => p + (view[i] / vn) * 0.5)
const boxM = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, boxPos[0], boxPos[1], boxPos[2], 1]
await page.evaluate((b) => window.__addBox(b), { size: [0.12, 0.12, 0.12], M: boxM })
const t2 = Date.now()
const withCube = await page.evaluate(() => window.__render(12))
console.log('with-cube render', Date.now() - t2, 'ms', savePng(withCube, 'splat-with-cube.png'))

fs.writeFileSync(path.join(OUT, 'render.json'), JSON.stringify({
  gl, splat_ready_ms: t0 === 0 ? null : undefined, render_ms: renderMs,
  splat_only: true, with_cube: true, camera: cam.camera, resolution: [cam.width, cam.height],
  spark: '2.4.0', three: '0.180.0',
}, null, 1))

await browser.close()
server.close()
console.log('done')
