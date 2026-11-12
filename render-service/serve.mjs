// Static server for the Spark splat page + PNG sink (no browser automation here).
import http from 'node:http'
import fs from 'node:fs'
import path from 'node:path'

const ROOT = path.resolve('.')
const PUB = path.join(ROOT, 'public')
const PLY = path.resolve(ROOT, '..', 'assets', 'scenes', 'world_586c79650960', '1', 'source', 'scene.ply')
const OUT = path.resolve(ROOT, '..', 'outputs', 'phase2-render-validation')
const PORT = Number(process.env.PORT || 8900)
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.wasm': 'application/wasm',
  '.json': 'application/json', '.png': 'image/png', '.ply': 'application/octet-stream' }

const server = http.createServer((req, res) => {
  const u = new URL(req.url, 'http://x')
  if (req.method === 'POST' && u.pathname === '/save') {
    const chunks = []
    req.on('data', (c) => chunks.push(c))
    req.on('end', () => {
      const body = Buffer.concat(chunks).toString('utf8')
      const name = (u.searchParams.get('name') || 'frame.png').replace(/[^a-zA-Z0-9_.-]/g, '_')
      const b64 = body.includes(',') ? body.split(',')[1] : body
      fs.mkdirSync(OUT, { recursive: true })
      fs.writeFileSync(path.join(OUT, name), Buffer.from(b64, 'base64'))
      res.writeHead(200, { 'Content-Type': 'application/json' })
      res.end(JSON.stringify({ saved: name, bytes: Buffer.from(b64, 'base64').length }))
    })
    return
  }
  if (u.pathname === '/health') { res.writeHead(200); res.end('ok'); return }
  let file
  if (u.pathname === '/' || u.pathname === '/index.html') file = path.join(PUB, 'index.html')
  else if (u.pathname === '/app.js') file = path.join(PUB, 'app.js')
  else if (u.pathname === '/scene.ply') file = PLY
  else if (u.pathname === '/camera.json') file = path.join(OUT, 'camera.json')
  else if (u.pathname.startsWith('/node_modules/')) file = path.join(ROOT, u.pathname)
  else { res.writeHead(404); res.end('not found'); return }
  fs.readFile(file, (err, data) => {
    if (err) { res.writeHead(404); res.end('missing'); return }
    res.writeHead(200, { 'Content-Type': MIME[path.extname(file)] || 'application/octet-stream' })
    res.end(data)
  })
})
server.listen(PORT, '127.0.0.1', () => console.log('render page at http://127.0.0.1:' + PORT, 'ply', fs.existsSync(PLY)))