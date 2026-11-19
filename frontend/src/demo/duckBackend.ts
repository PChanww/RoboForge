import { api } from '../api'
import type { ProjectConfig } from '../types'
import type { DemoAsset } from './backend'

type Template = { config: ProjectConfig; assets: DemoAsset[] }
export const defaultChassis = 'imported/four-wheel-robot-chassis-82e8e5'
async function template(seed = 0, asset?: string, chassis = defaultChassis): Promise<Template> {
  const query = new URLSearchParams({ seed: String(seed), chassis })
  if (asset) query.set('asset', asset)
  const response = await fetch(`/api/v1/duck-demo?${query}`)
  if (!response.ok) throw new Error(`鸭子场景不可用：${await response.text()}`)
  return response.json()
}

export async function listDuckAssets() {
  return (await duckCatalog()).objects
}
export async function duckCatalog(): Promise<{ objects: DemoAsset[]; chassis: DemoAsset[] }> {
  const response = await fetch('/api/v1/duck-demo/assets')
  if (!response.ok) throw new Error('生成模型目录读取失败')
  return response.json()
}
const projects = new Map<string, Promise<ProjectConfig>>()
async function prepare(asset: string, seed: number, chassis: string) {
  const key = `tracking-v2:${asset}:${chassis}:${seed}`
  let pending = projects.get(key)
  if (!pending) {
    pending = (async () => {
      const data = await template(seed, asset, chassis)
      if (data.config.task.asset !== asset) throw new Error('鸭子资产不匹配')
      const id = 'duck_demo_' + crypto.randomUUID().replace(/-/g, '').slice(0, 12)
      data.config.project_id = id
      return api.saveConfig(data.config, id)
    })()
    projects.set(key, pending)
    pending.catch(() => projects.delete(key))
  }
  return pending
}

export async function previewDuck(asset: string, seed: number, chassis = defaultChassis) {
  const project = await prepare(asset, seed, chassis)
  const session = await api.createSession(project.project_id)
  try {
    const images: Record<string, string> = {}
    for (const [id, frame] of Object.entries(session.observation.images)) {
      if (!frame) throw new Error(`相机 ${id} 暂无首帧`)
      images[id] = frame.jpeg_b64
    }
    return images
  } finally { await api.deleteSession(session.id) }
}

export async function collectDucks(asset: DemoAsset, episodes: number, seed: number, chassis = defaultChassis) {
  const project = await prepare(asset.id, seed, chassis)
  return api.createJob('collect', {
    project_id: project.project_id, config_version: project.version,
    dataset_id: 'duck_demo_' + crypto.randomUUID().replace(/-/g, '').slice(0, 12),
    realtime: true, provenance: { asset, chassis, demo: 'duck_conveyor_tracking_v2' },
    plan: Array.from({ length: episodes }, (_, i) => ({ asset: asset.id, seed: seed + i })),
  })
}
