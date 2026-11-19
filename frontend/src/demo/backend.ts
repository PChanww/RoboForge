import { api } from '../api'
import type { ProjectConfig } from '../types'

export type DemoAsset = { id: string; name: string; source: string; generation_id: string; height_m: number; chassis_generation_id?: string }

// These are the two existing imported assets verified for this small grasp demo.
// Names are display labels; mesh, dimensions, mass and origin stay on the server.
const labels: Record<string, string> = {
  'imported/lux3d-blue-live-5887561': '蓝色杯 · Aholo',
  'imported/lux3d-ivory': '陶瓷杯 · Aholo',
}

export async function listDemoAssets(): Promise<DemoAsset[]> {
  const library = await api.assetLibrary()
  return library.assets
    .filter(item => labels[item.id] && item.source === 'lux3d' && item.generation_id)
    .map(item => ({ id: item.id, name: labels[item.id], source: item.source,
      generation_id: String(item.generation_id), height_m: item.height_m }))
}

const projects = new Map<string, Promise<ProjectConfig>>()

async function prepareProject(asset: string, seed: number) {
  if (!labels[asset]) throw new Error('请选择当前 Demo 支持的后端 Aholo 物品。')
  const key = `${asset}:${seed}`
  let pending = projects.get(key)
  if (!pending) {
    pending = (async () => {
      const config = await api.getConfig()
      const projectId = 'aholo_demo_' + crypto.randomUUID().replace(/-/g, '').slice(0, 12)
      config.project_id = projectId; config.version = 1
      Object.assign(config.scene, { seed, environment: null, sensor_renderer: 'mujoco',
        use_default_table: true, colliders: [], table_color_rgb: [.29, .37, .39] })
      Object.assign(config.task, { id: 'cup_pick_place', plugin: 'cup_pick_place', asset,
        object_color_rgb: null, time_limit_s: 10, goal_xy: [.135, .065], goal_surface_m: .009,
        spawn_min: [-.145, -.065], spawn_max: [-.10, -.035],
        instruction: `将${labels[asset].split(' · ')[0]}放入右侧托盘` })
      for (const sensor of config.sensors) {
        if (sensor.id !== 'front') continue
        const scale = 384 / sensor.resolution[1]
        sensor.resolution = [512, 384]
        sensor.intrinsics.fx *= scale; sensor.intrinsics.fy *= scale
        sensor.intrinsics.cx = 256; sensor.intrinsics.cy = 192
      }
      // The existing config endpoint validates the real physics scene before
      // saving a new project. Never PUT to the user's default project.
      return api.saveConfig(config, projectId)
    })()
    projects.set(key, pending)
    pending.catch(() => { projects.delete(key) })
  }
  return pending
}

export async function previewDemo(asset: string, seed: number) {
  const project = await prepareProject(asset, seed)
  const session = await api.createSession(project.project_id)
  try {
    const images: Record<string, string> = {}
    for (const [camera, frame] of Object.entries(session.observation.images)) {
      if (!frame) throw new Error('后端相机首帧不可用，请重试。')
      images[camera] = frame.jpeg_b64
    }
    if (!images.front || !images.wrist) throw new Error('后端未返回完整相机画面。')
    return images
  } finally {
    await api.deleteSession(session.id)
  }
}

export async function collectDemo(asset: DemoAsset, episodes: number, seed: number) {
  const project = await prepareProject(asset.id, seed)
  return api.createJob('collect', {
    project_id: project.project_id, config_version: project.version,
    dataset_id: 'aholo_demo_' + crypto.randomUUID().replace(/-/g, '').slice(0, 12), realtime: true,
    provenance: { asset, demo: 'aholo_backend_viewer_v1' },
    plan: Array.from({ length: episodes }, (_, i) => ({ asset: asset.id, seed: seed + i })),
  })
}
