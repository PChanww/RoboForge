import type { Observation, ProjectConfig } from './types'

async function req<T>(method: string, path: string, body?: unknown): Promise<T> {
  const res = await fetch(path, {
    method,
    headers: body !== undefined ? { 'Content-Type': 'application/json' } : undefined,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  })
  if (!res.ok) {
    let detail: any = await res.json().catch(() => ({}))
    const msg = typeof detail.error === 'string' ? detail.error
      : Array.isArray(detail.issues) ? detail.issues.join('; ')
      : JSON.stringify(detail)
    throw new Error(`${res.status}: ${msg}`)
  }
  return res.json() as Promise<T>
}

export const api = {
  capabilities: () => req<any>('GET', '/api/v1/capabilities'),
  getConfig: (project = 'default') => req<ProjectConfig>('GET', `/api/v1/config?project=${project}`),
  saveConfig: (cfg: ProjectConfig, project = 'default') =>
    req<ProjectConfig>('PUT', `/api/v1/config?project=${project}`, cfg),
  validateConfig: (cfg: ProjectConfig) =>
    req<{ valid: boolean; issues: string[] }>('POST', '/api/v1/config/validate', cfg),
  createSession: (project = 'default', robot_id?: string, controller_id?: string) =>
    req<{ id: string; epoch: number; observation: Observation }>('POST', '/api/v1/sessions', { project, robot_id, controller_id }),
  compat: (robot_id?: string, model = 'state') =>
    req<{ compatible: boolean; mismatches: string[]; robot: string }>('GET', `/api/v1/compat?model=${model}${robot_id ? `&robot_id=${robot_id}` : ''}`),
  reset: (sid: string, seed?: number) =>
    req<{ observation: Observation; epoch: number }>('POST', `/api/v1/sessions/${sid}/reset`, { seed }),
  observation: (sid: string) => req<Observation>('GET', `/api/v1/sessions/${sid}/observation`),
  acquire: (sid: string, owner: string) =>
    req<{ token: string }>('POST', `/api/v1/sessions/${sid}/control/acquire`, { owner }),
  release: (sid: string, token: string) =>
    req('POST', `/api/v1/sessions/${sid}/control/release`, { token }),
  action: (sid: string, token: string, action: number[], epoch?: number) =>
    req<Observation>('POST', `/api/v1/sessions/${sid}/action`, { token, action, epoch }),
  expertStep: (sid: string, token: string, epoch?: number) =>
    req<Observation>('POST', `/api/v1/sessions/${sid}/expert_step`, { token, epoch }),
  result: (sid: string) => req<{ observation: Observation; metrics: any }>('GET', `/api/v1/sessions/${sid}/result`),
  deleteSession: (sid: string) => req('DELETE', `/api/v1/sessions/${sid}`),
  // jobs / library / datasets / deployments
  createJob: (kind: string, params: Record<string, unknown> = {}) =>
    req<any>('POST', '/api/v1/jobs', { kind, params }),
  jobs: () => req<{ jobs: any[] }>('GET', '/api/v1/jobs'),
  job: (id: string) => req<any>('GET', `/api/v1/jobs/${id}`),
  jobLogs: (id: string) => req<{ lines: string[] }>('GET', `/api/v1/jobs/${id}/logs`),
  stopJob: (id: string) => req<any>('POST', `/api/v1/jobs/${id}/stop`, {}),
  projects: () => req<{ projects: any[] }>('GET', '/api/v1/projects'),
  datasets: () => req<{ datasets: any[] }>('GET', '/api/v1/datasets'),
  dataset: (id: string) => req<any>('GET', `/api/v1/datasets/${id}`),
  assetLibrary: () => req<{ assets: any[] }>('GET', '/api/v1/asset-library'),
  deployments: () => req<{ deployments: any[] }>('GET', '/api/v1/deployments'),
  generationStatus: () => req<any>('GET', '/api/v1/generation-jobs'),
  sceneAssets: () => req<{ scenes: any[] }>('GET', '/api/v1/scene-assets'),
  sceneAsset: (id: string) => req<any>('GET', `/api/v1/scene-assets/${id}`),
  sceneValidate: (id: string) => req<any>('POST', `/api/v1/scene-assets/${id}/validate`, {}),
  sceneImport: (body: any) => req<any>('POST', '/api/v1/scene-assets/import', body),
  bindScene: (pid: string, body: any) => req<any>('POST', `/api/v1/projects/${pid}/bind-scene`, body),
}