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
  createSession: (project = 'default') =>
    req<{ id: string; epoch: number; observation: Observation }>('POST', '/api/v1/sessions', { project }),
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
}