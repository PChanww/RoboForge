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
  expertStep: (sid: string, token: string, epoch?: number, steps = 1) =>
    req<Observation & { metrics?: any }>('POST', `/api/v1/sessions/${sid}/expert_step`, { token, epoch, steps }),
  policyStep: (sid: string, token: string, model_id: string, epoch?: number, steps = 1, instruction?: string) =>
    req<Observation & { metrics?: any }>('POST', `/api/v1/sessions/${sid}/policy_step`, { token, model_id, epoch, steps, instruction }),
  models: () => req<{ models: any[] }>('GET', '/api/v1/models'),
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
  datasetPreviewFrames: (id: string, episode = 0) => req<any>('GET', `/api/v1/datasets/${encodeURIComponent(id)}/preview-frames?episode=${episode}`),
  assetLibrary: () => req<{ assets: any[] }>('GET', '/api/v1/asset-library'),
  deployments: () => req<{ deployments: any[] }>('GET', '/api/v1/deployments'),
  generationStatus: () => req<any>('GET', '/api/v1/generation-jobs'),
  sceneAssets: () => req<{ scenes: any[] }>('GET', '/api/v1/scene-assets'),
  sceneAsset: (id: string, revision?: number) => req<any>('GET', `/api/v1/scene-assets/${encodeURIComponent(id)}${revision ? '?revision='+revision : ''}`),
  scenePointCloud: (id: string, revision = 1, maxPoints = 24000) =>
    req<any>('GET', `/api/v1/scene-assets/${encodeURIComponent(id)}/revisions/${revision}/point-cloud?max_points=${maxPoints}`),
  sceneValidate: (id: string, revision?: number) => req<any>('POST', `/api/v1/scene-assets/${encodeURIComponent(id)}/validate${revision ? '?revision='+revision : ''}`, {}),
  sceneImport: (body: any) => req<any>('POST', '/api/v1/scene-assets/import', body),
  bindScene: (pid: string, body: any) => req<any>('POST', `/api/v1/projects/${pid}/bind-scene`, body),
  generateScene: (params: Record<string, unknown>) => req<any>('POST', '/api/v1/generation-jobs', {kind: 'generation', params}),
  pollGeneration: (id: string) => req<any>('POST', `/api/v1/generation-jobs/${id}/poll`, {}),
  getScene: (pid: string) => req<any>('GET', `/api/v1/projects/${pid}/scene`),
  commitScene: (pid: string, body: any) => req<any>('PUT', `/api/v1/projects/${pid}/scene`, body),
  setSceneInstance: (pid: string, body: any) => req<any>('PUT', `/api/v1/projects/${pid}/scene/instance`, body),
  setColliders: (pid: string, colliders: any[]) => req<any>('PUT', `/api/v1/projects/${pid}/scene/colliders`, { colliders }),
  lux3dStatus: () => req<any>('GET', '/api/v1/lux3d/status'),
  lux3dGenerate: (body: any) => req<any>('POST', '/api/v1/lux3d/generate', body),
  deploymentDetail: (name: string) => req<any>('GET', `/api/v1/deployments/${name}`),
  deploymentCompat: (name: string, robotId: string, project = 'default') =>
    req<any>('GET', `/api/v1/deployments/${name}/compat?robot_id=${robotId}&project=${project}`),
  // generation loop (round 4)
  genCapabilities: () => req<any>('GET', '/api/v1/generation/capabilities'),
  genRemoteTasks: (provider = 'lux3d', page = 1, pageSize = 20) =>
    req<any>('GET', `/api/v1/generation/remote-tasks?provider=${encodeURIComponent(provider)}&page=${page}&page_size=${pageSize}`),
  genSessions: () => req<{ sessions: any[] }>('GET', '/api/v1/generation/sessions'),
  genSession: (id: string) => req<any>('GET', `/api/v1/generation/sessions/${id}`),
  genCreateSession: (body: any) => req<any>('POST', '/api/v1/generation/sessions', body),
  genMessage: (id: string, content: string, expected_revision: number) =>
    req<any>('POST', `/api/v1/generation/sessions/${id}/messages`, { content, expected_revision }),
  genPatchSpec: (id: string, body: any) => req<any>('PATCH', `/api/v1/generation/sessions/${id}/spec`, body),
  genAnswer: (id: string, question_id: string, answer: string, expected_revision: number) =>
    req<any>('POST', `/api/v1/generation/sessions/${id}/answer`, { question_id, answer, expected_revision }),
  genCompile: (id: string, body: any = {}) => req<any>('POST', `/api/v1/generation/sessions/${id}/compile`, body),
  genBranch: (id: string, reason: string, expected_revision: number) =>
    req<any>('POST', `/api/v1/generation/sessions/${id}/branch`, { reason, expected_revision }),
  genConfirm: (id: string, body: any) => req<any>('POST', `/api/v1/generation/sessions/${id}/confirm`, body),
  genTasks: (id: string) => req<any>('GET', `/api/v1/generation/sessions/${id}/tasks`),
  genPoll: (id: string, jobId: string) =>
    req<any>('POST', `/api/v1/generation/sessions/${id}/tasks/${jobId}/poll`, {}),
  genRetryNode: (id: string, jobId: string, node: string) =>
    req<any>('POST', `/api/v1/generation/sessions/${id}/tasks/${jobId}/retry`, { node }),
  genRecoverTask: (id: string, jobId: string, remote_id: string, provider?: string) =>
    req<any>('POST', `/api/v1/generation/sessions/${id}/tasks/${jobId}/recover`, { remote_id, provider }),
  genImportRemote: (id: string, body: { remote_id: string; provider?: string; client_request_id?: string; expected_revision?: number }) =>
    req<any>('POST', `/api/v1/generation/sessions/${id}/import-remote`, body),
  recipeValidate: (id: string, projectId = 'default') => req<any>('POST', `/api/v1/scene-recipes/${id}/validate?project_id=${projectId}`, {}),
  recipeApply: (id: string, body: any) => req<any>('POST', `/api/v1/scene-recipes/${id}/apply`, body),
  recipeFromSession: (id: string, body: any) => req<any>('POST', `/api/v1/generation/sessions/${id}/recipe`, body),
  partSplitAsset: (slug: string, body: { segments?: number; axis?: string; remote?: boolean } = {}) =>
    req<any>('POST', `/api/v1/assets/${encodeURIComponent(slug)}/part-split`, body),
  assembleRobotParts: (body: { project_id?: string; robot_id: string; parts_slug: string; mapping: Record<string, string>; replace_primary?: boolean }) =>
    req<any>('POST', '/api/v1/robots/assemble-parts', body),
  robotUrdfUrl: (projectId = 'default', robotId?: string) =>
    `/api/v1/projects/${encodeURIComponent(projectId)}/robot.urdf${robotId ? `?robot_id=${encodeURIComponent(robotId)}` : ''}`,
  setupHackathonDemo: (body: { project_id?: string; with_world_background?: boolean; support_plane_z?: number } = {}) =>
    req<any>('POST', '/api/v1/hackathon-demo/setup', body),
}
