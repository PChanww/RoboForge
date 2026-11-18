import { create } from 'zustand'
import { api } from './api'
import type { Observation, ProjectConfig, Selection } from './types'

interface LogEntry { t: string; msg: string; level: 'info' | 'error' }

interface State {
  config: ProjectConfig | null
  dirty: boolean
  observation: Observation | null
  metrics: any | null
  sessionId: string | null
  token: string | null
  epoch: number
  activeRobotId: string
  selection: Selection
  setRobot: (id: string) => Promise<void>
  running: boolean
  busy: boolean
  error: string | null
  logs: LogEntry[]
  init: (project?: string) => Promise<void>
  select: (s: Selection) => void
  edit: (fn: (c: ProjectConfig) => void) => void
  save: () => Promise<void>
  reloadSession: () => Promise<void>
  reset: (seed?: number) => Promise<void>
  step: () => Promise<void>
  expertStep: () => Promise<void>
  toggleRun: () => void
  log: (msg: string, level?: 'info' | 'error') => void
}

let stopRunTimer: (() => void) | null = null
let stepping = false
let initQueue = Promise.resolve()
let hideListenerInstalled = false

function startExpertLoop(fn: () => void): () => void {
  if (typeof Worker !== 'undefined' && typeof Blob !== 'undefined') {
    try {
      const code = 'const id = setInterval(() => postMessage(1), 60); self.onmessage = () => clearInterval(id);'
      const url = URL.createObjectURL(new Blob([code], { type: 'application/javascript' }))
      const w = new Worker(url)
      w.onmessage = () => fn()
      return () => {
        w.postMessage('stop')
        w.terminate()
        URL.revokeObjectURL(url)
      }
    } catch {
      // Fall back to main-thread interval when Workers are unavailable
    }
  }
  const id = setInterval(fn, 60)
  return () => clearInterval(id)
}

export const useStore = create<State>((set, get) => ({
  config: null,
  dirty: false,
  observation: null,
  metrics: null,
  sessionId: null,
  token: null,
  epoch: 0,
  activeRobotId: '',
  selection: null,
  running: false,
  busy: false,
  error: null,
  logs: [],

  log: (msg, level = 'info') =>
    set((s) => ({ logs: [...s.logs.slice(-199), { t: new Date().toLocaleTimeString(), msg, level }] })),

  init: async (project) => {
    const load = async () => {
    try {
      set({ busy: true, error: null })
      const config = await api.getConfig(project || get().config?.project_id || 'default')
      const changed = get().config?.project_id !== config.project_id || get().config?.version !== config.version
      set({ config, dirty: false, activeRobotId: config.robots.some(r => r.id === get().activeRobotId) ? get().activeRobotId : config.robots[0].id, selection: null })
      if (changed || !get().sessionId) await get().reloadSession()
      get().log(`已加载配置 v${config.version}，会话就绪`)
      if (!hideListenerInstalled) {
        window.addEventListener('pagehide', () => {
          if (stopRunTimer) { stopRunTimer(); stopRunTimer = null }
          const { sessionId, token } = get()
          if (sessionId && token) {
            fetch(`/api/v1/sessions/${sessionId}/control/release`, {
              method: 'POST', headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ token }), keepalive: true,
            }).catch(() => {})
          }
        })
        if (typeof document !== 'undefined') {
          document.addEventListener('visibilitychange', () => {
            if (get().running) get().expertStep()
          })
        }
        hideListenerInstalled = true
      }
    } catch (e: any) {
      set({ error: e.message })
      get().log(e.message, 'error')
    } finally {
      set({ busy: false })
    }
    }
    initQueue = initQueue.then(load, load)
    await initQueue
  },

  select: (selection) => set({ selection }),

  edit: (fn) => {
    const c = get().config
    if (!c) return
    const copy = structuredClone(c)
    fn(copy)
    set({ config: copy, dirty: true })
  },

  reloadSession: async () => {
    const { sessionId, token } = get()
    if (stopRunTimer) { stopRunTimer(); stopRunTimer = null; set({ running: false }) }
    if (sessionId) {
      if (token) await api.release(sessionId, token).catch(() => {})
      await api.deleteSession(sessionId).catch(() => {})
    }
    const cfg = get().config!
    const robot = cfg.robots.find((r) => r.id === get().activeRobotId) ?? cfg.robots[0]
    const has5R = ['waist', 'shoulder', 'elbow', 'wrist_pitch', 'wrist_yaw'].every((n) => robot.joints.some((j) => j.name === n))
    const { id, observation, epoch } = await api.createSession(cfg.project_id, robot.id, has5R ? 'cartesian' : 'joints')
    const { token: tok } = await api.acquire(id, 'manual')
    set({ sessionId: id, token: tok, epoch, observation, metrics: null })
  },

  setRobot: async (id) => {
    set({ activeRobotId: id, selection: null })
    await get().reloadSession()
    get().log(`切换活动机器人 → ${id}`)
  },

  save: async () => {
    const c = get().config
    if (!c) return
    try {
      set({ busy: true, error: null })
      const v = await api.validateConfig(c)
      if (!v.valid) {
        set({ error: v.issues.join('; ') })
        get().log('校验失败: ' + v.issues.join('; '), 'error')
        return
      }
      const saved = await api.saveConfig(c, c.project_id)
      set({ config: saved, dirty: false })
      await get().reloadSession()
      get().log(`已保存并应用配置 v${saved.version}`)
    } catch (e: any) {
      set({ error: e.message })
      get().log(e.message, 'error')
    } finally {
      set({ busy: false })
    }
  },

  reset: async (seed) => {
    const { sessionId } = get()
    if (!sessionId) return
    try {
      const r = await api.reset(sessionId, seed)
      const res = await api.result(sessionId)
      set({ observation: r.observation, epoch: r.epoch, metrics: res.metrics })
      get().log(`重置 episode（seed=${r.observation.seed}, epoch=${r.epoch}）`)
    } catch (e: any) { get().log(e.message, 'error') }
  },

  step: async () => {
    const { sessionId, token } = get()
    if (!sessionId || !token || stepping) return
    stepping = true
    try {
      const obs = await api.action(sessionId, token, [0, 0, 0, 1], get().epoch)
      const res = await api.result(sessionId)
      set({ observation: obs, metrics: res.metrics })
    } catch (e: any) { get().log(e.message, 'error') }
    finally { stepping = false }
  },

  expertStep: async () => {
    const { sessionId, token } = get()
    if (!sessionId || !token || stepping) return
    stepping = true
    try {
      const steps = typeof document !== 'undefined' && document.hidden ? 10 : 2
      const obs = await api.expertStep(sessionId, token, get().epoch, steps)
      const metrics = obs.metrics ?? (await api.result(sessionId)).metrics
      set({ observation: obs, metrics })
      if (metrics?.success) { get().log('抓放成功（闭环伺服专家）'); if (get().running) get().toggleRun() }
    } catch (e: any) { get().log(e.message, 'error'); if (get().running) get().toggleRun() }
    finally { stepping = false }
  },

  toggleRun: () => {
    const { running } = get()
    if (running) {
      if (stopRunTimer) stopRunTimer()
      stopRunTimer = null
      set({ running: false })
      get().log('暂停运行')
    } else {
      set({ running: true })
      get().log('开始运行脚本专家')
      stopRunTimer = startExpertLoop(() => { get().expertStep() })
    }
  },
}))

export const activeRobot = (s: State) =>
  s.config ? (s.config.robots.find((r) => r.id === s.activeRobotId) ?? s.config.robots[0]) : null

if (import.meta.env.DEV) (window as any).__vlabStore = useStore
