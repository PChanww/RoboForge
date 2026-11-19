import { create } from 'zustand'
import { api } from './api'
import type { Observation, ProjectConfig, Selection } from './types'

export const OMNI_VLA = 'so100_smolvla_omni'

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
  vlaStatus: string | null
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
  stopRun: () => void
  runVla: (instruction: string) => Promise<void>
  log: (msg: string, level?: 'info' | 'error') => void
}

let runTimer: ReturnType<typeof setInterval> | null = null
let activeVlaRun: object | null = null
let stepping = false
let initQueue = Promise.resolve()
let hideListenerInstalled = false

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
  vlaStatus: null,
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
      if (!hideListenerInstalled) window.addEventListener('pagehide', () => {
        get().stopRun()
        const { sessionId, token } = get()
        if (sessionId && token) {
          fetch(`/api/v1/sessions/${sessionId}/control/release`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ token }), keepalive: true,
          }).catch(() => {})
        }
      })
      hideListenerInstalled = true
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
    get().stopRun()
    set({ vlaStatus: null })
    const cfg = get().config!
    const robot = cfg.robots.find((r) => r.id === get().activeRobotId) ?? cfg.robots[0]
    const { id, observation, epoch } = await api.createSession(
      cfg.project_id, robot.id, undefined, cfg.project_id === 'hackathon_demo',
    )
    let tok: string
    try {
      tok = (await api.acquire(id, 'manual')).token
    } catch (error) {
      await api.deleteSession(id).catch(() => {})
      throw error
    }
    set({ sessionId: id, token: tok, epoch, observation, metrics: null })
    if (sessionId) {
      if (token) await api.release(sessionId, token).catch(() => {})
      await api.deleteSession(sessionId).catch(() => {})
    }
  },

  setRobot: async (id) => {
    const previous = get().activeRobotId
    set({ activeRobotId: id, selection: null })
    try {
      await get().reloadSession()
      get().log(`切换活动机器人 → ${id}`)
    } catch (error) {
      set({ activeRobotId: previous })
      get().log(String(error), 'error')
    }
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
      set({ observation: r.observation, epoch: r.epoch, metrics: res.metrics, vlaStatus: null })
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
      const obs = await api.expertStep(sessionId, token, get().epoch)
      const res = await api.result(sessionId)
      set({ observation: obs, metrics: res.metrics })
      if (res.metrics.success) { get().log('抓放成功（闭环伺服专家）'); if (get().running) get().toggleRun() }
    } catch (e: any) { get().log(e.message, 'error'); if (get().running) get().toggleRun() }
    finally { stepping = false }
  },

  toggleRun: () => {
    const { running } = get()
    if (running) {
      get().stopRun()
    } else {
      if (stepping) return
      set({ running: true, vlaStatus: null })
      get().log('开始运行脚本专家')
      runTimer = setInterval(() => { get().expertStep() }, 60)
    }
  },

  stopRun: () => {
    if (!get().running) return
    if (runTimer) clearInterval(runTimer)
    runTimer = null
    if (activeVlaRun) set({ vlaStatus: 'VLA 已暂停' })
    activeVlaRun = null
    set({ running: false })
    get().log('暂停运行')
  },

  runVla: async (instruction) => {
    const { sessionId, token, epoch, observation, running, busy, dirty } = get()
    if (!instruction.trim() || !sessionId || !token || !observation || running || busy || dirty || stepping) return
    const run = {}
    activeVlaRun = run
    stepping = true
    set({ running: true, vlaStatus: '正在检查指令并准备推理…' })
    try {
      get().log(`VLA · ${instruction}`)
      const firstTick = observation.tick
      const lastTick = firstTick + 160
      let holding = false
      for (let tick = firstTick; tick < lastTick && activeVlaRun === run; ) {
        const steps = Math.min(2, lastTick - tick)
        const sentAt = performance.now()
        const result = await api.policyStep(sessionId, token, epoch, OMNI_VLA, instruction, steps, tick === firstTick)
        if (get().sessionId !== sessionId || get().epoch !== epoch) return
        set({ observation: result, metrics: result.metrics })
        if (activeVlaRun !== run) return
        tick = result.tick
        if (result.instruction_warning) {
          const message = `指令不支持 · ${result.instruction_warning}`
          if (!holding) get().log(message)
          holding = true
          set({ vlaStatus: tick >= lastTick ? message : `${message} 仿真继续运行。` })
          const remaining = steps * result.timing.control_period_s * 1000 - (performance.now() - sentAt)
          if (remaining > 0 && tick < lastTick) await new Promise((resolve) => setTimeout(resolve, remaining))
          continue
        }
        if (result.metrics.success) {
          set({ vlaStatus: 'VLA 搬运成功' })
          get().log(`VLA 搬运成功 · tick ${tick}`)
          return
        }
        set({ vlaStatus: tick >= lastTick ? 'VLA 本轮未完成搬运' : 'VLA 搬运中…' })
      }
    } catch (error) {
      if (activeVlaRun === run) {
        const message = error instanceof Error ? error.message : String(error)
        set({ vlaStatus: `VLA 运行失败 · ${message}` })
        get().log(message, 'error')
      }
    } finally {
      stepping = false
      if (activeVlaRun === run) {
        activeVlaRun = null
        set({ running: false })
      }
    }
  },
}))

export const activeRobot = (s: State) =>
  s.config ? (s.config.robots.find((r) => r.id === s.activeRobotId) ?? s.config.robots[0]) : null

if (import.meta.env.DEV) (window as any).__vlabStore = useStore
