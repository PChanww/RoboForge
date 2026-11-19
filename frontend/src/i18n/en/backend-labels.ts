/**
 * English copy for labels the backend returns as Chinese text.
 *
 * These are data, not UI copy: the API sends finished labels for phases,
 * subtasks and verdicts. They are translated at the render site and fall back to
 * whatever the backend produced, so a new backend label never breaks the page.
 */
export const BACKEND_LABELS: Record<string, string> = {
  // Phases
  '持续输送': 'Conveying',
  '追踪目标': 'Tracking target',
  '跟随接近': 'Approach while following',
  '跟随下降': 'Descend while following',
  '跟随夹紧': 'Close jaws while following',
  '确认抬升': 'Confirm lift',
  '收拢准备搬运': 'Tuck to transport',
  '导航搬运': 'Navigate with payload',
  '到位转向并停车': 'Align to bin and stop',
  '越过目标箱沿': 'Cross the bin rim',
  '箱内下降': 'Descend into bin',
  '松爪释放': 'Release jaws',
  '撤离并检查': 'Retract and verify',
  '抓取或搬运失效': 'Grasp or transport failed',

  // Subtask labels
  '传送带抓取': 'Pick from belt',
  '移动到箱旁': 'Move to bin',
  '稳定入箱': 'Stable placement',
  '从箱内重新抓取': 'Regrasp from bin',
  '抬出箱沿并保持夹持': 'Lift above rim and hold',
  '沿原路线返回': 'Return along route',
  '放回传送带': 'Place back on belt',
  '拿起放回': 'Pick and return',

  // Task kinds
  '送入箱子': 'Deliver into bin',
  '入箱再取出': 'Bin then retrieve',
  '入箱取出再返回': 'Bin, retrieve, return',

  // Verdict labels
  '本次专家基线通过': 'Expert baseline passed for this run',
  '本次验证未通过': 'This validation did not pass',
  '验证未完成': 'Validation not completed',
  '数据无效': 'Data invalid',
  '等待结果': 'Awaiting result',

  // Run statuses
  '通过': 'Passed',
  '未通过': 'Not passed',
  '运行中': 'Running',
  '排队中': 'Queued',
  '生成资产': 'Generating assets',
  '待确认资产': 'Awaiting asset review',
  '正在停止': 'Stopping',
  '已停止': 'Stopped',
  '已中断': 'Interrupted',
}
