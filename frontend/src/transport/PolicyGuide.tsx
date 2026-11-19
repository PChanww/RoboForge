import './policy-guide.css'

const models = [
  { id: 'expert', name: '状态专家基线', inputs: '仿真真值与预设控制逻辑', use: '第一次测试场景或采集训练示范时选它。可以先确认任务、物体和目标箱设置是否合理。' },
  { id: 'smolvla', name: 'SmolVLA · 移动动作头', inputs: '双相机画面、任务文字、机器人状态和任务参数', use: '用来试验从传送带抓取到移动入箱的完整任务。输出机械臂、夹爪和底盘的 7 维动作。' },
  { id: 'visual_bc', name: '移动双相机 BC', inputs: '双相机画面、机器人状态和选定任务参数', use: '模仿成功示范中的动作。与状态 BC 使用同样任务配置，比较加入视觉画面后的效果。' },
  { id: 'state_bc', name: '移动状态 BC', inputs: '机器人状态和选定任务参数，不使用相机', use: '作为行为克隆对照，检查只靠机器人状态和任务参数能做到哪一步。' },
  { id: 'privileged_bc', name: '移动特权 BC', inputs: '机器人状态、任务参数和物体真实位姿与速度', use: '仅用于仿真诊断。即使策略能读取物体真值，也要用闭环评测确认是否能完成任务。' },
  { id: 'smolvla-libero', name: 'SmolVLA · LIBERO 实验适配', inputs: '双相机画面、任务文字和机械臂状态', use: '适合测试机械臂和夹爪响应。底盘指令为零，跨场景抓取也需要实测。' },
] as const

export default function PolicyGuide({ current }: { current: string | undefined }) {
  return (
    <section className="tt-card tt-policy-guide" aria-labelledby="tt-policy-guide-title">
      <h2 id="tt-policy-guide-title">策略怎么选</h2>
      <p className="tt-policy-guide__steps">选好策略后，回到「任务要求」先检查方案，再启动仿真。任务是否完成以「可行性结果」为准。</p>
      <div className="tt-policy-guide__grid">
        {models.map(model => <article key={model.id} className={model.id === current ? 'is-current' : undefined}>
          <h3>{model.name}{model.id === current && <span>当前策略</span>}</h3>
          <p className="tt-policy-guide__input">{model.inputs}</p>
          <p>{model.use}</p>
        </article>)}
      </div>
    </section>
  )
}
