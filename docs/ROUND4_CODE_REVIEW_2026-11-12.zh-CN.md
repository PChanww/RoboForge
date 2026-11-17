# 第四轮代码审查：对话式资产与场景生成

审查日期：2026-11-12。审查对象为当前工作区，包括未提交与未跟踪文件，重点是第四轮新增生成模块及其前端、导入和场景应用路径。本次未修改业务源码，未提交、未推送。

结论：**发现并复现 10 项缺陷（7 项 P1、3 项 P2），目前不能验收“对话到生成到组合场景”的完整闭环。** 现有杯子、参数化代理和 MuJoCo 路径可用，但它们不足以证明新资产和场景生成链路可靠。

## 本轮验证及证据范围

实际执行：

```powershell
# 在 outputs/vla-lab 中执行
& ../../work/venv/Scripts/python.exe -m pytest tests/test_generation_sessions.py tests/test_generation_jobs.py tests/test_scene_recipe.py -q
# 26 passed，39.65s

& ../../work/venv/Scripts/python.exe ../review-round4-20261112/reproduce_review.py
# 10/10 缺陷复现成功，具体结果在同目录 results.json

# 在 frontend 中执行
npm run build
# TypeScript 与 Vite 构建通过；存在既有大包警告（主 JS 约 2.30 MB）
```

本轮没有重跑全套 64 项测试，没有重新训练模型，没有调用真实 LLM 或创建远程付费生成任务。复现使用本地 mock provider、临时数据库/作业目录，以及现有真实杯子 GLB；场景验证实际构建并步进 MuJoCo。

复现脚本与机器可读结果：

- [reproduce_review.py](C:/Users/31295/Documents/Codex/2026-11-09/https-www-aholo3d-cn-hackathon-utm/outputs/review-round4-20261112/reproduce_review.py)
- [results.json](C:/Users/31295/Documents/Codex/2026-11-09/https-www-aholo3d-cn-hackathon-utm/outputs/review-round4-20261112/results.json)

仓库自带配方 API 测试会向 `qa_scene_recipe` 测试项目保存版本及测试会话；新增的缺陷复现使用临时目录隔离，没有更改正式项目配置。测试通过并不能覆盖下列问题。

## F01 · P1：幂等检查不原子，前端也没有保留同一次确认的请求 ID

位置：[object_provider.py:173](C:/Users/31295/Documents/Codex/2026-11-09/https-www-aholo3d-cn-hackathon-utm/outputs/vla-lab/vlab/generation/object_provider.py:173)、[Generation.tsx:250](C:/Users/31295/Documents/Codex/2026-11-09/https-www-aholo3d-cn-hackathon-utm/outputs/vla-lab/frontend/src/components/Generation.tsx:250)。

`submit()` 在锁外查询 client_request_id，之后才加锁创建作业。两个请求可以同时发现“没有任务”，各自生成一个 jid 并调用 provider。线程屏障复现：**同一个 client_request_id 创建 2 个本地作业，provider.submit 调用 2 次**。这在单后端进程内也成立，不依赖多进程部署。

前端每次点击确认都生成新 UUID，没有针对冻结规格保留确认 ID；第一次响应丢失后再次点击，也无法复用后端幂等键。按钮在 busy 时禁用只能减少单页重复操作，不能处理跨标签页、刷新和响应丢失。

修复：将幂等查找与占位创建放入同一临界区或数据库唯一约束；请求 ID 绑定 session、规格及编译版本，前端持久化到确认结果明确为止。相同 ID 携带不同载荷应拒绝，不应静默复用。补并发与响应丢失重试测试。

## F02 · P1：重新生成复用旧目录，第二个任务成功却返回第一次的资产

位置：[orchestrator.py:90](C:/Users/31295/Documents/Codex/2026-11-09/https-www-aholo3d-cn-hackathon-utm/outputs/vla-lab/vlab/generation/orchestrator.py:90)、[object_provider.py:284](C:/Users/31295/Documents/Codex/2026-11-09/https-www-aholo3d-cn-hackathon-utm/outputs/vla-lab/vlab/generation/object_provider.py:284)。

slug 只由名称和 session ID 后缀构成，没有作业/分支/资产版本。用户在同一会话修改外观再生成，远程确实创建新任务，但导入发现旧 manifest 已存在就跳过注册，也不比对 remote_id、spec_hash 或文件 hash。

复现：先生成 mock-1，再修改外观生成 mock-2；两个任务均 succeeded，asset_id 相同，manifest 的 generation_id 仍为 mock-1，spec_hash 与第二个任务不一致。真实服务下会产生新生成费用，但交付旧资产，来源追踪也错误。

修复：每次显式新生成使用不可变资产 revision/唯一输出目录；只有同一作业重试且来源/hash 匹配时允许复用。不能通过覆盖旧目录修复，因为已有项目可能引用旧版本。

## F03 · P1：从工作台创建 World 背景会话后，读取会话返回 500

位置：[api.py:185](C:/Users/31295/Documents/Codex/2026-11-09/https-www-aholo3d-cn-hackathon-utm/outputs/vla-lab/vlab/generation/api.py:185)。

创建接口先持久化默认 task_object/dynamic 会话，再直接赋字符串 kind，没有同步 role 或重新运行 AssetSpec 校验。带初始描述创建 environment_splat 时，存入了 schema 明确禁止的 environment_splat/dynamic 组合；后续 `SessionStore.get()` 重新反序列化触发 ValidationError。

实际 API 复现：POST 返回 200，kind=environment_splat、role=dynamic；紧接着 GET 同一 session 返回 **500**。这正是工作台选择 World 类型、输入需求、进入详情页的正常路径。

此外，不填初始描述时新 kind 没有被保存，刷新会恢复默认类型。应把创建参数一次性校验后写入数据库，为背景/机器人外观设置合法角色，未知类型返回 4xx。已有非法会话需要明确修复或错误提示。

## F04 · P1：SceneRecipe 的 asset_nodes 没有进入项目和仿真，缺失资产也能校验通过

位置：[scene_recipe.py:151](C:/Users/31295/Documents/Codex/2026-11-09/https-www-aholo3d-cn-hackathon-utm/outputs/vla-lab/vlab/generation/scene_recipe.py:151)、[scene_recipe.py:383](C:/Users/31295/Documents/Codex/2026-11-09/https-www-aholo3d-cn-hackathon-utm/outputs/vla-lab/vlab/generation/scene_recipe.py:383)。

配方接受节点、位姿、缩放和依赖，但映射与应用只处理 colliders、单个 background_ref 和 task_asset_ref。独立道具网格根本没有映射到项目场景。节点校验检查依赖结构和 collider_ref，却没有解析节点资产引用、确认文件存在或将节点放入 MuJoCo。

复现：加入 asset_ref=imported/DOES_NOT_EXIST 的静态节点，真实 MuJoCo 校验仍返回 valid=true，只有“无 collider，作为视觉对象”的警告；转换后的项目配置完全没有该节点。这个警告也不准确，因为它连视觉都没有出现。

修复：提供真正的多实例场景配置、资源解析、可视网格装载和编辑器渲染，并将节点位置/缩放及关联代理落地；尚未支持的节点类型应明确拒绝应用。补“生成静态道具→配方→保存读回→MuJoCo 网格/位姿存在”的测试，不能只检查代理数量。

这是本轮场景资产生成的主要功能缺口：目前“工作台+工具箱”的验证主要是两个手工 box 代理，并未证明生成的工具箱网格能够自动进入场景。

## F05 · P1：通用导入忽略宽深尺寸，却标记为已标定且来自用户

位置：[import_object.py:55](C:/Users/31295/Documents/Codex/2026-11-09/https-www-aholo3d-cn-hackathon-utm/outputs/vla-lab/asset_pipeline/import_object.py:55)、[import_object.py:70](C:/Users/31295/Documents/Codex/2026-11-09/https-www-aholo3d-cn-hackathon-utm/outputs/vla-lab/asset_pipeline/import_object.py:70)。

函数接收 width/depth，但仅按 height 等比缩放；宽深只进入 manifest，并未用于实际几何或碰撞。只要传入宽或深，box 代理还被标记为 basis=user_explicit。

使用真实杯子 GLB 作为受控输入，要求托盘 0.20×0.15×0.02 m，得到的代理约为 **0.01436×0.01414×0.02 m**；manifest 同时保存用户要求的宽深并声明 calibrated=true。模型、碰撞和规格因此有三个不同含义。

这不是要求生成器保证形状，而是导入后程序必须处理尺寸约束。修复应提供清晰缩放策略：按单轴等比、明确允许的 bbox 非等比、或报告剩余尺寸不满足。未满足的用户硬约束不能标记为已标定，代理来源必须反映真实计算方法。补实际 bbox 与碰撞尺寸的数值断言。

## F06 · P1：确认生成丢弃用户审核过的提示词，重新编译另一份请求

位置：[orchestrator.py:40](C:/Users/31295/Documents/Codex/2026-11-09/https-www-aholo3d-cn-hackathon-utm/outputs/vla-lab/vlab/generation/orchestrator.py:40)。

编译预览允许 prompt_override 并保存至 session.compiled，但 confirm 不使用该结果，而是再次调用 `compile_session(s, provider=provider)`。因此用户编辑、重新编译并审核的文字在实际生成时被替换。

复现：预览保存“blue cup + CUSTOM_REVIEW_DETAIL”，confirm 后作业里的 prompt 不包含这两项，退回默认 cup 描述。现有 spec_hash 也不包含 override/payload，所以不能证明预览与实际请求一致。

修复：确认时要求明确的 compiled revision/request hash，检查规格尚未变化，然后冻结并提交审核过的 payload；规格变化应要求重新编译。提示词修改与尺寸/否定硬约束冲突时必须校验，而不是通过确认时悄悄丢弃修改规避问题。

## F07 · P1：点击澄清问题答案后，答案既不更新规格，也不会传给下一次语言模型

位置：[api.py:248](C:/Users/31295/Documents/Codex/2026-11-09/https-www-aholo3d-cn-hackathon-utm/outputs/vla-lab/vlab/generation/api.py:248)、[planner.py:94](C:/Users/31295/Documents/Codex/2026-11-09/https-www-aholo3d-cn-hackathon-utm/outputs/vla-lab/vlab/generation/planner.py:94)。

answer 接口只写 questions[].answer 并可能把状态改成 ready。它不追加对话消息、不运行规划合并，也没有字段映射。Planner.plan 又只发送文本消息和 asset_spec，没有把 questions/answers 放入上下文。

复现：回答宽度“20 cm”，question 显示 answered，session 显示 ready，但 width.value_m 仍为 null；捕获下一次规划请求也找不到“20 cm”。用户正常点击问题选项无法真正补齐需求。

修复：把问题及答案作为持久化的用户输入交给规划器或有类型的字段更新路径，完成校验后更新规格及用户来源，再判断是否就绪。应支持自由文本答案，当前没有 options 的问题只给“是/否”，无法回答尺寸等开放问题。

## F08 · P2：submit_unknown 无法在原作业上绑定找回的 remote_id

位置：[orchestrator.py:48](C:/Users/31295/Documents/Codex/2026-11-09/https-www-aholo3d-cn-hackathon-utm/outputs/vla-lab/vlab/generation/orchestrator.py:48)。

创建超时后，使用相同 request ID 并填 remote_id 再确认，会因找到旧任务直接返回，不采用找回的 ID。poll 与 retry 又要求作业已有 remote_id，所以原任务无法前进。

复现：原任务 submit_unknown；带 known-remote 再确认后仍是同一任务、remote_id=null、submit_unknown。前端换 UUID 可以绕成新本地任务，但这不是恢复原任务，会遗留未知任务并丢失关联。

修复：增加受校验的 recover/attach_remote 操作，只允许在合适状态绑定已知 ID，保持原作业、规格、request ID 与审计记录；不重新提交远程生成。远程任务类型/来源不匹配应拒绝。

## F09 · P2：下载失败残留文件被当作完整缓存，重试下载仍不会下载

位置：[object_provider.py:269](C:/Users/31295/Documents/Codex/2026-11-09/https-www-aholo3d-cn-hackathon-utm/outputs/vla-lab/vlab/generation/object_provider.py:269)。

下载函数直接写 result.glb，异常可能留下截断文件。后续下载节点只要发现 result.* 就复用，不检查上次节点是否完成、文件签名/完整性或可信 hash。显式 retry download 也走同一逻辑。

复现：放入模拟断流残留文件后 poll，再显式重试 download，provider.download 调用次数仍为 **0**，任务一直在导入阶段报 GLB 无效。

修复：下载使用 .part 并在完整校验后原子改名；仅复用已完成节点且 hash 匹配的缓存，显式重新下载能够废弃损坏产物，清除过期下游状态。不要将“对截断文件现算 hash”当成完整性证明。

## F10 · P2：语言模型可以覆盖用户已确认的摩擦参数

位置：[planner.py:174](C:/Users/31295/Documents/Codex/2026-11-09/https-www-aholo3d-cn-hackathon-utm/outputs/vla-lab/vlab/generation/planner.py:174)。

尺寸与质量合并使用硬约束保护，但 friction 没有同样检查，直接赋模型建议并覆盖 provenance。

复现：用户确认 [0.95, 0.005, 0.0001] 后，模型建议 [0,0,0]，结果直接变成后者并标记 model_inferred。另一个问题是这里直接修改已存在的 Pydantic 对象，没有对整个 PhysicsSpec 重新校验，非法长度/数值也没有统一受控处理。

修复：所有物理字段使用一致的硬约束保护和候选整体校验；模型建议只能进入待接受的变更。补已确认摩擦保留、非法摩擦拒绝，以及模型失败时会话不丢失用户消息的测试。

## 验收与修复顺序

1. 先修 F01/F02/F06：防止重复付费、旧资产冒充新结果，以及审核内容与提交内容不一致。
2. 修 F03/F07：恢复场景会话创建及对话答案闭环。
3. 修 F04/F05：生成资产真正进入多实例场景，且尺寸/碰撞满足声明。此处完成前，不应把组合生成描述为完整交付。
4. 修 F08/F09/F10：恢复流程与物理参数保护。
5. 将每个复现转成隔离回归测试，再跑现有 26 项、相关渲染/场景回归和前端构建。

下一轮最小可验收演示：对话或表单确定一个托盘/静态道具规格 → 审核实际请求 → 生成或恢复任务 → 独立资产 revision 入库 → 资产节点真正进入项目配置和 MuJoCo → 核对尺寸及碰撞 → 修改外观再生成，确认新旧资产能同时存在且来源/hash 对应。

在线 LLM 与真实新生成尚未在本次审查调用；修好本地逻辑后再分层验证。当前既有专家/采集成功，只能证明那个项目版本的杯子任务与手工代理成立，不能替代上述生成链路的验收。
