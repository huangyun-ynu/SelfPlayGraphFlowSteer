# SWE Director 上下文与执行链路完整修复方案

日期：2026-09-27。状态：完整设计方案；已实施下述关系审计隔离、[三项 SWE 工程修复](SWE_ENGINEERING_FIXES_20260927.zh-CN.md)、公开功能测试。用户最新选择是[恢复包含 thinking 的增量式上下文](DIRECTOR_INCREMENTAL_CONTEXT_20260927.zh-CN.md)，概率审计仅留离线；历史 thinking 剥离方案已撤回。后文重建快照及其他未明确实施的内容仍是设计方案，未启动新模型实验。

## 0. 已实施：关系审计详情只在离线记录中保留

- 在 `Canvas._graph_state_snapshot()` 生成展示数据时排除 `last_relation_decision.policy`，因此当前快照、反馈及其后续历史都不再重复关系概率、log probability、token ID、模型和请求校验详情。保留关系端点、类型、选择及选择前后的状态；实际边仍由 `actual_relations` 报告。
- 原始 `CanvasStep.relation_decision`、`DirectorTurn.relation_decision` 和 `binary_policy_audit` 保存完整审计内容，事件序列化、行为概率、训练 token 与反事实关系选择继续使用原始数据。
- 当前保留 `snapshot_dedup`、`append_only`，兼容 legacy 与统一提交协议；正式入口默认增量追加 thinking、动作和反馈，关系概率审计仅留离线。第三种 `delta_timeline` 已删除，完整训练 token 与前缀验证均保留。
- 新采集目录的 `director_context_policy.json` 增加 `relation_audit_visibility=offline_only_v1`。已有标记与新策略不一致时拒绝混合续跑，应使用新输出目录；不改写旧实验记录。
- 回归验证：新增 21 项上下文/审计/续跑隔离测试通过，现有 Director、Canvas、rollout、统一提交、SWE 修复及关系反事实相关测试共 160 项通过；另有 6 项真实 tokenizer 测试因本地模型缺失跳过。新测试通过无需模型下载的可逆 token fixture 验证追加模式的真实请求前缀一致性。未开展新 SWE 模型实验，尚不能据此判断解决率或整体上下文超限率的变化。

## 1. 目标、依据和范围

目标是让 Director 在有界输入内持续获得真实、足够的决策信息，同时保证 SWE 的调查、修改、测试、提交链路及后续 RL 数据一致。优先修复执行正确性，再用真实实验判断任务解决率；不承诺仅靠上下文压缩就能解决代码题。

依据：

- 用户提供的 [Self-Play × Graph Multi-Agent FlowSteer 指导文档](/root/.codex/attachments/7d88971c-8020-464e-ba19-466cf638219b/selfplay_graph_multiagent_flowsteer_implementation.md)，已完整阅读。尤其是第 10–18、20–21、32、34、39–43、47–48 节。
- `full-v5` 的 [128 题诊断](../state/audits/unified-protocol-20260926/full-v5/SWE_DIAGNOSIS.zh-CN.md)与[离线核验](../state/audits/unified-protocol-20260926/full-v5/swe-128-diagnosis-verification.json)。历史运行以冻结源码为准，不能用后续工作区变化改写历史。
- [Director 上下文排查](../state/audits/unified-protocol-20260926/DIRECTOR_CONTEXT_EXHAUSTION.zh-CN.md)。
- FlowSteer 官方源码提交 `1c9f2abf55cb9b8ea2ca2e3359cdb91acb9964e9` 的主训练、评测调用链；其主入口每轮重建状态，普通反馈最多 800 字符，向量化评测最多 500 字符，不传完整 history。通用 Builder 的追加历史路径另行区分。

本次设计覆盖 Director 上下文及 SWE 配套执行修复；与 Graph RL、关系反事实、Self-Play 的衔接给出完整约束。不会把修复任务扩展成同时重写整个 Proposer、图核和奖励系统。

## 2. 文档原则与当前已确认协议如何衔接

| 文档要求 | 本方案的落实方式 |
|---|---|
| 一轮一个原子编辑，多轮依反馈编排 | 保持；整理上下文属于请求准备，不产生隐式图编辑 |
| 通用 Agent，职责由 prompt 定义 | 保持；不加入固定 Investigator/Coder/Verifier 流水线 |
| Canvas 只返回事实反馈 | 只展示状态、错误、约束与合法动作，不输出 `NEXT=Add verifier` 等策略建议 |
| 图结构由策略和任务结果学习 | 不增加必须多节点、必须双向、必须 checker 的约束或奖励 |
| 单向依赖、同层双向两阶段执行 | 保持；压缩不能改变信息可见性或同步屏障 |
| Dirty 子图和缓存以真实输入变化为准 | 保持；上下文显示变化不等于 Worker 输入变化 |
| 历史轨迹保留，think/action 参与训练 | 完整实际调用留档；在线只读取有界事实视图 |
| MVP 不引入 memory 系统 | 不增加跨题记忆库、检索技能、学习型摘要器；仅从本题现有事件确定性生成输入 |
| 文档早期动作包含 SET_OUTPUT | 沿用会话后续已确认的 `task_result + finish(target)`；最终提交时仍唯一选择一个目标 |

`result_scope` 是交付范围，不是固定角色库。探索期间允许多个 task_result 或暂时没有 task_result；正式提交时检查目标、真实可达性和产物有效性。

文档的完整历史 H_t 继续保存。在线使用 Z_t = Project(q, G_t, runtime_t, H_t; context_policy_version)。这是一个需要验证的观测设计，不能声称它与读取全部历史的策略在数学上等价。必要信息是否遗漏必须通过回放和对照实验检验。

## 3. 三种数据分别管理

### 3.1 权威运行状态

由 Canvas/Runtime 维护完整真实状态：图、节点配置、实际边、dirty 闭包、候选输入绑定、SWE 工作区与补丁版本、测试证据、预算、在途执行、提交事务。

它是执行和提交检查的唯一依据，不从 Director 的自然语言摘要反推状态。

### 3.2 Director 在线视图

由纯函数从权威状态与本题事件生成，包含决策需要的事实。每次请求重新构造；不把上一轮 prompt 原文或完整 reasoning 追加进来。

只使用当时已公开给本题求解过程的信息。官方隐藏测试、评分金标、未来事件及反事实分支结果不得进入这份视图。

### 3.3 完整审计与训练记录

每次真实模型请求保存：

- 原始 messages、实际 prompt token IDs、chat template/tokenizer 版本。
- 实际生成的 completion token IDs、thinking/action、停止原因、生成参数。
- behavior log probabilities、关系二元支持集与概率。
- 当前图/工作区/候选快照引用、执行结果、工具账本。
- 上下文策略版本、投影输入哈希、输出哈希、纳入和省略的事件引用。

未进入在线视图的数据仍可留在审计记录中。把审计字段移出模型输入不等于删除反事实训练所需概率。

## 4. 每轮 Director 输入的结构

建议新增显式模式 `bounded_snapshot_v1`，名称为待实现配置，不是假称当前已存在。

```text
Protocol and action schema
Task: original public problem
Current authoritative state:
  phase / pending configuration / graph version
  nodes: id, layer, current assigned instruction, result_scope, status
  real directed edges and bidirectional components
  relevant off decisions and their configuration version
  executed / reused / dirty / blocked nodes
Resources:
  edits, shared tool calls, Worker tokens, existing recovery allowance
SWE state:
  workspace version, changed files, patch reference
  latest test profile, scope, exit status, tested workspace version
Candidates:
  artifact, effective input binding, patch status, submit blockers
Last action:
  requested / accepted or rejected / factual effects / exact error code
Recent events and unresolved facts:
  bounded records with provenance
Legal action parameters:
  existing admission results, without recommended topology
```

### 4.1 必须保留的控制信息

- 原始任务；协议；本轮合法动作、目标和必要参数。
- 当前实际图与层级；节点当前职责；配置未完成和 dirty 状态。
- 提交候选、阻塞、输入/工作区/测试版本及真实预算。
- 最新动作的接受/拒绝结果和精确错误码。
- 待决关系的端点、合法 on/off 含义；候选关系不冒充真实边。

这些信息不能通过“保留字符串前 800 字符”处理。若完整节点 instruction 过大，则先删除可选历史、扩展到硬输入预算；仍放不下时明确记录状态过大。不能只向模型展示一个摘要，却声称它看到了完整配置。

### 4.2 可以限长的叙述内容

- Worker 答案和证据的展示预览。
- 已解决的历史错误、较旧的动作结果。
- 重复出现、同来源同版本的工具证据。
- 不再活动节点的历史详情。

预览带来源、完整记录引用、是否截断和省略数量。引用用于可追溯性，不假称 Director 能通过一个引用自动读到未提供内容。

### 4.3 应移出后续在线输入的字段

- 历史完整 reasoning、原始响应全文。
- policy/logprob 明细、tokenizer 证明、HTTP 请求元数据。
- 已被当前权威快照替代的完整旧快照。
- 重复的原始工具输出和全文轨迹。

每轮仍允许 Director 正常推理，其当前实际生成内容照常留档和训练。

## 5. 有界事实历史：保留过程信息，但不增加新 Agent

第一版采用确定性事件选择，不调用额外摘要模型，不维护跨任务经验库。

建议起始设置（均为工程候选参数，需消融，不来自指导文档的硬性规定）：

- 最近事件窗口最多 6 条，同时不超过 2,048 token。
- 仍未解决的历史事实池最多 4,096 token。
- 同源同版本事实去重；重复错误保留错误码、计数、首末事件和当前是否阻塞。
- 优先保留当前候选及其依赖证据、当前失败测试、未解除的阻塞；已解决旧事件先退出在线视图。

预算是软分配，最新控制状态永远优先；总长按实际聊天模板重新 tokenize，不把分段 token 数简单相加作为最终长度。

事实项区分：`runtime_fact`（工具/运行时直接证明）、`agent_claim`（模型分析）、`unresolved`（尚无结论）。不得把 Worker 自述“已经修好”压缩成受信任的测试通过。

同一输入事件序列、状态及策略版本应生成一致视图。时间戳、字典遍历顺序、日志噪声不参与内容选择。改变显示顺序不改变 Agent ID，也不改变图核的置换不变规则。

不保证有限视图能容纳所有历史证据；被省略内容必须可计量。真实实验中若发现遗忘关键约束，则调整事实选择与预算，再单独复测。

## 6. 精确 token 准入与上下文生命周期

保留当前服务的 32,768 token 窗口；第一轮实验不同时更换模型、窗口或 Director 生成预算。

定义：

```text
L_effective = min(serving_context_limit, verified_training_sequence_limit)
B_input = L_effective - requested_generation_reserve - template_safety_margin
```

`requested_generation_reserve` 按当前动作/设 prompt/关系选择/协议修复的实际调用类别计算。不能为了塞进输入把正常输出额度一路压到十几或几百 token。

建议安全余量初值 512 token，普通输入软目标 24,576 token。它们不构成新的 Director 动作额度。训练长度尚未覆盖采集长度时应在启动前校验失败；源码默认值不是正式训练配置的证据。

每次发送之前：

1. 从权威状态重新生成视图。
2. 去掉重复内容和可选历史，按字段优先级缩减预览。
3. 使用实际 tokenizer 与聊天模板计数。
4. 若超过软目标，继续减少已解决历史和旧预览；必要时使用软目标之外、硬预算之内的空间。
5. 确保输入、保留输出和安全余量均满足有效窗口。
6. 控制信息本身仍放不下时，记录 `director_context_state_too_large` 等明确原因，不发送注定超限的请求。

控制信息过大是配置/上下文基础设施问题，不能自动按模型解题失败归零。不得删除任务、图约束或候选版本来勉强提交。

上下文整理只改变下一次 Director 的输入，绝不修改 graph version、Worker input signature、dirty 集合、预算余额、无进展计数、候选资格或 reward。

断点恢复固定 `context_policy_version`、模板哈希、模型/tokenizer、投影参数；版本不一致使用新实验目录。

## 7. 与 masked GRPO 的精确衔接

每次调用分别保留真实序列：

```text
x_t = actual prompt token IDs
y_t = actual sampled completion token IDs
training sequence = x_t || y_t
loss mask = 0 on x_t, 1 on eligible sampled y_t
```

由 Canvas 确定性构造的快照、错误摘要和历史事实全部 mask=0。Director 当前真实生成的 think/action 继续参与训练。一个旧动作出现在下一轮摘要时只是输入，不再次获得 loss。

重要边界：

- 去掉历史思考的后续在线回灌，不等于去掉该思考在原调用中的训练。
- 行为概率必须对应原调用的真实 x_t。不能对压缩前输入采样的动作，换成压缩后输入重新标记 behavior probability。
- 原始 `full-v5` 轨迹仍可做旧策略分析，不能冒充 `bounded_snapshot_v1` 的新采样轨迹。
- 每一次真实补动作调用分别记录；控制器补造的动作不产生策略梯度。
- 缺少真实 token/logprob 证据的调用不能伪造为合格的 on-policy 数据；记录覆盖率及阻塞原因。

当前代码已有逐调用 `policy_calls`，`tokenize_director_policy_calls()` 禁止不连续截断，可以扩展其版本绑定。`timeline` 合并仅在实际 token 前缀完全一致时成立；快照重建通常不满足，应回退 `call`/`micro`，不能强行拼接来节约训练成本。

验收包括初始 behavior/learner 概率校验、mask 对齐、关系 token 定位，以及超长单调用在采集前被处理，而非训练时删去中间 token。

## 8. SWE 配套一：共享预算与重复操作

保持此前约定：Director 编辑 24 次，SWE 整题共享工具 32 次。初始、修订、重跑、节点重建不产生新工具额度。现有 token 与恢复保护照常存在，不另加一个独立 Director 总决策次数上限。

### 8.1 修正跨角色资源保护

把现有修改/测试保留规则从仅 task_result 生效，改为读取同一份整题状态，对所有角色一致生效。

- 当前尚无修改且仍走补丁路径时，保留完成最小 edit+test 所需的 2 次工具。
- 修改后尚未测试时，保留一次测试额度。
- 已有有效成果、或有证据支持的无补丁失败收尾，按其实际所需资源判断。
- 不强迫无依据改文件，不强迫新增特定角色。

这仍是保守的工程保留策略，不是任务成功的数学必要充分条件；会影响末尾探索自由度。必须记录拒绝原因，并单独做保留策略的消融。它是同一账本内的准入规则，不是恢复初始/修订两份预算。

### 8.2 重复只读操作

重复判定前移到环境调用和工具扣额之前。相同工作区/资源版本下、已成功完成过的相同只读请求，返回已有事实引用与重复说明。

新工作区版本、确实变化的资源或此前瞬态失败不简单视为重复成功调用。

区分实际工具调用数、拒绝尝试数、模型调用数与 token 成本。免费拒绝仍累计题级重复/停滞记录，沿用有界终止保护，不允许通过重建节点清零后无限循环。修改计费语义应单独版本化，并在对照报告中披露。

### 8.3 多工具响应

明确 SWE 有状态工具的顺序观察契约，不能静默把一批未执行调用当作完成。第一版保留现有串行语义，改善协议提示和反馈；是否支持批量只读作为独立能力后续评估，不能与本次预算修复混在一起。

## 9. SWE 配套二：统一执行准入与实际进展

### 9.1 所有执行路径使用同一准入器

覆盖首次配置、dirty 自动执行、显式 run_agent、完整性恢复、双向修订。没有产物或处于 dirty 状态只说明“需要工作”，不能直接证明“还有资源完成工作”。

| 状态 | 准入结果 |
|---|---|
| 输入匹配的有效产物可复用 | 不重复执行 |
| 仍需新的代码修改/测试但工具为零 | 不启动注定缺少前置资源的工作 |
| 已有充分证据，仅需修复结果封装 | 在现有题级恢复额度和 token 内允许有界收尾 |
| 当前候选有效，仅图连接阻塞 | 保留候选，等待 Director 的合法原子操作 |
| 没有可提交产物且无可执行续行 | 结束并记录具体阻塞，保留结果状态的证据边界 |

任何实际工作区读写、补丁应用或测试仍按工具契约计费，不能把它伪装成免费的纯文本收尾。

### 9.2 把状态变化和任务进展分开

持续进展证据包括：新的受信任代码观察、实质补丁变化、新测试结果、当前输入下的有效产物、提交阻塞被解除。

仅换节点 ID、改写摘要、上下文压缩、消耗预算、反复试 finish 不算任务前进。内容哈希和历史集合可排除精确重复，但不能声称能完全识别任意自然语言改写或无意义代码变化。

合法配置需要多步完成：创建、设 prompt、设模型/层级、选择关系不能因每一步尚无补丁而立即被判停滞。给予与未完成配置实际步骤对应的有限窗口，窗口受已有编辑预算约束；在完成相关执行后评估是否产生进展。

合法 off 关系决策是一次真实策略选择，必须保留和训练，但不等于新增证据或连通性改善。相同配置下反复考虑同一关系、重复 on/off 循环不能无限重置停滞。

这些进度信号只用于准入与循环保护，不增加“产生补丁+奖励”“增加边+奖励”等人工 task reward。

## 10. SWE 配套三：候选、增量执行和测试

### 10.1 单一候选评估

统一以下读取者使用同一份事实判断：Director 状态、合法 finish targets、完整性风险、续行准入、实际提交。

关键字段：

```text
artifact_current / result_scope / has_nonempty_patch
workspace_binding / tested_workspace_version
protocol_complete / blocking_integrity_risks
graph_ready / submit_ready / blockers
```

删除对旧 `selected_as_output` 的业务依赖，按当前职责和受信任执行证据评估。只豁免已证明不会破坏候选的尾部重复只读拒绝；真实写入失败、版本冲突、未知环境错误不能被一起清除。

### 10.2 不可变候选与最新尝试分开

同一有效输入下，新一次失败不抹去仍然有效的候选。输入、职责、依赖、工作区或节点生命周期变化时，旧候选不能自动提交。

新增 A→B 时，B、其双向组件和下游照常失效；不能为了保留补丁绕过文档要求的 dirty 传播。已有补丁可以作为可追溯的继续工作材料，在合法信息可见性内应用/复核；需要新测试时仍消耗共享工具。

仅整理 Director 上下文、或不改变候选有效输入的结构清理，不应导致该候选重算。迟到响应不得覆盖新输入版本的产物。

`finish(target)` 只做事务性校验和提交，不触发 Worker，不重新解题，不借机改变职责。

### 10.3 功能测试

先按仓库准备公开、可运行的测试 profile，保留语法检查但准确命名。

测试记录必须包含仓库/版本、测试范围、工作区版本、命令、退出状态、超时和依赖问题。执行过测试与功能测试通过分别展示；失败测试也是有效反馈，不额外加入正确性 Judge。

本地工具观察允许提供公开测试结果，官方隐藏测试继续只用于独立最终评测。不能把后者回灌来挑补丁。

## 11. 结果归因与远程评测

补齐 `total_action_budget_exhausted` 名称映射，但不把所有 unknown 默认置零。

结果记录分别表达：是否提交非空补丁、是否完成评分、最终停止条件、过程中确认的框架/模型/工具证据。`policy_failure` 是现有分类标签，不作为框架无责证明。

`django-12304` 后续恢复原补丁的远程评分；不重复解题、不按多次得分择优。其他不完整评分也采用预先确定的基础设施重试规则。

框架/环境故障、待评分和无法归因的数据不能直接制造 Self-Play 的成功—失败边界。训练对不完整组如何等待、重评或排除，应固定规则并报告覆盖率，不挑有利样本。

## 12. 关系反事实和 Self-Play 的衔接

### 12.1 不丢失关系概率

关系选择概率、支持集、对应 token、policy checkpoint 和真实 request hash 保存在审计记录，供训练及 probe 调度读取。在线只展示实际选择与图上的事实。

不能让新上下文模式使用旧完整历史下的关系概率。旧概率只属于原请求。

### 12.2 文档要求的同 prefix probe

若实现文档的同 Canvas prefix 分支，需要冻结完整恢复包：图、节点 prompt、实际可见 packets、dirty/缓存状态、工作区快照、候选输入绑定、已花费及剩余预算、随机状态、Director 在线投影版本与真实请求。

两个分支共享关系决策前的公开信息和随机设置；只强制该关系 off/on。随后因关系产生的合法反馈差异允许进入各自后续决策，不能人为把不同环境观察压成相同内容。

模型 API 的 seed 不一定保证完全确定性；需记录后端可复现能力、temperature 和复验结果，不能仅凭相同 seed 宣称已消除噪声。

当前工作区 `counterfactual.py` 可见 `full_graph_v1` 路径，并拒绝旧的 prefix-artifact replay。它与“恢复同一运行 prefix 后继续”不同。本次不得把它悄悄改名为文档方案；精确 prefix 恢复是单独实现/验证项。未完成前，保留原估计口径并限制结论，不混用两类 relation credit。

### 12.3 奖励与图表示

保持任务 reward、graph-level GRPO、关系局部 credit、成功图密度校正的既有方法边界。上下文长度、截断次数、压缩次数作为监控，不给结构奖励。

上下文摘要、日志字段、压缩编号不进入 canonical graph feature。Proposer 仍只生成可验证任务，不接收 Director 的思考、图审计或新建的 Workflow Judge。

先让固定题集的求解与训练调用一致性通过，再恢复完整交替 Self-Play。

## 13. 建议代码落点

以下是实施规划，不代表这些新文件/配置已经存在。

| 位置 | 变更 |
|---|---|
| 新 `director_context_projection.py` | 纯函数投影、字段优先级、去重和精确 token 装配 |
| `director.py` | 每轮重建输入，所有普通/关系/修复调用统一接入并留存真实 tokens |
| `director_timeline.py` | 显式新模式、策略版本和恢复校验；与旧模式隔离 |
| `canvas.py` | 面向模型的事实状态与离线审计分离；更新实际进展检测 |
| `unified_submission.py` | 提交/续行评估一致性，零资源准入 |
| `runtime.py`、`dataset_actions.py` | 跨角色共享保留、重复操作处理、自动执行与恢复准入 |
| `swebench.py`、测试 profile 配置 | 仓库功能测试与工作区/测试版本证据 |
| `outcome_admission.py` | 错误码映射和归因边界 |
| `rollouts.py`、`training.py` | context hash 绑定、逐调用 mask/logprob、训练长度和前缀验证 |
| `counterfactual.py` | 审计引用适配；精确 prefix 恢复另行阶段实施 |

先保留 `full-v5` 完整基线和冻结源码，在单独候选版本中实施。不能直接修改历史实验包。其他聊天可能在修改共享工作区，落代码前重新检查实际差异和使用状态。

## 14. 分阶段实施与对照

| 阶段 | 内容 | 目的 |
|---|---|---|
| P0 | 固定基线、故障样本、配置及请求记录 | 确保历史可复核 |
| P1 | 候选评估一致性、共享预算保护、统一续行准入 | 先恢复执行正确性 |
| P2 | 新 Director 在线投影与精确 token 准入 | 验证消除历史膨胀且保留决策事实 |
| P3 | 逐调用训练、版本隔离及概率校验 | 防止推理修好但 RL 数据错位 |
| P4 | 分仓库公开功能测试、评分恢复 | 补足修复反馈与结果解释 |
| P5 | 固定题集真实对照，之后扩大与训练 | 判断实际收益和代价 |
| P6 | 单独验证文档要求的 prefix probe，再接 Self-Play | 避免框架噪声进入图边界奖励 |

可以按 P1→P2 集成修复，但实验必须保留可分离开关：

- A：原 full-v5 基线。
- B：仅移除在线审计字段，其他行为保持一致。
- C：完整有界 Director 输入，运行时保持基线，用于观察上下文方案效果。
- D：共享运行时缺陷修复＋原上下文。
- E：共享运行时缺陷修复＋有界上下文。
- F：E 加公开功能测试；测试能力变化单独报告。

预算保留和重复调用计费变化也应有独立标记，不能把节省的额度解释成模型策略自然提升。所有对照固定题目、模型、采样设置、路由规则和原有 24/32 额度；不同时加工具额度或重写解题提示。

## 15. 验证矩阵

### 15.1 离线状态和请求验证

- 历史反复增长而图/事实不变时，在线输入长度不随旧全文线性增长。
- 原始任务、当前节点职责、实际 on/off、预算、dirty、候选及测试版本不被截掉。
- 相同输入的投影与序列化确定；无未来事件、金标、隐藏测试或审计概率泄露。
- 上下文变化不触发 Worker 重算、不改变无进展计数。
- 普通动作、prompt 配置、关系选择、协议修复均走同一长度准入。
- 断点恢复禁止上下文版本混用；关键状态过大时有明确错误而非无声截断。

### 15.2 SWE 真实失败条件回放

- 首个 subtask 在工具只剩 2 次时，保护与 task_result 一致。
- 重复只读不访问环境；免费拒绝仍会触发有界停滞保护。
- 零工具时新建/改 prompt/显式续行/自动 dirty/双向修订均正确准入。
- `django-14315` 的补丁就绪与尾部重复读取不再触发错误重跑。
- `pylint-6903` 新增真实依赖仍触发必要失效；不能偷偷提交旧输入产物。
- 同输入失败不抹去有效候选；不同输入、已删除节点、迟到响应不能污染当前候选。
- 空补丁、语法通过但功能失败、依赖缺失、远程报告不完整正确区分。

回放只验证工程行为，旧模型输出不能充当新 prompt 下的真实采样结果。

### 15.3 训练与反事实验证

- 每次调用的输入/生成 token 与采集时一致，behavior 概率和 mask 正确对应。
- 历史动作作为输入不重复获得 loss；Canvas 摘要始终 mask=0。
- relation 概率和局部 credit 绑定原始 call_id/上下文，不依赖在线审计正文。
- 不满足真实前缀的调用回退逐调用计算，不伪造 timeline。
- 若开展 prefix probe，除指定关系之外恢复状态一致，资源和失败重试规则一致。

### 15.4 真实实验与指标

先使用预先固定、覆盖预算耗尽/上下文耗尽/候选丢失/正常成功的诊断子集，再在固定 128 题上完成同条件对照。受影响子集不能替代总体效果。

主要指标：

- 非空补丁提交率、确认解决数、未提交率、待评分数。
- 曾就绪但未提交数量；成功修改题数；功能测试覆盖。
- Director 每次输入/输出 token 的中位数、P95、最大值；整理次数与省略类别。
- 上下文超限、输出截断和补动作次数。
- Worker 与 Director 成本分别统计；工具真实调用与被拒绝尝试分开。
- 工具归零后执行次数及其是否产出有效收尾。
- 编排轮次、有效编辑、图大小、真实边、缓存命中、输入失效与候选保存情况。
- 训练合格调用覆盖率、概率校验、逐调用计算成本。

不预设必须答对某两道历史失败题，不按多次尝试挑最好成绩。工程验收要求资源与版本规则正确；效果验收要求报告固定完整题集的真实结果和成本，包括退化与未知。

涉及共享 Director 模块时，对其他数据集做固定样本回归，先确认协议和输出行为，再决定是否推广配置；不因 SWE 改动自动替换所有正式实验基线。

## 16. 推荐最终形态

Director 持续做原子图编辑；Canvas 返回真实、精简的当前状态；完整历史和概率进入审计与逐调用训练；Worker 仍依真实图关系执行，必要变更仍传播 dirty；预算、候选和提交使用一致事实。

核心收益假设是减少重复历史和无法完成的执行，使资源更多用于实际修复。该假设需要对照验证，不能以输入变短或 unknown 变少代替解决率提升。
