# 全任务统一增量执行与提交协议（实验实现，分阶段验证中）

日期：2026-09-26。工作目录：`SelfPlayGraphFlowSteer-output-contract-v1`。

本文是当前推荐方案，替代此前仅覆盖文本任务的设计。范围包括 AIME、NQ、HotpotQA、HealthBench、ALFWorld、WebShop、SWE，以及它们在新版协议下启用的执行模式。所有新版任务均移除独立 `set_output`；旧协议仅保留用于历史回放和对照，不作为新任务的隐式回退。

实现位于实验目录，新增统一职责、执行准入和提交事务；正式目录与历史成绩保持原样。验证按离线、每集 1 题、每集 10 题、全量顺序执行，进度与证据保存在 `state/audits/unified-protocol-20260926/`。下文仍包含恢复能力的设计要求；当前实现对跨进程已有事务采取锁定并要求核对，不宣称已有自动恢复或端到端恰好一次语义。

## 1. 核心规则

1. 配置 Agent 时明确它交付局部工作还是完整任务结果，允许普通增量修改和双向职责转换。
2. 执行器根据真实输入、依赖与环境状态计算或继续执行，不根据最终选谁提交改变职责。
3. Director 使用统一 `finish(target)` 提交该节点已经产生的、当前有效的结果。
4. finish 不调用 Worker、不隐藏补算、补搜索、补代码修改或补测试；必要的工作通过正常图修改或显式 `run_agent(target)` 完成。
5. finish 可以执行适配器声明的确定性提交操作，例如提交已经暂存的购买；不能临时决定买什么、生成补丁或选择别的节点。
6. 结果可提交不等于任务成功。真实失败按既有评分/归因规则保留，基础设施不确定性不能冒充答案错误。
7. 节点可删除，外部副作用不能凭删除撤销。提交在途或结果不确定时锁定事务，防止二次提交。

## 2. 四个独立概念

| 概念 | 含义 | 权威来源 |
|---|---|---|
| 节点职责 | 当前 Agent 被委派做什么 | Director 的合法配置 + 系统任务合约 |
| 任务产物 | 答案、环境执行记录、购买提案、补丁或可证实失败 | 实际执行及 Runtime 来源记录 |
| 执行资源 | 环境 session、代码工作区、资源版本、剩余额度 | 适配器生命周期与 Runtime 账本 |
| 最终提交 | 本题正式采用哪份产物 | 真实 Director finish + Runtime 提交凭据 |

完整结果职责不会自动获得任意工具、独占会话或外部提交权限。工具权限与共享资源互斥由适配器声明；实际最终提交权由 finish 事务持有，不永久挂在某个 Agent 上。

## 3. 统一动作协议

保留 add_agent、set_prompt、set_model、set_layer、consider_relation、set_relation、remove_relation、delete_agent。

修改 set_prompt，增加明确交付范围：

```text
result_scope = subtask | task_result
```

此前文本设计中的 task_answer 在统一版中改称 task_result，避免将环境结果和代码补丁叫作答案文本。它是职责的一部分，不是模型输出后自报的标签。

新增/修改的动作：

```json
{"action":"run_agent","target":"agent_3"}
{"action":"finish","target":"agent_3"}
```

- run_agent：对 Runtime 已确认待执行、可继续或可修复的节点进行有界执行。具体是首次执行、继续环境交互还是修复失败，由当前状态决定并记录。健康且已完成的候选不能无依据反复抽样。
- finish：选择并提交已有有效结果，零 Worker 调用。
- 移除 set_output，不引入 replace_output，不另外保留面向某些数据集的“选输出”别名。

保留现有配置/编辑后执行受影响闭包的调度方式。run_agent 用于“输入未改但实际工作未完成/失败需要继续”的场景，替代旧 SET_OUTPUT 或 FINISH 中隐藏的完成阶段。它不要求所有普通修改再多调用一次执行动作。

新协议不接受缺少 target 的 finish，不通过自动挑答案补全。原始模型动作、解析结果、拒绝与执行结果都完整记录。

## 4. 各任务交付什么

| 任务 | task_result 的实际交付 | finish 的确定性处理 |
|---|---|---|
| AIME / NQ / HotpotQA | 回答完整公开原题的文本产物 | 校验来源与版本，确定性提取并固定提交文本 |
| HealthBench | 对完整公开对话的专业回复 | 固定完整回复，不在 finish 内另找模型润色 |
| ALFWorld | 绑定当前题目与 episode 的可信环境执行结果 | 固定真实执行记录；不再开局或执行环境动作 |
| WebShop | 真实 session 中准备好的购买提案，或符合现有终止政策的可信负结果 | 对选定、未过期提案执行唯一购买事务；可信负结果按规则结束 |
| SWE | 绑定 instance/repo/base/patch hash 的补丁与相关测试证据，或可证实的类型化失败 | 固定并导出该补丁/失败记录，后续官方评测独立进行 |

各适配器负责解释其产物，公共控制器使用同一个评估/执行/提交接口。不同 payload 类型不构成 Director 动作例外。

### 4.1 不把成功当准入条件

- 文本节点被要求回答完整原题，不代表其内容真的正确或完整；不增加隐藏模型裁判。非空错误/答案格式错误沿用当前评分协议。
- ALFWorld 的可证实失败、既有规则允许的提前停止结果不能因没有成功而消失。
- WebShop 买错商品或 reward 为零仍是一次真实终局，不能再买一个来覆盖成绩；无购买的负结果须满足既有 Runtime 归因规则，不能只凭模型说“没买”。
- SWE 当前规则允许最新修改后的测试失败记录进入提交评估；“有对应测试证据”不能偷换成“所有测试通过”。无补丁时，只接受既有证据规则认可的类型化失败。
- 基础设施异常、环境状态未知、缺少可信证据单独记未知/未提交或评分失败，不默认归零为模型错误。

## 5. Runtime 拥有产物记录

建议引入通用 TaskResultRecord，字段至少包括：

```text
run_id / task_identity / dataset / protocol_version
agent_id / agent_incarnation / result_scope / contract_version
artifact_id / execution_generation / semantic_input_signature
payload_kind / payload_ref / payload_hash
upstream_refs / peer_refs / source_execution_refs
resource_id / resource_version / resource_scope
integrity_status / terminal_failure_evidence
```

这些字段由 Runtime 从合法配置、工具观察和实际执行生成，不能接受模型自填权威字段。AgentArtifact 的文本报告与可信资源结果分别保存：环境已经完成真实操作后，模型 JSON 报告失败不能抹除真实结果。

对于文本，subtask 的产物不能事后升级。对于环境已有的不可变终局，职责改变后由适配器重新核验它是否确实对应完整原任务；可以根据真实终局证据重新形成结果记录，但不能靠改标签、重放动作或重开局伪造完成。新的职责版本和原执行来源必须同时保留。

结果状态至少区分：局部结果可用、完整任务候选、候选过期、需要继续、可修复、可提交、可信失败、未知、已提交。

## 6. 普通增量修改与职责转换

subtask ↔ task_result 双向转换均通过 set_prompt 完成，不需要删除重建。

| 操作 | 规则 |
|---|---|
| 修改 prompt / result_scope / 模型 | 执行输入变化，旧候选失效；正常执行受影响闭包 |
| 增删真实依赖 | 按旧、新依赖及双向组件传播失效 |
| consider_relation 为 off | 不新增边，不声明已传递证据 |
| 删除任意未提交节点 | 移出活动图、撤销候选，处理资源生命周期，保留审计 |
| 暂时没有 task_result 节点 | 可以继续构图和执行，不能把局部产物当正式结果提交 |
| 新建 task_result 节点 | 按新职责执行，资源与预算策略照常生效 |
| 无意义重复修改 | 不触发额外执行、不当成进展 |
| finish(target) | 不改变其他节点职责、Worker 输入或缓存 |

图变更合法性与预算准入应在使现有候选失效之前完成。非法编辑不能先毁掉候选再回滚。合法输入变更后，即使新执行失败，旧输入产物也不能直接回退提交。

对状态化任务，“重新执行”通常表示从当前可信资源状态继续，不表示重放已经执行的环境动作。已终止 episode 不再接受动作；真实终局保持记录。

## 7. 环境资源与 Agent 身份分离

Runtime 以 resource handle 管理环境，包含题目、session/episode、版本、执行租约、状态与累计预算。Agent ID 和资源 ID 不再混为一谈，Agent 重建使用新 incarnation。

### 7.1 真实声明资源模式

- per_agent_episode：当前 ALFWorld 和 Native WebShop 已有独立、可继续的节点会话。修改同节点 prompt/model 使用原会话；删除释放该会话并取消未提交候选；新节点使用新的独立会话，消耗题级尝试/动作预算，不继承旧节点私有历史。
- WebShop 旧单一 first-owner 模式仅保留给旧协议。新版 generic 和 native Worker 都显式使用 per_agent_episode 生命周期：各节点独立 session，同节点修改复用该 session，整题动作预算共享。此选择在开始运行前固定并记录，不在失败后切换资源模式。题级共享 session 的租约转交尚未实现，新版不冒充支持该模式。
- isolated_workspace：SWE 使用隔离工作区和不可变补丁。当前实现每次执行创建工作区，后继执行通过公开可见、身份匹配的补丁引用继续，不能声称会自动保留原进程文件系统。

新版不同模式使用相同 Director 动作，模式和预算预先写入配置与运行清单。不能在失败时偷偷从共享环境切换成多个新 episode，也不能把旧 first-owner 的删除禁令作为新版输出身份限制保留下来。

### 7.2 删除与不可逆状态

- 普通工作阶段允许删除节点，包括当前唯一完整结果节点。
- 删除在途执行节点前必须取消/等待并隔离迟到结果；迟到结果进历史，不能使节点复活。
- 每节点会话删除时关闭，题级共享会话由题级生命周期管理；资源清理失败记录为清理问题，不抹去已完成结果。
- 已发生的官方结果和动作始终保留在题级账本，删除图节点不删除成本、失败或成功事实。
- 提交事务已进入可能产生外部副作用的阶段，暂不允许删除、改职责或换 target；事务确认后结束，确认未执行则可以回到工作阶段。

## 8. WebShop：准备候选与真正购买分开

所有新版 WebShop 模式都要求暂存购买提案：Worker 的 Buy Now 意图生成与当前 session/page/product/options/version 绑定的候选，不直接执行官方购买。旧 execute_on_output/raw purchase 模式不能作为新版隐式回退。

1. Worker 搜索、查看、选择选项，使用真实页面动作形成提案。
2. 普通 prompt/关系修改允许同 session 修订；合法新输入使旧提案失效。
3. run_agent 在剩余预算内继续购物或修复，不因为“被选为输出”才获得最后一次执行机会。
4. finish(target) 只提交该 target 的当前提案；不能提交别的节点的候选，也不能临时替换商品。
5. Graph/职责/资源版本和提交预算全部核验后才调用后端 commit。
6. 成功提交后锁定本题；商品不合要求或 reward 为零也不允许重买。

当前 Native WebShop 已有“隔离 session + 修改时复用页面 + finish 提交”的基础；旧模式需要迁移生命周期和激活条件。当前 sidecar 的 commit_id 去重存在内存状态和写入确认窗口，不能据此宣称已经具备跨崩溃的恰好一次语义。

## 9. ALFWorld 与 SWE 的具体迁移

### ALFWorld

- task_result 要求处理原始完整目标；subtask 可以探索/提供局部观察，不能仅凭文字报告取得完整结果资格。
- 修改职责/模型时保留同一 episode 的当前位置、物体状态及剩余步数。
- Worker 在阶段预算内未完成但仍可继续时，开放 run_agent；不得靠 finish 隐式接着走环境。
- finish 固定真实 episode 执行结果。正负结果均按现有评估规则记录；步骤超限、主动停止、环境故障分开。
- 没有 session 状态确认能力时，丢失动作响应不能被当作动作未执行并盲重试。

### SWE

- task_result 从配置时就包含交付补丁及修改后测试证据的职责，替代“选为输出后才获得 code_commit 职责”。允许多个节点各自准备补丁，最终只提交一个。
- 原 SET_OUTPUT 的 final-fix pass 迁入正常职责执行或显式 run_agent。
- 诊断节点改成 task_result 属于真正职责扩展，需要执行；不能把“建议修改某文件”的 prose 当补丁。
- 补丁绑定 instance/repo/base_commit/hash；不同实例或分支的补丁不能串用。
- 测试证据绑定实际补丁/工作区版本，修改后旧测试不能冒充新版本测试。测试失败与没测是不同状态。
- finish 固定实际 patch bytes/hash，不补编辑、不执行 Worker 测试、不做 git push。官方 harness 在评测阶段处理，只评价已选补丁，不将隐藏测试结果反馈给未结束的策略。
- 导出后临时工作区清理失败不改变已持久化补丁；持久化本身失败则不能声称已提交。

## 10. 统一 finish 状态机

所有数据集都经过同一个状态机，适配器只实现结果检查和确定性终结。

```text
WORKING
  → 只读评估 + 绑定 Director 看到的快照
  → PREPARED：持久化提交意图、锁定 target/result/resource
  → COMMITTING：按适配器确定性终结（纯文本可立即完成）
  → COMMITTED：持久化凭据，结束策略执行
  → SCORING / SCORED：异步或同步评测已固定结果
```

错误分支：

- 检查失败：无副作用，返回 WORKING 与明确 blocker。
- 后端明确拒绝且确认未发生副作用：记录事务失败，允许有预算的后续修正；不清计数。
- 后端响应丢失、超时或执行结果不明：进入 COMMIT_UNKNOWN，锁定本题提交目标；只允许对原事务核对，不允许换节点重买。
- 已确认提交而本地收据写入失败：按原持久化提交意图恢复收据，不再次执行外部动作。
- 确认提交失败且没有合法续行：结束并记录可信失败或未知，不伪造成功 receipt。

不增加 Director 的 prepare/commit 两次动作：这些是一次 finish 内部的事务步骤。后端恢复沿用那次真实 Director 调用及目标，不生成新的虚假 finish。

### 10.1 通用检查

target 存活且职责匹配；候选及资源来源可信；输入和依赖当前有效；图合法且保留节点均可到达 target；没有冲突的在途执行；预算允许；Director 调用身份与它实际看到的候选版本匹配。

检查与 PREPARED 之间要防并发修改。若 Director 返回时候选已经改变，拒绝过期动作，不悄悄提交最新另一份。

### 10.2 外部提交不确定性

事务标识包含 run/task、结果引用和资源版本，持久化后复用；不能以每次重试的新 UUID 产生新操作。

WebShop 需要后端支持提交状态查询及可靠去重。仅存在 HTTP 缓存或当前 session 内的 commit_results，不足以覆盖“操作执行后、结果写入前崩溃”的窗口。若权威状态无法恢复，保持 UNKNOWN 并停止该题的副作用操作，不能为了给出结果盲重试。

同一事务重复 finish 返回已有状态/凭据。提交后的不同 target 被拒绝。评分失败只重试评分，不重做购买、环境动作或代码修复。

## 11. 适配器接口与统一状态评估

建议接口职责如下（名称为设计名）：

```text
assess_result(node, artifact, resource_snapshot) -> ResultAssessment
assess_execution(node, reason, resources, budgets) -> ExecutionAdmission
prepare_submission(result_ref, observed_snapshot) -> CommitPlan
commit_submission(plan, transaction_id) -> CommitResult
query_submission(transaction_id) -> CommitStatus
dispose_or_detach_node(agent_incarnation) -> CleanupResult
```

评估接口无模型调用、无业务副作用。prepare 生成精确提交计划并保存意图；commit 只执行计划中的确定性操作。缺少必要后端能力时明确拒绝新版配置或进入未知结果，不能回退 set_output。

统一评估输出至少包含：candidate/current/submit_ready、blockers、真实负结果依据、可继续/可修复动作、资源状态、剩余预算、是否需要外部 commit。控制快照、动作菜单、finish、恢复和日志都读取它。

## 12. 统一凭据、真实失败与评分

将现有文本 SubmissionReceipt 扩展为所有任务的通用凭据：Director 调用、目标 incarnation、图哈希、职责/输入版本、结果引用、payload 类型、资源版本、事务 ID、执行/提交证据、预算账本摘要。

文本答案、episode 记录、购买结果和补丁分别有适配器载荷，不强制把一切转换成 answer 字符串。

Receipt 表示正式提交/结束了哪份结果，不表示答对或任务成功。正负评分由可信 verifier/环境结果产生；评分待定与未提交分开。控制器预算耗尽而没有真实 finish 的情况保留无提交凭据状态，可依据既有可信失败账本作归因，不补造 Director 行为。

所有尝试按原计划题数统计；删除节点不清失败记录，不从隐藏 reward 中选最优候选。局部节点即使碰巧说中答案，也不能用金标指导后补职责。

## 13. 候选保护与资源版本

纯文本同一有效输入下，失败尝试不应抹掉仍有效的候选。状态化任务额外要求同一资源版本：新执行已改变页面/环境/补丁后，旧候选通常不可复用，即使 prompt 没变。

缓存键包含职责、任务、输入依赖、环境与节点生命周期；不包含最终选定输出的标签。禁止复用另一个 Agent 的可变 session、私有历史或已过期购买 target。

缓存命中不重放环境副作用。资源版本不符、session 已关闭、候选过期、节点 incarnation 改变均禁止提交旧结果。真实环境终局和补丁引用的历史记录独立保留。

## 14. 预算、继续执行与停滞

- 保留每题 token、时间、Director 轮次总约束，增加统一的节点创建/episode 尝试与恢复账本；实际上限显式配置并参与对照。
- 修改 prompt/model、角色转换、删除重建不清题级累计额度。共享 episode 不刷新步数；新独立 episode 是新的计费尝试，不是免费恢复。
- run_agent 的原因由 Runtime 确认：尚有工作、输入失效或可修复失败；同一未解决问题的恢复额度有限。纯重试上限初值可沿用设计建议 2，但须作为新配置验证，不能假装原基线已有相同额度。
- token 仍有余量不意味着环境步数、工具额度或最终提交空间足够。
- 模型工具调用数与真实环境动作数分别计账。WebShop 暂存 Buy Now 与实际 commit 保留对应关系，购买步预留并且只按既定预算口径消耗一次，禁止提交阶段越过额度。
- 正常继续与失败重试分开记录；不声称仅靠 prompt 哈希能识别所有语义等价绕行，题级硬预算仍是最终上限。
- 变化 target、Agent ID、artifact ID 或错误文案不构成修复进展。真实连通性、有效资源进展、产物完整性与提交阻塞才是依据。
- 连通性修复覆盖真实边及多步合法调整；既有循环保护迁移到新协议，不每换图版本就无限重置。
- 冻结后保留有效且有预算的继续/修复，否则明确结束，不留必败 finish-only 菜单。

## 15. 示例

### AIME

A 被配置 task_result，产生有效 754；finish(A) 固定已有答案，不重算。若 A 原本只负责局部子题，则先改职责并执行，不能事后提升旧局部产物。历史未提交成绩不修改。

### ALFWorld

A 已在自己的 episode 中拿到杯子；修改 prompt 后 A 从当前位置继续，不重开房间。若删除 A、创建 B，在 per_agent 模式下 B 是新的独立 episode，步数/尝试计入题级总账；B 不会凭 A 的文字报告声称自己已经拿到杯子。

### WebShop

A 已准备商品 X 的购买提案但未购买。修改 A 的 prompt 后，旧候选失效，A 在原 session 修订并准备 Y。finish(A) 校验 Y 的当前绑定，只执行 Y 的购买。若 commit 超时，固定该事务进行核对，不能改 finish(B) 去买另一件。

### SWE

A 有补丁 P 和对应测试记录。修改 prompt 后继续修复得到 P2，则 P 的测试不能自动证明 P2 测过；完成对应执行后 finish(A) 固定 P2。若删除 A 并新建 B，B 只有通过合法可见补丁引用在自己的工作区应用/验证后，才能产生自己的结果。

## 16. 代码迁移落点

| 位置 | 必须迁移的内容 |
|---|---|
| actions / delegation / Director | 统一 scope、run_agent、finish(target)；全部新版任务无 set_output |
| graph / canvas | 针对 target 的纯验证、依赖失效、统一候选菜单、事务锁及生命周期删除 |
| output_contract / runtime | 去掉最终选择对职责/缓存的影响；按 task_result 生成数据集职责；所有修订/协议修复分支一致 |
| dataset_actions | 废除新协议 execute_on_output/commit_pending_on_output 激活；提交方式与资源模式明确声明 |
| ALFWorld 生命周期 | 同节点持续 episode、删除清理、未知状态隔离及 Runtime 结果引用 |
| WebShop / Native / sidecar | 暂存/修订/删除统一接口；共享与独立资源模式；持久事务与查询、去重 |
| SWE 生命周期与进度 | 完整补丁职责不再依赖独占 output code_commit；补丁与测试版本；移出隐藏 final-fix |
| adaptive / submission_contract | 去掉 finish 隐藏执行，所有任务统一 receipt 与真实 Director 来源 |
| outcome / evaluation / training / PATS / replay | 载荷化评分、版本隔离、资源/预算 lineage、反事实分支资源隔离 |

运行清单显式记录资源模式、模型、预算、候选/动作/提交协议版本。旧动作轨迹以旧语义回放，不能把历史 set_output 静默重解释为新 finish，也不能无条件复用旧缓存。

反事实回放必须创建隔离资源、保留原始目标选择语义并独立记账；不能拿真实运行的购买事务重复提交，也不能因为某分支得分更高就替换正式结果。

## 17. 验证矩阵

### 通用模拟

单节点、多节点、全部局部、多个完整候选、双向转换、模型/职责修改、删除唯一完整节点、节点满额删除重建、空图重建、迟到结果、真实依赖与 off 候选、双向组件、多终端图、同输入候选保护、过期候选、无预算、冻结、重复 finish、不同 target 重复提交、旧协议/缓存混用。

### ALFWorld

持续 episode、局部到完整职责转换、成功和失败终局、结束后不再 step、删除后资源清理、新 episode 计入总账、错误 action ID、响应丢失状态未知、模型报告失败但真实环境证据仍在。

### WebShop

独立与共享模式分别验证；候选修订、删除、新节点不能继承私有 session、候选 TTL/页面变化、stage 后无余量 commit、真实购买 reward=0、不完整负结果、未选候选不购买、网络重试、双重 finish、同事务并发、提交前/执行后/写凭据前崩溃、后端重启和状态无法确认。

### SWE

完整职责直接改代码、局部诊断升级、补丁为空、类型化失败、修改后没测、测试失败、测后再次改代码、跨 repo/base 补丁、导出失败、清理失败、harness 延迟/故障、只重试评分不重修代码。

### 真实验证

先固定旧故障轨迹做复现，再按新协议重放故障条件（明确是模拟），再调用真实模型和环境。先 AIME 15/22 及三个环境的代表性成功/失败任务，再固定全量/子集对照。HotpotQA 128、NQ 128、AIME 30 保持题目并行 40 和 DeepSeek 并行 40。

模型路由并行 40 不等于同一个有状态 session 可并行写；环境并发由资源隔离和后端容量限定并单独记录。环境数据或服务不具备时报告该项未验证，不能把 mock 通过写成真实环境通过。

此前 1–4 节点、16,609 种配置的结构穷举只证明该有限范围内的可达性目标判断一致；不代表本方案事务、环境恢复或模型效果已验证。本轮没有新增运行测试。

## 18. 验收与实施顺序

先固定全任务旧行为与关键回归，随后完成职责/结果/资源三种记录与统一评估；接入新版动作及文本路径；迁移 ALFWorld、SWE、两类 WebShop 生命周期；完成通用事务与所有任务 receipt；最后检查训练/回放和真实对照。全部任务支持完成前，不宣称“统一版完成”，也不为未迁移环境悄悄保留新协议 set_output。

硬条件：所有新版任务 set_output 调用为 0；finish 引起 Worker 调用为 0；无依据重复执行为 0；局部/过期/已删除节点结果被提交为 0；候选边冒充已有边为 0；未选候选产生最终提交为 0；重复购买为 0；删除重建刷新题级预算为 0；确定已提交结果被失败报告抹掉为 0；已知必败 finish-only 死路为 0。

效果指标按完整计划统计：任务正确率/EM/F1/环境 reward/代码 resolved、提交与未提交率、真实负结果与基础设施未知、模型 token 与调用、工具调用与真实环境动作、session/工作区创建、必要重算、恢复次数、提交延迟与未知事务、清理异常。不能只报告提交成功样本。

每次运行独立冻结源码、配置、任务输入和协议，保存逐步图、scope 变化、资源版本、请求响应、工具轨迹、候选引用、预算、提交意图/确认/凭据、评分及失败。保留原有评测与全部失败尝试。


## 两份独立预算

新评测策略将 Director 的成功图编辑与 Worker 的工具调用分开：`canvas.max_director_edits` 及 `max_director_edits_by_dataset` 控制整题实际图修改；`canvas.action_budget_policy = "shared_total_v1"` 让所有 Worker 共享整题工具账户。在 `director_budget_policy = "edits_v1"` 下，Director 决策轮数只统计，不执行旧 `max_rounds` 上限；无效动作仍受无进展限制约束。旧模式保留历史轮数语义用于轨迹兼容。

初始节点、prompt 和模型配置计入编辑额度；关系提案或无变化选择不计，真正增删边计一次。执行和提交不扣编辑额度。额度不足以完成必需配置时不再允许创建节点。即使 Worker 后续失败，已生效的编辑也计数。节点重建与模型切换不能重置任一账户，只有新任务 reset 可以重置。

控制快照提供两份额度 `director_edit_budget`、`action_budget`，并用 `decision_statistics` 记录决策次数，不再提供第三份 `round_budget`。工具的 initial/revision 标签仅表示调度阶段，不再分别分配可调用次数。`finish(target)` 始终只提交已有合格产物，不消耗编辑和工具额度。无进展计数不会因预算计数器变化而重置。
