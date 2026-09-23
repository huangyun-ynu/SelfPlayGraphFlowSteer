# WebShop 跨版本轨迹与代码设计审计

审计日期：2026-09-23。结论：存在当前可复现的代码缺陷，也存在已经修复的历史缺陷；
继续增加通用思考提示前，应先修复观察事实与控制接口的一致性。但不能把全部失败
归因于这些问题，也不能把“暴露于问题的题数”当作“修复后增加的成功题数”。

本次只读取轨迹、源码和商品库，增加离线审计脚本与报告；未修改运行逻辑、正式配置、
历史结果或部署服务，也未调用模型重跑评测。

## 范围与方法

覆盖 7 组完整 128 题评测（896 条），外加 closure original/removed/best_so_far
三组各 30 题、option-fixes 21 题、option-fixes retry 20 题，共 **1027 条轨迹**。
子集有选择偏差，只用于排障，不与完整测试集直接比较准确率。

| 版本 | 配置简述 | 严格成功 | 完成购买 | 未完成购买/无有效结果 |
| --- | --- | ---: | ---: | ---: |
| V1 | 早期 GPT Worker，有静态 Director skill | 23/128 | 76 | 52 |
| V2 | 早期 GPT Worker，无 skill | 25/128 | 87 | 41 |
| V3 | 全量索引、DeepSeek、不思考 | 53/128 | 112 | 16 |
| V4 | thinking、retain_page_text | 58/128 | 113 | 15 |
| V5 | thinking、legacy，正式历史参考 | 62/128 | 119 | 9 |
| V6 | legacy + 重复搜索/返回反馈 | 58/128 | 122 | 6 |
| V7 | legacy + LASER 页面核对 | 62/128 | 123 | 5 |

成功与得分以各运行 `records.jsonl` 为准；动作、状态和拒绝原因来自对应轨迹。
按轨迹内 artifact_id 去重，避免把 Canvas 输出选择时再次携带的执行记录重复计数。
除历史原有审计外，本次不使用隐藏目标指导模型；所有复现均为本地纯函数或假客户端。

当前源码能复现不代表每个历史版本的所有代码相同。本报告将当前可复现问题、
历史故障证据和待验证效果分别说明。历史早期缺陷主要依据当时保留的专项审计。

## 1. P1：legacy 删除商品身份字段，后续逻辑却继续依赖它

**审计时状态：可复现；历史正式基线和当时最近两组实验均实际暴露。**

后续按用户要求已修复：通过当前合法 target_id 恢复 ASIN，接通访问状态、候选记忆、
动作辅助、语义记忆及同商品目标纠正，保留 legacy 页面格式。新的 128 题实验中身份
错误标记从 136 次降为 0，但相对预算关闭版严格成功从 63 降至 61。
见 [身份修复实验](WEBSHOP_IDENTITY_FIX_EXPERIMENT.zh-CN.md)。下文保留原始审计证据。

代码路径：

1. `webshop.py::_legacy_webshop_subaction` 删除搜索商品动作中的 `asin`、`title`、`price`。
2. `runtime.py::_annotate_webshop_search_state` 仍用 `item.get("asin", "")`
   查找已查看商品；没有从 `open_product:<位置>:<ASIN>` 恢复身份。
3. `_update_webshop_candidate_ledger` 因 ASIN 为空跳过搜索候选的记忆更新。
4. `_webshop_action_decision_support` 因 ASIN 为空，同时给出
   `already_inspected=false` 和 `may_add_product_page_evidence=false`。

因此 legacy 虽保留了可执行 target_id，却破坏了依赖 ASIN 的辅助信息：看过的商品被
标为 `not_inspected`，而新候选又被标为不会增加商品页证据。这些不是模型推理结果，
是程序主动提供的错误或误导性布尔值。

| 运行 | 看过的商品再次被标成未看过的题数 | 其中失败 | 同时发生商品重访的题数 |
| --- | ---: | ---: | ---: |
| V5 正式参考 | 51 | 34 | 39 |
| V6 环境反馈 | 51 | 36 | 31 |
| V7 LASER 提示 | 42 | 29 | 32 |

“重访”可能是合理的最终购买或补充查证，不能把全部重访视为错误；但错误身份状态是确定的。

案例：V5 `goal-00352` 请求黄色、可机洗凳套，同一商品 `B09PBDCN4P` 被打开 5 次。
16 次预算用完后购买，得分 2/3。V7 `goal-00090` 同一被套商品 `B005Y8GE6C`
被打开 4 次，最终得分 1/4。这些轨迹同时包含错误的“未查看”提示，符合循环探索的
风险路径；本次没有做修复后配对重跑，不能证明所有重复都是该提示造成的。

修复建议：为内部逻辑保留稳定商品身份，或统一从公开 target_id 解析 ASIN；
不要把“对模型少展示字段”和“删除内部判断依赖的身份”混为一件事。无需恢复全部
结构化商品描述，也无需改变合法动作或增加动作次数。

## 2. P1：详情证据摘要混入购物请求，并把截断内容标记为已保留

**状态：当前可复现；轨迹中保存的 section_evidence 确实包含请求文本。**

`runtime.py::_webshop_section_evidence` 按换行拆页面，仅在整行等于 `Instruction:`
时跳过下一行；实际 WebShop 页面通常是一行 `[SEP]` 文本，例如：

```text
Instruction: [SEP] Buy waterproof shoes [SEP] Back to Search [SEP] < Prev [SEP] Description: cotton shoes
```

现函数把整段存入商品 `section_evidence`，包括 “Buy waterproof shoes”。
用户想要防水鞋，并不等于商品描述证明它防水。这会污染“要求”和“已观察证据”的边界。

| 运行 | 商品详情记忆中含请求文本的题数 | 其中失败 |
| --- | ---: | ---: |
| V4 | 62 | 36 |
| V5 | 77 | 43 |
| V6 | 77 | 46 |
| V7 | 76 | 38 |

这些计数检查的是实际 artifact 的 `product_inspections.section_evidence`，
不是仅凭函数存在推测暴露。它仍保留 `Instruction:` 字样，模型有机会辨认，因此
不能认定这些题都被误导；但字段语义与内容不一致需要修复。

另外，摘要只保留 1400 字符，而商品动作可能被标为 `already_observed_and_retained`，
配套提示还说再次打开不增加证据。V5 有 26 题、V7 有 29 题曾看到超过 1400 字符的
详情页。该暴露计数不代表关键事实必然落在截断位置，但“部分保留”不应被描述为
“证据已经完整保留”。本次全部扫描未发现观察页超过 8000 字符的情况，因此没有
证据把 runtime 的 8000 字符头尾裁剪认定为这批实验的主要损失源。

修复建议：按真实 renderer 的 `[SEP]` 边界提取章节正文，排除任务与导航；
记录摘要是否截断、保留范围及来源，允许为缺失信息重新查看。不要在记忆摘要中
把用户需求当成商品事实。

## 3. P1：收尾预算转移依赖 sidecar 从不返回的 remaining_steps

**状态：当前接口不匹配，离线复现确认功能无法激活；历史有明确未用完总预算的零分。**

- `webshop_sidecar.py::WebShopSession._project` 返回 `steps`，没有 `remaining_steps`。
- `webshop.py::WebShopSessionLifecycle.closure_budget_context` 要求
  `remaining_steps` 为整数，否则返回 `official_remaining_steps_unknown`。
- `runtime.py` 因而无法调用 `ActionBudgetLedger.begin_webshop_closure` 转移
  剩余初始/修订额度，只能继续使用原有分阶段预算。
- `_webshop_has_feasible_completion_path` 和部分 `completion_budget` 提示也依赖
  同一缺失字段，其预期功能同样受影响。

V5/V6/V7 分别有 **27/27/26 份执行记录**报告
`transfer_status=official_remaining_steps_unknown`，没有一份完成该预算转移。
这是执行记录数，不是失败题数；其中不少任务仍用原来的 4 次修订额度完成购买。

强案例：V4 `goal-00203`（白色 66×66 遮光帘）已经选好尺寸、颜色，查看详情后返回
商品页，最终只执行 6 次环境动作。记录同时显示 `total_remaining=10`，但修订预算
已经用完，收尾转移又失败，最终没有 Buy Now，得分 0。这个案例同时有早期 token
门槛干预，不能把全部责任单独归给一个条件。

修复建议：统一预算权威。项目自己的 16 次动作预算应由 ActionBudgetLedger 管理；
若底层 WebShop 不提供独立步数限制，应明确表达这一点，而不是要求一个不存在的
“官方剩余步数”。保证同一会话中未用额度可以按设计转移，但不增加原定总预算。

## 4. P1：Director 词汇校验会误拒普通表述，且结果受附加输出约定影响

**状态：当前可复现；V6 有明确因此未进入后续收尾的失败，V7 有前置调度失败。**

`canvas.py::_compile_responsibility` 把 `self.task` 传给
`delegation.py::delegation_task_alignment_issue`。这个文本包含附加的
`Submission contract`，并不只是原始购物请求。

V6 `goal-00428` 的同一份委派内容可以稳定复现：

- 用原始购物请求校验：接受。
- 用实际 Canvas 文本（加上 Submission contract）校验：把 `meet`、`requirements`
  判成新增商品约束，拒绝。

附加输出约定中的普通词改变了词汇邻接规则的判断。请求“确认商品满足需求”本身
并没有修改商品约束。该轨迹经历 4 次类似拒绝，最后 `director_no_progress_exhausted`，
没有输出 Agent，得分 0；此时旧购物 Worker 已用 12 次动作，仍有修订额度。

V7 `goal-00387` 则在第一个 Worker 配置前连续被拒绝，检查报出 `pc`、`user`、`vs`，
最终没有任何 Worker 执行。同题请求本身包含 router 与 PC 零件的歧义，Director
的改写也确实值得约束；但应区分“分析歧义”与“擅自改写购买要求”，不能仅凭词汇
命中就反复卡死调度。

| 运行 | 出现 responsibility_violation 的题数 | 其中失败 |
| --- | ---: | ---: |
| V5 | 8 | 6 |
| V6 | 11 | 5 |
| V7 | 13 | 9 |

此表包括其他责任约束，不代表每次拒绝都是误报。严格复现“附加输出约定使本来合法
内容被拒”的真实例子是 V6 `goal-00428`。

修复建议：仅对原始公开购物请求检查约束一致性；普通过程词不应作为硬拒绝依据。
对明确新增品牌、数量、价格等约束保留限制；重复校验失败时提供具体可修复字段，
避免把任务卡在无法完成配置的节点上。修复不应取消所有约束检查。

## 5. P2：商品页上一页被错误描述为返回同一商品

**状态：当前确定错误；尚无直接失分归因。**

`runtime.py::_annotate_webshop_product_state` 对商品页和详情页的 `previous_page`
一律写入 `navigation_effect=return_to_current_product_page`。

真实行为不同：详情页上一页返回同一商品；商品页上一页返回搜索结果，并清除当前
商品选择。`webshop_sidecar.py::click` 已经按真实规则处理，因此错误在辅助注释层。
离线调用真实 sidecar 投影/点击代码（使用假环境响应）可复现该矛盾。

V5 的 128 题、V7 的 127 个进入 Worker 的题目都见过这一错误注释。但在扫描的
真实轨迹里，记录到的 `previous_page` 点击来自详情页，未找到“在商品页按错注释
点击后丢失选择”的直接案例。因此这是应修复的事实错误，不应给它虚构成功率损失。

修复建议：按实际 page_type 标注导航效果，保持侧车与 Worker 辅助说明一致。

## 6. 预算估算与重复结构化信息：历史工程损失明确，最近实验影响较小

**审计后的处理：** 用户随后要求关闭请求级估算拦截。现代码已将 WebShop 的
请求额度分配、收尾预留和请求前估算接入 `remaining_token_admission_enabled`；
当前正式配置为 `false`，普通评测、完整图复评及收尾路径均不再执行这些预测检查。
实际用量仍在执行报告返回后检查总上限，动作预算不变。定向回归测试覆盖开关两种
状态、历史额度残留及实际用量超限。下文保留原始审计证据，不代表修复后的重跑结果。

**补充核对：配置已经关闭预测准入，但该开关没有覆盖 WebShop 请求级检查。**
实际加载 `formal_training.toml`、`webshop_official_eval.toml`、
`webshop_env_feedback_eval.toml`、`webshop_laser_checklist_eval.toml`，
`canvas.remaining_token_admission_enabled` 均为 `false`，WebShop 总 token 上限均为
350000。正式及 LASER 配置的当前哈希与 LASER 启动清单一致；9 月 18 日提交
`e46b384` 的正式配置也已将该开关设为 `false`。V5 的运行清单没有保存这个开关，
因此不能仅凭该清单声称恢复了 V5 全部实际配置。

该开关控制 `canvas.py::_new_agent_token_admission` 和 `_execution_token_admission`
两处调度预测检查。但 `_execute_dirty` 的 WebShop 分支仍不检查此开关，继续分配
`_runtime_token_credit` 及收尾预留；`runtime.py` 将其传入 `request_token_credit`，
`llm.py::_request_credit_admission` 再依据序列化请求估算进行硬拒绝。
V5 案例实际记录的阶段是 `webshop_request_token_credit_exhausted`，明确来自后者。
所以“预算检查仍在拦截”不应被解释为用户没有关闭配置，或实际 token 已耗尽；
准确说法是关闭调度预测后，独立的请求额度与预留检查仍在执行。配置注释明确保留
硬上限，但目前该硬检查依赖阶段额度和保守估算，不能等同于实际用完 350000 token。

`webshop_budget.py::request_budget_quote` 使用请求 JSON 的 UTF-8 字节数加 2048
作为输入 token 上界，再预留输出和收尾额度。这是刻意保守的估算，不是 tokenizer
得到的真实 token 数；同时 `runtime.py::_webshop_context_for_prompt` 又增加一份
动作说明、约束矩阵等信息。V5/V7 的累计 prompt 字符量相对原 context 分别增加
约 **14.62% / 13.15%**，并不是只做压缩。

V5 `goal-00197` 是很强的工程损失案例：

- 已经在 `B082N82QWM` 商品页选中 `baby blue`。
- V3、V6、V7 在同一题购买同一商品、同一颜色获得满分。
- V5 只执行 13/16 次动作，仍剩 3 次；Canvas 执行账本累计 Worker token 为 113513。
  原先统计保留的最终 Artifact 得到 107561，漏掉双向执行中被后续修订覆盖的
  5952 token；这里已改用执行报告累计值，不改写历史原始记录。
- 最后请求所需上界为 55403，而当前调用额度为 53624，请求在发送前被拒绝。
- 最终没有购买，得分 0。它不是单纯“模型找不到正确商品”。

其中 53624 的计算为：收尾前整题余额 243817，调度估计 `call_count=4`，均分后
`per_execution_credit=243817//4=60954`；该 Worker 先消耗 6123+1207=7330，
本轮额度剩余 `60954-7330=53624`。收尾执行报告实际只运行 agent_1，agent_2 被复用；
此时整题账面余额仍有 `243817-7330=236487`。这个数字是旧分配机制的局部余额，
不是用户配置的单次上限或模型上下文容量；上述开关修复已停用关闭状态下的这一分配。

另一个典型触发因素是大选项表：`goal-00365` 有 204 个动作，多个版本在很低的
实际累计 token 消耗下就触发保守请求上界限制，然后依靠后续收尾购买成功。

| 运行 | 出现请求准入阻断的题数 | 其中失败 |
| --- | ---: | ---: |
| V4 | 14 | 12 |
| V5 | 11 | 9 |
| V6 | 2 | 0 |
| V7 | 3 | 0 |

这个机制当前仍在，但最近两组里暴露题均最终成功，不能将它称为最新 66 个失败的
直接主因。历史与当前其他代码差异、轨迹长度和提示规模都可能改变其触发频率。

修复方向：统一 token 计量与各阶段信用分配；优先去掉重复表示，而非丢弃合法动作；
需要保守上界时明确其与“实际 token 预算”的区别，避免还有动作和总预算时被过早
关闭购买路径。不要通过偷偷增加测试预算来掩盖控制逻辑问题。

另有较小的调度浪费：V5 有 12 份、V7 有 2 份所有者执行，在初始 12 次已用完、总
额度仍剩余时以初始阶段再次运行，没有执行任何动作。应检查 prompt revision 与
revision/closure 阶段的映射；本次未证明它独立造成严格成功率损失。

## 已修复的历史缺陷与外部故障

1. **小索引部署。** V2 当时使用 99995 文档索引，仅覆盖 128 题中 12 个原目标
   ASIN；全量索引覆盖 128 个。同批已记录查询的离线回放，目标进入 top-10 的题数
   从 7 增至 50。这是有实质影响的部署问题，但替代商品也可能满分，不能说其余
   116 题全部不可解。当前已切换全量索引。
2. **选项规范化不一致。** V3 的 20 题暴露过官方 `/` → ` | ` 转换后选项未被
   正确认作选项的问题，如 `goal-00071`、`goal-00436`；这会让选项已点击却未记录，
   引发重复操作。已有修复及真实环境 smoke 记录。
3. **合法动作静默截断至 100 项。** V3 的 `goal-00135`、`goal-00365` 出现合法
   选项在被截掉的位置，随后验证器拒绝。已有修复，115/204 项动作的保留得到验证。
4. **模型后端故障。** V1/V2/V3 分别有 29/9/4 个失败题包含 Worker 后端失败标记；
   option-fixes 的 21 题中有 20 题后端失败，其重试不能作为纯适配收益证明。
   这是早期成绩的重要干扰，应与模型能力和代码机制分别统计。V7 没有 Worker
   终止性后端失败，只有一次请求级 502 记录。

依据：`state/sota-20260916/webshop-index-audit.md` 和
`state/sota-20260916/webshop-full-index-deepseek-no-skill-v1/followup_audit.md`。

## 没有被证据支持的归因

- 没有发现已暂存购买被 Canvas 丢失、从而未完成购买的扫描案例。
  暂存后限制继续修改也对应环境购买终止语义，不应随意改成可以撤销购买再试。
- 最近运行没有证据显示 `semantic_no_progress` 硬熔断造成大量失败。
- 超长证据字符串的 schema 拒绝不消耗动作额度，V5 的两例随后都成功执行了购买；
  不能将这些拒绝直接当作整题失败原因。
- 不是所有失败都来自买不到或提交失败：V7 的 66 个失败中，61 个已经购买，只得
  部分 reward，4 个用完 16 次动作未买，1 个未进入 Worker。
- 同题同商品也可能因选项不同得分不同。例如 `goal-00201` 的窗帘商品 `B073SRVMRY`，
  `white | silver` 得 2/3，V6 选 `white | off white` 得 1；这需要进一步分析选项语义
  与官方评分匹配，不能直接归为适配器失效。

## 建议修复顺序与验证边界

1. 先修复 legacy 身份丢失、详情正文提取、导航说明这三类事实不一致。
2. 修复 Director 校验的输入文本及普通过程词误报，保留明确约束漂移的检查。
3. 接通预算转移接口，确保原定 16 次动作不会因阶段切换丢失；再优化 token 准入。
4. 在同一当前源码、同一部署上保留原配置控制组，再做修复组；保持 thinking、
   题目、skill、并行和预算一致。先用已知问题轨迹做离线/小样本验证，再报告全量成绩。

无法据当前审计给出“修复后一定从 48% 到多少”的可靠数字。高置信结论是：至少
五个事实/接口/校验问题可确定复现，历史也存在本来接近成功却被工程机制中断的题；
修复这些问题比继续叠加通用思考提示更有依据。

## 审计产物与复现

- `state/experiments/webshop-design-audit-20260923/summary.json`：各版本聚合计数。
- `trajectory_index.jsonl`：1027 条去重轨迹的动作、预算、拒绝、状态索引，不复制思考内容。
- `additional_analysis.json`：详情证据污染、输出约定诱发词汇误拒的逐题清单。
- `reproductions.json`：五项当前代码问题的离线复现结果。
- `option_collisions.json`：额外排查的选项同名情况；只有一个观察商品存在跨组选项同名，
  对应实际轨迹没有选择碰撞值，故未列为已导致失分的核心问题。

```bash
.venv/bin/python scripts/formal/audit_webshop_design.py
PYTHONPATH=src .venv/bin/python scripts/formal/reproduce_webshop_design_issues.py
```

五项离线复现断言通过；两份审计脚本 lint 通过，`git diff --check` 通过。
本次未把这些复现作为要求失败行为长期保持的回归测试；正式修复时应改为验证正确行为。
