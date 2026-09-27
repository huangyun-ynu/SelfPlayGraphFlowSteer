# QA 历史源码、答案整理器与提示词说明

## 当前采用的评分约定

根据用户明确要求，HotpotQA / NQ 使用 `flowsteer_qa` 作为评分规则，独立保留官方答案
EM 和 token-F1 审计。训练奖励、通过率、EM、F1 分别报告，不把通过率当 EM。
历史 HotpotQA exact-match 口径保留为历史记录，不要求正式训练切回该评分器。

当前实现已经在生成 `answer_submission.submitted_answer` 后计算
`task.metadata.qa_official_metrics.answer_em/answer_f1`；评分器再对同一提交答案评分。
训练汇总与 benchmark 报告继续分别输出 EM/F1。
新增回归覆盖“FlowSteerQA 给 0.7 分且通过，但 EM=0”的实际 Solver 路径，避免二者被混为一项。
相关 QA 和报告测试共 29 项通过。

## 找到的历史源码

原仓库是浅克隆。通过显式 Git 目录读取并从配置的 origin 补齐历史后，找到了以下提交；
完整受 Git 管理的文件已分别导出到独立目录，当前工作区未切换到旧提交。

| 提交 | Git 记录的时间（UTC+8） | 内容 | 恢复目录 |
|---|---|---|---|
| `496ab8e` | 2026-09-18 09:43:50 | 正式评测、检索、ALFWorld 修复检查点 | `state/restored-versions/qa-checkpoints-20260925/496ab8e/` |
| `fb73ce4` | 2026-09-19 22:51:19 | 正式实验代码与 NQ 固定证据流程 | `state/restored-versions/qa-checkpoints-20260925/fb73ce4/` |
| `b28aedc` | 2026-09-20 16:26:36 | NQ128 冻结清单和外部资源配置 | `state/restored-versions/qa-checkpoints-20260925/b28aedc/` |

完整 commit ID、时间和所选 QA 文件哈希保存在该目录的 `manifest.json`。
对比结果及差异文件保存在 `state/audits/qa-history-search-20260925/`。

源码确认结果：

- 三个提交中的 `runtime.py`、`director.py`、`delegation.py` 分别逐字节一致。
- `answer_submission.py` 在 `496ab8e` 到 `fb73ce4` 之间变化：新增了固定证据 span、问题类型等处理。
- 三个旧提交的 Worker 输出提示均没有当前 `qa_role_conditioned_v2` 的中间/输出角色分支。
- 当前 Worker 明确区分“中间 Agent 回答分配到的局部问题”和“选定输出 Agent 回答整题”。
- 09-18 的 formatter 输入使用完整 `task.prompt`；当前版本抽取 `original_question`，并增加
  `question_type` 等字段。关闭证据 span 开关并没有把完整 formatter 提示和输入恢复成 09-18 版本。

因此，前面“没有找到历史源码”的表述需要修正：**现在已经找回历史代码检查点。**
仍未证明的是哪一次 128 题运行精确使用了哪个提交，以及运行时是否存在未提交修改。
记录中的 `executor_bundle_signature` 是配置 manifest 的哈希，不能证明源码版本。
恢复目录中的代码是有 Git 来源的真实历史版本，不应直接冒充两次最高 EM 运行的逐文件源码快照。

## 答案整理器是什么

当前处理顺序是：

```text
Director 编排 → Worker 工作流 → 输出 Agent 的 answer
                              ↓
                  AnswerFinalizer（答案整理器）
                              ↓
                    submitted_answer
                              ↓
                FlowSteerQA 评分 + EM/F1 审计
```

`AnswerFinalizer` 是工作流之后的提交处理步骤，不是 Director 创建的另一个 Agent。
用户逐项确认选择 A 后，正式配置只按明确格式确定性提取答案，不再调用模型。
当测试配置显式开启 `qa_model_enabled` 时，处理器才可能调用 `answer-formatter` 请求角色。
历史最高 EM 实验及固定对照配置使用 `gpt` 路由；模型只接收公开问题、原始输出与摘要，
不接收参考答案。固定对照中的 QA 证据重选开关关闭，候选必须来自原始 answer。

它可能改变用于评分的字符串。真实历史例子：

```text
输出 Agent 的 answer：Entropy, 1999
整理后的 submitted_answer：1999
```

两组历史记录已经有这一步：HotpotQA 120/128 条、NQ 81/128 条的提交方式为
`qa_model_source_span`。分别只有 2 条、1 条记录的提交字符串实际发生变化。
所以这一步不是此次更新新增，也不是每次调用都会改动答案。

补充逐次历史审计：09-16 两数据集各 128 题确实关闭了整理器；09-18～09-20 保存的
后续实验以及本机 09-24 试跑仍开启。上面的 120/81 是**采用模型候选数**，不是调用次数；
调用失败或候选被拒绝后也会记为确定性提取。完整统计见
[答案整理器历史开关审计](QA_FORMATTER_HISTORY_20260925.zh-CN.md)。

## “完整提示”具体指什么

“完整提示”是发送给模型的整套输入消息，不是“仪式”。需要区分三层：

| 输入 | 谁生成 | 谁接收 | 主要内容 |
|---|---|---|---|
| Agent 委派内容 | Director | 指定 Worker | role、objective、scope、expected_output；例如查找某电影上映年份 |
| Worker 完整请求 | 执行器将委派与通用规则、上下文组合 | Worker 模型 | 通用系统提示、公开题目、委派、上游/同层结果、历史结果、可用工具与观察、JSON 输出要求 |
| formatter 完整请求（正式版已关闭） | `AnswerFinalizer` | 额外的答案整理模型 | “从原答案抽取短答案”的系统提示，以及 question、raw_answer、raw_summary 等输入 |

之前确认“委派提示一致”，比较的是 Director 写出的职责字段与系统附加的输出合同。
这不自动证明执行器围绕它添加的整套 Worker 消息一致，更不证明图外 formatter 的请求一致。
现在已经可以用刚找回的旧源码明确比较这些层次，避免再用笼统的“prompt 一致”描述它们。
