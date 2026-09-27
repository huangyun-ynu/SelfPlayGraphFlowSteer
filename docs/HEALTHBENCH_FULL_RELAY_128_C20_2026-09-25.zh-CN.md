# HealthBench 完整正文传递修复与 128 题重跑

正式部署后续决定：用户选择 46.96 分的上一版代码作为正式训练基线，已撤回正式源码中的
完整正文传递改动。本报告记录实验完成时的状态；该实验的冻结代码、固定路由和结果继续
独立保留。当前正式设置见 [正式同步记录](HEALTHBENCH_FORMAL_PROMOTION_2026-09-25.zh-CN.md)。

修复 `MultiAgentRuntime._packet()` / `_bounded_artifact()`：由 Canvas 管理的
`action_adapter=healthbench_professional` 节点传递完整 answer，覆盖上游结果、同伴提案、
自身旧稿和输出恢复。摘要及辅助字段仍按原有限制传递；非 HealthBench 保持旧规则。
判定依据是节点的系统元数据，Worker 不能通过生成文本关闭传递限制。

正式训练共用源码已包含本修复。实验从上一轮冻结代码复制，仅替换 runtime.py，
避免工作区其他数据集的同期改动影响对照。基线：`healthbench-unified-contract-gpt-low-128-c20-20260925-164301`。

## 配置与完成情况

- 同一批 128 条 HealthBench Professional，31 条多轮，共 258 个评分项。
- 数据 SHA-256：`991ed13ac248a3670e4caf7bdc83a770548c4209562c67a034631d4f0432042b`。
- Qwen3.5-9B Director thinking；GPU 1、端口 18604；题目并发 20。
- GPT Worker 池 gpt/gpt_eco/gpt_student，low；Judge 首选 student，并发 10。
- 评分失败重试同一答案，备用 GPT Judge 仅在请求失败时使用，不按得分挑选答案。
- 128/128 条完成、258/258
  评分项具有与最终答案 SHA-256 匹配的成功回执。
- 当前源码及实验冻结版本均通过 216 项回归；修复前实际请求测试
  能复现正文为空，非 HealthBench 对照通过。
- 收尾正式源码再通过 14 项 HealthBench
  专项测试；MultiAgentRuntime 的方法 AST 与实验快照一致，保留工作区其他数据集的同期修改。

## 正文传递审计

历史基线中 21 题的 88 次实际消费引用会清空 answer；本轮同时审计消息构造和实际 API 输入：

- 消息构造检查 118 次，原始正文与消息正文哈希不一致
  0 次。
- 实际 Worker 请求包含 130 次引用
  （包括格式恢复），与原始正文不一致 0 次。
- 最终成功轨迹含 110 次消费引用，
  涉及 26 题；均找到完整正文的实际请求证据。
- 其中 2
  次引用的中间稿未保留在最终事件快照中，改用消息构造时实时记录的原始 artifact 哈希
  与实际 API 输入交叉校验；没有据此跳过检查或改动答案、评分。
- 各类实际消息：`{"peer_proposal": 57, "self_proposal": 56, "upstream": 1}`。
- 统一答案规则仍在全部 284 次 Worker 请求中生效；
  冲突规则 0 次。Director 完整公开上下文
  128/128 条验证通过。

## 与上一轮比较

分数为 100 分制。

| 指标 | 修复前 | 修复后 | 变化 |
|---|---:|---:|---:|
| 长度调整后平均分 | 46.96 | 41.90 | -5.06 |
| 原始 rubric 平均分 | 55.00 | 52.13 | -2.87 |
| 平均答案字符数 | 4735.89 | 5480.44 | +744.55 |
| 平均总 token | 7128.15 | 9586.13 | +2457.98 |
| 31 条多轮题调整后均分 | 52.43 | 34.86 | -17.57 |
| 97 条单轮题调整后均分 | 45.21 | 44.15 | -1.07 |
| 项目阈值通过题数 / 128 | 75 | 73 | -2 |
| 正向评分项命中 / 214 | 148 | 140 | -8 |
| 负向评分项触发 / 44 | 18 | 15 | -3 |
| Director 未主动结束题数 | 2 | 3 | +1 |

逐题原始分：14 条提高、21 条下降、
93 条不变。调整分配对 bootstrap 95% 区间为
[-12.53, 2.68] 分。

本次调整后均分变化 -5.06 分，可算术分解为：原始 rubric
均分变化 -2.87 分，长度修正额外变化
-2.19 分。
平均答案长度从 4736 增加至 5480
字符。正向评分项命中由 148 降至
140，负向项触发由 18
降至 15。这只是分数构成，不能当作因果归因。
本次没有观察到整体分数提升；配对区间跨越 0，单次重跑不足以确定稳定收益或损失。

预先按基线锁定的 21 条正文曾被清空的题目，调整后均分变化
-9.70 分，原始分变化 -8.23 分。
该分组在本轮完成前固定，但子集仍小且受生成随机性影响，不构成因果估计。

前后全由 student 评分的共同子集为 126 条；完整统计及备用路由使用情况见
`comparison.json`。成功 Judge 请求路由：`{"gpt_student": 257, "gpt": 1}`。
成功 Worker 请求路由：`{"gpt_eco": 85, "gpt_student": 93, "gpt": 89}`。
重试事件：`{"judge_route_failure": 2, "inference_retry": 1, "judge_fallback_success": 1, "complete": 1}`。

这是同题、单次配对重跑。完整中间正文会影响后续回答、图结构和路由调度；总分差同时
包含生成随机性及端点波动，不能等同于丢失正文的纯因果损失。bootstrap 只覆盖样本重采样。
项目 score≥0.5 的通过题数不是官方准确率。

## 资源与产物

收尾 Qwen API HTTP 200，PID 14783 存活，GPU 1 模型保持加载。
没有执行模型停服、GPU 释放或 pod 释放。

- 实验目录：`state/experiments/healthbench-full-relay-gpt-low-128-c20-20260925-173438`。
- 逐题结果：`results/records.jsonl`；评分审计：`verified_results.json`。
- 传递审计：`relay_audit.json`；逐题配对：`case_comparison.json`。
- 实际消息哈希：`provenance/relay_packet_checks.jsonl`、`provenance/worker_request_contract_checks.jsonl`。
- 回归测试：`tests/test_healthbench_relay.py`；补丁：`experiment_versions/patches/healthbench-full-relay-v1.patch`。
