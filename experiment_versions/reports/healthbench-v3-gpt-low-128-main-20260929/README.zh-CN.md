# HealthBench Professional：main V3，GPT-5.5（low），固定 128 题

按用户选择沿用 GPT-5.5（low）Worker，使用正式源码
`9803f058b7306c7bf2a3d5b874edb3e4db16acd6` 完成固定 128 题推理。
源码和正式训练配置未修改，未训练或更新模型参数。

| 指标 | 本轮 V3 | 9 月 25 日旧基线 |
|---|---:|---:|
| 长度调整分：全部 128 题、未提交计零 | **43.68** | **46.96** |
| 原始 rubric 分：全部 128 题、未提交计零 | **53.91** | **55.00** |
| 有效提交并完成评分 | **122/128** | 128/128 |
| 实际完成的 rubric 评分 | **249/258** | 258/258 |
| 项目阈值通过数（不是准确率） | 69/128 | 75/128 |

HealthBench 使用连续评分，上表不是准确率。按全部 128 题、缺失计零的口径，
长度调整分低 **3.28 分**，原始 rubric 分低 **1.10 分**。
仅对本轮实际完成评分的 122 题取均值，长度调整分为 **45.83**，
原始 rubric 分为 **56.56**；这一子集不能当作完整 128 题成绩。
先求样本分数均值，再将数据集均值裁剪到 [0, 1]，没有逐题截去负分。

## 未提交与接口恢复

- **5 题**：`director_context_budget_exhausted`，Director 上下文耗尽，未产生有效提交。
- **1 题**：`director_no_progress_exhausted`。图中只有局部任务节点，未形成可提交的
  `task_result`，运行时保留为 `unsubmitted_unknown`。上表按缺失计零处理，
  原始记录中仍为未知分数，没有伪造 rubric 判定。
- **1 题 Worker 接口中断**：主轮记录为 `worker_backend_unavailable`，随后以同样
  GPT Worker 池和 V3 协议单题补跑一次，成功提交并评分。补跑成绩照实计入，
  没有挑选较高分；原始失败记录仍保留。
- **1 题 Judge 接口失败**：原答案已提交。student 路由持续返回 400 上游兼容错误，
  保持答案、完整对话和 rubric 不变，改用同为 GPT-5.5（low）的 GPT Responses
  备用评分线路，成功补齐 2 条评分。没有重新生成该题回答。

主轮原始分布为 120 题已评分、1 题待评分、5 题运行时策略失败、2 题未提交未知。
完成上述两项接口恢复后为 122 题已评分、6 题未提交。协议失败题没有补跑。
最终 249 条 rubric 判定中，247 条返回模型 `gpt-5.5-2`，2 条为 `gpt-5.5`；
均为 Responses API、low 推理强度。6 条 Judge 请求错误保留在私有审计日志中。

## 评分与轨迹核对

离线审计逐条检查了 122 个 V3 提交回执、答案哈希和 249 条 Judge 回执，
独立重算 rubric 权重和长度调整公式，全部一致。
122 次被接受的 `finish(target)` 均未执行 Worker；全轮没有 `set_output` 调用。
已对全部已评分轨迹核对 Director 的完整公开对话输入；其任务元数据不含私有 rubric。
运行前后全部源文件哈希一致。

本轮已提交答案平均 5650.3 字符，旧轮为 4735.9 字符。按全 128 题口径，
平均长度扣分由 8.04 分增至 10.23 分。总分下降中约 2.19 分来自长度扣分增加，
约 1.10 分来自按缺失计零后的原始 rubric 均分下降；这是分数的算术分解，
不是协议的因果消融结论。

## 冻结配置与可比性

- 数据与旧基线字节一致，SHA-256：
  `991ed13ac248a3670e4caf7bdc83a770548c4209562c67a034631d4f0432042b`。
- Director：Qwen3.5-9B，thinking 开启；复用本项目 GPU 4 的 `127.0.0.1:18605/v1`，
  服务上下文上限 32,768，`SPGFS_DIRECTOR_CONTEXT_MODE=append_only`。
- 实际协议：`director_action_json_v3` / `unified_task_result_v1`。
  配置通用提示词字段仍为 v2.2，HealthBench 的按数据集覆盖实际启用 V3，已从运行清单和回执核对。
- Worker：逻辑路由 `gpt`，池成员 `gpt` / `gpt_eco` / `gpt_student`，全部 low；
  最终选定轨迹记录 264 次成功 Worker 请求，166 次返回 `gpt-5.5`、98 次返回 `gpt-5.5-2`。
  端点尝试失败另见私有 pool 日志，不计入这个成功请求数。
- seed=0；主轮题目并发 20；三个 Worker 端点并发分别为 10/10/5；Judge 共用 student 路由。
  单题预算 240,000 token、最多 4 个 Agent、24 个 Director 回合，relay 上限 1,000 字符；Skill、检索关闭。
- 历史基线使用外部无限重试及 Judge 自动备用线路包装；本轮运行当前正式代码的有限重试，
  仅单独恢复上述接口故障。旧轮 Director 服务实例、源码和 student 并发也有差异。
  因此不能把全部分差归因于 V3。

[审计汇总](audited-summary.json)、[冻结配置](config.toml)、[运行清单与源码哈希](manifest.json)。

完整私有产物目录：
`/mnt/ssd/test/codex-students/student02/SelfPlayGraphFlowSteer/state/healthbench-v3-gpt-low-128-main-20260929/`。

`results/records.jsonl` 是未经覆盖的主轮 128 条记录；
`recovery/results/records.jsonl` 是单题接口补跑；`rescore/records.private.jsonl`
是冻结答案补评分；`audited-records.private.jsonl` 是按上述规则合并的评测副本。
`audit_results.py` 可离线重算；`paired-scores.json` 保存逐题新旧分数；
`evaluated-code.tar.gz` 是本轮代码快照。原始对话、rubric、回答及轨迹只保存在本地忽略目录。
