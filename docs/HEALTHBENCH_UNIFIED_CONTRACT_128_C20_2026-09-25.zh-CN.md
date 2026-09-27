# HealthBench 答案规则统一修复与 128 条重跑

修复 Worker 系统提示要求 answer 完整，而题目末尾提交规则要求支持性说明放到 summary/evidence 的矛盾。新增纯文本模块 `healthbench_protocol.py`，让正常生成、格式恢复和 `submission_contract()` 共用同一份完整答案定义。必要解释、证据摘要、限定条件和不确定性必须包含在 answer 中；summary/evidence 保留内部协作用途。中间节点仍只完成分配给自己的职责。

## 对照与配置

对照为上一轮已修复 Director 完整对话输入的实验 `healthbench-director-context-gpt-low-128-c20-20260925-155258`。Worker 原有系统提示文字逐字保持一致，本轮仅将题目末尾的旧通用提交规则替换为同一完整答案规则。非 HealthBench 提交规则保留原行为。

- 同一批 128 条 HealthBench Professional，31 条多轮、97 条单轮，共 258 个 rubric。
- 数据 SHA-256：`991ed13ac248a3670e4caf7bdc83a770548c4209562c67a034631d4f0432042b`。
- 题目并发 20；judge 首选 gpt_student，全局评分并发上限 10。
- Qwen3.5-9B Director thinking 开启，复用 GPU 1 的同一服务，端口 18604。
- GPT Worker 路由池 gpt/gpt_eco/gpt_student，reasoning effort 为 low；其他配置与对照一致。
- 推理缺失有效结果才重试；评分失败则重试同一答案。已授权的备用 GPT judge 路由只在请求失败时使用，不按分数高低挑选结果。

## 验证

- 210 项相关回归测试通过。新增断言在修复前准确触发 6 项失败，覆盖正常生成、格式恢复、初次执行、修订执行，以及实际求解入口。
- 128/128 条题目和 258/258 个评分项完成；每个评分项都有与最终答案 SHA-256、题目 ID、rubric 索引和布尔判断相符的成功 judge 回执。
- 在真实 API 调用入口记录 272 次 Worker 请求，其中 59 次为恢复请求；覆盖 128 条题目。系统提示词和 public_task_context 均含统一规则，冲突规则出现 0 次。
- 128 条首轮 Director 请求均保留完整公开对话；私有 rubric、参考回答和 canary 不进入生成上下文。
- 成功评分请求路由：`{"gpt_student": 256, "gpt_eco": 2}`。
- 成功 Worker 请求路由：`{"gpt": 90, "gpt_eco": 91, "gpt_student": 87}`。
- 重试事件计数（含完成事件）：`{"judge_route_failure": 2, "judge_fallback_success": 2, "complete": 1}`。
- 收尾时 Qwen `/v1/models` 返回 HTTP 200，PID 14783 仍在运行，模型保留在 GPU 1。

## 配对结果

除字符数、token 和计数外，分数均为 100 分制。

| 指标 | 修复前 | 修复后 | 变化 |
|---|---:|---:|---:|
| 长度调整后的平均分 | 42.03 | 46.96 | +4.93 |
| 原始 rubric 平均分 | 48.24 | 55.00 | +6.77 |
| 平均答案字符数 | 4112.76 | 4735.89 | +623.13 |
| 平均摘要字符数 | 298.43 | 271.44 | -26.99 |
| 平均总 token | 6713.88 | 7128.15 | +414.27 |
| 多轮题（31 条）调整后平均分 | 31.09 | 52.43 | +21.34 |
| 多轮题（31 条）原始平均分 | 36.28 | 59.58 | +23.30 |
| 单轮题（97 条）调整后平均分 | 45.52 | 45.21 | -0.31 |
| 单轮题（97 条）原始平均分 | 52.06 | 53.54 | +1.48 |
| 项目阈值通过题数 / 128 | 66 | 75 | +9 |
| 正向评分项命中数 / 214 | 137 | 148 | +11 |
| 负向评分项触发数 / 44 | 18 | 18 | +0 |
| 少于 300 字符的答案数 | 8 | 4 | -4 |
| 摘要长于答案的题数 | 3 | 0 | -3 |
| Director 未主动结束题数 | 0 | 2 | +2 |

逐题原始 rubric 分数：22 条提高、12 条下降、94 条不变。调整分平均差的配对 bootstrap 95% 区间为 [-2.05, 11.96] 分。该区间跨过 0，本次重跑不能证明总体分数有稳定变化。

前后均由 student judge 完成全部评分的共同子集有 127 条，该子集调整分变化 +4.91 分，原始分变化 +6.82 分。逐题路由与排除名单保存在 `comparison.json`。

## 解释边界

本次测量统一提示规则后的整套生成与评分结果。提示变化可能改变回答内容、后续 Director 决策、图结构和路由调度；不能把所有分差等同于原先“写在 summary 而未提交”的纯内容损失。配对 bootstrap 覆盖样本题目的重采样波动，不覆盖端点变化或重复运行方差。

“项目阈值通过”是 score≥0.5 的项目策略，不是 HealthBench 官方准确率。本轮没有重跑官方单模型基线，也没有更改节点间答案传递机制。

## 文件

- 共享规则：`src/selfplay_graph_flowsteer/healthbench_protocol.py`。
- 接入：`src/selfplay_graph_flowsteer/answer_submission.py`、`src/selfplay_graph_flowsteer/runtime.py`。
- 回归测试：`tests/test_healthbench_answer_protocol.py`、`tests/test_healthbench_director_context.py`。
- 独立补丁：`experiment_versions/patches/healthbench-unified-contract-v1.patch`。
- 实验目录：`state/experiments/healthbench-unified-contract-gpt-low-128-c20-20260925-164301`。
- 结果审计：`verified_results.json`；配对统计：`comparison.json`。
- 实际请求规则检查：`provenance/worker_request_contract_checks.jsonl`。
- 完整结果：`results/records.jsonl`、`results/samples/`；原始评分回执位于实验私有审计目录。
