# HealthBench 完整答案协议修复：128 条配对实验

完成日期：2026-09-25（Asia/Shanghai）。128/128 条均已评分，258/258 个 rubric 逐项与 judge 原始成功请求的题目 ID、答案 SHA256、rubric 索引和判断匹配。

后续检查补充：本页记录的分数和实验改动仍有效，但生产入口还残留旧 submission contract，Director 也未接收初始多轮历史，节点传递还会压缩掉 answer。上一轮测试未覆盖这些完整路径，不能据此认为端到端协议已经一致。详见[剩余失分检查](HEALTHBENCH_RESIDUAL_ERROR_AUDIT_2026-09-25.zh-CN.md)。

只修改 HealthBench 的 Worker 最终答案协议：在正常生成和格式恢复中都要求 `answer` 包含完整结果、必要说明、依据、限定条件和不确定性；`summary/evidence` 保留内部协作用途。中间节点仍只完成自身职责，最终节点提交完整用户回答。没有追加整理模型调用。

| 指标 | 修复前 | 修复后 | 变化 |
|---|---:|---:|---:|
| 长度修正均分 /100 | 30.4587 | 41.3251 | +10.8664 |
| 原始 rubric 均分 /100 | 29.1034 | 46.5514 | +17.4479 |
| 正向标准命中 /214 | 92 | 134 | +42 |
| 负向标准触发 /44（越少越好） | 17 | 19 | +2 |
| 项目 0.5 阈值通过 /128 | 51 | 70 | +19 |
| 平均答案字符数 | 1539.03 | 3777.65 | +2238.62 |
| 原始零分题数 | 55 | 37 | -18 |
| 平均 Worker token（项目口径） | 6000.81 | 6638.44 | +637.62 |

这些是 rubric 分数，不能称作准确率。原始分 34 题提高、6 题下降、88 题不变。配对 bootstrap（10,000 次，seed=20260924）长度修正分差的 95% 区间为 [3.79, 17.80]。

## Judge 恢复与同路由对比

127 条使用原 student judge，最后 1 条的 student 请求持续 HTTP 400。用户授权临时更换 GPT 路由后，仅恢复这 1 条未完成任务，以 GPT eco、low、Responses 接口评分。由于原进程在评分结束前没有落盘答案，该题恢复时重新生成了答案，2 个 rubric 全部对应新答案；没有按分数高低选择结果。恢复前后，其余 127 条 sample 和 trajectory 文件 SHA256 完全相同。

排除这 1 条，**同为 student judge 的 127 条**：长度修正分 29.9591 → 40.9424，提升 10.9834 分，95% 区间 [4.07, 18.25]；原始分 28.5452 → 46.1305，提升 17.5853 分。因此本轮提升并非由最后一题的 judge 切换所造成。

## 对“内容没有完整提交”的量化解释

这次协议修改的观测净收益约 **10.87 个长度修正分**，原始 rubric 收益约 **17.45 分**。答案变长使平均长度修正贡献从 +1.3552 变为 -5.2263，抵消了约 6.58 分收益。正向内容覆盖改善明显，负向错误没有同步减少。

EDI 案例从 184 字符答案、0/3 标准命中，变为 12,625 字符、3/3 命中；长度修正后为 68.76 分，说明完整性改善与冗长惩罚同时存在。少于 300 字符的答案由 46 条降为 8 条，summary 长于 answer 的题目由 54 条降为 3 条。未正常结束的 Director 轨迹由 4 条变为 2 条，这些题目仍有答案且已完整评分。

这里测量的是**修改协议后的整体效果**。提示变化也会影响生成内容、后续 Director 决策和路由分配，不能把全部提升解释为原先被遗漏文字的纯因果损失。一次配对重跑及其 bootstrap 不涵盖跨次生成随机性、provider 变化；这 128 题已用于诊断，也不能视为独立留出集。

## 配置、验证与文件

数据、seed=0、Qwen3.5-9B 开思考、GPT Worker 池 low、轨迹并发 10、judge 并发 10、skill-context off 保持原配置。初始 judge 为 student，最后一题有上述 GPT eco 恢复例外。Qwen 在 GPU 1、端口 18604，实验收尾健康检查 HTTP 200，继续保留服务。

69 项相关测试通过；冻结快照中的 4 项协议测试通过；72 组提示比对确认其他数据集的正常和恢复提示与修改前逐字一致。合成测试验证完整 answer 传输、内部字段不参与评分、公开多轮上下文保留以及私有 rubric 不进入生成提示。

- 实现：[runtime.py](../src/selfplay_graph_flowsteer/runtime.py)
- 回归测试：[test_healthbench_answer_protocol.py](../tests/test_healthbench_answer_protocol.py)
- 单项代码补丁：[healthbench-full-answer-v1.patch](../experiment_versions/patches/healthbench-full-answer-v1.patch)
- 可版本化统计：[healthbench-full-answer-128-20260925.json](../experiment_versions/reports/healthbench-full-answer-128-20260925.json)
- 本地完整报告、原始结果和审计：`/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/experiments/healthbench-full-answer-gpt-low-128-20260924-231723`；其中 `REPORT.zh-CN.md` 为详细报告，`results/records.jsonl` 为 128 条结果，`comparison.json` 含逐题分差，`private/judge/` 为评分审计。题目原文、答案和私有 rubric 仅保留本地。

数据 SHA256：`991ed13ac248a3670e4caf7bdc83a770548c4209562c67a034631d4f0432042b`。
