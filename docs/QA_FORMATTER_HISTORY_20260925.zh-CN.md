# QA 历史实验答案整理器开关审计

## 结论和范围

确实存在未开启整理器的历史版本：`qwen35-9b-director-full-v3`（09-16）。
HotpotQA 和 NQ 各 128 条记录均保存 `enabled=false`、`qa_model_enabled=false`，
提交方式全部为 `legacy_passthrough`，formatter token 为 0，提交字符串未改变。

但 HotpotQA EM=73/128 的 09-18 实验、NQ EM=50/128 的 09-19 冻结证据实验，
两个开关均为 true，且确有模型整理调用。保存下来的 09-20 实验及 09-24 试跑也仍然开启。
现有证据未找到“后续某次起持续关闭整理器”的实验序列；不能因此断言用户未进行过未归档实验。

审计读取了迁移包内所有 `records.jsonl` 中的 QA 记录，以及本机 09-24 QA 评测记录，
共 1,196 条题目运行记录（跨实验有重复题目），另外检查了 NQ 训练试跑的 20 条 rollout。
目录清单中有 13 个 QA 相关运行未附逐题 records；其中一些只保留汇总或尚未完成。
这些运行不按完整轨迹计数，也不根据“没有模型提交记录”推断开关关闭。

机器可读结果及可重跑脚本：

- `state/audits/qa-formatter-history-20260925/results.json`
- `state/audits/qa-formatter-history-20260925/audit.py`

## 完整 128 题运行

“尝试整理”根据模型提交或 `fallback_used=true` 统计；不代表调用成功。
“采用”指 `qa_model_source_span`，候选可能与原答案完全相同。
通过率使用每次历史记录自身评分口径，因此不适合直接跨行比较。EM/F1 独立重新计算。

| 日期 / 运行 | 数据集 | 两个开关 | 尝试整理 | 采用模型候选 | 提交字符串变化 | 历史通过率 | EM | F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 09-16 full-v3 | HotpotQA | false / false | 0 | 0 | 0 | 21.88% | 21.88%（28） | 32.24 |
| 09-16 full-v3 | NQ | false / false | 0 | 0 | 0 | 15.63% | 15.63%（20） | 31.63 |
| 09-18 qa-sota DeepSeek | HotpotQA | true / true | 126 | 120 | 2 | 57.03% | 57.03%（73） | 75.74 |
| 09-18 qa-sota DeepSeek | NQ 在线检索 | true / true | 128 | 120 | 3 | 27.34% | 27.34%（35） | 44.55 |
| 09-19 frozen DeepSeek | NQ 固定证据 | true / true | 127 | 81 | 1 | 56.25% | 39.06%（50） | 54.02 |
| 09-19 aligned-v4 | HotpotQA | true / true | 128 | 121 | 4 | 53.13% | 53.13%（68） | 70.75 |
| 09-20 format-baseline-v2 | HotpotQA | true / true | 128 | 108 | 7 | 74.22% | 50.00%（64） | 67.71 |
| 09-20 role-conditioned-v2 | HotpotQA | true / true | 127 | 77 | 26 | 53.13% | 33.59%（43） | 54.25 |
| 09-20 role-conditioned-v3 | HotpotQA | true / true | 128 | 111 | 5 | 75.00% | 49.22%（63） | 68.05 |

关键运行目录：

- `state/sota-20260916/qwen35-9b-director-full-v3`
- `state/formal-eval/qa-sota-no-skill-deepseek-c10-20260918`
- `state/experiments/nq-frozen-v1/run-deepseek-qwen35-9b-128-20260919`
- `state/formal-eval/hotpot-flowsteer-aligned-no-skill-deepseek-c10-20260919-v4`
- `state/experiments/hotpotqa-format-baseline-20260920-v2`
- `state/experiments/hotpotqa-role-conditioned-20260920-v2`
- `state/experiments/hotpotqa-role-conditioned-20260920-v3`

以上均位于 `state/imports/nq-hotpotqa-migration-20260925/extracted/` 下。
逐题开关位置是 `trajectory.task.metadata.model_roles.answer_submission`；
实际执行证据在 `trajectory.answer_submission`。

## 小样本和训练试跑

- NQ `run-8`、`run-inline-8`、`run-inline-8-fixed-20260919-r3`、
  `run-span-regression-8`、`run-span-type-8`：每组 8 条，两开关均 true，每组均尝试整理 8 次。
- 本机 09-24 HotpotQA / NQ 评测各 2 条：两开关均 true，4 条均有 formatter token。
- 本机 NQ 训练试跑 cycle-0000 / cycle-0001 各 10 条：两开关均 true，20 条均有 formatter token。
  cycle-0001 的 10 条最终全部为 `qa_deterministic_extraction`，但 10 条均为调用后回退，
  不是关闭整理器。这说明不能单凭提交 method 推断是否调用模型。

历史 NQ128 的 46 次回退中，43 次记录 `BackendRequestError`，3 次记录候选校验 `ValueError`。
所以“采用了 81 条模型结果”不能写成“只调用了 81 次”；其余大部分也进入过模型调用路径。

另一个文件 `formatter-ablation-qwen35-9b.jsonl` 比较的是 `current` / `complete` 两种
Qwen 抽取提示，各 256 条。它没有 formatter-off 分组，也不是关闭整理器的整套工作流实验。

## Git 和当前开关

已找回完整 Git 历史。09-16 `cfab652` 的正式配置没有 `[answer_submission]` 段；
09-18 `496ab8e` 开始显式配置 `enabled=true`、`qa_model_enabled=true`、`runtime_route=gpt`。
后续保存的正式配置提交 `e46b384`、`3049203`、`fb73ce4`、`687eb83` 均保持开启。
这证明受版本管理的配置变化，不能代替每次运行的源码快照或证明不存在未提交修改。

逐项确认前的正式配置为：

```toml
[answer_submission]
enabled = true
qa_model_enabled = true
runtime_route = "gpt"
qa_evidence_spans_enabled = false
```

- `enabled=false`：关闭 QA 提交整理，原答案直接通过。
- `qa_model_enabled=false`：停止额外模型调用，仍可保留确定性格式提取。
- `qa_evidence_spans_enabled=false`：仅关闭从证据 passages 重新选择答案的分支；模型仍可从原答案抽取。

此次历史审计未改变上述开关，也未执行新的模型评测或训练。
评分继续按用户要求使用 FlowSteerQA，同时独立报告 EM/F1；该评分约定与 formatter 开关是两回事。

后续用户明确选择 A：三份正式训练/评测配置现已将 `qa_model_enabled` 改为 false，
`enabled` 保持 true，仅保留确定性格式提取。以上历史记录和审计结果保持原样；
`qa_best_recorded_eval.toml` 作为独立历史对照仍开启模型整理。
