# 正式评测指标

每个 cycle 的 `primary_outcomes.json` 在 `datasets.<dataset>.benchmark` 保存基准汇总。
已有 W&B 数值上报器自动上传到 `eval/datasets/<dataset>/benchmark/`（评测周期）
或 `cycle/datasets/<dataset>/benchmark/`（训练周期）。训练周期成绩不应当作测试成绩。

## 分母与重复采样

- `observed`：仅在可评分样本上计算，同时报告 observed_slots/planned_slots。
- `missing_as_zero`：计划执行但未获得指标的槽位按零计入，是明确标记的缺失处理方案。
- 每个 source_id 内先平均重复轨迹，再对 source 等权平均；缺少 source_id 时使用 task_id。
- 95% CI 对 source 均值做 1000 次固定种子 bootstrap；只有一道题时不输出区间。
- 该区间估计题间不确定性，不等同于 HealthBench 论文的题内/题间方差分解。
- 旧 task_pass_at_k 继续保留，其含义是至少一次成功，与重复采样平均准确率不同。

## 主指标

| 数据集 | benchmark.metrics |
| --- | --- |
| AIME | accuracy |
| NQ-open | answer_em、answer_f1、success_rate |
| HotpotQA | answer_em、answer_f1、support_em、support_f1、joint_em、joint_f1 |
| WebShop | score、success_rate |
| ALFWorld | success_rate |
| SWE-bench | resolved_rate，另有 swe_status 数量汇总 |
| HealthBench Professional | length_adjusted_score、raw_rubric_score |

分数使用 0 到 1 的比例单位展示；HealthBench 原始单题分数允许负值，总分先平均再截断。
展示百分制时乘 100。HealthBench 训练奖励不是这里的基准分数。

## 分组与诊断

按 source_split（无该字段时用 split）、task_type、difficulty、use_case 和 interaction_type 分组。
ALFWorld 优先使用 alfworld_task_type。没有标签时不会推断标签或制造子集结果。
QA precision/recall、提取后 EM/F1、HealthBench 答案字符数与正负 rubric 命中数量也进入数值汇总。

## HotpotQA 证据协议

可评分的证据必须由模型在最终输出中显式提交 JSON：

```json
{"answer": "answer text", "supporting_facts": [["Document title", 0]]}
```

句子编号从 0 开始。金标准必须位于 TaskSpec.private_verifier_payload.supporting_facts；
评分器不从答案、轨迹或参考证据反推模型引用。缺失金标准时 support/joint 为 null，
报告 evidence_coverage。存在金标准但未合法提交证据时按零分处理。

当前正式物化 HotpotQA 池未保留 supporting_facts；现有运行仍是 Answer-only。
要报告完整 joint 成绩，需要另行从对应官方样本恢复私有金标准和带句号编号的上下文，
并给模型启用此提交协议后重新评测。新增评分路径本身不能补出既有运行的证据成绩。

SWE 的汇总不能把未评分一律认定为 empty patch 或 harness error；详细故障类型继续看原有诊断。
这些改动仅增加观测统计，不改变训练奖励、任务提交合法性或训练准入。
