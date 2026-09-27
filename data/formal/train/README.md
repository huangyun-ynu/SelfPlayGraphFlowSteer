# Formal training pools

This directory contains the seven fixed training pools used by the formal
training configuration. Each JSONL file has 512 records, all marked with
`split=train`.

The pools are copied from the validated ADS task pools generated under the
local training state. Experiment logs, trajectories, checkpoints, retrieval
corpora, model files, virtual environments, and other runtime state are not
included here.

| Dataset | Records |
| --- | ---: |
| AIME | 512 |
| ALFWorld | 512 |
| HealthBench Professional | 512 |
| HotpotQA | 512 |
| NQ-open | 512 |
| SWE-bench Verified | 512 |
| WebShop | 512 |

HotpotQA 于2026-09-27替换为 FlowSteer 分层抽样512题（easy98/medium327/hard87；bridge414/comparison98），已重算ADS，排除公开128测试题重叠。抽样清单位于 ../sources/flowsteer/hotpotqa_train_512.selection.json。

SWE-bench 于2026-09-27对齐SkillFlow：512条训练记录、372个独立实例，四档难度195/276/39/2，与新128条测试集零重叠；ADS已重算。详见 docs/SWE_SKILLFLOW_DATASET_20260927.zh-CN.md。

WebShop 已按用户要求恢复为切换SkillFlow之前的512条训练数据及128条测试数据，原ADS特征和实际训练池同步恢复。SkillFlow重跑记录仅作历史保留。
