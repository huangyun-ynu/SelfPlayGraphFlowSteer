# AIME 2026：当前 main 的 V3 协议 30 题推理

按用户要求，使用当前 main `42dd925` 完成固定 AIME 2026 全部 30 题的新协议推理。
本次只做评测，没有训练或修改模型参数。

| 指标 | 本轮 V3 |
|---|---:|
| 完成 / 计划 | 30/30 |
| 数值答案正确 | **20/30（66.67%）** |
| 有效提交回执 | **30/30** |
| 未评分 / 调度失败 | 0 / 0 |

已独立把 30 个最终提交整数与输入文件的 `reference` 比对：20 个一致，
与系统逐题分数完全相同。所有结果的 `outcome_status=scored`，
`submission_contract_version=unified_submission_v1`；30 条轨迹均有该版本回执，
实际图协议为 `unified_task_result_v1`。30 次被接受的 `finish(target)`
均未执行 Worker，轨迹里没有 `set_output` 动作。

## 运行设置

- 固定输入：`data/formal/eval/aime_official_test.jsonl`，30 个唯一 ID，SHA-256
  `f50a3ba12616c8d87b37d1fd7a0a2c2b5e440c18388db0991855423856a610c7`。
- 使用本次 main 源码和旧 AIME 评测配置作底，只为 AIME 明确启用
  `unified_task_result_v1`；生效 Director 提示词为 V3。
- Qwen3.5-9B Director thinking 开启，复用本项目现有 `127.0.0.1:18605/v1`
  服务（GPU 4）；没有额外启动或停止模型服务。
- Worker 固定 DeepSeek Flash，thinking 关闭；单题 240,000 token、最多 4 个
  Agent 和 24 轮；AIME 本地 Python 工具及其调用预算保留。
- seed=0；题目并发上限 30，DeepSeek 路由并发上限 50；Skill 关闭。
- 启动脚本、冻结配置、输入/配置指纹及全部原始结果在
  `state/aime-v3-30-main-20260929/`。首次预检因沿用旧配置的 GPU 1 资源编号
  被配置校验拒绝，改为实际复用服务所在 GPU 4 后才开始本轮推理。

## 与旧协议结果对照

2026-09-28 旧协议使用同一批 30 题、seed=0、同名 Qwen3.5-9B Director
与 DeepSeek Flash Worker，得到 **24/30 正确、28/30 有效提交**。
本轮 V3 为 **20/30 正确、30/30 有效提交**。逐题配对后，19 题两轮都正确、
5 题两轮都错、5 题由正确变错（05、10、11、18、23），1 题由错误变正确（21）。

这不是只改协议的受控消融：运行源码版本、Director 服务实例和服务环境也不同。
本轮说明 V3 提交路径已经跑通，但不能仅凭两轮差值断定准确率下降完全由协议造成。
本轮错误的 10 道题 ID、提交答案、参考答案和文件哈希见 [审计清单](audit.json)。

完整轨迹：`state/aime-v3-30-main-20260929/results/records.jsonl`，共 30 条。
汇总：`state/aime-v3-30-main-20260929/results/aggregate_summary.json`。
旧版逐题记录保存在同一学生工作区的
`SelfPlayGraphFlowSteer-hotpot-answer-contract/state/hotpot128-aime30-c50-20260928-204627/aime/results/records.jsonl`。
