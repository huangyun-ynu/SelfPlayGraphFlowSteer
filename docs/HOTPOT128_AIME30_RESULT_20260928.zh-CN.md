# HotpotQA 128 → AIME 30 推理结果（2026-09-28）

## 本轮结果

按用户要求，先跑修订版HotpotQA 128，全部完成后再跑现有正式AIME 2026共30题。两轮都完成了全部计划轨迹；系统执行异常为0。

| 数据集 | 正确数/全部题数 | 主准确率 | 有效提交 |
|---|---:|---:|---:|
| HotpotQA 本地修订版 v1，严格EM | 111/128 | 86.72% | 128/128 |
| AIME 2026，数值答案正确率 | 24/30 | 80.00% | 28/30 |

HotpotQA答案F1为92.05%，FlowSteer通过数为119/128（92.97%）。严格EM已独立用冻结输入中的reference/target_answers重新计算，与轨迹中记录的结果一致。

AIME有4道提交了错误答案，2道未提交。主成绩分母始终是全部30道，未提交也计为失败。原聚合文件中的`pass_rate=85.71%`仅对应已提交的24/28，不能当作本轮全体准确率；全体准确率对应`successful_submission_rate_all=80.00%`。

### AIME未通过的题

| ID尾号 | 本轮答案 | 参考答案 | 状态 |
|---|---|---|---|
| 14 | 空 | 681 | 未产生有效提交回执 |
| 16 | 32 | 178 | 答错 |
| 21 | 8 | 50 | 答错 |
| 22 | 空 | 754 | 未产生有效提交回执 |
| 26 | 144 | 132 | 答错 |
| 28 | 111 | 107 | 答错 |

这里的ID尾号沿用本地固定数据集`aime-2026/2026/<编号>`，不是重新解释成某一张试卷上的题号。未提交记录的原因字段为`missing_runtime_submission_receipt`；本轮未做补跑来替代失败，也未修改评分代码。

## 配置与版本

- 推理源码：Hotpot实验副本，commit `698ef78eb000876678dc3a72d6519da2a461babd`，包含已在16题验证的证据/summary先于answer的Hotpot输出契约，以及14条数据修订。
- Hotpot数据：`flowsteer-hotpotqa-corrected-v1`。SHA256 `6e4096785ad5b869b6541c868beb25ad7f157cc64ae8968c0db3cbe2f185e27c`。
- AIME数据：项目现有`data/formal/eval/aime_official_test.jsonl`，30题。SHA256 `f50a3ba12616c8d87b37d1fd7a0a2c2b5e440c18388db0991855423856a610c7`。
- Qwen3.5-9B Director，prompt v2.2，thinking on；DeepSeek Flash Worker，thinking off，唯一Worker路由`deepseek`。
- 两轮使用同一份配置：题目并发上限50，DeepSeek路由并发上限50，`FIRST_COMPLETED`完成一题即补一题；AIME只有30题，最多同时运行30题。
- 两个数据集的单题token预算均为240000，最多4个Agent、24轮；技能上下文off，检索off。AIME保留项目已有的本地Python计算工具。
- 启动时检测所有GPU并选中GPU1，服务地址`127.0.0.1:18632`。选择依据与检测快照已保存，没有抢占或终止其他服务。
- SWE关闭，未启动SWE远程评测服务器。未做训练或参数更新。

实际运行清单的模型、thinking、Worker路由、路由并发、题目并发、token预算、执行语义均逐字段核对一致，详见`shared-config-check.json`。

Hotpot原始128题历史成绩102/128保持原样。本轮同时采用修订数据与候选推理代码，不能将分数差归因于单独一项模型改进，也没有重写历史结果。

## 调用与用量

| 数据集 | 去重Worker请求事件 | 记录输入token | 记录输出token |
|---|---:|---:|---:|
| HotpotQA | 264 | 605,317 | 58,171 |
| AIME | 389 | 1,441,411 | 182,662 |

653条去重Worker事件均为`backend_request_success`，路由全部为`deepseek`。652条带`provider_model`的记录均为`deepseek-flash`；AIME ID18有1条缺失provider_model/usage_source，已单独列出其event_id，不能声称这1条具有完整供应商返回元数据。其路由仍为deepseek，冻结配置仅包含Flash Worker，未配置Pro或其他模型回退。token表是应用日志记录值，未换算账单金额。

## 执行顺序与资源清理

- HotpotQA：2026-09-28T20:48:59+0800 → 2026-09-28T20:52:39+0800。
- AIME：2026-09-28T20:52:39+0800 → 2026-09-28T20:56:33+0800。
- 清理完成：2026-09-28T20:56:35+0800；本轮Director确认`running=false`，18632端口已关闭。
- 全部158条轨迹的ID、prompt及reference与冻结数据一致，源码哈希在运行前后保持一致。

## 可复核文件

- 原始运行目录：`/mnt/ssd/test/codex-students/student02/SelfPlayGraphFlowSteer-hotpot-answer-contract/state/hotpot128-aime30-c50-20260928-204627`。
- `hotpotqa/results/records.jsonl`、`aime/results/records.jsonl`：完整158条结果和轨迹。
- `audited-summary.json`：按全部题数计算的汇总及用量。
- `failed-answers.json`：Hotpot 17条严格EM失分、AIME 6条未通过记录。
- `manifest.json`、`config.toml`：冻结的数据/源码哈希与配置。
- `resource-check.json`：停止服务、端口及GPU状态核查。
- `audit_results.py`：离线复核脚本，仅读取已生成结果，不调用模型。

汇总、配置与审计副本已保存到项目内`experiment_versions/reports/hotpot128-aime30-c50-20260928-204627`；大型原始轨迹保存在上述state目录。
