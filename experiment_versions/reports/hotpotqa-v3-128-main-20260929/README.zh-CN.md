# 当前 main V3：HotpotQA 修订版 128 题重跑

按用户要求，使用恢复后的正式源码 `93c901d` 完成全部 128 题推理。
该提交的源码、测试及配置内容与 AIME 复核实验前的 `29e4f2f` 完全一致；
AIME 实验修改已经撤回。本次没有训练或更新模型参数。

| 指标 | 本轮 V3 | 2026-09-28 v2.2＋legacy |
|---|---:|---:|
| 严格答案 EM | **115/128（89.84%）** | 111/128（86.72%） |
| 答案 F1 | **93.58%** | 92.05% |
| 有效提交 | **128/128** | 128/128 |
| 项目 verifier 通过数 | 121/128（94.53%） | 119/128（92.97%） |

主准确率使用严格 EM，不能用项目 verifier 通过率替代。逐题配对后，
110 题两轮均正确、12 题两轮均错、5 题由错变对、1 题由对变错。
本轮调度异常为 0，128 题全部正常评分并具有 `unified_submission_v1` 回执。
轨迹内 128 次被接受的 `finish(target)` 均未执行 Worker，没有 `set_output`。

## 数据与运行配置

- 输入为 `data/formal/eval/hotpotqa_flowsteer_corrected_v1_128.jsonl`，
  版本 `flowsteer-hotpotqa-corrected-v1`；与旧轮输入 SHA-256 完全一致：
  `6e4096785ad5b869b6541c868beb25ad7f157cc64ae8968c0db3cbe2f185e27c`。
- Qwen3.5-9B Director，thinking 开启；复用本项目 GPU 4 上的
  `127.0.0.1:18605/v1` 服务。评测结束后继续保留该原有服务。
- DeepSeek Flash Worker，thinking 关闭，唯一 Worker 路由 `deepseek`。
- seed=0，题目与 Worker 路由并发上限均为 50；单题预算 240,000 token、
  最多 4 个 Agent、24 个 Director 回合；Skill 与外部检索关闭。
- 保留既有 Hotpot 证据优先、答案最后的输出契约。
- 冻结配置的通用提示词字段沿用 v2.2；Hotpot 的按数据集覆盖实际启用
  `director_action_json_v3` / `unified_task_result_v1`，已从运行清单和回执核对。

本轮去重 Worker 请求 152 次，全部成功且供应商模型均记录为 `deepseek-flash`。
输入 token 359,953，输出 token 35,880，共 395,833；这是应用记录用量。

## 复核与记录

离线脚本独立实现答案规范化、严格 EM 和 token F1，逐题结果与系统记录一致。
输入绑定核对包含加载器对题目外部空白的裁剪，以及冻结数据中已有的答案别名；
没有修改问题、参考答案、别名或评分代码。运行前后所有源文件哈希一致。

同题同模型名称的单轮对比并非只改变协议的受控消融：旧轮源码和 Director
服务实例不同；本轮净增 4 题不能全部归因为 V3。

- [汇总及哈希](audited-summary.json)
- [全部逐题对照](paired-cases.json)
- [13 道严格 EM 失分题](failed-answers.json)
- [冻结配置](config.toml)及[运行清单](manifest.json)

原始运行目录：
`/mnt/ssd/test/codex-students/student02/SelfPlayGraphFlowSteer/state/hotpotqa-v3-128-main-20260929/`。
其中 `results/records.jsonl` 保存完整 128 条结果和轨迹，`dataset.jsonl` 为冻结输入，
`run.sh` 为启动脚本，`audit_results.py` 可离线重新核算结果。
