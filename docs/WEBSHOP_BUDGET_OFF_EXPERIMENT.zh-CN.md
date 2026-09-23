# WebShop 关闭请求预算拦截：128 题对照实验

2026-09-23，按用户要求重跑完整 128 题，比较关闭 WebShop 请求前预算估算拦截前后
的严格成功率、平均得分和 token 用量。128/128 完成，任务级异常 0，进程退出码 0。
实验结束后 Qwen 服务与额外显存占位进程继续运行。

## 对照与配置

- 对照：`state/formal-eval/webshop-laser-checklist-c24-20260923-182224/`。
- 新运行：`state/formal-eval/webshop-budget-off-laser-c24-20260923-191922/`。
- 审计目录：`state/experiments/webshop-budget-off-laser-c24-20260923-191922/`。
- 配置：`configs/webshop_laser_checklist_eval.toml`，同一批官方 128 题、seed 0、
  24 路轨迹；DeepSeek Worker 的服务并发仍为 20。
- Qwen3.5-9B Director：复用 `http://127.0.0.1:18623/v1`，常规调用 thinking 开启，
  关系二选一沿用原有非思考策略。
- Worker：DeepSeek `deepseek-flash`，thinking 开启、reasoning effort 为 low、
  单次输出上限 16384。legacy 页面，LASER 提示，无 skill，环境反馈关闭。
- 初始 12 次、修订 4 次、合计 16 次动作；实际 Worker token 总上限 350000，
  仍在执行报告返回后核验。

两组配置文件与数据哈希一致。对照组虽然已经将
`remaining_token_admission_enabled=false`，旧代码仍执行请求额度分配、收尾预留和
请求前估算。新代码将这些路径一起接入该开关。与对照启动清单相比，仅
`application.py`、`canvas.py`、`config.py`、`runtime.py` 发生变化，差异均为该修复及
相关审计字段。对照源码通过当时 Git 提交和保留补丁重建，并逐文件校验原始哈希。
完整差异保存在 `change_from_baseline.patch`，实验过程中未发生源码或配置漂移。

## 结果

| 指标 | 关闭前：上一轮 LASER | 关闭后 | 变化 |
| --- | ---: | ---: | ---: |
| 严格成功 | 62/128 | 63/128 | +1 题 |
| 严格成功率 | 48.4375% | 49.2188% | +0.7813 个百分点 |
| 平均得分（百分制） | 69.5052 | 71.0221 | +1.5169 |
| 平均 Worker token | 69492.09 | 68111.69 | -1380.40（-1.99%） |
| 平均 Qwen Director token | 41883.66 | 41688.06 | -195.60（-0.47%） |
| 平均合计 token | 111375.75 | 109799.75 | -1576.00（-1.42%） |
| 128 题合计 token | 14256096 | 14054368 | -201728 |
| 平均题目耗时（秒） | 108.34 | 105.03 | -3.30（-3.05%） |
| 请求预算拒绝 | 4 次，涉及 3 题 | 0 | -4 次 |

逐题成功变化：55 题两次均成功、58 题两次均失败、8 题失败变成功、7 题成功变失败。
配对 McNemar 精确双侧检验 p=1.0。单次实验观察到小幅净提升及 token 减少，
不足以认定关闭预算拦截带来稳定准确率收益。

## 原来被拦截的三题

| 题目 | 原拒绝次数 | 关闭前得分 | 关闭后得分 | Worker token：前 → 后 |
| --- | ---: | ---: | ---: | ---: |
| goal-00365 | 2 | 1.00 | 1.00 | 145867 → 145015 |
| goal-00135 | 1 | 1.00 | 1.00 | 148986 → 124462 |
| goal-00031 | 1 | 1.00 | 0.75 | 160214 → 143060 |

这些题在关闭前已通过后续执行完成满分购买，因此本轮不存在“原被拦截失败的题直接
变成成功”的三题证据。goal-00135 的 Qwen 用量从 179078 降至 46934，减少的
请求与调度开销是本轮总 token 差值的重要组成；具体轨迹仍可能受采样影响。

上一轮在 Director 配置阶段失败、未进入 Worker 的 goal-00387 本轮成功；该题的
原失败并非请求预算拒绝，不能将其改善直接归因于本次修复。此前审计提到的
goal-00197，在本次选用的 LASER 对照和关闭后运行中均为满分；其原始零分案例属于
更早的正式历史参考运行，不能混用对照版本。

## token 统计方法

Worker 用量从轨迹中的执行报告累计，并对重复携带的同一报告去重，包含双向执行中
被后续修订替换的中间执行。逐题与 Canvas 反馈中的累计账本核对，两组均无差异。
本次两组的这一统计也与原 `records.jsonl` 的 token_cost 一致。

Qwen 用量按各 Director turn 的 `backend_request_events` 中 `completion_usage`
统计，按 event_id 去重，包含分开的 thinking 和动作生成请求。两组均没有缺失
用量的成功 Director 请求。合计为 Worker 加 Qwen，未把两者重复相加。
这些是已记录的输入/输出 token，用量包含重复输入，不是美元费用或预算估算。

| 用量明细 | 关闭前 | 关闭后 |
| --- | ---: | ---: |
| Worker 输入 | 8145573 | 8031953 |
| Worker 输出 | 749414 | 686343 |
| Qwen 输入 | 5123138 | 5100248 |
| Qwen 输出 | 237971 | 235824 |

## 生效检查与服务保留

- 128 题均进入 Worker，169 份去重执行记录全部走 DeepSeek。
- 169 份记录均为 `request_admission_enabled=false`，活动请求额度 0，
  请求预算报价 0，请求预算拒绝 0。
- 156 份会话所有者执行使用 LASER 提示，13 份无状态评审按原设计不注入。
- Qwen 常规 graph_action 641 次，requested/effective thinking 均为 true；
  12 次关系二选一均为 false，符合原策略。
- DeepSeek 1198 个成功请求，无已记录请求失败；1041 次报告正 reasoning tokens，
  2 次报告 0，155 次未报告该字段。缺失字段不解释为关闭 thinking。
- 服务结束检查正常：Qwen API server PID 37099、engine PID 39080 继续运行。
- GPU 1 的额外显存占位 PID 19576 继续运行，保留 2415919104 字节（2.25 GiB），
  设备保留约 1 GiB 的运行余量。本次未停止模型或释放占位。

结果与逐题数据：`budget_comparison.json`、`budget_paired_tasks.csv`。
提示与 thinking 审计：`guidance_audit.json`。
源码、配置与服务保留检查：`experiment_manifest.json`、`post_run_integrity.json`、
`retained_services.json`、`gpu_after.csv`。

本次没有改写用户选定的历史正式基线，也未替换历史结果。

复核命令：

```bash
.venv/bin/python scripts/formal/report_webshop_budget_ablation.py \
  --baseline state/formal-eval/webshop-laser-checklist-c24-20260923-182224 \
  --candidate state/formal-eval/webshop-budget-off-laser-c24-20260923-191922 \
  --output state/experiments/webshop-budget-off-laser-c24-20260923-191922
```
