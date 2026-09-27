# W08 与合并提示版：128 题公平复测（2026-09-24）

合并版本轮为 **66/128（51.5625%）**，重新运行的 W08 为 **58/128（45.3125%）**，净增 8 题。平均 reward 从 0.657031 升至 0.723359；平均 Worker token 降低 14.26%，平均合计 token 降低 12.57%。这是同一开发集上的单次配对结果，严格成功率尚未达到统计显著，不能据此宣称稳定泛化收益。

## 结果

| 指标 | W08 原版，本次重跑 | 合并版，本次重跑 | 差值（合并 − 原版） |
|---|---:|---:|---:|
| 严格成功数 | 58 | 66 | +8 |
| 严格成功率 | 45.3125% | 51.5625% | +6.2500 个百分点 |
| 平均 reward（0–1） | 0.657031 | 0.723359 | +0.066328 |
| 完成购买数 | 119 | 123 | +4 |
| 部分得分任务数 | 58 | 56 | -2 |
| 零分任务数 | 12 | 6 | -6 |
| 未完成购买数 | 9 | 5 | -4 |
| 平均 Worker token | 71,626.41 | 61,414.36 | -10,212.05 |
| 平均 Director token | 19,408.62 | 18,178.87 | -1,229.76 |
| 平均合计 token | 91,035.04 | 79,593.23 | -11,441.81 |
| Worker 总 token | 9,168,181 | 7,861,038 | -1,307,143 |
| Director 总 token | 2,484,304 | 2,326,895 | -157,409 |
| 合计总 token | 11,652,485 | 10,187,933 | -1,464,552 |
| 平均动作数 | 8.2500 | 7.3984 | -0.8516 |
| 平均任务耗时（秒） | 86.63 | 82.76 | -3.87 |
| 整组墙钟时间（秒） | 602.57 | 602.44 | -0.14 |
| 任务耗时中位数（秒） | 65.24 | 59.49 | -5.75 |
| 任务耗时 P95（秒） | 198.72 | 225.57 | +26.85 |
| Worker HTTP 请求数 | 1,245 | 1,135 | -110 |
| Worker 输入 token | 8,433,079 | 7,171,799 | -1,261,280 |
| Worker 输出 token | 735,102 | 689,239 | -45,863 |
| Director HTTP 请求数 | 714 | 680 | -34 |
| Director 输入 token | 2,300,361 | 2,151,257 | -149,104 |
| Director 输出 token | 183,943 | 175,638 | -8,305 |

WebShop 没有官方 F1，本报告将 F1 记录为不适用，不用平均 reward 替代 F1。平均 reward 的 100 分制分别是 **65.703125** 和 **72.3359375**。两组均 128/128 正常结束，执行异常 0、模型 HTTP 错误 0，未用其他题补位。

平均任务耗时下降，但 P95 从 198.72 秒上升至 225.57 秒，整组墙钟时间基本相同；不能概括为所有耗时指标均改善。

## 逐题配对与不确定性

| 配对结果 | 题数 |
|---|---:|
| 两组均成功 | 51 |
| 原版失败、合并版成功 | 15 |
| 原版成功、合并版失败 | 7 |
| 两组均失败 | 55 |

严格成功的精确双侧 McNemar p=0.133801。任务配对 bootstrap（10000 次，seed 20260924）得到成功率差值 95% 区间 **[-0.78125, +13.28125] 个百分点**，平均 reward 差值区间 **[+0.014961, +0.120560]**。这些区间用于本次开发集的探索性分析；单个 seed 不能消除采样、服务版本、缓存和负载的波动。

历史 W08 的 63/128、平均 reward 0.710221 保留作历史参考。主对比使用这次新跑的 58/128，不将历史 63 与新结果混合归因。当前仍保留 W08 基线身份，合并版作为有正向结果的候选；本轮没有启动训练或额外提示调参。

## 公平性与审计

- 数据集 SHA-256：`35fbda3aef098050f649ca569a8c8e1c8f3a4dbf0e8c800ba3fade74f7335328`；同一 128 题、seed 0、24 个轨迹并发，分别新建输出目录顺序执行。
- Qwen3.5-9B 基础 Director、thinking 开启；DeepSeek `deepseek-flash` Worker、thinking 开启、`reasoning_effort=low`、温度 0、最大输出 16384 token，接口并发上限 20。没有训练和 Skill。
- 初次/修订/总动作预算 12/4/16，Worker token 预算 350000，W08 请求预算拦截关闭。两组均没有请求预算拦截事件，也没有超过 16 次动作或 350000 Worker token 的任务。
- 只比较恢复的 W08 源码与其合并候选；没有使用夹杂后续 Native 修改的主目录。两组独立冻结源码，差异仅三个提示接入文件；有效配置仅 `webshop.worker_guidance_policy` 不同。
- 两组共用独立的基础模型服务和固定官方环境；环境 checkout `64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd` 干净，商品库、目标集哈希与准备清单一致。独立服务已停止，原有用户服务保留。
- **128/128 道题首次 Director 请求正文与参数逐字一致**。两组所有模型请求均有响应且任务关联完整；逐题 Worker/Director 的 API 用量与执行账本完全一致；提示生效及 thinking 审计均通过。
- API 缓存 token 单列；Worker reasoning token 只对服务端明确返回的请求求和，原版 154 次、合并版 158 次未单独报告，不能当作零。Director 没有单列 reasoning/cache token。输出 token 已包含服务端计入的推理消耗，不将 reasoning token 再加一次。没有推测货币费用。
- 第一次基线启动因 SDK 使用 `httpx2`、初版归档器遗漏模型报文而整轮停止，保留在 `setup-attempt-incomplete-http-audit/`。补齐归档并通过实际 SDK 模拟传输检查后，两组从头完整运行。中断轮不参与任何指标，没有按题目结果挑选或续接。

## 可复核产物

- [完整指标 JSON](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/experiments/webshop-w08-merged-128-20260924-run1/comparison/comparison.json)：两组逐题与汇总指标、分布、失败类型、请求用量、参数、统计区间及审计结果。
- [128 题配对 CSV](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/experiments/webshop-w08-merged-128-20260924-run1/comparison/paired_tasks.csv)：成功/失败、reward、token、动作、耗时等逐题差异。
- [原版请求索引](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/experiments/webshop-w08-merged-128-20260924-run1/comparison/baseline_request_index.csv)、[合并版请求索引](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/experiments/webshop-w08-merged-128-20260924-run1/comparison/candidate_request_index.csv)：每次请求的用量、耗时、状态及原始请求/响应文件位置。
- [归档说明](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/experiments/webshop-w08-merged-128-20260924-run1/README.zh-CN.md)：源码、配置、数据、模型/环境来源、逐题轨迹、请求/响应、环境交互、服务与评测日志、离线 W&B、中断产物的目录说明。
- [产物校验清单](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/experiments/webshop-w08-merged-128-20260924-run1/artifact-index.json)：每份文件的大小及 SHA-256。
