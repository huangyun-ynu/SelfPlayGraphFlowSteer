# 当前代码＋侧边栏修订测试集与测评器：GPT 128 题

当前主目录提交 `1322e43`，累计修复、自动记忆 v2、完成动作预算预留 v2、调研调度与工程修复全部保留。评测从正式配置生成，冻结运行源码、goals 和测试集；侧边栏评分版本 `webshop-quality-20260930-v1`。这是一套项目修订评分规则，源自原官方环境，未称为官方发布的新版本。

本次是 128 题全量真实模型购物，每题一次新尝试；没有按成绩筛选重试。128 个 ID 与原测试集一致，28 条题面修订。开始时间 2026-09-30T03:04:47.767307-07:00（America/Tijuana）。

| 指标 | 本次结果 |
|---|---:|
| 满分 | 84/128（65.62%） |
| 已购买 | 128/128 |
| 未购买 | 0 |
| 已购买未满分 | 44 |
| 未提交 | 0 |
| 平均奖励 | 0.857682 |
| 运行异常 | 0 |

固定 GPT 逻辑路由，学生网关 `gpt_student`（`lab-gpt-5.5-2`，Responses）并发 15、Nexus 非 eco `gpt`（`gpt-5.5`，Chat Completions）并发 5；两者 reasoning_effort=low；`gpt_eco` 从池中移除。题目并发 15；Qwen3.5-9B Director thinking=true，32768 上下文；seed=0；技能上下文关闭；每题 16 个环境动作。现有 Director 服务复用；本次独立 CPU WebShop 服务已关闭。

实际成功请求分布：{"gpt": 228, "gpt_student": 753}；eco 请求 0。请求池失败尝试：{"queue_timeout": 262}；本地 `queue_timeout` 表示触及成员限额后按现有策略转到另一个端点，不等于题目运行失败。全部请求均经配置中的成员限流器调度，未在本次调整池分流算法。

与上一轮 DeepSeek 评测核对：代码、测试集、goals 与评分源码相同；变更仅为 Worker 路由、成员限额、题目并发及运行输出目录，详见 `comparison-controls.json`。

购买终态独立重算：128 条修订分数与环境结果一致，原标签分数也一致。原标签下同一批购买的满分数为 67，平均奖励 0.751497；这些是同一次购买的附加记录，28 条题面已变化，不能当作原题重新评测的准确率。

未购买：无。已购买未满分的逐项评分检查和商品选项见 `per-task.json`、`rescore.json`。失败检查按规则类型计数：{"option": 45, "attribute": 15}；一题可以有多个失败检查。

运行目录：`/mnt/ssd/test/codex-students/student02/SelfPlayGraphFlowSteer/state/formal-eval/webshop-current-quality-gpt128-20260930-100446Z`。冻结源码与数据完整性：{"frozen_source_unchanged": true, "current_source_matches_snapshot": true, "frozen_goals_unchanged": true, "formal_config_unchanged": true}。

全局正式训练配置未被本次评测覆盖。本次配置、任务及原始轨迹见运行目录中的 `config.toml`、`tasks.jsonl` 与 `results/`；本报告目录保存启动器与独立复核脚本。
