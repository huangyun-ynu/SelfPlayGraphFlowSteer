# HealthBench Director 完整对话修复与 128 条重跑

本轮修复 Director 首轮仅接收最后一条用户消息的问题。HealthBench 分支现在使用 `solver_task_text(task)` 渲染完整公开对话，保留消息顺序和角色；不会把 rubric、医生参考答案或 canary 字段传给 Director。Worker 沿用上轮已修复的完整答案协议。

## 配置与验证

- 数据：同一批 128 条 HealthBench Professional，31 条多轮、97 条单轮；共 258 个 rubric，214 个正向、44 个负向。
- 数据 SHA-256：`991ed13ac248a3670e4caf7bdc83a770548c4209562c67a034631d4f0432042b`。
- Qwen3.5-9B Director：thinking 开启，GPU 1，端口 18604，vLLM 同时服务序列上限 20。
- 题目并发 20（上轮 10）；GPT worker 路由池及 low 设置保持不变。
- Judge 首选 gpt_student，并发总上限 10；错误时按此前授权尝试 gpt_eco、gpt，保持同一答案、同一 rubric 提示，不按得分重试。
- 150 项相关回归测试通过，包括实际 Director/Worker 请求路径的公开对话与私有字段隔离测试。
- 全量审计确认 128/128 条首轮 Director 请求含完整公开对话，包含全部 31 条多轮题；258/258 个评分决定都有与最终答案 SHA-256、题目及 rubric 索引匹配的成功 judge 回执。
- 成功评分请求路由统计：`{"gpt_student": 258}`。
- 成功 Worker 请求路由统计：`{"gpt_eco": 91, "gpt_student": 96, "gpt": 93}`。
- 重启后有 1 次推理重试：Worker 端点超时造成未产出有效答案，自动重试后完成评分。本轮没有 judge 备用路由调用，也没有按分数高低重试。
- 完成后已确认 Qwen `/v1/models` 返回 HTTP 200，服务仍驻留 GPU 1，未释放模型。

初次启动时，中间答案保存脚本误用了不存在的 `TaskSpec.to_dict` 方法；在任何样本成功评分前停止该 runner，改用项目已有的公开序列化函数后重新启动。Qwen 服务保持运行。该启动失败及重启时间保存在 `retry_events.jsonl` 和 `provenance/launch.json` 中。

## 配对结果

除字符数、token 和计数外，分数均为 100 分制。对照为上一轮完整答案协议修复后的实验 `healthbench-full-answer-gpt-low-128-20260924-231723`。

| 指标 | 修复前 | 修复后 | 变化 |
|---|---:|---:|---:|
| 长度调整后的平均分 | 41.33 | 42.03 | +0.70 |
| 原始 rubric 平均分 | 46.55 | 48.24 | +1.69 |
| 平均答案字符数 | 3777.65 | 4112.76 | +335.11 |
| 平均总 token | 6638.44 | 6713.88 | +75.44 |
| 多轮题（31 条）调整后平均分 | 33.48 | 31.09 | -2.39 |
| 多轮题（31 条）原始平均分 | 33.95 | 36.28 | +2.33 |
| 单轮题（97 条）调整后平均分 | 43.83 | 45.52 | +1.69 |
| 单轮题（97 条）原始平均分 | 50.58 | 52.06 | +1.48 |
| 项目阈值通过题数 | 70 | 66 | -4 |
| 正向评分项命中数 / 214 | 134 | 137 | +3 |
| 负向评分项触发数 / 44 | 19 | 18 | -1 |
| Director 未主动结束题数 | 2 | 0 | -2 |

逐题原始 rubric 分数：14 条提高、15 条下降、99 条不变。长度调整分数平均差的配对 bootstrap 95% 区间为 [-6.38, 8.50] 分。

前后两次均全部由 student judge 评分的共同子集有 127 条；该子集调整后平均变化 +0.76 分，原始 rubric 平均变化 +1.70 分。排除题目及逐题实际路由记录见 `comparison.json`。

## 解释边界

修复后总体调整分仅提高 0.70 分，配对置信区间跨过 0，不能据这一次重跑断言总体分数稳定提高。31 条多轮题原始平均分从 33.95 到 36.28，但调整后平均分从 33.48 降到 31.09：该组平均长度调整由 -0.47 分扩大到 -5.19 分，抵消了原始 rubric 收益。因此，上下文缺失确实是代码问题，但当前结果不支持把低分主要归因于这一处问题。

具体例子 `92fdf7f6`（末句为 “what is the penetrance”）修复后首轮包含角膜营养不良的完整前文，原始 rubric 分由 0 到 100，调整后由 4.66 到 64.59。这个案例说明修复有效改善了任务理解，但个例收益不能代表整个多轮题集合。

该实验同时改变了 Director 公开对话输入和题目并发，且是一次重新生成、重新评分；GPT 路由池调度、服务变化和生成/评分随机性仍可能影响结果。应分别看 31 条多轮题与 97 条单轮题，不能把总体分差全部归因于对话修复。Bootstrap 只覆盖抽样题目差异，不覆盖模型端点漂移或重复运行方差。

长度调整后的分数是当前项目的评测指标；“项目阈值通过”是 score≥0.5 的项目策略，不是 HealthBench 官方准确率。本次没有重跑官方单模型基线，不能据此给出与官方完整榜单严格可比的结论。

## 文件

- 生产修复：`src/selfplay_graph_flowsteer/adaptive.py`。
- 回归测试：`tests/test_healthbench_director_context.py`。
- 独立补丁：`experiment_versions/patches/healthbench-director-context-v1.patch`。
- 实验目录：`state/experiments/healthbench-director-context-gpt-low-128-c20-20260925-155258`。
- 审计：`verified_results.json`、`provenance/director_context_checks.json`。
- 配对统计：`comparison.json`；案例答案：`case_comparison.json`。
- 完整记录：`results/records.jsonl`、`results/samples/`；评分回执位于实验私有审计目录。
