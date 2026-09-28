# SWE 128 题重跑记录

沿用最近 30 题运行配置；只替换运行输出路径，重新执行冻结的全部 128 题。

轨迹并发 15，滚动补位；GPT-student 并发 10，GPT 非 eco 并发 5；远程测评并发配置 4。
Director Qwen3.5-9B thinking，端口 18605；整题实际 usage 发送阈值 350,000。
上轮讨论的候选恢复新方案尚未实施；本轮使用 launch_manifest.json 指定的源码快照。

| 项目 | 数量 |
| --- | ---: |
| 已保存逐题结果 | 128/128 |
| 官方通过 | 65 |
| 官方未通过 | 39 |
| policy_failure | 15 |
| unsubmitted_unknown | 9 |

官方通过数占固定全部题目：65/128 = 50.78%。
Benchmark 退出码：0；远程不计费关机确认：True。
启动及收尾脚本均不停止 Director。

未评分结果保持未知，不按官方未通过计。部分中断执行的旧 token_cost 字段存在已知少计问题，用量以每题 usage 账本为准。
原始结果见 results/records.jsonl；源码、配置和题目指纹见 launch_manifest.json。
