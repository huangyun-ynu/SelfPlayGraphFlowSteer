# WebShop 详情记忆取消 1,400 字符上限：128 题对照实验

对照为刚完成的身份修复版本 `webshop-identity-fix-c24-20260923-202622`，新运行为 `webshop-detail-unlimited-c24-20260923-205311`。

严格成功从 61/128（47.6562%）变为 60/128（46.8750%）。
平均得分从 70.7552 变为 67.6953。
128/128 完成，任务级运行异常 0，退出码 0。

## 改动范围

仅修改 `runtime.py` 三处单段详情截断：`_webshop_section_evidence` 的返回值、
`_update_webshop_product_inspections` 的 `section_evidence` 写入，以及
`_webshop_progress_prompt` 的决策检查点 `retained_section_evidence`。
清理空白及导航文本的原有逻辑沿用。配置文件没有修改。

整体提示记忆仍沿用原来的 6,000 字符淘汰阈值：超出时优先移除候选条目，再移除商品记录等，
不对当前决策检查点中的单段详情重新切片。商品记忆最多六项、每商品最多保留两个详情分区的规则沿用。
因此本次取消的是单段详情字符上限，不能解读为所有记忆容量限制都已取消。

3 个回归案例（1,401 / 3,012 / 9,000 字符）验证长文本末尾在提取、持久化、后续分区更新及当前决策检查点中保留，
共 112 项相关测试通过，lint 通过。

## 控制变量

同一官方 128 题、seed 0、24 路轨迹，DeepSeek Worker 服务并发 20。
Qwen3.5-9B 常规 graph_action thinking 开启，关系二选一沿用关闭 thinking 的原设置。
DeepSeek `deepseek-flash` thinking 开启、reasoning effort low、max_tokens 16384。
legacy 页面、LASER checklist、无 skill、环境反馈关闭，动作预算 12+4=16。
预算预测准入和请求额度拦截持续关闭；实际 Worker 总 token 上限 350000 的执行后核验沿用。
运行源码快照相对对照仅 `runtime.py` 修改，配置、数据、启动脚本哈希一致；运行结束无源码漂移。

## 结果

| 指标 | 1,400 上限开启 | 单段上限关闭 | 变化 |
|---|---:|---:|---:|
| 严格成功题数 | 61/128 | 60/128 | -1 题 |
| 严格成功率 | 47.6562% | 46.8750% | -0.7812 个百分点 |
| 平均得分（百分制） | 70.76 | 67.70 | -3.06（-4.32%） |
| 平均 Worker token | 72622.79 | 73625.73 | +1002.94（+1.38%） |
| 平均 Qwen Director token | 40371.02 | 45475.84 | +5104.82（+12.64%） |
| 平均合计 token | 112993.80 | 119101.56 | +6107.76（+5.41%） |
| 平均题目耗时（秒） | 103.86 | 114.19 | +10.33（+9.94%） |
| 全部题目合计 token | 14463207 | 15245000 | +781793 |
| 请求预算拒绝 | 0 | 0 | 0 |

逐题配对：4 题失败变成功，5 题成功变失败；
56 题两次均成功，63 题两次均失败。
McNemar 精确双侧 p=1.000000。这是一次历史运行与一次新运行的比较，模型采样和服务状态仍可能产生波动，
不能仅据此将每个变化归因于记忆长度。

## 取消上限是否实际生效

| 实际保存的详情记忆（去重） | 上一轮 | 本轮 |
|---|---:|---:|
| 条目数 | 116 | 119 |
| 小于 1,400 字符 | 81 | 91 |
| 恰好 1,400 字符 | 35 | 1 |
| 超过 1,400 字符 | 0 | 27 |
| 最大值 | 1400 | 3012 |
| 中位数 | 975.0 | 1031 |
| 能确认被单段长度上限截短 | 33 | 0 |

所有本轮保存条目均与实际工具观察的完整清理后文本匹配。统计按任务、Worker、商品、分区、文本去重，
不重复计算多个 artifact 快照或镜像字段，字符数不是 token 数。两轮决策路径不同，条目数量不要求相同。

上一轮确有截断的 28 题中，严格成功 9 → 9；
其余 100 题为 52 → 51。
该分组仅说明历史暴露情况，不是额外的随机对照实验。

## 用量与保留状态

Worker 用量按去重执行报告累计并与 Canvas 账本核对；Qwen 按去重请求 completion_usage 累计，包含 thinking/action。
两轮核对差异均为 0；本轮 Director 成功请求缺失用量 0 次。
路由、思考设置、LASER 提示已由轨迹审计核对，审计问题 0。

Qwen API PID 37099、engine PID 39080、GPU 占位 PID 19576 保留，
模型与 WebShop 健康检查通过；当前占位张量约 44.50 GiB，预留 1024 MiB 运行余量。
本轮没有停止服务或释放占位。当前工作区保留单段详情上限关闭的代码，历史正式基线及既有实验文件未覆盖。

产物位于 `state/experiments/webshop-detail-unlimited-c24-20260923-205311/`：`budget_comparison.json`、`budget_paired_tasks.csv`、
`memory_audit.json`、`guidance_audit.json`、`changed_tasks.json`、`previously_truncated_groups.json`、
`change_from_baseline.patch`、`experiment_manifest.json`、`post_run_integrity.json`、`retained_services.json`。
