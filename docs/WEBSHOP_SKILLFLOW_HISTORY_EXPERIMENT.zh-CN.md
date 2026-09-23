# WebShop SkillFlow 完整历史模式：128 题实验

对照：`webshop-detail-unlimited-c24-20260923-205311`（已取消单段 1,400 字符上限，旧事实记忆模式）。
本轮：`webshop-skillflow-history-c24-20260923-213558`，`worker_memory_policy = "skillflow_history_v1"`。

128/128 完成，任务级运行异常 0，退出码 0。
严格成功从 60/128（46.8750%）变为
60/128（46.8750%），平均得分
67.6953 → 71.8945。

## 结果

| 指标 | 上一轮事实记忆 | SkillFlow 完整历史 | 变化 |
|---|---:|---:|---:|
| 严格成功题数 | 60/128 | 60/128 | +0 题 |
| 严格成功率 | 46.8750% | 46.8750% | +0.0000 个百分点 |
| 平均得分（百分制） | 67.70 | 71.89 | +4.20（+6.20%） |
| 平均 Worker token | 73625.73 | 57157.30 | -16468.43（-22.37%） |
| 平均 Qwen Director token | 45475.84 | 35836.04 | -9639.80（-21.20%） |
| 平均合计 token | 119101.56 | 92993.34 | -26108.23（-21.92%） |
| 平均题目耗时（秒） | 114.19 | 90.03 | -24.16（-21.16%） |
| 128 题合计 token | 15245000 | 11903147 | -3341853 |
| 请求预算拒绝 | 0 | 0 | 0 |

逐题配对：6 题失败变成功、6 题成功变失败；
54 题两次均成功、62 题两次均失败。
McNemar 精确双侧 p=1.000000。这是一次历史运行与一次新运行的配对对照，
模型采样和服务条件仍可能导致波动；不得将单次结果解释为已证实的稳定因果效果。

## 实际更改与控制变量

候选 TOML 与对照 TOML 仅相差 worker_memory_policy。该模式的代码效果包括：
完整页面/动作历史进入每次 Worker 输入，跨同一会话所有者的修订及最终报告保留；
历史不按商品数量或正文字符数裁剪，当前页面绕过 Worker 的 8,000 字符投影裁剪；
移除 Worker 输入中的旧商品记忆摘要及依赖它的访问/详情标记、动作辅助与约束矩阵。
内部旧 ledger 仍维护用于兼容和审计，六商品上限不限制完整历史。

因此，这次比较的是上述完整历史适配模式，不能把结果只归因于放大某一个容量参数。
完整实现和与 SkillFlow 原版的差异见 `WEBSHOP_SKILLFLOW_HISTORY_ADAPTATION.zh-CN.md`。

沿用同一官方 128 题、seed 0、24 路轨迹、DeepSeek Worker 服务并发 20、legacy 页面、
LASER checklist、无 skill、环境反馈关闭。DeepSeek 为 deepseek-flash，thinking 开启，
reasoning effort low，max_tokens 16384。Qwen3.5-9B 常规 graph_action thinking 开启，
关系二选一沿用关闭 thinking 的设置。动作预算 12+4=16，token 预测准入和请求额度拦截关闭，
执行后的实际 Worker 总 token 上限 350000 沿用。数据及旧配置哈希一致；运行期间无源码漂移。

## 完整历史与模型设置审计

- 有完整历史的任务：128/128；无历史任务：[]。
- 所有者执行记录 138 份，修订记录 10 份，
  无状态评审记录 9 份。
- 每题最终历史步数：最大 16，中位数 6.0。
- 每题最终历史页面正文累计字符数：最大 34176，
  中位数 5371.5；55 题超过 6,000 字符，
  41 题超过 8,000 字符。
- 历史启用、跨执行连续性、评审隔离、动作预算和旧状态提示投递审计问题：0。
  有中间 artifact 未呈现在最终快照的任务数：0；此类差异在审计中单独记录。
- 路由与提示审计问题：0。Director 回合统计：`{"graph_action:requested=True:effective=True": 595, "relation_choice:requested=False:effective=False": 8}`。
- Worker 请求事件统计：`{"backend_request_success": 1035, "provider_reports_reasoning_tokens": 897, "positive_reasoning_tokens": 894, "reasoning_tokens_unreported": 138}`。

历史字符数仅统计页面正文，不等于 token 数或完整请求长度。线上执行产物保存历史计数及字符数，
不保存完整后端请求；输入连接与不裁剪行为由冻结源码及适配阶段 281 项测试（含 12 项专项）验证。
本轮连续性审计将累计计数与去重后可见轨迹步骤核对；不会用仍受限的内部 product_inspections
来冒充模型可见的完整历史。

## 用量与服务保留

Worker token 按去重执行报告累计，与 Canvas 账本核对；Qwen 按去重请求 completion_usage
累计，包含 thinking/action。两轮用量核对差异均为 0，本轮 Director 成功请求缺失用量
0 次。

Qwen API PID 37099、engine PID 39080、GPU 占位 PID
19576 保留，模型及 WebShop 健康检查通过。当前占位约
44.50 GiB，预留 1024 MiB 运行余量。
候选配置保留完整历史模式，既有事实记忆配置和历史实验保留。

产物目录：`state/experiments/webshop-skillflow-history-c24-20260923-213558/`。主要文件：`budget_comparison.json`、
`budget_paired_tasks.csv`、`history_audit.json`、`guidance_audit.json`、`changed_tasks.json`、
`experiment_manifest.json`、`post_run_integrity.json`、`retained_services.json`。
