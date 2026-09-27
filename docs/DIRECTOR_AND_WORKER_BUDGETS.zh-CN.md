# Director 与 Worker 两份次数额度

最新确认配置如下。Director 决策次数仅作统计，不再执行独立的 24 轮截止。

| 数据集 | Director 成功编辑上限／题 | Worker 共享工具上限／题 |
| --- | ---: | ---: |
| AIME | 24 | 10 |
| HotpotQA | 24 | 不启用工具 |
| NQ | 24 | 不启用工具 |
| HealthBench | 24 | 不启用工具 |
| ALFWorld | 24 | 100 |
| WebShop | 24 | 16 |
| SWE | 24 | 32 |

## 编辑计数

成功改变节点、prompt、模型、层级或关系的一次动作计 1 次；初始建图也计入。一个动作内部的多项图更新只计一次。失败、无变化、关系提案不扣编辑额度；关系选择只有实际增删边才扣额度。run_agent 与 finish 不扣编辑额度。已提交的编辑即使随后 Worker 失败，也不会退回次数。创建节点前检查是否还够完成必要的 prompt 和模型配置。

编辑额度用尽后，仍允许已有合格结果的 finish，以及符合运行条件的 run_agent。若无任何可用结果和合法后续动作，记录明确的编辑额度终止原因，不伪造答案。基础设施故障不能被这个原因覆盖。

## 工具计数

整题所有 Worker 共用一个工具账本。首次执行、读取同伴结果后的执行、修改 prompt／职责／模型后的执行及 run_agent 都使用同一余额。删除重建节点、环境状态指纹变化也不能重置；新题 reset 才重置。initial/revision 仅保留为审计标签，不再各自限制调用次数。实际工具失败仍按既有调用记账规则计费；未分派的预检拒绝与真实执行在轨迹中区分。

finish 只提交已有合格结果，不运行 Worker，不扣编辑次数或新增工具调用。仍保留无进展检测、token 限制、环境自身步数限制以及失败恢复保护；它们不改变上述两份动作额度的定义。

## 实现与验证

新模式使用 canvas.director_budget_policy = "edits_v1"、canvas.max_director_edits = 24、canvas.action_budget_policy = "shared_total_v1"。AIME 工具总额由 aime_actions.max_total_calls = 10 设置。旧 max_rounds 与阶段工具字段仅用于旧模式兼容，在新模式不生效。

控制快照分别显示 director_edit_budget、action_budget；decision_statistics 只显示已进行多少决策。原始模型请求／响应、每次工具执行、图修改、预算快照和汇总指标全部留存在审计目录。

最终新策略的首轮真实验证配置是 state/audits/unified-protocol-20260926/one-v14/config.toml。one-v12、one-v13 均在启动前被用户后续参数调整替代，并记录 superseded.json，不混入真实测试结果。
