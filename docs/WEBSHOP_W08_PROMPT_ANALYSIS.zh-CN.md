# W08 基线与 FlowSteer 提示设计对照

日期：2026-09-24。用户选择 W08 budget-off 作为新的比较基线，并要求参考 FlowSteer 分析 prompt 改进。本文记录当时的源码分析和待测方案。随后用户选择实施 P3（提示去重与优先级整理），已完成独立合并候选，见 [W08 提示合并版](WEBSHOP_W08_MERGED_PROMPT.zh-CN.md)；P1/P2 未实施，真实评测未启动。

## 基线及来源

- W08：`webshop-budget-off-laser-c24-20260923-191922`，63/128，严格成功率 49.21875%，平均 reward 0.7102213541666668，无 F1 指标。
- 源码 ID：`webshop-budget-off-128`；恢复到 `state/restored-versions/webshop-budget-off-128`，索引中 92 个文件的 SHA-256 全部验证通过。基础提交通过 Git 获取，主工作区未切换版本。
- `configs/webshop_official_baseline.json` 已改选 W08；原 W05 选择文件原样存入 `configs/history/webshop_legacy_20260918.json`。当前主目录及旧 official 启动脚本不是恢复后的 W08，故不把它们登记为 W08 精确评测入口。
- W08 保留 LASER、旧事实记忆及图工具协议；不包含后续 W09 身份修复、W10 详情扩容、W11 完整历史或 Native。无 skill、无参数训练。
- FlowSteer：官方仓库 `beita6969/FlowSteer`，本次读取提交 `1c9f2abf55cb9b8ea2ca2e3359cdb91acb9964e9`。下载文件及哈希在 `state/research/flowsteer-prompt-20260924/manifest.json`。此版本仓库树没有 WebShop 路径，README 的实验集列表没有 WebShop；参考的是通用工作流提示，不是已经证明有效的 WebShop 专用模板。它也不是本项目先前引用的 SkillFlow。

## 核对结果

| 方面 | FlowSteer 源码 | W08 已有内容 | 可研究的差异 |
|---|---|---|---|
| 职责委派 | `PROMPT_REQUEST_TEMPLATE` 要求短提示、职责具体、保留原始事实，避免改写数值和关系 | Director 分 role/objective/scope/expected_output；Worker 已获得完整 public_task_context | 明确原任务是约束权威，委派只分配职责，不放宽约束；并非补上原来不存在的任务输入 |
| 检查与修订 | 系统提示鼓励 Checker；有独立 Review/Verify/Revise 模板 | Director 已可分配评审，Worker 已报告 evidence/unresolved_issues | 把泛泛的核对要求变成可定位的“需求—证据—状态—缺口”报告，复用已有字段 |
| 完成条件 | Plan 不等于解答；末尾要求 Format | 已说明推荐不等于环境完成，实际购买由暂存候选和 Canvas 提交完成 | 不照搬 Format。准确区分有候选、候选满足需求、已提交三个事实，减少报告概念混淆 |
| 商品决策 | 没有 WebShop 专用规则 | 原 Worker 指令和 LASER 均包含价格、规格、详情、未知证据、预算及部分匹配规则 | 先精简重复及澄清优先关系，不能把这些规则再次添加后称为新方法 |

FlowSteer 的 Review 仅在对错误有超过 95% 的信心时否定答案。这不适合直接迁移到证据缺失的购物任务：没有发现错误不等于已经证明满足需求。其 Verify 强调独立重新计算，也不等于可以在未看到商品页面时重新推导商品事实。我们应只借用明确检查对象和反馈驱动修订，不照搬固定算子顺序、强制多节点或完整解题过程输出。

## 优先候选：逐项证据对应（P1）

W08 已有 `verified_requirements`、`unresolved_constraints` 和公开约束矩阵，缺口不是再建一套检查接口。本方案只调整 Worker 的证据表述要求，保留当前 JSON schema、工具参数和购买提交协议：

> Use the original public request as the source of requirements. For each requirement relevant to this candidate, distinguish supported, contradicted, and unknown, and tie support to an observed fact about that same product and applicable variant. In the existing evidence fields, record concise requirement-to-evidence correspondences; keep missing evidence and contradictions explicit in unresolved fields. A prior Agent's conclusion alone does not verify a requirement.

这里的三个状态是证据条目的内容约定，不新增顶层 JSON 字段，也不要求输出内部推理。继续使用已有预算内部分匹配兜底；不把 unknown 变成禁止一切购买的硬规则。已有明确证据时不强制读取所有详情页。

预期：减少只列出已满足条件、遗漏一个未满足条件却宣称完全匹配的情况。这是待验证假设，没有 W08 原始轨迹在当前工作区可供重新统计该失败比例。

## 第二候选：职责保真及针对性评审（P2）

只修改 Director 委派文字和已有评审 Agent 的提示，不新增强制调用或固定图拓扑。独立于 P1 从 W08 对照：

> Delegate a distinct responsibility in concise terms. The original public task remains authoritative; preserve all explicit constraints without weakening or adding requirements. When review is useful, specify which claim or uncertainty it must resolve, and return concrete evidence gaps or contradictions that the environment owner can act on. A review report is not itself an environment completion.

W08 已经要求职责具体、共享原任务及单一环境所有者，故增量应限于约束保真和评审的可操作反馈。不要把 1–3 句话当硬截断，导致要求遗漏；不要默认增加多个评审节点。

## 第三候选：提示去重与决策优先级（P3）

W08 的公共 Worker 规则及 LASER 都讨论选项、详情、购买和预算。可保持协议原文，仅重写购物决策段，按原任务约束、已见证据、当前可执行动作、剩余额度的顺序组织。避免“继续核查”和“预算不足则买最佳候选”散落多处，使模型过早启用兜底。

这是一组语义精简消融，仍从 W08 单独比较；不在同一轮顺带移除 LASER、切换历史模式或迁入 Native。页面分支按当前状态动态选择属于后续提示编排实验，不能与纯静态文字改写混为一组。

## 公平验证

- 对照使用恢复后的 W08，而非当前主目录加一个配置开关。P1/P2/P3 分别与对照比较，不先叠加再归因。
- 固定 Qwen3.5-9B、DeepSeek-flash、thinking、Worker reasoning_effort=low、采样设置、环境/索引、skill off、seed=0、同一 128 题、24 轨迹并发、12+4=16 动作额度；沿用 W08 的 350000 Worker token 执行后检查及关闭的请求准入，不能把它宣称为硬性逐请求限额。
- 同样的预算不保证实际计算量相同；单列 Director/Worker token、请求数、动作数和成本变化，避免把额外评审开销隐藏起来。
- 主指标严格成功数/128，辅指标平均环境 reward；另报未购买、部分匹配、预算耗尽、格式/服务错误及逐题新增/丢失成功。没有 F1，不混入十题 Native 成绩。
- 历史 128 题已用于开发，达到更高成绩后还需独立题集验证。原始失败与中断记录不删，不逐题拼接多轮最优结果。
- W08 有已知内部商品身份误判问题；prompt 不能修复错误的程序状态。若另做代码修复，需要单独消融，不能在提示实验里隐含升级到 W09。

## 原始参考

- [FlowSteer Director 与 prompt 生成模板](https://github.com/beita6969/FlowSteer/blob/1c9f2abf55cb9b8ea2ca2e3359cdb91acb9964e9/src/interactive/prompt_templates.py)
- [FlowSteer Review、Verify、Revise 等模板](https://github.com/beita6969/FlowSteer/blob/1c9f2abf55cb9b8ea2ca2e3359cdb91acb9964e9/scripts/prompts/prompt.py)
- [W08 实验记录](WEBSHOP_BUDGET_OFF_EXPERIMENT.zh-CN.md)
- [LASER 局部适配及已有规则边界](WEBSHOP_LASER_CHECKLIST_ADAPTATION.zh-CN.md)
