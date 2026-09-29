> 2026-09-29 NQ 更新：当前正式训练已采用 R2D2 1,702,133 段语料的运行时检索，
> 每次 top-8、每题最多 4 次。下文 NQ 冻结证据设置和成绩是历史记录；当前入口、
> 数据准备及验证以 [NQ 正式同步记录](NQ_R2D2_FORMAL_PROMOTION_20260929.zh-CN.md) 为准。
> `qa_best_recorded_eval.toml` 继续保留为历史对照，不是当前正式 NQ 配置。
>
> 2026-09-27 更新：默认 HotpotQA 测试集已切换为 FlowSteer 公开128题。下文历史成绩仍对应旧题集；默认路径不再代表历史输入。见 [数据切换与训练难度说明](HOTPOT_FLOWSTEER_DATASET_20260927.zh-CN.md)。

# HotpotQA / NQ：正式训练与历史对照（2026-09-25）

正式训练采用可核验的 QA 证据输入，输出答案仅做确定性提取，Worker 模型由 Director 自主选择，
Qwen Proposer 和 Solver/Director 均开启 thinking。固定 DeepSeek 路由只存在于独立测试配置。
正式训练与历史最高 EM 实验不是同一个完整配置，也尚未重新测出准确率。

逐题一致性复核另见 [实验一致性审计](QA_EXPERIMENT_CONSISTENCY_20260925.zh-CN.md)：
输入、基础 Director 提示与委派合同已核验；评分器和答案整理策略按用户决定保留差异，
Worker 完整提示和历史源码绑定仍有差异或未验证项，独立对照配置也不能视作精确历史复现。

## 逐项合并决定

1. 2026-09-26 用户进一步要求删除模型答案提取，取代此前“关闭但不删除”的决定。
   实验副本已删除模型提取实现、提示词、后端接入、专用 token 统计及配置选项。
   `[answer_submission]` 仅保留 `enabled`，控制确定性提取；不再调用额外 GPT 整理答案。
   当前独立对照配置也使用确定性提取。历史配置与审计快照保持原记录；旧配置中的
   模型提取选项不再生效，不能用旧开关重新启用此功能。
   相关提交、评分、配置和路由回归测试共 372 项通过；未发起真实模型评测。
2. 用户选择 A：保留现有角色区分的实验实现（`qa_role_conditioned_v2`），不回退旧版通用短答案提示。
   用户明确限定本次工作为实验版本对齐，不授权新增架构改造。
   核验过程中额外加入的身份提示、恢复 schema 和输出角色切换重跑改动均已撤销；
   `runtime.py`、`canvas.py` 的相关逻辑恢复到本项确认前的实现。
   已核对正常/恢复请求的 QA 角色提示与迁移包源码相同；撤销后相关 195 项测试通过。
   后续仅逐项核对实验差异；不将自动输出架构或自行发现的修复混入此次版本对齐。
3. 用户确认采用正式版 DeepSeek 并发上限 24：HotpotQA 和 NQ 均保持 24，不加入按数据集降到 20 的覆盖。
   历史 HotpotQA 的 20 仅保留在独立对照配置中。正式三份配置原本均为 24，无需修改。
4. 用户确认 Skill 设置暂时保持现状：正式训练允许注入已审核可用的 Skill，评测默认不注入。
5. 用户明确正式版的**轨迹并发为 24**，DeepSeek 路由请求上限也为 24；两者是不同层的限制。
   课程配置原本为 `rollout_workers=24`，但正式入口显式 `--workers 35` 会覆盖它，
   已将入口改为 `--workers 24`。文档统一使用“轨迹并发”，不称为“任务级并发”。

## 历史成绩与可恢复范围

以下是迁移档案中完整 128 题运行的最高 EM 记录，F1、通过率分别列出。
通过率是各次运行记录的 `passed` 比例，不能替代 EM，也不保证不同历史评估器口径相同。

| 数据集 | 历史运行 | EM | 答案 F1 | 记录通过率 |
|---|---|---:|---:|---:|
| HotpotQA | `qa-sota-no-skill-deepseek-c10-20260918` | 57.03%（73/128） | 75.74 | 57.03%（73/128） |
| NQ | `nq-frozen-v1/run-deepseek-qwen35-9b-128-20260919` | 39.06%（50/128） | 54.02 | 56.25%（72/128） |

原始档案位于 `state/imports/nq-hotpotqa-migration-20260925/extracted/`；逐运行重算表为
`state/imports/nq-hotpotqa-migration-20260925/verified_metrics.json`。
核对结论和恢复数据的哈希保存在
`state/audits/formal-qa-baseline-promotion-20260925/provenance.json`。

评分约定已确认：HotpotQA/NQ 继续使用 FlowSteerQA，另行保留 EM/F1 审计。
已从 Git 历史恢复 09-18/19/20 源码检查点；详见
[历史源码与提示词说明](QA_HISTORY_AND_PROMPTS_20260925.zh-CN.md)。

两次运行合计 256 条轨迹的首轮 Director system prompt 均与当前 `v2.1` 的对应提示一致。
因此继续使用该提示，没有为本次设置迁移重写 Director 协议。
档案没有保留两次运行时逐文件对应的完整运行源码快照；当前 Worker 提示和执行代码已有演进，
本次不能宣称恢复了完全相同的历史版本，也不能将历史分数直接归给修改后的正式训练。

## 正式训练设置

适用配置：`configs/formal_training.toml`、`configs/formal_training_h200.local.toml`、
`configs/formal_eval_h200.local.toml`。

| 项目 | 正式配置 | 独立历史对照配置 |
|---|---|---|
| QA Worker 候选 | GPT、Grok、Gemini、DeepSeek、MiniMax，Director 自选 | 仅 DeepSeek |
| QA 路由覆盖 | 无强制定向 | 无隐式定向，候选池显式只有 DeepSeek |
| DeepSeek 请求并发上限 | HotpotQA / NQ 均为 24（用户确认） | HotpotQA 20、NQ 24 |
| 轨迹并发 | 七数据集混合训练共享 24 | HotpotQA 10、NQ 24 |
| Qwen Proposer thinking | 开启 | benchmark 不使用 Proposer |
| HotpotQA Director thinking | 开启 | 开启 |
| NQ Director thinking | 开启 | 关闭，与该次历史最高 EM 记录一致 |
| Skill | 保留正式 PATS 学习流程 | 关闭 skill context |
| HotpotQA 证据 | 提供原有题目上下文，并开放 search | 相同证据模式 |
| NQ 证据 | 每道公开问题预取 top-8，固定内联，Worker 不开放 search | 恢复的历史 128 题 top-8 |
| 答案整理 | 仅确定性格式提取，无额外模型调用 | 保留 GPT 原答案抽取，关闭证据重选 |

`models.proposer.enable_thinking`、`models.solver.enable_thinking` 均显式设为 `true`。
模型请求层实际读取这些值；正式配置没有按数据集关闭 thinking 的覆盖。
测试配置使用 `director.thinking_by_dataset` 区分历史 HotpotQA/NQ。
显式 CLI `--director-thinking` / `--no-director-thinking` 仍可覆盖测试运行。

正式 QA 候选池经过新鲜 route report 的可用性筛选后交给 Director，选择某模型不会再被
QA 数据集覆盖暗中改成 DeepSeek。GPT 的同类 endpoint 池轮换继续生效；其它数据集已有的
候选池配置独立保留。正式 QA 不调用图外 GPT 答案整理器；GPT 仍可由 Director 选为 Worker。

## NQ 输入准备与边界

`scripts/formal/run_experiment.sh` 在采集前调用
`python -m selfplay_graph_flowsteer.nq_frozen_context`，从原始正式任务池生成
`state/formal-training/qa-baseline/task_pool.jsonl`。
仅 NQ 行附加固定证据；行顺序、ID、答案、split 以及其它数据集内容保留。
发给检索服务的只有公开问题；参考答案不进入检索请求或证据缓存。
缓存按问题、服务地址和 top-k 复用；完整准备成功后才发布新任务池。
Solver 在发出模型请求前拒绝缺失这 8 条内联证据的正式 NQ 输入。

Proposer 的 NQ 候选预览和 anchor 优先使用 `metadata.original_question`，不展示冻结证据。
池内 anchor 从对应任务读取原问题；独立 seed 从自身 metadata 读取。
未提供原问题的旧输入沿用原预览截断规则。选题只改变展示内容，返回的任务仍保留
完整问题、八段证据与原有评分数据；Director/Worker 的求解输入不变。

恢复的历史评测文件是
`state/formal-training/qa-baseline/eval/nq_open_frozen_128.jsonl`，只用于对照测试；
正式训练启动脚本不会将它注入训练池。
训练问题的证据需由本机固定的 Search-R1 检索服务生成。

## 对照测试入口

配置：`configs/qa_best_recorded_eval.toml`。
预先启动 Qwen3.5-9B（`127.0.0.1:18603/v1`），HotpotQA 还需检索服务
（`127.0.0.1:18010`）；凭据从 `.env` 读取。

```bash
bash scripts/formal/run_qa_best_recorded_eval.sh hotpotqa
bash scripts/formal/run_qa_best_recorded_eval.sh nq_open
```

默认题数 128，HotpotQA 并发 10，NQ 并发 24。
第二、三个参数分别可指定数据文件和输出目录；旧的 `run_nq_frozen_eval.sh` 已转向该独立入口。
正式训练入口不调用上述脚本。

本次完成本地配置、路由、thinking、证据输入和答案整理回归验证，未启动正式训练，
也未发送新的付费模型评测请求。移除 `set_output`、由最终输出 Agent 自动承担全题回答的架构
仍是另一项设计；不应把此次配置迁移视为该架构已经实现。
