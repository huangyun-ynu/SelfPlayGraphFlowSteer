# 跨数据集实验记录（WebShop 更新至 2026-09-25）

**2026-09-25 正式训练选择更新：M02（合并＋身份修复）。** 历史62/128，EM48.4375%，平均分73.5221。正式训练保留Director选择逻辑模型、程序轮换同模型接口和Qwen thinking；不沿用实验固定DeepSeek，不导入S01人工Skill卡。本次没有新训练或128题成绩。见[同步记录](WEBSHOP_FORMAL_PROMOTION_2026-09-25.zh-CN.md)。

**2026-09-25 WebShop 版本总账已更新。** 完整列出17组128题结果、8组10题开发实验、47/128题提前停止轮、定向子集及撤回/未实施方案；逐版记录继承关系、改动、LASER/Skill状态、EM和平均分。见[最新版本总账](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/docs/WEBSHOP_EXPERIMENT_LEDGER_2026-09-25.zh-CN.md)、[JSON](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/experiment_versions/reports/webshop-version-ledger-20260925.json)、[CSV](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/experiment_versions/reports/webshop-version-ledger-20260925.csv)。近期6组完整128题、7组10题和中止轮47题均已用逐题轨迹核对指标；迁移前版本引用保留指标档案。其他数据集及下方逐次运行附录保留其原统计日期。

新增结果：合并＋身份修复62/128，平均73.5221分；再修记忆来源61/128，平均68.1380分；Director Skill版61/128，平均70.8789分；SkillFlow内部接入Director为38/128，平均51.6016分。历史after2扩展128计划在47题时停止，17/47正确、平均55.5851分，不能当作完整128题成绩。09-25讨论的工程修复仍是方案，尚无修复后推理成绩。此前总账整理仅更新记录；随后的M02正式训练选择更新见页首。

**2026-09-24 WebShop Director Skill v1 的128题测试已完成。** 以“合并＋身份修复＋记忆来源修复”的封存版本为底座，仅启用12条轨迹编排Skill（初始检索top-3，1024 token）。严格成功61/128（47.65625%），与上一轮持平；平均reward由0.6813802083升至0.7087890625，10题转成功、10题转失败。模型、low推理强度及预算保持相同；128个首请求移除Skill段落后与对照完全一致。逐卡记录384次任务级注入、42次可确认的采用行为、16次观察性帮助（非因果增益），附384条逐题技能证据。完整原始产物已保存，Director保留。见[实验报告](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/experiments/webshop-director-skills-128-20260924-run1/comparison/report.zh-CN.md)及[技能计数](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/experiments/webshop-director-skills-128-20260924-run1/skill_audit/skill_counts.csv)。该实验不自动替换正式基线。

最新无策略提示的 Native 推理机制迭代见 [2026-09-24 机制实验](WEBSHOP_NATIVE_MECHANISM_EXPERIMENT_2026-09-24.zh-CN.md)。

Native提示分支见[2026-09-24 WebShop Native 完成状态迭代](WEBSHOP_NATIVE_COMPLETION_EXPERIMENT_2026-09-24.zh-CN.md)。主表已补齐09-24至09-25记录；10题、提前停止及撤回尝试在最新总账中单列。

**2026-09-24 W08 提示合并的 128 题配对复测已完成。** 两组均重新运行：W08 原版 58/128（45.3125%，平均 reward 0.65703125），合并版 66/128（51.5625%，平均 reward 0.723359375）。逐题改善 15、退步 7，净增 8 题；平均合计 token 降低 12.57%。严格成功的 McNemar p=0.1338，尚不足以证明稳定提升。模型、推理强度 low、预算、seed、并发固定，未训练、无 Skill。全部中间产物、完整指标及逐题对比见 [复测报告](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/docs/WEBSHOP_W08_MERGED_128_2026-09-24.zh-CN.md)。首次归档缺陷导致的中断轮单独保留并整轮排除；历史 63/128 未被覆盖或混入本次主对比。

本记录汇总最近的真实评测：WebShop、ALFWorld、NQ Open、HotpotQA、AIME、HealthBench Professional、SWE-Bench。当前代码、历史代码补丁、实验配置及汇总指标进入 Git；原始轨迹、题目/答案、日志、模型、服务状态、显存占位状态和凭据保留在本地。

**2026-09-24 用户已改选 W08 budget-off：63/128（49.21875%）作为当前提示实验基线。** 见 [W08 提示分析](WEBSHOP_W08_PROMPT_ANALYSIS.zh-CN.md)。以下历史选择说明截至 09-23：当时正式基线为 WebShop legacy 的原始 62/128（48.4375%）。 retain_page_text 的 58/128（45.3125%）为历史档案。关闭请求 token 预算门槛的改动继续保留。后续实验和 native 第一版不自动替换正式基线；当前代码也不能直接冒充当时的基线源码。

## 1. 证据、口径与复现范围

- 历史运行索引：[run_catalog.json](../experiment_versions/run_catalog.json)，截至09-23包含69次运行、配置白名单、汇总文件哈希、评分器及提交器计数。附录保持该历史范围；新增WebShop版本见上方最新总账。
- 历史源码：[版本说明](../experiment_versions/README.md)、[版本索引](../experiment_versions/index.json)。8 个源码版本均提供补丁和逐文件 SHA-256；早期缺少完整快照的版本明确保留为“指标/配置档案”，不伪造源码。
- 主要依据为本地 `aggregate_summary.json`、`summary.json`、`run_manifest.json`，并核对现存轨迹的 verifier 和提交器。文中的 `state/...` 是本地证据位置，不是承诺上传的文件。
- `completed` 表示评测器完成记账，包括答错、环境失败、无答案等；不等于答对。`failed` 是运行器层面的失败，也不等于所有模型/API 失败。无完整汇总的运行不报成完整评测。
- WebShop 成功率为严格成功；分数为 reward 均值 ×100。ALFWorld 使用环境成功率；AIME 使用数值答案正确率；SWE 使用项目 `swe_outcome` 判定。HealthBench 的 rubric 分数不等同于“准确率”。
- NQ/HotpotQA 的旧 `multi_answer_exact_match`、`token_f1`、`flowsteer_qa` 不能混用。`flowsteer_qa` 的通过条件是归一化 token-F1 ≥0.5；其 1.0/0.7/0.4/0.2 分段 reward 不是官方 EM，也不是连续 F1。详见[指标说明](BENCHMARK_METRICS.zh-CN.md)。
- 通用索引中的 `mean_token_cost` 只记录 Worker，**不含 Director**。WebShop 下表另列请求 usage 核算的 Director 和合计；不同口径不拼接比较。
- 全部列出的正式运行均未更新模型参数。单次 128 题或定向小样本结果只支持本轮观察，不足以证明稳定因果收益。改变模型、数据索引、代码、提示或评分器时会标注混杂因素。

## 2. WebShop：128 题版本主线

以下128题主线使用固定WebShop题集；正式/历史基线记录的数据SHA-256为 `35fbda3aef098050f649ca569a8c8e1c8f3a4dbf0e8c800ba3fade74f7335328`。W07–W11及合并系列使用Qwen3.5-9B Director、DeepSeek Worker、12+4=16次动作预算。S01启用Director Skill；F01为独立SkillFlow架构分支、10题并发，旧合并轮为24题并发。表格不是逐行继承关系；EM指严格成功率，平均分为reward均值×100。

| 版本 | 核心设置/本轮改动 | 严格成功 | 成功率 | 分数 /100 | 相对前版的观察 |
| --- | --- | ---: | ---: | ---: | --- |
| W01 fixed | 修正目标映射；实际 GPT-5.5 Worker；Director thinking 关闭；结构化观察 4,000 字符；含 static Director skill | 23/128 | 17.9688% | 37.3307 | 修正前错误映射结果已废弃 |
| W02 no-skill | 移除 static skill，仍为 GPT-5.5、结构化观察 | 25/128 | 19.5313% | 41.9661 | +2 题，分数 +4.64 |
| W03 full-index | 完整商品索引 + DeepSeek Worker（thinking 关闭），Director thinking 关闭 | 53/128 | 41.4063% | 63.3568 | +28 题；索引和 Worker 同时变化，不能单独归因 |
| W04 retain_page_text | 保留原页面文本、字符配置为 0；Director/DeepSeek thinking 开启 | 58/128 | 45.3125% | 64.8359 | +5 题；用户指定历史存储 |
| **W05 legacy 正式基线** | legacy 页面呈现，max_observation_chars=0；沿用 thinking | **62/128** | **48.4375%** | **69.0625** | +4 题、分数 +4.23；用户选定的正式成绩 |
| W06 env-feedback | 添加重复查询、离开商品页面的环境反馈 | 58/128 | 45.3125% | 67.5195 | 相对正式基线 −4 题、分数 −1.54；未显示收益 |
| W07 LASER checklist | 在W05上加入按页面核对需求、价格、属性和选项的提示；环境反馈关闭 | 62/128 | 48.4375% | 69.5052 | 对照为W05；不是在W06上累计保留环境反馈 |
| W08 budget-off | 让关闭预算检查真正覆盖请求准入、每次执行 credit 及 closure 预留 | 63/128 | 49.2188% | 71.0221 | +1 题、分数 +1.52；请求预算拦截归零 |
| W09 identity-fix | legacy 展示删减后仍保留/恢复内部 ASIN，修正已访问和证据判断 | 61/128 | 47.6563% | 70.7552 | −2 题、分数 −0.27；身份误判消除，未体现整体准确率提升 |
| W10 detail-unlimited | 删除每段详情记忆 1,400 字符截断 | 60/128 | 46.8750% | 67.6953 | −1 题、分数 −3.06；本轮 token 增加 |
| W11 SkillFlow history | 完整私有 Observation/Action history，取消旧 shopping ledger 等重复提示，保留图执行工具协议及 LASER | 60/128 | 46.8750% | 71.8945 | 成功数不变、分数 +4.20；合计 token −21.92% |
| W08-R1 原版重新对照 | 原W08源码重跑；LASER仍开启 | 58/128 | 45.3125% | 65.7031 | 新一轮采样；不覆盖历史63/128 |
| M01 W08提示合并 | 合并公共Worker指令与LASER，去重并明确优先级；不改Director | 66/128 | 51.5625% | 72.3359 | 同期对照W08-R1；保留核对内容，取消重复注入 |
| M02 合并＋身份修复 | 在M01移植W09内部ASIN和访问/详情身份修复 | 62/128 | 48.4375% | 73.5221 | 历史对照M01；与原W09是不同分支 |
| M03 合并＋身份＋记忆来源修复 | 在M02修正裁剪后的访问事实来源，区分已读但未保留；仍有may_add价值推断 | 61/128 | 47.65625% | 68.1380 | 历史对照M02；后续修复方案尚未实施 |
| S01 Director Skill v1 | 在M03仅启用12条Director编排Skill，初始top-3、1024 token | 61/128 | 47.65625% | 70.8789 | 10题转对、10题转错；使用同一128题轨迹开发Skill |
| F01 SkillFlow内部接入Director | 直接在SkillFlow改造reset_react/react_step编排；新执行器、会话和FINISH提交 | 38/128 | 29.6875% | 51.6016 | 无Skill、无旧LASER；已有工程缺陷诊断，未实施修复 |

W02 的旧索引为 99,995 条，仅覆盖固定测试集的 12/128 个目标商品；W03 使用 1,181,370 条索引并覆盖全部目标。这个环境修复是主要混杂因素，不能把 W02→W03 全算成模型提升。W01/W02 的 manifest 声明可用路由池，实际调用审计显示使用 GPT-5.5。

W05 的完整 128 题确实开启了 Qwen thinking；更早未完成的启动尝试不计入这次成绩。W04 与 W06 都是 45.3125%，但它们是不同代码与设置，不能混称“45 版本”。legacy 只描述页面呈现层，W05–W11 仍存在结构化内部状态和图工具协议。

### 2.1 请求预算与 token 对照

| 版本 | 平均 Worker token | 平均 Director token | 平均合计 token | 平均耗时 /秒 |
| --- | ---: | ---: | ---: | ---: |
| W07 LASER | 69,492.09 | 41,883.66 | 111,375.75 | 108.34 |
| W08 budget-off | 68,111.69 | 41,688.06 | 109,799.75 | 105.03 |
| W09 identity-fix | 72,622.79 | 40,371.02 | 112,993.80 | 103.86 |
| W10 detail-unlimited | 73,625.73 | 45,475.84 | 119,101.56 | 114.19 |
| W11 history | 57,157.30 | 35,836.04 | 92,993.34 | 90.03 |

来源为各实验本地 `budget_comparison.json`。Director 使用请求事件中的 completion usage，Worker 对执行报告去重后核算；不是只数最终一条回复。关闭准入前有 4 次请求预算拦截、涉及 3 题，关闭后为 0；+1 题的配对 McNemar 双侧 p=1.0，不支持显著提升的结论。实际累计用量检查和环境动作上限仍存在，**不是所有预算机制都关闭**。

identity-fix 将 45 道题中的 136 次已访问误判降为 0，重复打开次数从 44 降为 29。修正内部身份不要求恢复整套面向模型的结构化商品字段。详情改动只取消单段 1,400 字符限制，当时仍保留最多 6 个商品、每商品 2 个详情区段及其他整体展示限制，不能称为“所有记忆无限”。

### 2.2 Native 第一版与停止状态

`skillflow_native_v1` 是显式选择的执行策略，默认仍为 `graph_tools_v1`。第一版采用原生 `search[...]` / `click[...]`、每 Agent 隔离环境 session 与完整私有 history；保留 Director 自主生成通用 Agent、Canvas 原子编辑、图边通信、增量记忆、冻结首轮和一次修订。Buy Now 暂存，`SET_OUTPUT` 选择输出，`FINISH` 才提交选中 Agent 的最新候选；动作预算仍全图累计。

| 同一组预先选定的 10 题 | 严格成功 | 分数 /100 | 完成购买 | 平均 Worker token |
| --- | ---: | ---: | ---: | ---: |
| W11 history 的匹配子集 | 4/10（40%） | 70.8333 | 10/10 | 41,589.90 |
| native 首轮真实推理 | 1/10（10%） | 37.1667 | 5/10 | 44,936.80 |

首轮有 5 道题使用多 Agent；Director 平均轮次从匹配子集的 4.8 增至 12。结果下降，不能据此采用为正式版本。原 `comparison10.json` 的 Director token 按 turn token-ID 长度统计，与上表 128 题请求 usage 口径不同，所以这里不混列 Director 或合计。

之后的Director/Canvas模拟测试发现并修正四个问题：native标志跨数据集误作用、未闭合think内容被当作动作、一个session清理失败阻断后续清理、FINISH提交回写污染历史Artifact/Canvas。**09-23停止时**尚未进行修复后复测；09-24已继续完成before/after/after2和m1–m4各有效10题实验。after2扩展128轮后来在47题停止。上述历史停止状态不能当作当前状态；完整成绩和改动见[最新版本总账](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/docs/WEBSHOP_EXPERIMENT_LEDGER_2026-09-25.zh-CN.md)。保存源码ID `webshop-native-10`仍对应修复前真实推理。

本次 Git 整理仅运行离线测试：8 个新增 WebShop 测试文件 **94 passed**，包括 native、Director 多 Agent、Canvas 三种增量模式、详情、环境反馈、history、身份、请求预算；未启动模型推理或修改 GPU 服务。8 个源码归档也逐一恢复并校验文件哈希。之前被中断的全量回归不能算作全量通过。

### 2.3 定向试验与版本入口

- closure 的 30 道失败题：original 0/30、分数 1.2778；removed 1/30、5.5556；best_so_far 2/30、27.4667。属于定向难题子集，不能和 128 题准确率直接比较。
- option-fixes：21 题首跑 1/21（20 题出现后端失败），重试 20 题为 7/20、分数 67.25；不能只报重试结果而省略服务故障。
- budget-aware 的 27 题为 0/27、分数 43.5494；价格区间单题为 0/1、分数 50。仅作机制诊断。
- 历史/正式配置：[历史 45](../configs/history/webshop_retain_page_text_20260918.json)、[正式 48](../configs/webshop_official_baseline.json)。当前可运行配置：[official](../configs/webshop_official_eval.toml)、[feedback](../configs/webshop_env_feedback_eval.toml)、[LASER](../configs/webshop_laser_checklist_eval.toml)、[native](../configs/webshop_skillflow_native_eval.toml)。这些当前配置需配合正确代码快照，不能单靠切换 TOML 就保证重现所有旧实验。`skillflow_history_v1` 已于2026-09-27移除，历史实验须使用归档快照；[最近128题复跑](../state/audits/webshop-engineering-20260927/online-history128-v2/REPORT.zh-CN.md)为52/128，事实记忆基线为56/128。
- 详细记录：[版本对比](WEBSHOP_VERSION_COMPARISON.zh-CN.md)、[feedback](WEBSHOP_ENV_FEEDBACK_EXPERIMENT.zh-CN.md)、[LASER](WEBSHOP_LASER_CHECKLIST_EXPERIMENT.zh-CN.md)、[预算关闭](WEBSHOP_BUDGET_OFF_EXPERIMENT.zh-CN.md)、[身份修复](WEBSHOP_IDENTITY_FIX_EXPERIMENT.zh-CN.md)、[详情上限](WEBSHOP_DETAIL_MEMORY_UNLIMITED_EXPERIMENT.zh-CN.md)、[history](WEBSHOP_SKILLFLOW_HISTORY_EXPERIMENT.zh-CN.md)、[native](WEBSHOP_SKILLFLOW_NATIVE_V1.zh-CN.md)。

## 3. ALFWorld

| 版本 | 改动/设置 | 成功 | 成功率 | 平均 Worker token |
| --- | --- | ---: | ---: | ---: |
| 多数据集 full-v3 | 早期联合评测配置 | 39/128 | 30.4688% | 31,108.63 |
| GPT no-skill | Qwen Director thinking 关闭、实际 GPT-5.5 Worker，无 skill | 45/128 | 35.1563% | 37,300.59 |
| DeepSeek no-skill | DeepSeek thinking 关闭；累计动作由 400 改为 200，无进展熔断由 20 改为 18 | 20/128 | 15.6250% | 87,928.34 |
| DeepSeek thinking-full | Worker thinking 开启；加入事实记忆、初始公共目标、成功锁定及 FINISH 防护 | 104/128 | 81.2500% | 43,949.91 |

后两版均为 24 并发、无 skill、Director thinking 关闭，episode 50、累计 200 动作。GPT 版与 DeepSeek 关闭 thinking 版还混有预算、熔断、代码及 API 可用性差异；不能据此做纯模型能力比较。

thinking-full 相对 DeepSeek 关闭 thinking 版增加 84 道成功题（+65.625 个百分点），配对为共同成功 18、新增成功 86、丢失成功 2、共同失败 22。同时进行了多项工程修复，因此不能把全部增益归给 thinking。104/128 中未成功的 24 题包含 23 个后端失败和 1 个其他失败。

定向测试按原样保留：事实记忆 3/7，FINISH 0/3，公共目标 2/2；这些小样本不是新的完整成绩。配置见 [alfworld_deepseek_thinking_eval.toml](../configs/alfworld_deepseek_thinking_eval.toml)，分析见 [全量版本比较](ALFWORLD_FULL_RUN_COMPARISON_2026-09-17.md)、[thinking 全量报告](ALFWORLD_THINKING_FULL_2026-09-17.md)。

## 4. NQ Open

| 版本 | 主要设置/改动 | 运行记录的通过率 | 分数 /100 | 平均 Worker token | 口径 |
| --- | --- | ---: | ---: | ---: | --- |
| 早期 full-v3 | 旧 QA 路径 | 20/128，15.6250% | 15.6250 | 1,043.23 | 旧多答案 exact match |
| 09-18 QA DeepSeek | DeepSeek Worker、10 并发，实时检索路径 | 35/128，27.3438% | 27.3438 | 8,021.18 | 多答案 exact match |
| frozen-128 | top-8 passages 预先冻结进 prompt；`provided_context_inline` 不开放运行时 search | 72/128，56.2500% | 59.5313 | 3,628.16 | `flowsteer_qa`，不能当 EM |

frozen-128 的独立严格 EM 复核为 **50/128（39.0625%）**，DeepSeek 语义复核为 62/128（48.4375%）；同一批回答的三种口径必须分别报告。固定证据提高了可控性、减少运行时检索调用；旧 EM 与新 verifier pass 的差值不代表真实 EM 提升。

后续最终提交 formatter 改为基于 passages 选择最短连续证据 span，并区分答案类型。对应 `run-span-regression-8` 与 `run-span-type-8` 都是 5/8（62.5%）verifier pass，平均 reward 分别 60 和 55；并未完成修改后的新 128 题评测。更早 `run-8` 为 5/8，inline 为 4/8，修复后的 r3 为 4/8；中间两次各 0 完成、8 个运行器失败，没有有效准确率。

配置/入口及原始口径说明见 [NQ 固定证据实验](NQ_FROZEN_CONTEXT_EXPERIMENT.zh-CN.md)、[prepare_nq_frozen_context.py](../scripts/formal/prepare_nq_frozen_context.py)、[run_nq_frozen_eval.sh](../scripts/formal/run_nq_frozen_eval.sh)。历史源码检查点 `fb73ce4`/`b28aedc` 保留，但不保证对应每个子版本的完整工作区。

## 5. HotpotQA

| 版本 | 主要设置/改动 | 通过 | 通过率 | 原汇总分数 /100 | 平均 Worker token |
| --- | --- | ---: | ---: | ---: | ---: |
| 早期 full-v3 | 旧 QA 路径 | 28/128 | 21.8750% | 21.8750 | 1,077.20 |
| 09-18 QA DeepSeek | DeepSeek、10 并发；多答案 exact match | 73/128 | 57.0313% | 57.0313 | 7,448.59 |
| FlowSteer-aligned v4 | 提供 distractor context；10 并发；实际 verifier 为 `token_f1` | 68/128 | 53.1250% | 71.5341 | 6,802.13 |
| role-conditioned v2 | 中间 Agent 局部职责与最终提交格式分离；24 并发；`flowsteer_qa` | 68/128 | 53.1250% | 72.5781 | 4,336.13 |
| role-conditioned v3 | 角色化提交路径后续迭代；24 并发；`flowsteer_qa` | 96/128 | 75.0000% | 75.0781 | 4,321.07 |
| format-baseline v2 | 格式基线对照；24 并发；`flowsteer_qa` | 95/128 | 74.2188% | 74.6094 | 4,369.93 |

09-20 三次完整运行均为 Qwen3.5-9B Director、DeepSeek Worker、双方 thinking 关闭、无 skill。当前实现把完整答案格式要求约束到选定输出 Agent，中间 Agent 可返回受分工限定的局部结果，避免所有 Agent 被强制回答整题。

现有记录没有保存 role-conditioned 各子版本的独立完整源码，不能把 v2→v3 的 +28 题完全归因于某一个函数修改。能核实的提交路径变化是：v2 中模型 span formatter 77 次、确定性提取 50 次、歧义提交 1 次；v3 对应 111/17/0；format-baseline v2 对应 108/20/0。v3 对格式基线只多 1 道通过题、Worker token 少约 1.12%，尚不足以证明稳定优势。

`token_f1` 的连续分数、`flowsteer_qa` 的分段奖励，以及旧 exact match 是不同口径。不能把 v4 的 71.5341 称为后续的同口径奖励，更不能把 75% 称为官方 HotpotQA EM 或 joint 准确率。现有正式池为 Answer-only，缺少所需 supporting_facts 金标准，不报 support/joint 指标。

原 role-conditioned 首跑只完成 57/128，34/57 通过（59.6491%），另有 3 个运行器失败；format-baseline v1 完成 0/128、失败 3。它们作为中断记录保存。待 GPU 的 Qwen thinking 启动脚本保留在代码中；没有完整结果的 thinking 尝试不计为已完成对照。

## 6. AIME、HealthBench Professional、SWE-Bench

| 数据集/版本 | 规模 | 指标 | 结果 | 平均 Worker token | 改动/效果与限制 |
| --- | ---: | --- | ---: | ---: | --- |
| AIME full-v3 | 30 | 数值答案准确率 | 21/30，70.0000% | 6,754.23 | 严格 AIME 答案提交；仅有这一份完整 30 题汇总，不能声称多版提升 |
| HealthBench Professional full-v3 | 128 | 原汇总 rubric 分数 | 21.6576/100 | 5,074.20 | verifier 为 `healthbench_rubric`；原 pass 为 37/128（28.9063%），不能当临床准确率或替代 rubric 主分数 |
| SWE full-c10 | 128 | 项目 `swe_outcome` resolved/pass | 71/128，55.4688% | 153,950.28 | Qwen thinking 开启、GPT Worker、10 并发、无 skill；属于当前抽样 128 题，不代表全量 SWE-bench Verified |

HealthBench 运行含无评分/API 失败记录。这里忠实保存当时 aggregate 分数，没有重新调用 grader，也没有把后来新增的 length-adjusted 指标追溯套用为本轮结果。原始题目、rubric 内容及回答不上传。

SWE 两次前置单题 smoke 都是 0/1；重试有实际 token 使用，但不能由此计算版本收益。完整 128 题轨迹包含 127 条 `swe_outcome` verifier 记录及 1 条未评分记录；分母仍保留 128。没有另一份同口径完整版本用于配对比较。

AIME 的 `dev-aime-v2/v3` 均为单题启动失败，v4 为 1/1。六数据集 smoke-v2 共 6 题、全失败；smoke-v3 中 AIME、ALFWorld 各 1/1，仅说明基本链路连通。

## 7. 无效结果与当前代码整理

早期 `qwen35-9b-director-full-v3` 原计划并记账 670 题，但其中 WebShop 128 题的 prompt 与环境目标映射无效，已从有效汇总删除，保留 **542** 题。索引保存这项排除说明，不能把原计数 670 当成 670 个有效样本，也不能使用被删除的 WebShop 成绩。

本次提交保留最近尚未提交的源码、配置、启动/审计/报告脚本和测试，包括：QA 局部职责与最终提交作用域、按数据集路由覆盖、兼容端点池、WebShop 环境反馈/LASER/history/native、请求预算开关、身份修复、详情记忆修复及 Director/Canvas 场景测试。没有替换既有历史实验记录，也没有启动真实推理。

下面的逐次索引从公开指标目录自动生成。它保留观察到的全部运行；没有指标的版本显示“无汇总”，而不是填 0% 准确率。`成功/样本` 列实际是该运行 verifier 的通过数；只有 WebShop 等二元环境任务才能直接读作严格成功。

<!-- RUN_TABLE -->

## 附录：逐次运行索引

路径以仓库根目录为起点；`sota-20260916` 等目录名表示组织批次，不据此推断每次实际结束日期。所有分数和通过率均乘 100，口径沿用各次 verifier。

| 本地运行路径 | 数据集 | 已完成/计划；运行器失败 | verifier 通过/样本 | 通过率 | 分数 | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| `state/experiments/hotpotqa-format-baseline-20260920-v1` | hotpotqa | 0/128；3 | 无汇总 | — | — | 不完整 |
| `state/experiments/hotpotqa-format-baseline-20260920-v2` | hotpotqa | 128/128；0 | 95/128 | 74.2188% | 74.6094 | 完成 |
| `state/experiments/hotpotqa-role-conditioned-20260920` | hotpotqa | 57/128；3 | 34/57 | 59.6491% | 73.1579 | 不完整 |
| `state/experiments/hotpotqa-role-conditioned-20260920-v2` | hotpotqa | 128/128；0 | 68/128 | 53.1250% | 72.5781 | 完成 |
| `state/experiments/hotpotqa-role-conditioned-20260920-v3` | hotpotqa | 128/128；0 | 96/128 | 75.0000% | 75.0781 | 完成 |
| `state/experiments/nq-frozen-v1/run-8` | nq_open | 8/8；0 | 5/8 | 62.5000% | 68.7500 | 完成 |
| `state/experiments/nq-frozen-v1/run-deepseek-qwen35-9b-128-20260919` | nq_open | 128/128；0 | 72/128 | 56.2500% | 59.5313 | 完成 |
| `state/experiments/nq-frozen-v1/run-inline-8` | nq_open | 8/8；0 | 4/8 | 50.0000% | 60.0000 | 完成 |
| `state/experiments/nq-frozen-v1/run-inline-8-fixed-20260919` | nq_open | 0/8；8 | 无汇总 | — | — | 不完整 |
| `state/experiments/nq-frozen-v1/run-inline-8-fixed-20260919-r2` | nq_open | 0/8；8 | 无汇总 | — | — | 不完整 |
| `state/experiments/nq-frozen-v1/run-inline-8-fixed-20260919-r3` | nq_open | 8/8；0 | 4/8 | 50.0000% | 60.0000 | 完成 |
| `state/experiments/nq-frozen-v1/run-span-regression-8` | nq_open | 8/8；0 | 5/8 | 62.5000% | 60.0000 | 完成 |
| `state/experiments/nq-frozen-v1/run-span-type-8` | nq_open | 8/8；0 | 5/8 | 62.5000% | 55.0000 | 完成 |
| `state/formal-eval/hotpot-flowsteer-aligned-no-skill-deepseek-c10-20260919-v2` | hotpotqa | ?/128；? | 无汇总 | — | — | 仅启动清单 |
| `state/formal-eval/hotpot-flowsteer-aligned-no-skill-deepseek-c10-20260919-v3` | hotpotqa | ?/128；? | 无汇总 | — | — | 仅启动清单 |
| `state/formal-eval/hotpot-flowsteer-aligned-no-skill-deepseek-c10-20260919-v4` | hotpotqa | 128/128；0 | 68/128 | 53.1250% | 71.5341 | 完成 |
| `state/formal-eval/hotpot-real-smoke-20260918` | hotpotqa | 1/1；0 | 1/1 | 100.0000% | 100.0000 | 完成 |
| `state/formal-eval/nq-real-smoke-20260918` | nq_open | 1/1；0 | 0/1 | 0.0000% | 0.0000 | 完成 |
| `state/formal-eval/qa-sota-no-skill-c10-20260918` | hotpotqa, nq_open | ?/256；? | 无汇总 | — | — | 仅启动清单 |
| `state/formal-eval/qa-sota-no-skill-deepseek-c10-20260918` | hotpotqa | 256/256；0 | 73/128 | 57.0312% | 57.0312 | 完成 |
| `state/formal-eval/qa-sota-no-skill-deepseek-c10-20260918` | nq_open | 256/256；0 | 35/128 | 27.3438% | 27.3438 | 完成 |
| `state/formal-eval/swe-full-c10-20260918-1805` | swe_bench | 128/128；0 | 71/128 | 55.4688% | 55.4688 | 完成 |
| `state/formal-eval/swe-full-debug` | swe_bench | ?/128；? | 无汇总 | — | — | 仅启动清单 |
| `state/formal-eval/swe-smoke-c10-20260918-174513` | swe_bench | 1/1；0 | 0/1 | 0.0000% | 0.0000 | 完成 |
| `state/formal-eval/swe-smoke-c10-retry` | swe_bench | 1/1；0 | 0/1 | 0.0000% | 0.0000 | 完成 |
| `state/formal-eval/webshop-budget-off-laser-c24-20260923-191922` | webshop | 128/128；0 | 63/128 | 49.2188% | 71.0221 | 完成 |
| `state/formal-eval/webshop-deepseek-inference-20260918-161911` | webshop | ?/128；? | 无汇总 | — | — | 仅启动清单 |
| `state/formal-eval/webshop-deepseek-inference-20260918-162136` | webshop | ?/128；? | 无汇总 | — | — | 仅启动清单 |
| `state/formal-eval/webshop-deepseek-inference-noskill-20260918-162212` | webshop | ?/128；? | 无汇总 | — | — | 仅启动清单 |
| `state/formal-eval/webshop-deepseek-inference-noskill-20260918-162427` | webshop | 0/128；3 | 无汇总 | — | — | 不完整 |
| `state/formal-eval/webshop-deepseek-inference-noskill-20260918-162524` | webshop | ?/128；? | 无汇总 | — | — | 仅启动清单 |
| `state/formal-eval/webshop-deepseek-inference-noskill-20260918-164206` | webshop | ?/128；? | 无汇总 | — | — | 仅启动清单 |
| `state/formal-eval/webshop-deepseek-inference-noskill-c24-20260918-165403` | webshop | ?/128；? | 无汇总 | — | — | 仅启动清单 |
| `state/formal-eval/webshop-deepseek-reasoning-noskill-c24-20260918-170329` | webshop | 128/128；0 | 58/128 | 45.3125% | 64.8359 | 完成 |
| `state/formal-eval/webshop-detail-unlimited-c24-20260923-205311` | webshop | 128/128；0 | 60/128 | 46.8750% | 67.6953 | 完成 |
| `state/formal-eval/webshop-env-feedback-c24-20260923` | webshop | 128/128；0 | 58/128 | 45.3125% | 67.5195 | 完成 |
| `state/formal-eval/webshop-identity-fix-c24-20260923-202622` | webshop | 128/128；0 | 61/128 | 47.6562% | 70.7552 | 完成 |
| `state/formal-eval/webshop-laser-checklist-c24-20260923-182224` | webshop | 128/128；0 | 62/128 | 48.4375% | 69.5052 | 完成 |
| `state/formal-eval/webshop-legacy-page-only-reasoning-c24-20260918-172320` | webshop | 128/128；0 | 62/128 | 48.4375% | 69.0625 | 完成 |
| `state/formal-eval/webshop-skillflow-history-c24-20260923-213558` | webshop | 128/128；0 | 60/128 | 46.8750% | 71.8945 | 完成 |
| `state/formal-eval/webshop-skillflow-native10-20260923-223240` | webshop | 10/10；0 | 1/10 | 10.0000% | 37.1667 | 完成 |
| `state/formal-eval/webshop-smoke-legacy-20260918-172159` | webshop | 1/1；0 | 0/1 | 0.0000% | 66.6667 | 完成 |
| `state/formal-eval/webshop-smoke-noskill-20260918-170142` | webshop | 0/1；1 | 无汇总 | — | — | 不完整 |
| `state/formal-eval/webshop-smoke-noskill-20260918-170237` | webshop | 1/1；0 | 0/1 | 0.0000% | 66.6667 | 完成 |
| `state/sota-20260916/alfworld-deepseek-finish-targeted-v1` | alfworld | 3/3；0 | 0/3 | 0.0000% | 0.0000 | 完成 |
| `state/sota-20260916/alfworld-deepseek-memory-targeted-v1` | alfworld | 7/7；0 | 3/7 | 42.8571% | 42.8571 | 完成 |
| `state/sota-20260916/alfworld-deepseek-public-goal-targeted-v1` | alfworld | 2/2；0 | 2/2 | 100.0000% | 100.0000 | 完成 |
| `state/sota-20260916/alfworld-deepseek-thinking-full-v1` | alfworld | 128/128；0 | 104/128 | 81.2500% | 81.2500 | 完成 |
| `state/sota-20260916/alfworld-qwen35-9b-deepseek-no-skill-v1` | alfworld | 128/128；0 | 20/128 | 15.6250% | 15.6250 | 完成 |
| `state/sota-20260916/alfworld-qwen35-9b-no-skill-v1` | alfworld | 128/128；0 | 45/128 | 35.1562% | 35.1562 | 完成 |
| `state/sota-20260916/dev-aime-v2` | aime | 0/1；1 | 无指标 | — | — | 仅静态测试摘要 |
| `state/sota-20260916/dev-aime-v3` | aime | 0/1；1 | 无指标 | — | — | 仅静态测试摘要 |
| `state/sota-20260916/dev-aime-v4` | aime | 1/1；0 | 1/1 | 100.0000% | 100.0000 | 仅静态测试摘要 |
| `state/sota-20260916/qwen35-9b-director-full-v1` | aime, alfworld, healthbench_professional, hotpotqa, nq_open, webshop | ?/670；? | 无汇总 | — | — | 仅启动清单 |
| `state/sota-20260916/qwen35-9b-director-full-v2` | aime, alfworld, healthbench_professional, hotpotqa, nq_open, webshop | ?/670；? | 无汇总 | — | — | 仅启动清单 |
| `state/sota-20260916/qwen35-9b-director-full-v3` | hotpotqa | 670/670；0 | 28/128 | 21.8750% | 21.8750 | 完成，有无效数据排除 |
| `state/sota-20260916/qwen35-9b-director-full-v3` | alfworld | 670/670；0 | 39/128 | 30.4688% | 30.4688 | 完成，有无效数据排除 |
| `state/sota-20260916/qwen35-9b-director-full-v3` | healthbench_professional | 670/670；0 | 37/128 | 28.9062% | 21.6576 | 完成，有无效数据排除 |
| `state/sota-20260916/qwen35-9b-director-full-v3` | nq_open | 670/670；0 | 20/128 | 15.6250% | 15.6250 | 完成，有无效数据排除 |
| `state/sota-20260916/qwen35-9b-director-full-v3` | aime | 670/670；0 | 21/30 | 70.0000% | 70.0000 | 完成，有无效数据排除 |
| `state/sota-20260916/qwen35-9b-director-smoke-v2` | hotpotqa | 6/6；0 | 0/1 | 0.0000% | 0.0000 | 完成 |
| `state/sota-20260916/qwen35-9b-director-smoke-v2` | webshop | 6/6；0 | 0/1 | 0.0000% | 0.0000 | 完成 |
| `state/sota-20260916/qwen35-9b-director-smoke-v2` | alfworld | 6/6；0 | 0/1 | 0.0000% | 0.0000 | 完成 |
| `state/sota-20260916/qwen35-9b-director-smoke-v2` | aime | 6/6；0 | 0/1 | 0.0000% | 0.0000 | 完成 |
| `state/sota-20260916/qwen35-9b-director-smoke-v2` | nq_open | 6/6；0 | 0/1 | 0.0000% | 0.0000 | 完成 |
| `state/sota-20260916/qwen35-9b-director-smoke-v2` | healthbench_professional | 6/6；0 | 0/1 | 0.0000% | 0.0000 | 完成 |
| `state/sota-20260916/qwen35-9b-director-smoke-v3` | hotpotqa | 6/6；0 | 0/1 | 0.0000% | 0.0000 | 完成 |
| `state/sota-20260916/qwen35-9b-director-smoke-v3` | webshop | 6/6；0 | 0/1 | 0.0000% | 0.0000 | 完成 |
| `state/sota-20260916/qwen35-9b-director-smoke-v3` | alfworld | 6/6；0 | 1/1 | 100.0000% | 100.0000 | 完成 |
| `state/sota-20260916/qwen35-9b-director-smoke-v3` | aime | 6/6；0 | 1/1 | 100.0000% | 100.0000 | 完成 |
| `state/sota-20260916/qwen35-9b-director-smoke-v3` | nq_open | 6/6；0 | 0/1 | 0.0000% | 0.0000 | 完成 |
| `state/sota-20260916/qwen35-9b-director-smoke-v3` | healthbench_professional | 6/6；0 | 0/1 | 0.0000% | 4.5541 | 完成 |
| `state/sota-20260916/qwen35-9b-director-webshop-fixed-v1` | webshop | 128/128；0 | 23/128 | 17.9688% | 37.3307 | 完成 |
| `state/sota-20260916/qwen35-9b-director-webshop-no-skill-v1` | webshop | 128/128；0 | 25/128 | 19.5312% | 41.9661 | 完成 |
| `state/sota-20260916/webshop-closure-ablation-deepseek-v1/best_so_far` | webshop | 30/30；0 | 2/30 | 6.6667% | 27.4667 | 完成 |
| `state/sota-20260916/webshop-closure-ablation-deepseek-v1/original` | webshop | 30/30；0 | 0/30 | 0.0000% | 1.2778 | 完成 |
| `state/sota-20260916/webshop-closure-ablation-deepseek-v1/removed` | webshop | 30/30；0 | 1/30 | 3.3333% | 5.5556 | 完成 |
| `state/sota-20260916/webshop-closure-ablation-v1/original` | webshop | ?/30；? | 无汇总 | — | — | 仅启动清单 |
| `state/sota-20260916/webshop-full-index-deepseek-no-skill-v1` | webshop | 128/128；0 | 53/128 | 41.4062% | 63.3568 | 完成 |
| `state/sota-20260916/webshop-option-fixes-deepseek-no-skill-retry-v1` | webshop | 20/20；0 | 7/20 | 35.0000% | 67.2500 | 完成 |
| `state/sota-20260916/webshop-option-fixes-deepseek-no-skill-v1` | webshop | 21/21；0 | 1/21 | 4.7619% | 4.7619 | 完成 |
| `state/sota-20260917/webshop-alfworld-gpt-nexus-no-skill-v1` | alfworld, webshop | ?/256；? | 无汇总 | — | — | 仅启动清单 |
| `state/sota-20260917/webshop-budget-aware-deepseek-no-skill-v1` | webshop | 27/27；0 | 0/27 | 0.0000% | 43.5494 | 完成 |
| `state/sota-20260917/webshop-price-range-deepseek-goal234-v1` | webshop | 1/1；0 | 0/1 | 0.0000% | 50.0000 | 完成 |
