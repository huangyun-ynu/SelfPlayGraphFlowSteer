# WebShop 实验版本对照

> 2026-09-25 当前正式训练已选择M02（62/128、平均分73.5221）；下文W08选择为历史记录。见[当前正式版本](WEBSHOP_BASELINE.zh-CN.md)。

**2026-09-25更新入口：** [完整版本总账](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/docs/WEBSHOP_EXPERIMENT_LEDGER_2026-09-25.zh-CN.md)已列出17组完整128题、全部有效Native 10题、提前停止和撤回记录，包含每版改动、LASER/Skill状态、EM及平均分；另有[CSV](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/experiment_versions/reports/webshop-version-ledger-20260925.csv)。下文保留09-23六版核查原文，其中“当前正式参考”等表述仅指当时状态；用户后来已选择W08作为提示实验基线。

核查日期：2026-09-23。V1–V6 是本表临时编号，不是项目 release 编号。依据为各运行的 aggregate_summary、manifest，以及全部历史轨迹中的配置、实际 Worker 模型、Qwen 思考字段和 skill 注入记录。

## 六个完整的 128 题版本

六组运行的题目 ID 集合完全相同；均为 Qwen3.5-9B Director、24 路评测并行、12+4=16 次动作预算、暂存购买开启。

| 编号 | 日期/版本 | 实际 Worker | Qwen 常规思考 | Director skill | 搜索观察模式 | 观察字符上限 | 严格成功率 | 平均得分 /100 |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| V1 | 09-16 webshop-fixed | GPT-5.5 / gpt 路由 | 关 | static_director_webshop | structured_only | 4000 | 23/128 = 17.97% | 37.33 |
| V2 | 09-16 webshop-no-skill | GPT-5.5 / gpt 路由 | 关 | 无 | structured_only | 4000 | 25/128 = 19.53% | 41.97 |
| V3 | 09-16 full-index-deepseek | deepseek-flash | 关 | 无 | structured_only | 4000 | 53/128 = 41.41% | 63.36 |
| V4 | 09-18 reasoning-noskill | deepseek-flash | 开 | 无 | retain_page_text | 0 | 58/128 = 45.31% | 64.84 |
| V5 | 09-18 legacy-page-only | deepseek-flash | 开 | 无 | legacy | 0 | 62/128 = 48.44% | 69.06 |
| V6 | 09-23 env-feedback | deepseek-flash | 开 | 无 | legacy + env_feedback | 0 | 58/128 = 45.31% | 67.52 |

V3 的 DeepSeek thinking=false；V4–V6 均为 true。V1/V2 manifest 虽声明多条可用 Worker 路由，实际轨迹中的 Worker artifact 全部为 gpt 路由、模型 gpt-5.5，不能据 manifest 将其当成多模型混合成绩。

Qwen 思考逐调用审计：V1/V2/V3 常规 graph_action 分别 658/640/651 次，全部关闭；V4/V5/V6 分别 770/721/639 次，全部开启且有效。关系二选一调用各版本均关闭思考。

字符上限 0 表示关闭 lifecycle 这一层的页面截断，不等于最终模型输入完全不截断；当前 runtime 对超过 8000 字符的页面仍会保留头尾各 4000 字符。

## 页面模式的实际差异

| 模式 | 搜索页面文本 | 搜索商品动作字段 | 其余结构化状态 |
| --- | --- | --- | --- |
| structured_only | 尝试精简可识别格式的商品列表；不识别的渲染格式会保留文本 | 保留商品信息及选项证据提示 | 保留 |
| retain_page_text | 保留页面文本 | 保留商品信息及选项证据提示 | 保留 |
| legacy | 保留页面文本 | 删除 open_product 上一组增强字段及 search_evidence_semantics | 仍保留动作表、商品页状态及 runtime 摘要 |
| legacy + env_feedback | 同 legacy | 同 legacy | 额外加入重复查询与返回提示 |

因此这些版本均不能统称为“模型只看纯文本页面”。特别是 V5 的实际历史轨迹已经包含 valid_subactions、product、selected_options、candidate_coverage、search_decision_state 等字段。V6 只新增环境反馈及其内部历史记录，没有重新创建这些已有观察结构。

## 变化如何解读

- V1 → V2：去掉静态 Director skill 后，单次成功数增加 2。不能凭一次运行证明 skill 必然有害。
- V2 → V3：成功数增加 28，但同时涉及 Worker 从 GPT-5.5 切到 DeepSeek、搜索索引切换等变化，不能把 21.875 个百分点全算给某一项。
- 早期 V2 搜索覆盖审计明确记录使用 indexes_100k，含 99995 文档，仅包含 128 题中 12 个原目标 ASIN；切换后的全量索引含 1181370 文档，包含这 128 个目标 ASIN。替代商品也可能获得满分，目标 ASIN 缺失不等于任务绝对不可成功。详见 [索引审计](../state/sota-20260916/webshop-index-audit.md)。V1 是否使用同一实际索引在本次未独立重建，不能只按运行顺序认定。
- V3 → V4：Qwen 和 DeepSeek 思考均开启、观察模式及截断改变，而且中间还有规格规范化、动作列表截断等修复。不能把 3.90625 个百分点解释为单独开启思考的收益。[中间修复审计](../state/sota-20260916/webshop-full-index-deepseek-no-skill-v1/followup_audit.md)
- V4 → V5：记录的 WebShop 配置主要差异为 retain_page_text → legacy，同日单次结果增加 4 个成功，平均得分增加 4.21875。它仍不是多随机种子的稳定增益证明。
- V5 → V6：增加反馈后本次减少 4 个成功，平均得分下降 1.54296875；当前源码与历史运行不完全相同，不能独立断言是反馈造成退化。[反馈实验报告](WEBSHOP_ENV_FEEDBACK_EXPERIMENT.zh-CN.md)

当前正式参考是 V5；V4 已归档；V6 是完成的辅助功能实验，没有替换正式参考。V4 与 V6 虽同为 45.3125%，但配置、平均得分和运行目录不同。

## 子集实验和排障运行

这些结果不与上述完整 128 题直接排名。

| 运行 | 题数 | 严格成功 | 平均得分 /100 | 范围与限制 |
| --- | ---: | ---: | ---: | --- |
| closure original | 30 | 0/30 = 0% | 1.28 | 选定的历史失败题；原购买提示 |
| closure removed | 30 | 1/30 = 3.33% | 5.56 | 同一子集；删除相关购买提示 |
| closure best_so_far | 30 | 2/30 = 6.67% | 27.47 | 同一子集；预算内购买已见最佳候选 |
| option-fixes | 21 | 1/21 = 4.76% | 4.76 | 规格/动作列表修复后的子集；20 条轨迹包含 WORKER_BACKEND_FAILURE 标记 |
| option-fixes retry | 20 | 7/20 = 35% | 67.25 | 重试子集；仍有 1 条轨迹含后端失败标记 |
| budget-aware | 27 | 0/27 = 0% | 43.55 | 预算与购买收尾策略子集 |
| price-range goal234 | 1 | 0/1 = 0% | 50.00 | 单题价格区间排障 |

closure 三组使用同一份 failed_active_30.jsonl。best_so_far 的购买次数由原组的 2 次升到 17 次，但严格成功仅由 0 升到 2；购买率和平均 reward 的改善不能当作同幅度的严格成功率改善。后端稳定性也是干扰因素。[原消融报告](../state/sota-20260916/webshop-closure-ablation-deepseek-v1/comparison.md)

09-18 的多次 inference/noskill/c24 启动记录设定 Qwen 和 DeepSeek 均不思考，但对应正式目录均没有完成样本，不能当作另一组“无思考 128 题成绩”。另有单题 smoke 运行，只能验证流程。

## 原始完整运行位置

- V1：[webshop-fixed](../state/sota-20260916/qwen35-9b-director-webshop-fixed-v1/aggregate_summary.json)
- V2：[webshop-no-skill](../state/sota-20260916/qwen35-9b-director-webshop-no-skill-v1/aggregate_summary.json)
- V3：[full-index-deepseek](../state/sota-20260916/webshop-full-index-deepseek-no-skill-v1/aggregate_summary.json)
- V4：[retain_page_text](../state/formal-eval/webshop-deepseek-reasoning-noskill-c24-20260918-170329/aggregate_summary.json)
- V5：[legacy 正式参考](../state/formal-eval/webshop-legacy-page-only-reasoning-c24-20260918-172320/aggregate_summary.json)
- V6：[legacy+反馈](../state/formal-eval/webshop-env-feedback-c24-20260923/aggregate_summary.json)

全部六组的配置计数、实际 Worker 模型、skill、Qwen 思考计数和题目集合哈希见 [机器可读核查结果](../state/experiments/webshop-version-comparison-20260923.json)。本次仅核查历史文件并整理报告，没有更改实验配置或模型服务。
