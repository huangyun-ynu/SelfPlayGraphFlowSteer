# W08 Worker 提示合并版

2026-09-24，按用户要求合并公共 Worker 购物指令与 LASER 清单，候选策略为 `merged_checklist_v1`。仅进行提示去重和优先级整理，未加入前一份分析中的逐项证据格式、Director 委派改写或强制评审。

## 合并后的顺序

1. 已暂存、已购买或已终止：进入原有完成协议，不继续购物。
2. 核对公开证据：类型、属性、价格、可用规格与当前已选规格；缺失证据仍为未知。
3. 证据支持需求且规格已选：暂存购买。
4. 尚有缺口，且有预算完成有用的调查与购买：优先解决缺口或矛盾。
5. 继续调查会导致无法完成购买时：保留返回商品页、选择规格和购买的额度，对已见最佳候选进行兜底；如实保留未解决条件。无可执行购买路径或无相关商品时允许不购买。
6. 四类页面说明只补充该页面如何获取信息；最后集中规定 purchase_evidence 与暂存提交协议。

保留 target ID、selected_options、价格区间不代表确定成交价、购买证据结构、一次状态动作、最终 JSON 等现有接口规则。未增加具体题目、商品、查询词、示例答案或隐藏评分信息。

## 接入方式

- `src/selfplay_graph_flowsteer/webshop_guidance.py` 中 `MERGED_PAGE_CHECKLIST` 为可审阅的合并文本。
- `runtime.py::_worker_output_instruction` 在该策略下替换原 WebShop 购物段。随后不再附加旧 LASER，避免两份规则同时出现。
- 默认 `baseline` 与原 `laser_checklist_v1` 行为保持。候选配置 `configs/webshop_merged_checklist_eval.toml` 与 LASER 配置的解析结果仅相差 `webshop.worker_guidance_policy`。
- 只对具有环境操作能力的 WebShop Worker 注入；所有者修订保持该策略，无状态评审和其他数据集不注入。Native 的配置边界保持不变。
- Director 提示、环境状态、工具 schema、动作调用、预算及模型设置没有随此候选更改。

## W08 单独对照包

当前主目录已含 W08 之后的其他改动，直接从主目录切换配置不能冒充与 W08 的单变量对照。

因此另外从已核验的 W08 副本生成独立候选：

`state/experiments/webshop-w08-merged-prompt-20260924/candidate/`

它只修改三份源码：guidance 文本与合法策略列表、runtime 的策略接线、application 的错误信息，并新增候选 TOML。其余源码与 W08 相同，Director 文件逐字相同。归档基线未修改，模型权重未更新。制作对照包时只做离线检查，随后完成了下述 128 题配对推理复测。

可移植的增量补丁：`experiment_versions/patches/webshop-w08-merged-checklist-v1.patch`。该补丁基于恢复后的 W08，而不是主目录：

```bash
.venv/bin/python scripts/formal/restore_experiment_version.py \
  webshop-budget-off-128 state/restored-versions/webshop-w08-merged-review
git -C state/restored-versions/webshop-w08-merged-review apply \
  "$PWD/experiment_versions/patches/webshop-w08-merged-checklist-v1.patch"
```

以上只恢复代码，不启动服务或评测。评测需使用这个候选副本的源码和候选配置，保持 W08 的 DeepSeek reasoning_effort=low、Qwen thinking、12+4=16 动作预算和原执行后 token 核验等设置；本次不变更 baseline 的 63/128 历史成绩。

## 验证与结果边界

审计目录：`state/experiments/webshop-w08-merged-prompt-20260924/`。

- W08 原始 92 个文件哈希再次核对通过；增量补丁通过 `git apply --check`。
- 独立进程分别导入 W08、W08 候选与当前主目录。原 baseline/LASER 输出指令逐字一致；两个候选的合并指令逐字一致，见 `prompt_audit.json`。
- 同一固定 Action schema 下，输出／动作指令与 guidance 合计从 6,190 字符降至 4,450 字符，减少 28.11%。这不是完整请求长度，也不是实际 token 节省率。
- 脚本化 Worker 对照检查实际发送的请求：合并块仅出现一次、旧两段被替换；相同动作序列下的观察、工具参数及预算一致。另覆盖路由执行、所有者修订、评审隔离、其他数据集及配置传递。
- 主工作区相关回归 283 项通过；独立 W08 候选的提示及预算专项 21 项通过。Ruff 与 `git diff --check` 通过。
- 提示字符缩短本身不构成准确率或 token 开销改善的证据；真实推理结果见下文。

## 128 题配对复测

2026-09-24，两组分别完整重跑相同 128 题，保持模型、`reasoning_effort=low`、thinking、seed、并发和预算一致，均正常完成且执行异常为 0。

| 指标 | W08 原版，本次重跑 | 合并版，本次重跑 |
|---|---:|---:|
| 严格成功 | 58/128，45.3125% | 66/128，51.5625% |
| 平均 reward | 0.65703125 | 0.723359375 |
| 平均 Worker token | 71,626.41 | 61,414.36 |
| 平均合计 token | 91,035.04 | 79,593.23 |

逐题改善 15 题、退步 7 题，净增 8 题；严格成功的精确 McNemar p=0.1338，尚不足以证明稳定提升。平均合计 token 降低 12.57%，任务耗时 P95 则增加，不能概括为所有效率指标均改善。F1 不适用于官方 WebShop。

完整报告及全部中间产物见 [128 题复测报告](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/docs/WEBSHOP_W08_MERGED_128_2026-09-24.zh-CN.md)。历史 W08 的 63/128 仍保留为历史记录，未用其代替本次对照。
