# WebShop main 恢复记录（2026-09-29）

按用户要求，main 的 WebShop 推理实现恢复到 `235e670fd83b089489aa95e902dedffd9eda9c65`。
该提交对应 2026-09-28 的固定 128 题评测：**64/128 完整成功（50%），126/128 已购买**，
平均分 72.06380/100。成功条件是 `purchased && reward >= 1.0`，不是答案字符串 EM。
这是历史运行成绩；本次只做代码恢复与离线验证，没有重跑 128 题，也没有启动训练。

## 恢复范围

- 全部 10 个 `src/selfplay_graph_flowsteer/webshop*.py` 原版文件与 `235e670` 逐字一致。
- 共享 Runtime 中 WebShop 提示、商品记忆、购买路径判断、暂存购买、闭环预算及错误反馈恢复原版。
- 移除后来添加的 `webshop_evidence.py`、`webshop_steps.py` 及对应实验测试；完整内容在实验副本中保留。
- WebShop 不再应用后来给 Student gateway 增加的文本解码和交互修复次数；其他任务继续使用现有实现。
- Canvas 对 WebShop 恢复原来的控制快照及待分配模型节点删除规则，保留 ALFWorld/SWE 的后续修复。
- 正式 WebShop 配置保持 M02 / `merged_checklist_v1` / `legacy` / `phase_split_v1`，动作预算仍为初始 12、修订 4、总共 16。

这是按数据集恢复，main 整库仍保留 SWE、ALFWorld、HotpotQA 等后续工作。
共享模块的版本号和源码哈希因此不能与整库 `235e670` 完全相同。

## 按用户选择保留的数据修正

继续使用修正目标索引、隔离测试集后的 **444 条 WebShop 训练数据**，不恢复旧的 512 条。
保留索引映射脚本、训练启动前检查和隔离清单；正式配置、本机评测配置、训练数据及清单的
SHA-256 均与恢复前相同。训练数据 SHA-256：
`7166743ec89973e30809b5c266a094821d98a39f801a128e63d0304a8ced2d60`。

## 实验副本

恢复前 main 为 `ce665fbd3cd6b41e1a0a2d724d9bc58ce734e995`，已完整保存在：

- Git 分支：`experiment/webshop-main-before-restore-20260929`。
- 本机源码归档：`state/webshop-main-restore-20260929/before-main.tar`。
- 归档 SHA-256：`21075d01d5dbae32d7594ad4cdcbc80000c863ebf1fdb8fe61906812f07d740d`。

归档包含该提交全部已跟踪文件；本机配置与训练数据另存于同目录的 `before/`。
此前的 WebShop 实验工作区保持原样，包括
`SelfPlayGraphFlowSteer-webshop-alf-align-v2` 中的
`codex/webshop-purchase-path-20260929`（`d3a7867`，含购买路径保护、候选判断及双向失败提交实验），
以及 `SelfPlayGraphFlowSteer-webshop-alf-v3` 等已有副本。本次没有推送远端。

查看恢复前的源码可使用：

```bash
git show experiment/webshop-main-before-restore-20260929:src/selfplay_graph_flowsteer/webshop.py
```

## 验证与证据

- 10 个 WebShop 专用文件逐字对照通过；45 个 Runtime WebShop 函数及 `WebShopConfig` 的 AST 对照通过。
- 106 个受保护文件（含其他任务的专用源码、正式/本机配置及训练数据）与恢复前哈希一致；原有 37 个未跟踪文件保持原样。
- 在原版干净工作区与恢复后的工作区运行相同离线探针，7 组结果完全一致，覆盖 17 次 Worker 请求、初始/修订、路由、异常购买参数恢复、购买暂存、Director 提示、控制快照与待分配模型节点删除。
- 原版生成的行为哈希保存在 `tests/fixtures/webshop_235e670_behavior_sha256.json`，由 `tests/test_webshop_baseline_restore.py` 检查，避免把恢复后的输出直接当作正确基准。
- 跨数据集回归：660 项通过。最后的 WebShop/Student/Canvas 状态检查中 161 项通过；额外一项 AIME 历史回放因本机缺少轨迹文件而无法运行断言。该旧轨迹只供这项测试读取，不是正式训练的运行依赖。

机器可读记录见 [恢复清单](../experiment_versions/promotions/webshop-main-235e670-20260929/manifest.json)、
[源码检查](../experiment_versions/promotions/webshop-main-235e670-20260929/source-audit.json) 和
[行为检查](../experiment_versions/promotions/webshop-main-235e670-20260929/behavior-comparison.json)。
测试结果与执行命令见同目录的 [验证记录](../experiment_versions/promotions/webshop-main-235e670-20260929/validation.json)。

历史评测的原始证据仍保留在
`state/formal-eval/webshop-main-v22-128-deepseek-c40-qwen-thinking-20260928/`，
包括 `run_spec.json`、`ANALYSIS.zh-CN.md` 和逐题结果。
代码恢复会同时恢复旧版本的行为和局限；离线一致性不能保证再次采样仍恰好得到 50%。
