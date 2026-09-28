# SWE 65/128 正式训练基线

2026-09-28，按用户确认，**仅将 SWE 同步到 65/128 的版本，其他六个数据集保留原提交协议**。

用户随后确认，同次正式发布还必须包含 HotpotQA **111/128（86.72%）** 的已验证版本。
已将其 `hotpot_evidence_first_v1` 输出契约和修订版 v1 数据一同纳入；Hotpot 继续使用 v2.2。
合并后两套数据集的 Worker 普通输出与恢复提示词哈希分别与各自评测记录一致。
详见 [Hotpot 同步记录](HOTPOT_FORMAL_PROMOTION_20260928.zh-CN.md)。

## 来源与成绩

- 版本标识：`swe-65of128-20260928`。
- 评测运行：`state/swe-128-rerun-20260928-205145`。
- 冻结源码：该运行的 `source_snapshot/src`。
- 题目 SHA256：`c56ba020f9bbf791c1e324cf71517c446fb53405a4436ed40c5398b2e17ffd38`。
- 128 题全部完成：官方通过 65，官方未通过 39，本地策略失败 15，未提交未知 9。
- 全部题目口径：**65/128 = 50.78125%**；已送测口径：65/104 = 62.5%。未知题保留未知。
- 15 道本地策略失败中，12 道到达 Worker 用量阈值，3 道 Director 重复非法动作。
- 评测未更新模型参数；50.78% 是这次固定模型、无 Skill 评测成绩，不能当成后续训练后的成绩。

可提交 Git 的逐题状态和版本指纹位于 [评测记录](../experiment_versions/reports/swe-65of128-20260928/)。完整轨迹、usage 账本、测评日志和私有资源保留在本机 state 目录。

## 正式入口

- 正式训练配置：`configs/formal_training.toml`。
- 正式训练脚本：`scripts/formal/run_experiment.sh`，继续加载当前 checkout 源码。
- 本机配置：`configs/formal_eval_worker08_main_v22.local.toml`，同时同步 SWE 设置。文件名中的 v22 表示其默认协议；SWE 有明确的 v3 覆盖。
- SWE 新产物与提交账本：`state/formal-training-swe65-v1/`。

配置中的关键选择：

```toml
[canvas.submission_protocol_by_dataset]
swe_bench = "unified_task_result_v1"

[runtime_routing.dataset_worker_routes]
swe_bench = ["gpt"]

[runtime_routing.dataset_endpoint_pools.swe_bench]
gpt = ["gpt_student", "gpt"]

[runtimes.gpt.max_concurrency_by_dataset]
swe_bench = 5

[runtimes.gpt_student.max_concurrency_by_dataset]
swe_bench = 10
```

全局 Director 仍为 v2.2、默认提交协议仍为 legacy。GraphCanvas 在创建每道 SWE 轨迹时选择统一提交协议，GraphDirector 随之使用与评测一致的 v3 提示词和动作 schema。其他六个数据集继续沿用原协议和路由池。

SWE 的 GPT-student 与非 eco GPT 复用原端点池排队、轮换、失败切换规则。仅 SWE 选择两成员池；其他数据集仍可使用原三成员 GPT 池。并发覆盖作用于 SWE 请求上下文，默认 GPT=10、GPT-student=5 的其他数据集设置不改动。

整题 Worker 用量继续采用 `reported_usage_threshold_v1`：350,000 发送阈值、所有 Agent/修订/重试共享实际 usage、每题最多一个本地在途请求、最多两个未结算请求。SWE 工具次数、测试环境、输出长度和远程测评身份沿用被评测版本。远程测评保留服务端并发 4，自动关机使用 `STOP_CHARGING`。

## 训练接入

正式七数据集混合训练继续使用已有 Proposer/Solver、PATS、五条同题轨迹、整体 24 条滚动轨迹和 GPU 角色分配。评测的 15 条轨迹并发属于复现实验参数，没有把全局混合训练并发改成 15，也没有把评测端口 18605/GPU 4 写入训练角色。

新增的接入代码只负责按数据集选择已有协议与池成员：

1. Canvas 根据数据集解析提交协议，在创建动作解析器和图之前完成选择。
2. 执行指纹同时记录默认协议及 SWE 的 Director/动作/提交协议覆盖；混合批次共享完整配置指纹，旧采样记录不能悄悄混入。
3. PATS 审核、修订、冻结、加载和上下文选择按 scope 使用对应契约：SWE=v3，其余=v2.2。旧 v2.2 审核不能充当 SWE v3 审核。
4. 新鲜路由报告必须覆盖 SWE 两成员池；只有 eco 可用时，不会误判 SWE 路由已经就绪。
5. 真正的远程未评分/基础设施异常继续排除奖励训练，未知结果不填零。
6. 批次绑定和学习入口按每条轨迹的数据集核对提交协议，允许 SWE v3 与 Hotpot v2.2
   组成同一合法混合批次；错用协议、缺少数据集身份和不同执行指纹仍被拒绝。

此次同步不实施 `SWE_SUBMISSION_RECOVERY_PLAN_20260928.zh-CN.md` 中尚未实现的候选检查点恢复、收尾裁剪、双向组件轮次发布等新方案。65/128 版本中已知的未提交和部分 token_cost 少计现象仍按原始证据记录；实际用量以 usage 账本为准。

## 归档、验证与发布

更新前源码、正式及本机配置和工作区说明，保存在：

```text
state/formal-training-promotions/swe-65of128-20260928/before/
state/formal-training-promotions/swe-65of128-20260928/before-sha256.json
```

被评测源码另存为同目录 `evaluated_source/`，不会因工作目录后续开发而改变。

针对性验证覆盖：SWE/其他六集协议隔离、两成员故障切换与 eco 隔离、实际用量边界、提交候选完整性、PATS 分数据集审核/冻结、训练契约，以及正式/本机配置可加载性。验证结论和 Git 发布指纹见 [版本清单](../experiment_versions/reports/swe-65of128-20260928/promotion.json)。

合并后共 564 项相关回归检查通过。完整检查中两份已退休 H200 本机配置仍保留历史
v2.1，因此不计入当前正式入口的验收；当前正式配置检查为 24 通过、2 项历史配置排除。
SWE 池负向测试最初被独立 Judge 资格检查提前拦截，修正测试夹具后通过；没有放宽生产资格检查。

本次操作是版本同步和 Git 发布，不启动训练或更新权重。已有 Director 保持占位；远程测评服务器保留此次评测结束时确认的不计费关机状态。
