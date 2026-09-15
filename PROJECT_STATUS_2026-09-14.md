# SelfPlayGraphFlowSteer + PATS 项目进度与交接

更新日期：2026-09-15（Asia/Shanghai）。本文件依据当前代码、正式配置、数据产物和环境检查整理；历史实验主体生成于 2026-09-12。

## 1. 当前结论

**PATS 主体集成代码、七数据集正式池和正式训练配置均已落地；SWE Verified 本地源码缓存与腾讯云官方验证端已接入。当前没有训练进程运行，正式 `route_report.json` 尚未生成，因此仍需路由预检和一轮 canary 后再开始长训练。**

当前正式池为 7 个数据集各 512 条，共 3584 条，每个数据集 16 个 ADS 簇。WebShop 已按官方 goal index 边界重建：训练只来自 `1500–12086`，128 条测试只来自 `0–499`，零重叠。SWE 使用 Verified 500 的内部无交叉划分：128 条保留测试，剩余 372 个唯一问题全部保留并按难度均衡重复到 512 条训练数据。旧普通 SWE train 及其 ADS 已删除。

最新迁移清单见 [MIGRATION_MANIFEST_2026-09-15.md](/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/MIGRATION_MANIFEST_2026-09-15.md)。含明文密钥的私密备份位于 `/home/bedicloud/sharestore2/iclr-users/2/SelfPlayGraphFlowSteer_PRIVATE_CONFIG_2026-09-15.md`，权限为 `0600`，不得提交 Git。

第二轮已完成采样、反事实评估、训练批次构建和 PATS 审查；Solver 在训练前的概率一致性检查中失败，所以这一轮没有完整提交。数值修复已有真实样本上的正向诊断结果，但尚未正式接入训练代码。新技能的独立语义检查也仍有误拒问题。

因此，当前不能表述为“全部跑通”“正式训练完成”或“已经验证 PATS 提升效果”。

## 2. 项目及重要路径

项目根目录：

```text
/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer
```

| 内容 | 绝对路径 / 入口 |
| --- | --- |
| 本进度文档 | [PROJECT_STATUS_2026-09-14.md](/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/PROJECT_STATUS_2026-09-14.md) |
| 项目说明 | [README.md](/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/README.md) |
| PATS 中文说明 | [docs/PATS.zh-CN.md](/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/docs/PATS.zh-CN.md) |
| PATS 详细接口说明 | [docs/PATS.md](/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/docs/PATS.md) |
| 主代码目录 | `/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/src/selfplay_graph_flowsteer` |
| 测试目录 | `/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/tests` |
| 七数据集正式训练目录 | `/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/state/pats-formal-20260912` |
| 正式训练配置 | [config.toml](/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/state/pats-formal-20260912/config.toml) |
| 正式课程 / 选题配置 | [formal_3500.toml](/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/configs/curriculum/formal_3500.toml) |
| 正式启动脚本 | [run_experiment.sh](/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/state/pats-formal-20260912/run_experiment.sh) |
| 正式环境变量脚本 | [environment.sh](/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/state/pats-formal-20260912/environment.sh) |
| 迁移清单 | [MIGRATION_MANIFEST_2026-09-15.md](/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/MIGRATION_MANIFEST_2026-09-15.md) |
| 正式七数据集 ADS 池 | `/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/state/pats-formal-20260912/private/datasets/ads/validated/validated_task_pool.jsonl` |
| SWE Verified 128 条保留测试 | `/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/state/datasets/swe-bench-verified/78f471bf655a3137b2e8a75af1501690ec009ec3/derived/stratified-128-test-balanced-512-train-seed-20260915/spgfs_test_public.jsonl` |
| HealthBench Professional 128 条保留测试 | `/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/state/datasets/healthbench-professional/349962fd46dd02343a0d8a606491baf59154ea1a/derived/stratified-128-test-balanced-512-train-seed-20260915/test.jsonl` |
| WebShop 官方 test 中的 128 条保留测试 | `data/formal/eval/webshop_official_test_128.jsonl` |
| Git 可拉取的加密正式数据包 | `data/formal/private/formal_data.tar.gz.enc` |
| SWE 本地源码缓存 | `/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/state/pats-formal-20260912/swe/repo-cache` |
| 历史三数据集架构验证产物 | `/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/state/pats-real-architecture-20260912` |
| 本地服务管理 | [manage_local.py](/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/state/pats-real-architecture-20260912/manage_local.py) |
| 数据预处理目录 | `/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/state/codex-pats-formal-preflight` |
| 测试与回归日志 | `/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/state/codex-pats-validation` |

上游仓库：https://github.com/huangyun-ynu/SelfPlayGraphFlowSteer 。当前分支：`feat/pats-skill-scaffold`。修改仍在本地工作区，包含新增文件，尚未提交为新的 Git commit，也未推送。

## 3. 项目在做什么，以及 PATS 接在何处

原链路是：数据预处理与候选检索 → Proposer 选题 → Solver 的 Director 构造 Agent 图 → 冻结 Worker 执行 → 验证器评分、Frontier 与关系反事实计算 → Proposer / Solver 分别更新参数。

PATS 作为 Director 的训练期技能组件：根据可信轨迹估计各任务作用域的表现，生成、修订或压缩技能卡；下一轮检索并冻结这些提示，模型按实际带技能的原始输入训练。默认最终评测关闭技能提示，以检查撤除支架后的行为。

参考实现为 https://github.com/shi-yipeng/PATS ，已核对版本 `bad468b5c73081c2f5aa74c4e0011c6fb2872dbf`。本项目做的是适配后的组件融合，不是复现该论文完整训练配方或额外技能条件 SFT。

### 已实现的主要内容

| 模块 | 实现内容 | 代码 |
| --- | --- | --- |
| PATS 控制器 | 分作用域维护、EMA、EXPAND / REVISE / COMPRESS / FORCED_PRUNE、原子编辑、幂等回执 | [pats.py](/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/src/selfplay_graph_flowsteer/pats.py) |
| 证据提取 | 真实动作、公开反馈、图关系、技能版本；排除不可信和未知归因样本 | [pats_evidence.py](/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/src/selfplay_graph_flowsteer/pats_evidence.py) |
| 技能审查 | 严格 JSON Schema、证据短 ID 映射、输入预算与输出审计 | [pats_refiner.py](/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/src/selfplay_graph_flowsteer/pats_refiner.py) |
| 接口语义检查 | 独立逐卡检查、拒绝或未审卡不暴露；审批绑定卡内容、版本、作用域与契约 | [pats_semantics.py](/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/src/selfplay_graph_flowsteer/pats_semantics.py) |
| 检索及冻结 | E5 检索、新卡优先、最多三卡、实际提示预算、旧快照恢复、使用账本 | [skill_evolution_v2.py](/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/src/selfplay_graph_flowsteer/skill_evolution_v2.py) |
| 执行入口 | 训练 / 评测开关、首次冻结前自动接入语义审查 | [cli.py](/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/src/selfplay_graph_flowsteer/cli.py) |

已修复过的实质问题包括：证据读取层级错误、Refiner 输出 ID / JSON 无效、新卡写入后检索不到、作用域使用统计混合，以及不完整终态图被误送去做可选反事实评估。

同组采样共享冻结提示；训练保留真实 prompt token、动作掩码和行为概率，不在 PPO 时重新检索新技能。Worker 与 Refiner 参数冻结。

## 4. 历史架构验证进度

以下内容仅记录旧的三数据集架构验证，不是正式训练。旧目录中的配置、课程和启动脚本已经删除，训练产物、日志、诊断和检查点继续保留为审计证据。

该历史实验原计划为 **3 轮训练 + 1 轮关闭技能的评测**，每轮 6 个任务，每个任务计划采样 5 条轨迹。

| 轮次 | 当前结果 |
| --- | --- |
| `cycle-0000`，第一轮训练 | 30 条原始轨迹，26 条训练可用；两角色更新、检查点和整轮状态已提交。首次 PATS 审查三组均失败，原始失败记录保留。 |
| `cycle-0001`，第二轮训练 | 29 条原始轨迹 + 1 个仅记分的超时终态；26 条进入训练批次。PATS 三作用域审查产生 4 张卡。Proposer 写出未共同提交的检查点，Solver 概率检查失败，整轮未提交。 |
| `cycle-0002`，第三轮训练 | 尚未开始；待验证新技能真正进入后续采样和参数更新。 |
| `cycle-0003`，关闭技能评测 | 尚未开始。 |

权威状态：[experiment_progress.json](/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/state/pats-real-architecture-20260912/experiment/experiment_progress.json) 的 `completed_cycles = 1`；[training_state.json](/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/state/pats-real-architecture-20260912/experiment/training_state.json) 的 `cycle = 1`、`global_step = 2`。

### 检查点位置与含义

- 已提交 Proposer：`/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/state/pats-real-architecture-20260912/experiment/checkpoints/proposer/step-00000001`
- 已提交 Solver：`/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/state/pats-real-architecture-20260912/experiment/checkpoints/solver/step-00000002`
- 未完整提交的第二轮 Proposer：`/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/state/pats-real-architecture-20260912/experiment/checkpoints/proposer/step-00000003`

第一轮处于学习率 warmup 的起点，实际应用学习率为 0：有反向计算和优化器状态更新，但参数改变量为 0。第二轮未提交 Proposer 检查点的独立比较确认 7,340,032 个元素非零变化，参数差 L2 约 0.002012；这不能替代两角色整轮提交证明。

关键记录：[第二次启动日志](/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/state/pats-real-architecture-20260912/experiment-resume.log)、[第二轮 PATS 回执](/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/state/pats-real-architecture-20260912/experiment/cycle-0001/pats_review.json)、[训练选择](/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/state/pats-real-architecture-20260912/experiment/cycle-0001/training_selection.json)、[第二次 Proposer 权重检查](/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/state/pats-real-architecture-20260912/proposer_second_update_audit.json)。

## 5. 当前未解决问题

### 5.1 训练与推理概率一致性

失败调用为 `task-3-r0:7:relation`。采样端 `P(on)` 约 0.705785，训练重算约 0.658418；完整二元分布的 KL 约 0.005112，超过原有 0.005 门限。尚未发现原始 token、掩码或检查点身份错误。

已完成隔离诊断：

- 去掉末尾 choice token：没有解决差异。
- 对 GDN 小投影使用 FP64：修复该调用，但引入其他调用失败，未采用。
- 对 Q/K 归一化内部使用 FP32，再转回原 dtype：本轮全部 **19 个二元调用通过原有门限**，最大 KL 约 0.004129。

最后一项是有源码依据的候选修复：Transformers fallback 与 vLLM 的归一化计算精度存在差别。但它目前**仅在隔离诊断脚本中实现，尚未接入正式训练模块，也尚未验证完整反向传播和全部普通动作调用**。没有放宽门限或替换原行为概率。

证据：[默认重放](/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/state/pats-real-architecture-20260912/binary_replay_diagnosis.json)、[FP32 归一化重放](/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/state/pats-real-architecture-20260912/binary_replay_fp32_qk_norm.json)、[诊断脚本](/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/state/pats-real-architecture-20260912/diagnose_binary_replay.py)。

### 5.2 生成技能及语义检查质量

正式生成卡存在接口过度泛化：例如把“修订已配置 Agent 需要证据”误写成“所有 SET_PROMPT 都必须上游证据”；AIME 卡还可能将输出格式修复设为选择输出前的必要条件，阻断合法修复流程。

独立语义门已实现，但真实检查仍有误判。最新 `semantic-interface-validation-v3` 对同一组 4 卡调用 3 次，批准 1 张、拒绝 3 张；NQ 卡仍被错误地以“不能拒绝 Canvas 提供的关系”为由拒绝，而 off 本来就是合法选择。

**语义检查仍需完善，不能把模型给出的批准 / 拒绝直接当成接口正确性的证明。** 三次验证均使用独立数据库，未改正式技能状态或历史回执。

记录：[v3 验证汇总](/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/state/pats-real-architecture-20260912/semantic-interface-validation-v3/validation_summary.json)、[NQ 原始判定](/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/state/pats-real-architecture-20260912/semantic-interface-validation-v3/scope-02.json)。

### 5.3 GPU 与服务

用户授权使用物理卡 1、5。此前 5 号卡被其他账号占用，当前账号终止进程时收到权限错误；这一历史事件记录在 [GPU 占用记录](/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/state/pats-real-architecture-20260912/gpu5-reclaimed-occupant.json)。

2026-09-14 本次检查未发现本项目训练进程或原 18601 / 18602 / 18603 服务启动进程。卡 1、5 当前仍有较高显存占用，不能假定资源空闲。恢复前应重新核验占用和模型服务。

曾讨论过单卡共置服务、串行训练的备选方案，但当前目录未发现已生成的单 GPU 配置文件，**该部署尚未落地或验证**。旧启动脚本仍按原配置运行，不应直接当成已修复的恢复入口。

## 6. 数据与环境

历史架构验证池共 **96 题**：AIME、NQ-open、HotpotQA 各 32 题；已完成真实 ADS 预处理与验证。它不会被正式启动脚本采用。

- 训练池：`/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/state/codex-pats-formal-preflight/ads/validated/validated_task_pool.jsonl`
- 训练池 manifest：`/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/state/codex-pats-formal-preflight/ads/validated/validated_task_pool_manifest.json`
- 原始抽样数据：`/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/state/codex-pats-formal-preflight/sources/raw_qa_train_96.jsonl`
- 验证器检查：`/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/state/codex-pats-formal-preflight/sources/verifier_validation.json`

正式七数据集任务池已经生成并绑定到 `SPGFS_FORMAL_TASK_POOL`：AIME、NQ-open、HotpotQA、WebShop、ALFWorld、HealthBench Professional、SWE-bench Verified 各 512 条，共 3584 条。池已通过逐行 JSON、公开字段和 16 簇 ADS 审计。WebShop 训练/测试已按官方 split 修复并确认零重叠；AIME 按最新决定维持当前 1983–2024 的 512 条唯一训练题。

SWE 本地缓存包含 11 个完整 Git mirror，共约 4.1 GB；训练需要的 372/372 个 base commit 均可解析。运行时在本地隔离工作区生成 patch，再通过 SSH 交给腾讯云服务器中的官方 SWE-bench Docker 镜像验证。`swe.enabled = true`；未部署的通用 retrieval 服务保持关闭。

主轨迹与关系反事实轨迹按各自的历史数据集平均耗时做长任务优先调度。正式启动脚本通过 `--historical-duration-priority` 开启该机制；每轮只使用此前轮次的持久化记录并冻结排序快照。

正式训练恢复三卡异步流水线：GPU 0、1 并行更新 Proposer/Solver，GPU 2 使用冻结策略采集下一 cycle。异步采集在当前 cycle 的 PATS 维护提交后启动，并把维护后的 PATS 快照、上下文契约及文件哈希绑定到策略谱系；策略最多滞后一次更新，PATS 视图不允许滞后或在恢复时变化。

环境路径：

- GPU Python：`/home/bedicloud/sharestore2/iclr-users/owner/.venvs/spgfs-pats-gpu`
- CPU 测试环境：`/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/.venv`
- 基模型：`/home/bedicloud/sharestore2/iclr-users/3/gpf/models/Qwen3.5-9B`
- E5 模型：`/home/bedicloud/sharestore2/iclr-users/owner/models/e5-base-v2`
- 共享缓存：`/home/bedicloud/sharestore2/iclr-users/owner/.cache/spgfs-pats`

GPU 环境记录版本：Python 3.11、PyTorch 2.13.0+cu130、Transformers 5.15.0、PEFT 0.20.0、vLLM 0.27.1。环境中 FlashInfer 的 Python 3.11 类型注解兼容补丁有独立脚本和记录，见实验目录的 `patch_dependencies.py`、`dependency_patch.json`。

## 7. 测试结论应如何理解

- 2026-09-15 最新完整回归：843 passed、6 skipped；使用本机 `/tmp` 作为 pytest basetemp，避免 NFS `nobody` 所有权映射触发 pytest 安全拒绝。
- WebShop 官方 split 重建审计：训练 512 条全部来自 `goal_index >= 1500`，测试 128 条全部来自 `goal_index < 500`，source ID 交集为 0；WebShop embedding 与 NLL 均按新样本 ID 重算。
- 便携部署验证：`configs/formal_training.toml` 完整校验通过，加密正式数据包成功回读并由 bootstrap 加载 3584 条任务。
- 历史全量回归：737 passed、11 skipped、1 failed；失败为旧后台线程测试未在时限内结束，之后单测与所在模块重跑通过。不能将该次全量记录写成全绿。
- PATS 集成和使用统计的两次重叠针对回归，共 129 个去重用例通过，见 `state/codex-pats-validation/latest-targeted-summary.json`。
- 新增 CLI 语义审查接线测试：7 passed，见 `state/codex-pats-validation/cli-semantic-tests.xml`。
- 不完整终态图反事实修复：相关 15 项测试通过，日志在 `state/codex-pats-validation/cf-terminal-precondition-3e_maywp/pytest.log`。
- 后续语义模块与冻结测试已有针对验证；以本节第一条的 843 passed、6 skipped 作为当前整套代码的最新完整结果。不同批次测试重叠，不应相加作为总测试数。

## 8. 下一步接手顺序

1. 迁移项目、私密配置和所需共享依赖，按迁移清单替换所有旧绝对路径并重建 Python 环境或验证复制环境。
2. 在新账号直连探测全部物理 API 端点并生成新鲜 `state/pats-formal-20260912/route_report.json`；模型 API 不得走代理。
3. 检查腾讯云实例已开机、SSH Host Key 未变化，并用空 patch 做一次官方验证端探测；自动开关机仍需要腾讯云 CAM API 凭据。
4. 使用正式入口做一轮可审计 canary，验证七数据集平衡选题、35 个共享主/反事实槽、主轨迹优先、8 个异步图复验槽及按数据集流水线。
5. canary 通过后从新的正式状态目录启动训练，不恢复旧三数据集检查点或进度指针。
6. 继续处理历史实验暴露的 Q/K 归一化概率一致性和技能语义误拒问题；不要用放宽门限掩盖差异。

旧目录中的 `audit_completed_experiment.py` 只适用于历史架构验证产物，不能作为七数据集正式训练的完成证明。
