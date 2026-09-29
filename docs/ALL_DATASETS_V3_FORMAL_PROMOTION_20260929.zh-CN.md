# 七个数据集统一 V3 正式协议（2026-09-29）

用户确认将原先仍使用旧协议的五个数据集全部切到 V3，包括 WebShop。
当前正式配置中的 AIME、NQ、HotpotQA、HealthBench Professional、WebShop、
ALFWorld、SWE-bench 均使用 `unified_task_result_v1`、Director V3 和
`unified_submission_v1` 提交回执。

## 提交行为

- Director 不再使用 `set_output`；创建责任时明确 `result_scope=subtask/task_result`。
- `subtask` 只提供局部证据；完整任务通过 `finish(target)` 提交指定的 `task_result`。
- FINISH 核验当前产物并提交，不重新调用 Worker。需要继续执行时使用 `run_agent`。
- 图依赖、产物新鲜度、环境状态和数据集输出要求仍必须满足。

| 数据集 | 此次生效的提交路径与保留设置 |
|---|---|
| AIME | V3 提交；保留数学工具、答案抽取和 240,000 token 上限 |
| NQ | V3 提交时核验目标节点可见引用；保留 R2D2 1,702,133 段语料、每次 top-8、每题共享最多 4 次检索 |
| HotpotQA | V3 提交；保留 `hotpot_evidence_first_v1`、公开上下文与修订后的评测集 |
| HealthBench Professional | V3 提交完整回答；保留完整公开对话与现有 Judge 设置 |
| WebShop | 各节点独立购物会话，共享总计 16 次工具动作；FINISH 只购买选中节点准备的商品 |
| ALFWorld | 延续已启用的 V3、真实环境任务绑定、成功候选保护与 usage 阈值 |
| SWE-bench | 延续已启用的 V3、补丁与编辑后测试校验、实际 usage 阈值 |

NQ 的 `insufficient_evidence` 仍是合法弃答并计零分；伪造引用不得生成有效提交回执。
检索服务故障仍属于未完成执行，不记作正常训练零分。FINISH 的证据检查不暗中增加
Worker 修复调用；继续执行和修复服从现有显式调度与检索协议预算。

WebShop 的局部搜索责任不再被强制改写成购买责任。完整任务仍必须准备真实购买，
不能凭“已购买”的文字报告获得成功。实际共享 16 次预算用尽后禁止创建空的新购物节点；
已有节点可按真实终止状态提交失败。保留 M02 商品身份校验、现有工具接口和
350,000 token 上限，没有合入后续商品记忆实验或 WebShop 的实际 usage 阈值实验。

## 正式入口与训练边界

`configs/formal_training.toml` 以及本机的以下三个已有配置已同步七种数据集覆盖表：

- `configs/formal_training_h200.local.toml`
- `configs/formal_eval_h200.local.toml`
- `configs/formal_eval_worker08_main_v22.local.toml`（文件名保留；当前七种数据集实际使用 V3）

原有全局 Director 默认值仍保留；七种数据集都由
`canvas.submission_protocol_by_dataset` 显式覆盖为 V3。执行清单、PATS 契约和训练
校验读取生效后的数据集协议；旧协议回执不能混入新协议的训练批次。

正式启动脚本将新的 experiment 写到 `state/formal-training-all-v3-20260929/experiment`，
提交日志写到同目录的 `submissions`。服务资产、路由报告、模型设置、并发设置和
数据路径保留原值。历史独立实验配置及已有 rollout 不改写，不作为此次 V3 的新采样结果。

## 来源与保存

切换前 main 为 `4678541bb06a74ea330e9c28037a2256b31e8bc1`，已保存到分支
`experiment/formal-before-all-v3-20260929`。这是含 NQ 新语料检索、WebShop 恢复版和
ALF/SWE V3 的完整旧正式版本。本机四份原始配置和切换前文件哈希另存于
`state/all-datasets-v3-promotion-20260929/`。

V3 主体沿用当前 main 已有实现；WebShop 的数据集级生命周期选择、局部责任说明及
预算耗尽后的新建节点限制参考 `fb2112d`。NQ 额外接通 V3 提交时的证据检查及回执评分。
未合并整个 WebShop 实验分支。原有独立实验副本、训练数据及 37 个未跟踪研究文件保留。
WebShop 继续使用修正索引并隔离测试集后的 **444 条训练数据**。

WebShop 的历史 **64/128（50%）** 属于切换前 v2.2 评测；当前 V3 改变了提交与会话行为，
不能继续称作“50% 那一版”。其它数据集的历史成绩同样不能视为此次 V3 的重测结果。

## 验证

相关离线回归 **1,137 项通过、1 项跳过**；另排除 1 项依赖本机缺失历史轨迹文件的
旧 AIME 回放。该回放文件不是正式训练依赖。验证覆盖：

- 七种数据集的 Director V3、PATS 和训练回执协议一致性；拒绝旧协议回执。
- NQ 引用、弃答、服务故障，以及训练反事实分支的独立证据和检索预算。
- WebShop 正式应用工厂创建独立会话，FINISH 只提交所选商品且零 Worker 调用。
- WebShop 训练反事实分支重新执行独立会话并购买；真实轨迹耗尽预算后不能新建空节点。
- ALFWorld / SWE 原有 V3 行为，文本任务、共享运行时和历史兼容路径。

未启动正式训练或付费模型评测；本次没有新的准确率结果。
机器可读的来源、文件保护及验证记录见
[版本目录](../experiment_versions/promotions/all-datasets-v3-20260929/)。
