# HealthBench 正式训练同步记录

当前生效版本已按用户明确选择设为 **46.96 分版**；下文原有同步过程作为历史记录保留。
当前状态如下：

- 保留 Director 完整公开对话、Worker 完整答案及统一 submission contract。
- 正式节点传递恢复为该版的 `relay_max_chars=1000` 行为，不采用后续完整正文传递补丁。
- Worker 由 Director 在 `gpt/grok/gemini/deepseek/minimax` 中自主选择；不引入测试的
  GPT-only、DeepSeek-only 或 `gpt -> student` 强制覆盖。
- Judge 经用户单独确认继续使用 student，并发恢复为迁移前的 5。
- 20 题并发、测试 Judge 并发 10、无限重试和备用 Judge 切换仍仅属于实验编排。
- 41.90 分实验的冻结代码、测试、补丁与结果保持原样；没有启动训练或释放模型资源。

本次选择、修改前备份和验证记录保存在
`state/formal-training/healthbench-baseline-selection-20260925/`。
本次 214 项回归测试通过，两份正式配置通过校验；Qwen API 返回 HTTP 200，模型继续运行。
46.96 是该代码基线在固定 GPT 路由实验中的历史分数，不代表正式自主选路训练的成绩。

## 首次同步记录

目标版本为 `healthbench-unified-contract-gpt-low-128-c20-20260925-164301`：
128/128 题、258/258 个评分项完成；长度调整后平均分 46.9593，原始 rubric 平均分
55.0028。实验详情见 [实验报告](HEALTHBENCH_UNIFIED_CONTRACT_128_C20_2026-09-25.zh-CN.md)。

正式训练并没有另一份 HealthBench 实现。`selfplay_runtime._collect_primary()` 通过
`AdaptiveSolverApplication.solve()` 使用同一套 `src/selfplay_graph_flowsteer` 源码。
以下修复已在该共用路径中确认：

- Director 初始接收带角色标记的完整公开对话；私有 rubric 不进入生成请求。
- Worker 的 `answer` 必须包含其职责所需的完整解释、证据、限定条件及不确定性；
  中间节点仍仅完成分配的子任务，最终节点输出完整用户回复。
- 正常生成、格式恢复和题目末尾 submission contract 使用同一份
  `healthbench_answer_instruction()`；HealthBench 最终答案原样送入 Judge。

本次补齐正式入口与实验的配置差异：

- `configs/formal_training.toml` 和 H200 本地配置移除 HealthBench 的
  `gpt -> gpt_student` 强制覆盖，Worker 使用 `gpt/gpt_eco/gpt_student` 池，均为 low。
- Judge 仍指定 student；student 端点并发由 5 改为 10。正式配置从
  `FLOWSTEER_API_KEY` 读取密钥，替代本 pod 不存在的旧密钥文件。
- 训练入口显式将当前 checkout 的 `src` 放到 `PYTHONPATH` 前面。
  未显式配置 `SPGFS_VENV` 时，优先沿用已有共享环境；共享环境不存在则使用项目 `.venv`。

这是 HealthBench 任务代码和路由配置的同步，不等于启动正式训练。正式训练的 PATS、
35 个采集 worker、GPU 角色分配、600 秒路由冷却和训练期限机制仍沿用训练配置。
实验 `run.py` 中为完成固定 128 题使用的无限重试、Judge 备用路由及请求观察器属于
实验外层编排，未注入训练主循环；因此不能声称正式训练拥有完全相同的重试行为。

验证走真实正式配置及训练所用 application 工厂，模型回复使用合成 mock：覆盖公开
历史传递、统一答案规则、私有标准隔离、完整答案送判和 rubric 原始分。外部 API
没有重新调用评分；Qwen 服务保持运行。275 项相关测试通过，正式与 H200 本地配置均
通过结构校验，两个入口脚本通过 Bash 语法检查。首次合并测试恰逢共享工作区的另一组
QA 配置修改尚未写齐；保留这些修改，在字段定义完整后重新执行，全部通过。
测试记录及文件摘要位于
`state/formal-training/healthbench-promotion-20260925/`。

## 节点传递限制仍存在

后续状态：同日曾尝试完整正文传递修复，随后用户明确选择 46.96 分版作为正式版本。
正式源码现已恢复下述传递行为；完整正文传递仍保留在
[独立实验](HEALTHBENCH_FULL_RELAY_128_C20_2026-09-25.zh-CN.md)中。

本次同步保留实验的 `relay_max_chars=1000`。`MultiAgentRuntime._bounded_artifact()`
在 `len(answer)+len(summary)>1000` 且 summary 非空时，会清空传递副本的 answer，
最多保留 1000 字符摘要以及部分其他字段。没有摘要时，才截取 answer 前 1000 字符。
这个限制按字符而非 token 计算，也不是模型的总上下文上限。

同一规则用于上游输出、同伴提案和自身旧稿，因此自我修订也可能看不到自己上一稿的
正文。原始 artifact 和最终选中答案没有被这个函数改写；受影响的是下一次模型请求
携带的 RelayPacket。新版协议将读者需要的内容放在 answer 中、使 summary 更短，
所以只保留摘要的旧传递策略与完整内容协作的需求不匹配。

按轨迹中去重后的 artifact 及其实际消费的 `source_artifact_ids` 离线重放：

| 实验 | 引用总数 | answer 清空次数 | 涉及题数 |
|---|---:|---:|---:|
| 首轮完整答案修复，20260924-231723 | 131 | 126 | 29 |
| 最新统一协议，20260925-164301 | 88 | 88 | 21 |

最新 88 次中，自身旧稿与同伴/上游各 44 次，没有找不到原始 artifact 的引用。
这些是引用次数，不是丢失评分的题数，也不等于已经测得的失分。接收者还拥有题目、
摘要及部分 evidence，可能重新推导出完整内容；因果影响需要单独消融实验。

例如最新实验题 `c6b2da2a85e39bd9b9018cf135476cf0` 的两份初稿分别有
1801/2508 个 answer 字符和 156/138 个 summary 字符；双方修订时收到的 answer
均为空。完整离线重放证据保存在该实验目录的 `private/relay_answer_replay.json`。
