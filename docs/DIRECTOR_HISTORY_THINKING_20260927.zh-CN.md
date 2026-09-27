# Director 历史 thinking 剥离（历史方案，已撤回）

日期：2026-09-27。状态：该方案曾实现，现已按用户最新要求撤回。当前采用[包含 thinking 的增量式上下文与离线概率审计](DIRECTOR_INCREMENTAL_CONTEXT_20260927.zh-CN.md)。以下仅保留当时的实现和验证记录，不能作为当前运行说明；未启动模型实验。

## 实际行为

- 当前支持 `snapshot_dedup`、`append_only` 两种 Director 上下文模式；用户后续要求删除的第三种 `delta_timeline` 已移除，包括差量快照编码器及专用提示。
- 后续 assistant 历史只保留 action channel，不再追加独立 reasoning channel；内嵌 `<think>...</think>`、`<think_only>`、预填充造成的只有闭合标签，以及未闭合 thinking 一并过滤。
- 保留原有动作、Canvas 反馈和增量状态顺序。没有切换到每轮重建的 FlowSteer 主入口模式。
- JSON 字符串参数里的字面量 `<think>` / `</think>` 属于动作数据，完整保留。
- thinking-only 响应不伪造动作，继续走原有解析拒绝和停滞保护。
- 当前轮是否启用 thinking、生成额度、图操作、SWE 工具额度及 Worker 输入均沿用现有设置。这是共享 Director 层的行为，适用于使用该层的各数据集。

## 训练边界

每轮原始 `raw_reasoning_text`、`raw_action_text`、实际 prompt/completion token IDs、behavior log probabilities 均完整保存。训练仍使用每次真实调用：

```text
input_ids = actual_prompt_ids + actual_completion_ids
loss_mask = 0 on prompt, 1 on trainable sampled completion
```

当前轮真实生成的 thinking 仍参与训练。下一轮提示里的旧动作属于输入，不重复赋予 loss。

剥离 thinking 后，上一轮完整采样序列通常不再是下一轮的精确 token 前缀。因此，动作和反馈仍增量追加，但不能再声称包含全部 thinking 的训练序列可以跨轮无损拼接。原有 `timeline_positions()` 逐 token 验证继续生效；不满足前缀条件时，`timeline` 训练自动回退到逐调用路径。没有通过重写 token 或行为概率来强行合并，代价是可能失去跨轮合并的计算节省。

## 版本与续跑

- Director context schema 升级为 `*_action_feedback_history_v2_no_thinking`。
- `director_context_policy.json` 增加 `history_thinking_visibility=offline_only_v1`。
- 已标记的旧上下文策略、以及未标记的旧采集目录，均拒绝按新策略续跑；新采集使用新输出目录。同策略标记允许正常续跑。
- 历史实验和已有训练轨迹保持原样；本次没有改写旧数据。
- 配置 `SPGFS_DIRECTOR_CONTEXT_MODE=delta_timeline` 会明确报错并列出当前支持的模式，不静默切换。旧差量模式采集目录不能作为其他模式续跑。
- 删除差量模式不会恢复被 thinking 剥离打断的 token 前缀；剩余两种模式仍执行真实前缀检查，并在必要时按逐调用方式训练。

## 验证

以下 329 项为移除第三种模式之前的 thinking 剥离验证记录；模式删除后的回归单独记录。

- 新增 34 项测试：标签边界、JSON 字符串保护、三种上下文模式、真实调用记录和 loss mask、训练回退、旧策略续跑隔离。
- 联合关系审计、Director JSON 协议、轨迹和 Qwen 模板测试，首批 88 项通过；模板测试使用本机 Qwen3.5-9B tokenizer，无需下载或运行模型。
- 补充 Canvas、训练、Director 编辑额度、统一提交、SWE 候选与执行准入回归 240 项，以及 Qwen 通道兼容测试 1 项，总计 329 项通过。
- 实际 SWE 解题率和上下文超限率仍需要新的采样实验验证。

移除第三种模式后，Director、历史 thinking、关系审计、轨迹、JSON 协议、Canvas、WebShop 场景及编辑额度回归共 238 项通过，包含本机 Qwen tokenizer 验证。静态检查通过；运行源码、配置及脚本不再引用差量模式。
