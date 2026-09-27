# Director 增量式上下文与离线概率审计

日期：2026-09-27。当前方案：用户明确选择按指导文档保留训练 thinking，继续原有 masked GRPO。Director 历史也保留 thinking，以维持此前要求的跨轮增量训练；概率审计继续隔离。先前的历史 thinking 剥离方案及尚未完成的 action-only 训练草稿均已撤回；未启动新模型实验。

## 与指导文档的对应

附件 `selfplay_graph_multiagent_flowsteer_implementation.md` 第 17 节明确要求对 Director 生成的 think/action token 计算梯度；Step 9 明确规定 action/think 的 mask=1、Canvas feedback 的 mask=0。普通图编辑使用图级优势，反事实 probe 的 relation token 使用局部关系优势。

文档没有规定“删除 thinking 后改用按优势加权的动作对数似然目标”。该方案是另一个训练目标，不能当成文档中的 masked GRPO。本次撤回对应的动作投影、训练目标切换和采集版本标记，继续保存并使用每次实际调用的原始 token 与行为概率。保留在线 thinking 是为维持现有跨轮 token 前缀一致性，并非文档明确规定的上下文序列化格式。

## 在线输入

`append_only` 按时间追加真实生成前缀、thinking、动作和 Canvas 反馈。通过 provider token IDs 解码还原实际 assistant 内容，保留原始空白、停止 token、截断 thinking 及独立 off/on 关系选择，避免重新拼装文本改变前缀。

关系审计明细由 Canvas 在构造在线快照时排除。关系端点、类型、on/off 选择和实际边仍可见；概率、logprob、token ID、模型/请求校验等 `policy` 明细只保存在原始事件、DirectorTurn 和二元策略审计记录中。反事实关系训练继续读取完整离线记录。

## 训练与启动

- `scripts/formal/environment.sh` 默认导出 `SPGFS_DIRECTOR_CONTEXT_MODE=append_only`，并允许外部显式覆盖。Director 服务和请求使用对应聊天模板；Worker/Proposer 的请求模板选择逻辑不变。
- `scripts/formal/run_experiment.sh` 默认使用 `--raw-policy-backward-mode timeline`；可用 `SPGFS_RAW_POLICY_BACKWARD_MODE` 或末尾 CLI 参数显式覆盖。
- GRPO 的原始输入、完整生成 token、行为概率、loss mask 均保留。thinking 与动作按原有规则参与训练，环境输入 mask=0。
- 每条轨迹仍通过真实 token 前缀检查；通过后使用现有 timeline 合并计算，异常或不连续轨迹回退逐调用。不会为了强行合并而重写采样 token 或概率。
- 这是恢复前缀可合并性，并非取消正确性检查。专用多 GPU Solver 数据并行实现仍按其原有配置强制逐调用；本次未改动该执行器。
- 直接运行 Python/CLI、且未加载正式环境脚本时，通用默认模式仍为 `snapshot_dedup`。手工采集要使用增量式，需显式设置 `SPGFS_DIRECTOR_CONTEXT_MODE=append_only`，服务也需使用相同模板。
- `snapshot_dedup` 保留为兼容选择，历史 thinking 同样恢复，但当前快照重建不保证跨轮精确前缀。此前已删除的 `delta_timeline` 不恢复。

## 采集策略版本

`history_thinking_visibility=online_and_training_v1`，`relation_audit_visibility=offline_only_v1`。context schema 为 `*_action_feedback_history_v3_with_thinking`。旧的 thinking 剥离采集目录或缺少策略标记的目录不能混用续跑；新采集使用新目录，旧实验不改写。

## 验证范围

覆盖完整/截断/独立通道 thinking、逐轮前缀、训练 token 和 mask、关系审计隔离及离线反事实数据、旧策略续跑隔离、正式环境默认设置，以及真实 Qwen tokenizer 模板。另以小型 CPU 因果模型核对现有 timeline 与逐调用计算的概率和梯度一致性；此测试不替代真实模型训练实验。

共 332 项不同回归测试通过（两组分别 91、258 项，重复 17 项），新增的增量上下文测试文件含 18 项。静态检查和启动脚本语法检查通过。

本次按指导文档撤回 action-only 草稿后，重新验证 217 项相关测试：首轮 215 项通过，两项续跑测试因模拟目录缺少上下文策略标记而提前失败。为这两个测试补齐合法的采集标记后，与四项旧策略/无标记续跑拒绝测试一起复测，6 项全部通过。生产代码仍拒绝无标记或策略不一致的目录续跑。验证包含真实本地 Qwen tokenizer、增量概率/梯度一致性、GRPO 训练器和 self-play 运行时；Ruff、差异空白检查与启动脚本语法检查通过。
