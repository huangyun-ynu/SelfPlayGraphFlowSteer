# ALFWorld / SWE：轮数边界与待配置节点删除修复

日期：2026-09-29。按本次要求修复审查中的前两个 bug，并同步到正式 SWE 项目的活动源码。

## SWE 是否受影响

**两个 bug 都可能在 SWE 触发，且已在两个项目各自的 SWE 离线场景中复现。**

- 正式项目 `configs/formal_training.toml` 为 SWE 启用 `unified_task_result_v1`，使用默认的 `rounds_v1` 和 `max_rounds = 24`。最后一轮提前清空合法动作的问题适用；使用 `edits_v1` 的配置没有这个轮数边界，但共享代码仍需修复。
- SWE 同样使用 ADD_AGENT → SET_PROMPT → SET_MODEL 的配置流程。删除等待选模型的 pending 节点时，旧代码也会留下已删除节点的 ID 和 `awaiting_model` 状态。
- 这些复现证明可触发，并不表示此前 SWE 65/128 的真实运行已触发。此前 ALFWorld 102/128 的轨迹也未直接触发这两项 bug，不据此调整旧成绩。

## 修复内容

1. **最后一轮动作**：统一提交协议的权威动作校验使用本轮计数前的合法状态快照。动作仍正常消耗一轮，返回的状态快照使用更新后的轮数，下一轮入口继续拒绝超限。
   - FINISH 可在最后一个合法轮次提交，不增加 Worker 调用。
   - RUN_AGENT 的执行检查识别本轮已经通过轮数准入，同时保留 Token、环境、SWE 执行条件及终态检查。
   - 新建节点的配置轮数预留不再因重复计算本轮而误拒；轮数耗尽时，等待提示词、模型或关系选择的状态也不再展示可执行动作。
2. **删除待配置节点**：删除 pending 节点后，无论此前等待提示词还是模型，都清空 `pending_agent_id` 并回到 `BUILDING`。等待配置时 DELETE_AGENT 的目标列表与状态门禁保持一致，只展示 pending 节点。

两个项目使用相同的生产代码补丁（Git stable patch ID：`d465294b047530735c3f8a9badc811a20d0b724a`），通过逐段应用同步，没有整文件覆盖两边已有的差异。

## 离线验证

新增 `tests/test_unified_round_and_pending_state.py` 的 22 项测试同时覆盖 ALFWorld 与 SWE。ALFWorld 使用本地模拟环境的真实生命周期接口；SWE 使用本地产物与已测试状态夹具；Director 使用 MockBackend。

| 检查 | ALFWorld 副本 | 正式 SWE 项目 |
| --- | --- | --- |
| 新增边界测试，修复前 | 16 失败、6 通过 | 16 失败、6 通过 |
| 新增边界测试，修复后 | 22 通过 | 22 通过 |
| 包含新增测试的回归组 | 338 通过、1 项缺失夹具失败 | 314 通过、1 项缺失夹具失败 |
| SWE 正式配置集成检查 | 不适用 | 14 通过 |

回归组覆盖统一提交、环境提交、Worker Token 限额、Director 编辑预算、Canvas/Director、提交回执与评分边界、SWE 执行准入、候选完整性及共享 Token 账本。副本还验证了 ALFWorld 在 Token 停发后，利用最后两轮删除孤立 planner 并 FINISH：候选、输入绑定、usage 账本保持一致，额外 Worker 调用为 0。

两边缺失夹具的是同一个既有测试：
`test_recorded_aime16_output_switch_trace_stops_after_first_unproductive_switch`。
它依赖的 `state/audits/aime30-finish-v1-deepseek-qwen-thinking-20260926/run/trajectories/064ef67eca667e6894335446.json` 不存在，读取时抛出 FileNotFoundError。未改写、跳过该测试，也未生成替代历史轨迹。测试输出（仅去除行尾空格，原始日志保留在 state/round-pending-fix-20260929）保存于本目录的 `before-tests.txt`、`after-tests.txt`，正式项目还保存 `formal-integration-tests.txt`。

## 边界与历史记录

- 本次外部模型/API 调用 **0 次**，没有重跑 128 题，没有启动 GPU 或远程 SWE 服务器。
- 本次未调整审查中的第三项候选保护范围缺口，也未修改历史测评点/图/产物恢复逻辑。
- 正式 SWE 存档 `state/formal-training-promotions/swe-65of128-20260928/{evaluated_source,release_source}` 中两个相关源文件的 SHA-256 均通过修复前后核对；历史运行与成绩保持原记录。
- 新代码已通过上述离线测试，尚无本次补丁的新真实题目成绩。运行时来源哈希会如实反映新源码，不把旧成绩标为新代码的测评结果。

详细检查命令、基线提交、源码哈希和同步证据见 `validation.json`。
