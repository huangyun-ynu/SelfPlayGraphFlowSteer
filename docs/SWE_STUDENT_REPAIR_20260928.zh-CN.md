# SWE：GPT-student 响应解析与公开测试环境修复

## 本次范围

沿用已有 `unified_task_result_v1`：Worker 在执行前获得 `task_result` 责任，Director 用 `FINISH(target)` 提交已完成的候选，提交不重新调用 Worker。本次修复响应解析与环境准备，Director 的追加历史方式保持原样。

## 响应解析

- Responses 网关保存 message 边界、response ID 和原始文本审计，只解析 assistant 的 output_text，排除 reasoning / analysis。
- 完整解析有上限的 JSON 流。对于串联动作，只执行第一条单动作请求；其余动作、重复动作、过早的最终答案不回放进对话。下一轮接收工具实际返回的观察。
- 拒绝夹杂说明文本、伪造 observation、重复键、截断 JSON、不明确后缀、单对象多动作等情况。拒绝时不执行任何动作，在同一工作区请求格式修复。
- 每次 Worker 执行最多修复 2 次，每题最多 6 次，继续使用原 token / 工具 / 时间额度。耗尽后显式记录 `worker_action_protocol_exhausted`。
- 仅 `responses_text` profile 使用新解析器；其它网关保持既有协议。

旧 33 道落盘题目中，304 条原先未解析出动作的响应离线回放结果：292 条恢复首个动作，12 条进入格式修复路径（10 条 JSON 流损坏、2 条后缀不明确）。这是协议恢复率，不是题目正确率。

## 环境

以 48 个安装槽准备当前 128 题覆盖的 10 个仓库、48 个版本组合，48/48 均通过非空公开 smoke。Python 和依赖按版本隔离；同一环境的源码安装与测试继续由锁串行保护。

本机没有 Matplotlib 所需的 FreeType / Qhull 开发文件，原配置仅要求使用系统库，导致编译失败。配方现在显式声明这两项原生依赖，通过 micromamba 安装到项目环境的 `native/`，测试进程设置相应头文件、链接和运行库路径。四个 Matplotlib 版本现均通过 smoke。

启动真实 benchmark 时批量检查全部所选版本；其它入口在绑定 SWE 任务时检查。缺失或配方过期时，在 Worker / Director 推理前报错并给出准备脚本位置。

每个版本的 smoke 使用一个公开 base commit；未运行全部 128 个 commit 的完整测试套件。数据库、网络、TeX、图像基准等可选测试仍取决于额外设施。公开测试不替代远程官方测评。

## 验证

- 166 项代码回归通过：解析、有限重试、共享额度、真实临时仓库的读改测、环境预检、统一提交和候选恢复。
- 真实 GPT-student 在 `pytest-dev__pytest-10051` 上完成搜索、读取、修改和公开测试。发生的串联响应被逐次处理；一次含不明确后缀的响应经格式修复继续执行。
- 实际导出非空补丁 439 字节；相关公开测试执行 15 项，14 通过、1 失败。模型错误地改变了共享日志列表的行为，因此该补丁仍有逻辑错误，未进行官方评分。
- 首次验证脚本设置 120k Worker 额度、3 轮 Director 上限，触发真实请求预算保护后完成收尾，但 3 轮上限阻止了 FINISH。将验证脚本轮数恢复为 24，使用已记录的真实响应回放、重新执行实际工具后，统一提交成功且 FINISH 的 `executed_agents=[]`；回放没有新增模型 API 请求。原始失败记录完整保留。

## 运行入口与证据

新入口：`state/swe-student-repair-20260928/run.sh`。
新配置：`configs/swe_student_repaired_20260928.local.toml`。

入口启动时按当前显存余量选卡，排除正在留给 Hotpot 的 GPU 1，并写入本次配置。采用统一提交协议 / Director v3；轨迹并发 15，端点池 GPT-student 10 + 非 eco GPT 5，远程测评并发 4，队列规则沿用原设置。退出清理继续检查云服务器 STOP_CHARGING 关机状态。尚未启动新的 128 题推理。

证据目录：`state/swe-student-repair-20260928/`。

| 文件 | 内容 |
| --- | --- |
| `protocol-replay.json` | 304 条旧响应的协议重放结果 |
| `environments.json` | 全部 48 个环境的就绪状态与 smoke 路径 |
| `prepare-native.jsonl` | 补齐原生库后的最终准备结果，退出码 0 |
| `regression.log` | 166 项回归测试结果 |
| `live/report.json`、`live/gateway.jsonl` | 真实模型响应、工具轨迹与首次验证记录 |
| `live-replay/report.json`、`live-replay/submissions/` | 响应回放和成功的统一提交收据 |

本次未启动 GPU 作业或远程测评服务器。

## 后续：修正该题补丁（2026-09-28）

用户要求继续修复后，在独立 checkout 中修正了 `pytest-dev__pytest-10051` 的失败补丁：

- `LogCaptureHandler.reset()` 仍创建新列表，保证 setup / call / teardown 各阶段相互独立。
- 新增 handler 的 `clear()`，原地清空当前列表并清空文本流；`caplog.clear()` 改用该方法，保持 `get_records()` 与当前记录同步。
- 新增回归测试覆盖重复清空、清空后继续写日志、保留 setup 阶段记录。该回归在原始 base commit 上实际失败，修复后通过。

扩大到日志模块全部测试时发现另一项环境问题：嵌套 pytest 项目的临时路径位于测试环境的 `pytest.ini` 下，导致 rootdir 与节点展示路径错误。未打补丁的 base commit 也能复现该失败。公开测试运行器现使用调用进程临时根下的独立临时目录，并将 pytest basetemp 指向该目录；本机新运行入口设置 `TMPDIR` 为项目外的 student02 `.tmp`。

最终验证：日志模块 **60/60 项通过**，包含原先失败项和新增回归；项目侧公开测试运行器回归 **26/26 项通过**。

产物在 `state/swe-student-repair-20260928/patch-fix/`：

- `solution.patch`：修正后的源码补丁。
- `regression.patch`：新增回归测试。
- `review.patch`：源码与回归测试的完整 diff。
- `report.json`：原补丁、新补丁哈希和验证摘要。
- `before.json`、`baseline-nested-root.json`、`after-logging-isolated.json`：修复前、环境基线与最终测试证据。

这是 Codex 辅助调试后得到的补丁；原始模型轨迹与失败补丁保留，未改写 GPT-student 自主测评成绩，也未进行官方远程评分。

## 同批 33 题重跑

已完成 `state/swe-33-repaired-20260928-164024/` 这轮重跑，输入与原先 33 个落盘样本完全一致，未注入人工诊断补丁。

- 通过数由 12/33 增至 16/33；非空补丁送评由 15 增至 26，空补丁送评由 9 降至 0。
- 本次还有 10 个官方未解决补丁、3 个 Director 上下文耗尽、1 个无进展终止、3 个 Worker 请求超时。
- 新发现 Sphinx 3.3 的特定测试缺少 `_testcapi`，以及一次 SymPy 目录测试超时；这些未被此前环境 smoke 覆盖，原始失败记录均保留。
- 本次 Director 已停止，GPU 4 已释放；云服务器已核验 `STOPPED / STOP_CHARGING`。

详细结果及逐题配对见该目录的 `REPORT.zh-CN.md`、`paired-comparison.csv` 和 `detailed-comparison.json`。
