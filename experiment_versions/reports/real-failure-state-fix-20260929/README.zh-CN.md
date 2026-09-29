# 两项状态机修复：真实失败轨迹回放与真实题目推理

日期：2026-09-29。修复版本：ALFWorld 副本 `478b38f`，正式 SWE 项目 `10a144c`。

## 真实题目重新推理

从既有失败记录中预先固定 2 道 ALFWorld 和 1 道 SWE，使用原始题目行、seed=0 和各自原配置，从头生成新轨迹。本次三题均合法提交并获得正式结果。

| 真实失败题 | 原记录 | 新结果 | Worker tokens | 物理请求 |
| --- | --- | --- | ---: | ---: |
| ALFWorld：冷却苹果放进微波炉（Apple / Microwave-19） | 图未连通、无进展终止，未提交 | **成功**，最终 episode 31 步 | 151,814 | 38 |
| ALFWorld：加热鸡蛋放进垃圾桶（Egg / GarbageCan-10） | Worker Token 停发，未提交 | **失败**，episode 50 步耗尽，但完成合法提交 | 221,374 | 51 |
| SWE：`django__django-13786` | 图未连通、无进展终止，未提交 | **官方 resolved** | 85,151 | 11 |
| 合计 | 三题原记录均未提交 | 2 成功、1 失败，3/3 合法提交 | **458,339** | **100** |

鸡蛋题失败轨迹显示后段重复搜索冰箱和台面，并拿着苹果；直到 50 步耗尽仍未完成鸡蛋的加热与放置。环境终止后 RUN_AGENT 被正确拒绝，随后 FINISH 正常提交失败结果。这不是本次修复的两个状态机 bug。

SWE 的本地公开测试和远程正式评测均通过。新补丁 SHA-256 为 `046ebb064c2620c61c37a67e27ca6ce2bfbcf0c1c99cae63ee1f403b1222a285`，与历史记录中的候选补丁一致；本次在线过程仍是从原题与仓库重新调用模型生成、测试和提交，没有将历史补丁或回放状态注入在线推理。远程日志确认正式 harness 完成，耗时约 28.9 秒。

### 实际配置和审计

- ALFWorld 固定 DeepSeek Flash，题目并发 2，路由并发仍为 50；共 89 次请求。
- SWE 题目并发 1，沿用 GPT-student / 普通 GPT 池及其并发 10 / 5；实际模型为 `lab-gpt-5.5-2` 6 次、`gpt-5.5` 5 次，没有 Pro。
- Qwen3.5-9B 普通 Director 决策 17 次，全部开启 thinking；另有 1 次原有协议的单 token on/off 关系选择。
- GPU 按启动时空闲显存动态选择，最终使用 GPU 1；本次 Qwen 最大同时计算序列数为 2。
- 三题均使用原来的 24 轮 Director 上限与 350,000 Worker tokens 发送阈值。三题账本与回执用量一致，未知用量和遗留在途请求均为 0。
- 三次 FINISH 都是 Director 显式发出，没有新增 Worker 执行。ALFWorld 环境题目和 game 指纹一致。
- 远程 SWE 最终状态明确为 **STOPPED / STOP_CHARGING**；本次启动的 Qwen 服务已停止，专用端口已关闭。

## 用真实失败轨迹验证两个 bug

首先核对上一轮全部 256 条记录：ALFWorld 最大 18 轮，SWE 最大 13 轮；都没有走到 24 轮边界，也没有在 AWAITING_MODEL 状态删除节点。因此不能把原失败归因于本次两个 bug，也不能把上面的新成功数当作补丁带来的准确率增幅。

使用苹果题和 `django-13786` 的真实失败记录，在隔离的测试 Canvas 中重放原始前 6 个 Director 动作及两份实际 Worker 产物：

- ALFWorld 的 **44 个历史环境动作**在原 TextWorld 游戏实际执行，每步观测、步数、终止状态与历史记录一致，最终环境结果和 game 指纹也一致。
- SWE 使用记录中的实际产物、补丁引用和测试证据；补丁文件 SHA-256 已核对。这部分是提交协议回放，不计为新的 SWE 测评成绩；新的正式测评结果来自上面的在线运行。
- 原轨迹中对未连通图的 FINISH，在新旧代码中都仍被拒绝，未放宽提交条件。

随后明确加入两个**诊断测试条件**，比较修复前源码与修复后源码：

| 条件（两个数据集各测一次） | 修复前 | 修复后 |
| --- | --- | --- |
| 在真实 ADD / SET_PROMPT 后删除等待选模型的节点，再 ADD 新节点 | 删除返回成功，但状态仍为 awaiting_model，新增被拒绝 | pending 清空，恢复 building，可以新增 |
| 在真实失败记录的产物上删除孤立 subtask，再于最后一个允许轮次 FINISH | 最后一轮误拒绝 | 正常提交，额外 Worker 调用 0 |

第二项诊断人为设置恰好剩两轮，并替换后续动作；第一项也人为插入了待配置节点 DELETE。两者均是基于真实输入的边界对照测试，**不是原始轨迹自然触发了这些条件**。回放没有调用付费模型，也没有实现生产流程中的历史测评点恢复。

## 证据与复核

完整运行目录：

`/mnt/ssd/test/codex-students/student02/SelfPlayGraphFlowSteer-hotpot-answer-contract/state/real-failure-state-fix-20260929-100808`

- `alfworld/results/records.jsonl`、`swe/results/records.jsonl`：三条新推理的完整记录。
- `offline/{alfworld,swe}/{before,after}/result.json`：四组新旧代码回放结果。
- `failure-inventory.json`：对原始 256 条记录的检查及源文件哈希。
- `audit.json`：输入、冻结源码、回执、物理请求账本、模型与资源清理检查。
- `cloud-final.json`、`cleanup.json`：不计费关机和本地服务停止证据。
- 各数据集的 `evaluated_source`、`manifest.json`、`config.toml`：实际运行源码及配置快照。

准备阶段修正了启动脚本的临时路径长度、遗漏包资源、配置相对路径三个问题，对应目录后缀为 `100150`、`100245`、`100644`；这些尝试均在付费模型推理前停止。只有 `100808` 完成了本次三题在线运行。

从 ALFWorld 副本根目录执行只读复核命令：

```bash
PYTHONPATH=src:. /mnt/ssd/test/codex-students/student02/.venvs/spgfs-pats-gpu/bin/python \
  scripts/audit_real_failure_state_fixes.py state/real-failure-state-fix-20260929-100808
```

所有原始记录的文件哈希保持不变。旧 ALFWorld 102/128、SWE 65/128 保持原成绩。本次没有扩大到 128，也没有修改候选保护范围或历史测评点恢复逻辑。
