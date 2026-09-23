# SkillFlow WebShop 第一版适配

本版本是显式选择的实验配置 `configs/webshop_skillflow_native_eval.toml`，不会覆盖正式 48.44% 基线。参考 SkillFlow 固定版本 `74be52bb6bd9f0e9e68dacb72636b75649197983`，保留用户提供的 SelfPlayGraph 多 Agent 核心文档约束。

## 迁入范围

Worker 使用 SkillFlow 的两份无 skill 原生提示模板、完整 Observation/Action 历史、当前页面文本与合法动作，直接输出 `search[...]` / `click[...]`。模板已逐字核对，历史不再截断为 1,400 字符或 6 个商品。

新 Worker 执行路径不经过旧的购物清单、`product_inspections`、语义停滞阻断和 `purchase_evidence` 购买格式检查。传给模型的动作不包含 `target_id` 或 `state_version`；内部仍用这些字段精确定位当前页面的操作。结构化数据继续用于传输与审计，模型的购物观察采用原始页面文本。

默认关闭 SkillFlow 的可选辅助反馈、可见状态摘要、动作重排与 skill。保留当前模型路由与思考配置：Worker 使用 DeepSeek；Qwen Director 的普通图动作开启 thinking，关系 off/on 选择保持原有配置。

## 为满足项目核心必须保留的差异

| 项目 | 本项目第一版 |
|---|---|
| Director | 继续由 Director 分配通用 Agent 的职责、选择模型、层次、关系及输出节点；不预设角色或拓扑 |
| Agent 会话 | 各 Agent 有独立 WebShop 会话，执行同一道公开任务；其后续修订继续自己的页面和历史 |
| 通信 | 只沿合法图边发送结构化 RelayPacket；不发送完整页面历史或推理链 |
| Worker 汇报 | 原生购物动作结束后，独立生成结构化结果包；职责不需操作环境时可直接汇报 |
| 购买 | `click[buy now]` 暂存可修订候选；`SET_OUTPUT` 只选输出；合法 `FINISH` 才提交选中 Agent 的最新候选 |
| 候选修订 | 真正重新执行节点时取消其旧候选并恢复同一页面；必须再次点击购买才能产生新候选 |
| Canvas 增量 | 保留现有 dirty closure、增量执行与缓存；缓存键增加自身环境状态和私有历史指纹 |
| 双向通信 | 首轮独立执行，次轮只读冻结的首轮结果包；不读取对方次轮更新 |
| 预算 | 整图累计初始 12 次、修订 4 次、合计 16 次动作；无效动作也计次；新增 Agent 不重置额度 |
| token 检查 | 延续已关闭的预测性准入和普通请求额度检查；不改变真实总量/时间/上下文上限及反事实专用预算 |
| 解析 | 支持原始动作和 `<action>`；过滤 `<think>` 后再回退解析，避免误执行思考中的动作例子 |
| 奖励 | 只提交最终选中候选后获取官方奖励；不按候选隐藏奖励挑选输出 |

这不是把整个系统替换成 SkillFlow 单 Agent 循环。迁入的是 WebShop Worker 交互层；Director、Canvas、分层图、双向同步以及训练/反事实框架继续由本项目管理。

## 验证与复现

初始相关回归共 429 项通过，其中新模式专项 18 项。旧测试的原始动作字段断言已更新；Director timeline 测试在包含 transformers 的完整虚拟环境中通过。结果记录保存在 `state/experiments/webshop-skillflow-native-v1-20260923-221018/offline_results.json`。

第一轮真实测试固定使用原 128 题 JSONL 的前 10 行（代码修改前已保存文件与哈希），seed=0，轨迹并行参数 24、实际最多 10 条；DeepSeek Worker、Qwen3.5-9B Director、skill off。独立 WebShop sidecar 端口 18021，原服务端口 18020 保留。

```bash
SPGFS_WEBSHOP_EVAL_DATASET=state/experiments/webshop-skillflow-native-v1-20260923-221018/webshop_first10.jsonl \
  bash scripts/formal/run_webshop_skillflow_native_eval.sh
```

测试结束后补充结果及 Director/Canvas 场景模拟记录。10 题用于检查真实链路和发现适配问题，不足以判定相较 128 题基线准确率提升。

## 第一轮真实测试（修复前快照）

原始结果不覆盖、不择优重跑替换。对照为上一版 `webshop-skillflow-history-c24-20260923-213558` 中相同 10 题，并非重新运行正式 48.44% 基线。

| 指标 | 上一版相同 10 题 | native 第一轮 |
|---|---:|---:|
| 严格成功题数 | 4.00 | 1.00 |
| 实际购买题数 | 10.00 | 5.00 |
| 平均分 / 100 | 70.83 | 37.17 |
| 平均 Worker token | 41,589.90 | 44,936.80 |
| 平均 Director token | 18,304.30 | 84,529.60 |
| 平均总 token | 59,894.20 | 129,466.40 |
| 平均 Director 调用次数 | 4.80 | 12.00 |

总 token 的口径为 Worker 记录的输入+输出 token，加 Director 真实 prompt/completion token IDs 长度；包含关系二选一调用。小样本、历史对照、远端模型采样都限制因果结论。

5 道未购买：goal-00055、00371、00069、00397、00453。00069 的 Agent 只用 3 次动作就汇报候选列表；Director 随后结束。00397 首个 Agent 也在 3 次动作后汇报，另一个 Agent 消耗了余下初始动作额度，仍未暂存购买。00371 用尽 12 次初始动作；00055、00453 的多 Agent 协作用尽全图 16 次动作，最终没有候选可提交。不能把“完成分析/汇报”当成“完成环境购买”。

首轮共有 5 题最终保留多个 Agent。新模式保留了多 Agent 与修订能力，同时使 Director 调用及 token 明显增加；目前没有准确率收益证据。没有更换题目、提升动作额度或加 skill 来掩盖结果。

## Director/Canvas 场景检查与修复

使用脚本化 Director、可控 Worker 和独立模拟 WebShop 会话，经过真实 GraphDirector → Canvas → Runtime → NativeLifecycle 调用链。三种 Director 上下文模式均覆盖：snapshot_dedup、append_only、delta_timeline；后两种使用本地真实 Qwen tokenizer 验证逐轮 token 前缀保持。

覆盖节点增删、职责修订、层次/关系变更、关系 off/on、旧双向连通分量和下游失效、无关分支缓存、独立会话、删除重建身份、切换输出、无候选结束、重复 FINISH、提交失败、后端中断后恢复、报告格式失败、过期 Canvas 版本、增量快照重建及完整图反事实重放。

新增检查发现并修复四类问题：

1. WebShop 专用判断缺少数据集边界，可能影响同一配置内其他数据集的提交能力。现在仅对 WebShop 生效。
2. 未闭合 `<think>` 的文本可能被当成动作。现在过滤到文本结尾，不执行其中的动作例子。
3. 一个 Agent 会话清理异常会中断其余会话清理。现在逐一尝试所有会话，最后上报首个错误。
4. 最终提交修改了共享 Artifact，导致早期 Canvas 轨迹也显示提交后的 reward。现在采用独立的提交结果副本，旧轨迹保持提交前状态；最终报告与完整图评估显式记录新状态。

第 4 项是轨迹时序问题：首轮购买请求实际发生在 FINISH，早期保存的 Canvas 控制快照和 Director 输入不含该终局 reward，但部分早期 execution.artifacts 在最终序列化时被回写。本轮原始日志保留，以 sidecar commit 请求与当时的控制快照判定操作时序。

首轮代码快照位于实验目录的 `inference_source/`；修复差异是 `post_simulation_fixes.patch`。修复后的同组 10 题复测另存，不用离线测试结果代替真实推理结果。

## 用户停止时的状态（2026-09-23 22:58）

按用户要求停止后续工作。首轮 10 题已完成（1/10，平均 37.17）；四类问题修复后的专项验证共 46 项通过（45 项合并运行，加 1 项完整图反事实集成测试）。扩大回归被中断，不能标记为全部通过；修复后同组 10 题复测尚未启动。Qwen 与显存占位按此前要求保留，未改动正式基线。详见实验目录 `user_stop.json`。
