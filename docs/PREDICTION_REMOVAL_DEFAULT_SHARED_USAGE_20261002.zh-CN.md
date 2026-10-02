# 默认共享实际 usage 与旧预测代码删除记录

2026-10-02 已直接更新现行训练与推理源码。正式 AIME 版本为 `aime-no-code-comments-shared-usage-default-20261002`，完整引擎与禁止代码注释的提示词继续使用。

## 当前范围与预算

| 数据集 | 默认机制 | 阈值 |
|---|---|---:|
| AIME（正式完整引擎及独立 Qwen 版）、NQ、HotpotQA、HealthBench | 整题共享实际 usage 发送阈值 | 240,000 |
| MATH-Hard、TriviaQA、MuSiQue | 整题共享实际 usage 发送阈值 | 240,000 |
| SWE、ALFWorld、MBPP+ | 整题共享实际 usage 发送阈值 | 350,000 |
| WebShop | 保留原有实际累计用量硬上限；删除旧预测限制 | 350,000 |

覆盖 `selfplay_graph_flowsteer`、`formal_aime`、`aime_qwen_eval`、`math_hard_ood`、`triviaqa_ood`、`musique_ood`、`mbppplus_ood`，包括 MBPP+ 实际加载的 `_shared/canvas.py`。主求解、修订、重试、格式修复、最终结果生成和训练反事实使用同一预算契约。

`CanvasConfig()`、省略策略表、空策略表和只提供策略名的配置都会得到完整默认策略。缺省数据集额度与显式额度合并；别名不能建立重复账户；显式发送阈值必须与数据集额度一致。独立 benchmark 默认选择各自的现行配置。当前正式评测基础配置也已同步 IID 策略与完整 AIME 版本。

共享账本只累计后端实际返回的 Worker 输入与输出 usage，Director/Judge 保持独立统计。同题串行发送，每次物理请求和重试独立记账，节点删除、重建、缓存与切换模型不重置账户。已确认用量小于阈值时允许发送，最后一个请求可以跨过阈值并保留完整结果。未知用量不会当作零；累计两个未结清请求后停止发送。独立 Canvas 即使 Director 使用相同随机种子，也有独立题目身份。

## 删除与保留

已删除旧时间/token 预测开关、路由估算器及调用、固定 8,192 token 图增长预留、按节点/调用分账、收尾与最终报告 token 预留、UTF-8 字节报价准入、`RequestTokenCredit` 和请求额度上下文。WebShop 的相关实现也已删除。

实际 API/工具/轨迹超时、环境动作额度、精确 tokenizer 上下文检查和每次请求配置的输出上限继续执行。耗时与 usage 的观测记录继续保留。旧节点分账字段只用于恢复状态时清理；SWE 历史诊断归因仅解释已发生的失败，不参与发送或分配额度。

WebShop 前后探针的六处差异全部是删除 `budget_partition` 诊断字段；请求正文、环境调用和购买流程一致。原有历史行为指纹保持不动，新指纹只覆盖经此对照确认的差异。

## 存档与验证

- 修改前源码、配置、脚本、测试和文档已存入 `experiment_versions/checkpoints/before-prediction-removal-20261002T093652Z/snapshot/`；随后在改写前补存四份 OOD manifest。另补存版本索引，共 1,271 个文件，全部 SHA-256 校验一致。
- 四份当前 OOD manifest 已记录新源码指纹和前一 manifest 的存档位置；原始来源指纹及历史冻结运行保持可追溯。MBPP+ 的源码快照检查通过，未关闭其校验。
- 检查七个现行包的 661 个 Python 文件：旧预测执行 API 不存在，AST 可解析，未定义名字检查通过。
- 检查 40 份非历史根目录配置：支持的数据集有效策略均为共享实际 usage；WebShop 保留硬上限。历史日期/H200 配置和冻结运行不覆盖；当前源码没有可由它们重新启用的旧预测实现。
- 合并回归结果按同一用例最后一次结果计算，正式完整 AIME 的独立验证单列：**2,109 项通过**。新增默认机制专项 **65/65** 通过；正式完整 AIME 预算专项 **20/20** 通过。
- 尚有六项既有测试问题：五项在修改前的源码快照中同样失败（一个 HealthBench 提示测试、两个 HealthBench 上下文测试、两个应用关闭测试）；另一项缺少历史 AIME 轨迹文件。它们未计入通过项，详情保留在 XML 和日志中。
- 本次使用离线 provider/工具替身验证实际物理请求边界，包括跨阈值、未知 usage、重试、缓存、节点重建、旧状态清理、精确上下文和真实历史 MBPP+ 检查点恢复。未调用真实模型或启动 GPU 作业；这些结果不代表新的准确率评测。

完整证据位于 `state/prediction-removal-20261002/`：`verification.json`、`effective-configs.json`、`defaults.json`、`module-paths.json`、`source-sha256.json`、各回归 XML/日志，以及 `webshop-probe-diff.json`。修改前后差异见 `changes-from-archive.json` 和 `implementation.diff`。
