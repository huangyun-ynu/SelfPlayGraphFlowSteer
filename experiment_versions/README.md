# 当前 WebShop 正式版本

2026-09-30 已选择 `M02-V3-engineering-20260930`（评测源码 `d662f39`），原始官方评分；唯一正式执行配置是 `configs/formal_training.toml`。见 [升级记录](../docs/WEBSHOP_ENGINEERING_FORMAL_PROMOTION_20260930.zh-CN.md)。

# 实验源码版本

> 2026-09-29 历史正式版本固定为 Git 标签 `unified-v3-20260929`。
> 七个数据集统一 V3，AIME 复核实验已撤回；包含最新 AIME、HotpotQA、HealthBench 评测记录。
> 版本范围、源码哈希和 91 项定向验证见 [版本快照](checkpoints/unified-v3-20260929/README.zh-CN.md)。

> 2026-09-29 最新正式协议：七个数据集均为 V3，包含用户确认切换的 WebShop。
> 切换前完整正式版保存于 `experiment/formal-before-all-v3-20260929`（`4678541`）。
> 下列 WebShop 50% 等成绩保留历史含义；不能直接视为当前 V3 成绩。见
> [统一 V3 记录](../docs/ALL_DATASETS_V3_FORMAL_PROMOTION_20260929.zh-CN.md)。

> 2026-09-29：main 的 NQ 已切换到 R2D2 1,702,133 段语料运行时检索。
> 切换前正式版保存为 `experiment/nq-frozen-before-corpus-20260929`（`68b9ead`）；
> 独立 NQ 实验副本继续保留。来源与验证见
> [NQ 正式同步记录](../docs/NQ_R2D2_FORMAL_PROMOTION_20260929.zh-CN.md)。

> 2026-09-29：ALFWorld / SWE 修复版已共同设为本地正式训练版本
> `alf-swe-statefix-20260929`。来源、归档和训练接入见
> [正式同步记录](../docs/ALF_SWE_FORMAL_PROMOTION_20260929.zh-CN.md)。远端发布状态以 push 核验为准。

> 2026-09-29：main 的 WebShop 推理实现已恢复到 `235e670`，对应历史 **64/128（50%）**；
> 训练数据保留修正后的 444 条。恢复前 main 的完整实验副本是
> `experiment/webshop-main-before-restore-20260929`（`ce665fb`）。
> 来源、保存位置和验证见 [恢复记录](../docs/WEBSHOP_MAIN_RESTORE_20260929.zh-CN.md)。
> 以下归档条目保留各自历史含义。

截至 2026-09-23。指标和修改效果统一见 [实验记录](../docs/EXPERIMENT_RECORDS.zh-CN.md)。

- `run_catalog.json`：69 次运行的公开指标索引，只含汇总指标、配置白名单、评分器统计和原文件 SHA-256，不含题目、答案、轨迹或凭据。
- `index.json`：8 份历史源码快照的范围、基础提交、补丁和逐文件 SHA-256。
- `patches/`：相对 `b28aedcb4d23f126bfef3770593a94d1af0a5e8a` 的源码补丁。它们是代码版本，不是原始实验产物。

恢复时必须使用一个不存在的新目录。脚本不修改主工作区、不启动模型或评测：

```bash
python3 scripts/formal/restore_experiment_version.py --list
python3 scripts/formal/restore_experiment_version.py webshop-budget-off-128 \
  state/restored-versions/webshop-budget-off-128
```

恢复脚本先导出基础提交、应用补丁，再校验快照中每个文件的 SHA-256。
恢复目录有独立的空 Git 仓库；它是源码检视副本，不会切换当前分支。
部署仍需自行配置未上传的数据、依赖、模型、环境服务、环境变量及密钥。
当年的启动参数、模型供应商及 tokenizer 变化也可能影响结果；相同代码不保证逐题重现。

| 源码 ID | 对应阶段 | 保存范围 |
| --- | --- | --- |
| `webshop-laser-128` | LASER 128 题 | 从 budget-off 完整源码快照逆向应用当时保存的预算修改，覆盖原 LASER 快照文件，并核对原记录哈希 |
| `webshop-budget-off-128` | 关闭请求预算门槛 128 题 | 全部 Python 源码 + 当时保存的配置/脚本 |
| `webshop-identity-fix-128` | 商品身份修复 128 题 | 全部 Python 源码 + 当时保存的配置/脚本 |
| `webshop-detail-unlimited-128` | 取消单段详情 1,400 字符上限 128 题 | 全部 Python 源码 + 当时保存的配置/脚本/测试 |
| `webshop-history-adaptation` | history 适配供审阅阶段 | 8 个文件的局部覆盖；不是完整运行快照 |
| `webshop-history-128` | ReAct history 128 题 | 全部 Python 源码 + 当时保存的配置/脚本/测试/说明 |
| `webshop-pre-native` | native 实现前 | 当时源码、非私有配置和已保存测试；没有新增推理成绩 |
| `webshop-native-10` | 首轮 native 10 题 | 首轮实际推理源码，**不包含**后续模拟测试发现的四项修复 |

除逐文件哈希列出的内容之外，其余文件来自基础提交，不能声称全部依赖、配置或测试都已经恢复到实验时刻。
`complete_src_with_selected_support_files` 指全量 Python 源码与选择性配套文件；`partial_overlay` 只保证列出的局部文件。

当前主目录还包含 native 模拟测试后的四项修复和新增场景测试；修复后的真实 10 题复测没有启动，不能套用修复前成绩。

早期版本保留现有 Git 提交、配置档案和逐次运行记录：

- `496ab8e`：正式评测、检索与 ALFWorld 修复的代码检查点。
- `e46b384`：完整 WebShop 页面观察代码检查点。
- `3049203`：正式训练池与评测更新。
- `fb73ce4`：NQ 固定证据流程。
- `b28aedc`：NQ128 冻结清单及外部资源配置。
- [正式 WebShop 基线配置](../configs/webshop_official_baseline.json)：原始 62/128，48.4375%。
- [历史 retain_page_text 配置](../configs/history/webshop_retain_page_text_20260918.json)：原始 58/128，45.3125%。

这些检查点不是每次实验的精确源码证明。早期 ALFWorld、NQ、HotpotQA、WebShop 环境反馈等运行没有完整的独立源码快照，无法从现有文件保证逐版本原样恢复；不把当前实现或推测的补丁冒充当时版本。
