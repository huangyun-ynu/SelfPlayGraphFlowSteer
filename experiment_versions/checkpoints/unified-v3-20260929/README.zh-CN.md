# 统一 V3 正式版本：unified-v3-20260929

按用户要求保存当前正式版本，并使用 Git 标签 `unified-v3-20260929` 固定其提交。
标签随 main 一起发布。保存前的 main 为 `c52c4c6`，本次只增加版本记录，
正式源码、训练配置和推理行为保持原样。

AIME、NQ、HotpotQA、HealthBench Professional、WebShop、ALFWorld、SWE-bench
七个数据集均通过 `configs/formal_training.toml` 的数据集覆盖启用 Director V3、
`unified_task_result_v1` 和 `unified_submission_v1`。Director 使用
`finish(target)` 提交；该步骤不再调用 Worker，不使用 `set_output`。

保留 NQ 的 R2D2 1,702,133 段语料检索、WebShop 修正并隔离测试集后的 444 条训练数据。
AIME 提交前复核实验已经撤回；当前源码、配置、脚本和测试与 `29e4f2f` 一致。
WebShop 的历史 50% 成绩属于旧协议版本，不能视为当前 V3 的重测成绩。

本次保存前核对了七种生效协议、WebShop 训练数据条数和源码哈希；
现有 V3 协议定向测试 **91 项全部通过**。完整树对象、文件哈希和验证命令见
[manifest.json](manifest.json)。

当前已保存的 V3 评测记录：

- [AIME 30 题](../../reports/aime30-v3-main-20260929/README.zh-CN.md)
- [HotpotQA 128 题](../../reports/hotpotqa-v3-128-main-20260929/README.zh-CN.md)
- [HealthBench 128 题](../../reports/healthbench-v3-gpt-low-128-main-20260929/README.zh-CN.md)

在新目录导出此正式代码版本：

```bash
git clone --branch unified-v3-20260929 --single-branch https://github.com/huangyun-ynu/SelfPlayGraphFlowSteer.git SelfPlayGraphFlowSteer-unified-v3
```

模型、外部语料索引、环境凭据和私有原始轨迹保持各自现有本地保存方式。
此前未跟踪的独立实验文件继续保留在工作区。
