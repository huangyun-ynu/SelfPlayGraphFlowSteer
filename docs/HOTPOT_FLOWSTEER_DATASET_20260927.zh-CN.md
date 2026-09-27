# HotpotQA 默认测试集切换与 FlowSteer 训练数据审计

2026-09-27 按用户要求，将默认 HotpotQA 测试集切换到 FlowSteer 公开 eval/hotpotqa.jsonl 的全部128题。两个工作目录 SelfPlayGraphFlowSteer 和 SelfPlayGraphFlowSteer-output-contract-v1 均已同步。

## 默认入口

- data/formal/eval/hotpotqa_official_test.jsonl：原有启动脚本继续使用此路径，现在内容为FlowSteer公开128题。
- data/formal/eval/hotpotqa_flowsteer_public_128.jsonl：固定规范副本，prepare_static_eval.py 从此读取，不再重新抽旧题。
- data/formal/sources/flowsteer/hotpotqa_eval_128.raw.jsonl：上游原始格式。
- 旧测试集备份已按用户要求删除。
- static_eval_split_manifest.json：Hotpot单项已更新SHA256、来源和选择策略；其他数据集未改动。顶层selection_seed为原有数据集的抽样种子，Hotpot使用其单项selection=flowsteer_public_eval_128。

新默认文件逐字节等于已完成重跑的inputs/hotpotqa.jsonl，SHA256=84db56807e8e8cd98224ba53b124b289f783faf463a55f5a7e6785b2127cbcc2。测试集切换不修改历史运行中的冻结inputs/records。训练池已按下文更新。此前基于旧128题的历史基线和输入一致性审计结论，仅适用于当时冻结数据，不能外推到新默认题集。

已核验两个目录的128个唯一题目、实际加载器prompt/reference、数据重建函数和清单哈希；test_prepare_static_eval.py的4项测试通过。

## FlowSteer公开训练集难度

实际读取train/train_12k.jsonl，共12000条；HotpotQA为2000条、2000个唯一原始ID。每条都带meta.id、meta.type和meta.level。文件没有按难度拆分，标签可用于自行分层。

|难度meta.level|数量|占比|bridge|comparison|
|---|---:|---:|---:|---:|
|easy|383|19.15%|302|81|
|medium|1277|63.85%|1039|238|
|hard|340|17.00%|277|63|
|总计|2000|100%|1618|382|

上游当前train_interactive.py:1416按source建立索引，每个来源随机抽samples_per_source（默认6）条，不依据meta.level分层，也不是easy→medium→hard课程。仓库另有cross_problem_sampler.py按领域/难度的辅助采样器，但当前主入口未引用它；该辅助类读取顶层difficulty/_difficulty，不读取本文件的meta.level，因此不能把辅助代码的存在描述为主训练已使用Hotpot难度标签。

## 实际重叠核验

公开Hotpot训练2000题与公开测试128题有12个相同ID，且12道problem正文也完全一致（占测试集9.375%）。具体ID保存在training-inspection.json。若采用公开训练文件，至少需要排除这些ID，Hotpot剩1988题；这不证明任何具体发布模型实际训练过这些重叠题。

新训练集已从上游 Hotpot 2000 条按难度×题型联合比例，用最大余数法分配512个名额，先排除12条测试重叠题再在各层内确定性抽样。easy=(bridge77,comparison21)，medium=(266,61)，hard=(71,16)。训练与测试的ID及完整prompt交集均为0。

两个工作目录的 data/formal/train/hotpotqa.jsonl 与实际入口 state/formal-data/validated_task_pool.jsonl 均已更新。其他六个数据集保持原内容。新题使用 Qwen3.5-9B 重新计算 ADS embedding、目标token NLL、128维PCA和16类聚类；没有沿用旧题特征。上游 easy/medium/hard 标签保存在 metadata.flowsteer_meta.level，独立于模型NLL难度分数。

原始抽样题、映射题和抽样清单在 data/formal/sources/flowsteer/hotpotqa_train_512.*；prepare_raw_pools.py 从新映射题读取，避免重建旧池。旧活动训练数据已被替换，旧测试集备份已删除；历史实验记录保留。此次只准备训练数据，未启动训练。


来源：
- https://huggingface.co/datasets/beita6969/FlowSteer-Dataset/blob/main/train/train_12k.jsonl
- https://huggingface.co/datasets/beita6969/FlowSteer-Dataset/blob/main/eval/hotpotqa.jsonl
- https://github.com/beita6969/FlowSteer/blob/1c9f2abf55cb9b8ea2ca2e3359cdb91acb9964e9/train_interactive.py#L1416
