# SWE-bench 对齐 SkillFlow 数据划分

测试集已替换为 SkillFlow test_iid_v3.json 中的128个 SWE-bench 实例，ID、顺序和 question 题面逐条一致。公开题面有部分长问题截断，沿用上游发布内容。未将上游 code_files、参考patch或私有测试加入模型输入；仍使用项目现有 SWE 真实执行验证流程。此数据是 SkillFlow 的 IID 验证划分，不能宣称覆盖其完整最终评测。

训练集保持512条，保留 SkillFlow 的全部372个独立训练实例，以其500条训练记录（含重复）的官方难度分布为比例基础，最大余数法确定各档名额：

|官方难度|SkillFlow记录|本地记录|独立实例|
|---|---:|---:|---:|
|<15 min fix|190|195|141|
|15 min - 1 hour|270|276|198|
|1-4 hours|38|39|31|
|>4 hours|2|2|2|

在每档内先保留每个实例一次，再按固定种子SHA256排序循环重复，合计140条重复。训练与测试按instance_id零重叠。372个独立实例重新计算Qwen3.5-9B embedding和目标patch NLL，再按512条权重执行PCA与16类KMeans。官方时间难度标签与ADS模型NLL分开保留。

默认测试入口：data/formal/eval/swe_bench_verified_test_128.jsonl。
默认单数据集训练入口：data/formal/train/swe_bench.jsonl。
实际七数据集训练池：state/formal-data/validated_task_pool.jsonl（3584条）。
重建训练用离线原始输入：state/source-data/swe_train_raw.jsonl（含仅供ADS使用的gold patch，不作为在线模型输入）。
完整划分清单：data/formal/eval/swe_bench_verified_split_manifest.json。

上游SkillFlow数据版本：07bb38bcc62fa8bebab6af86c39ba23b0293c97d。
官方Verified数据版本：78f471bf655a3137b2e8a75af1501690ec009ec3。

旧活动训练/测试数据已被替换，不保留此次替换前的完整数据备份；历史实验记录保留。其他数据集内容不变。本次只准备数据，未启动模型训练或评测。
