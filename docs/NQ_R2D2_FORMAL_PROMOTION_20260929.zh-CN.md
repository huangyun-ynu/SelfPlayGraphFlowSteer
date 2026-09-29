# NQ：R2D2 语料检索正式同步（2026-09-29）

> 后续更新：NQ 已与其它六个数据集统一切到 V3，检索语料和预算继续保留。
> 下文记录切换语料时的历史配置；当前提交行为见
> [七数据集 V3 同步记录](ALL_DATASETS_V3_FORMAL_PROMOTION_20260929.zh-CN.md)。

用户确认采用 R2D2 的 **1,702,133 段**新语料。当前 main 的正式 NQ 从预先内联
八段证据，切换为输入公开问题、由 Worker 在求解时检索固定本地语料。
正式配置是 `configs/formal_training.toml`；三个现有主机本地训练/评测配置同步了
同一检索设置，其余表保持原值。

## 当前设置与来源

| 项目 | 当前正式 NQ |
|---|---|
| 语料 | R2D2 pruned NQ，1,702,133 段 |
| 编码器 / 索引 | E5-base-v2 / FAISS FlatIP，768 维 |
| 服务 / profile | `http://127.0.0.1:19012/retrieve` / `nq-dense8-v1` |
| 输入模式 | `corpus_tool`；题面只含原始公开问题 |
| 检索预算 | 每次 top-8；单条轨迹所有 Agent、修订共享最多 4 次 |
| 提交要求 | 至少一次非空检索；答案引用当前 Agent 可见的证据编号与原文 |
| 证据不足 | 可明确提交 `insufficient_evidence`，按 NQ 无答案处理，得分 0 |
| 格式修复 | 全轨迹至多一次证据协议修复 |
| 外部回退 | 关闭网页检索回退 |
| Director / Worker 额度 | 正式默认 v2.2 / 240,000 tokens，沿用切换前设置 |
| 路由 | 保留正式候选池和 Director 自选；DeepSeek 请求并发 24 |

`evidence_token_budget=12000` 沿用实验参数，当前实现以 UTF-8 字节作保守计量，
并不表示可放入 12,000 个模型 token 的证据文本。

求解协议来自独立实验 `codex/nq-corpus-only-20260928` 的 `f7d1fcd`。
其核心源码与 `state/nq-r2d2-pilot-20260929/inference/results/run_manifest.json`
记录的运行文件哈希一致；另纳入该实验工作区的 R2D2 资产识别与健康检查支持。
随后未提交的 NQ 答案契约、预算改造和另一份 Hotpot 风格 NQ 研究代码继续留在实验副本中。
本次保留正式模型路由，没有套用 pilot 的固定 Worker 模型设置。

正式训练额外接通了 `evaluate_graph` 反事实分支的独立检索预算与证据核验：
不同分支不可借用主轨迹的引用；检索服务失败标为执行未完成，不能当成正常的训练零分。
混合任务池准备器按 JSONL 的 LF 分行，保留题目字符串中合法的 Unicode 行分隔符。

## 资产绑定

默认目录为 `state/formal-data/retrieval/r2d2-pruned-e5-v1/`：

- `r2d2.jsonl`：SHA-256 `a3628bc632895bebe4edd8aefba1ae5bc115acb8f13b2f25821cd68043007669`。
- `e5_Flat.index`：SHA-256 `f6dd966f17dbbe7c438dc9a4113139a1709e5e5567c40a4bd8750e0866d059d3`。
- 服务身份：`f8129c07a8c438a559bc4983f11dd1c8d48bc362a6035455cb8c20d6b28c9c9e`。

完整编码器文件校验值与原始压缩包标识见
`experiment_versions/promotions/nq-r2d2-20260929/asset-check.json`。
服务启动时校验实际文件，正式应用启动时拒绝身份不符的检索服务。
资产已在本机准备好并有运行中的服务；本次未重建索引或重启服务。
新主机需要复制上述固定资产及对应编码器，不能用旧 Wiki-18 文件替代。

## 训练与评测入口

`scripts/formal/run_experiment.sh` 默认读取正式配置；可用 `SPGFS_FORMAL_CONFIG`
显式指定配置。corpus 模式在采集前调用 `nq_corpus_tasks`，仅给 NQ 添加公开问题与
语料模式标记，生成 `state/formal-training-output-contract-v2/qa-corpus/task_pool.jsonl`。
它不预检索、不读取参考答案来构造查询；题目 ID、参考答案、split 和其它数据集行保持原值。

原始 `data/formal/train/nq_open.jsonl` 和正式 128 题均保留。新增
`data/formal/eval/nq_open_corpus_tool_128.jsonl` 是同一正式 128 题的模式转换版，
相邻 manifest 记录来源与身份。`configs/nq_corpus_eval.toml` 是独立的单 DeepSeek
评测配置；正式训练仍使用正式配置中的多模型候选池。

原有 WebShop 444 条训练数据及启动前数据校验保留。准备器曾在旧的 3,584 行本机缓存上
单独验证 NQ 转换，审计输出只存于本地状态目录，没有覆盖或启用这份旧缓存。
该检查只证明转换保持非 NQ 行不变，不表示旧缓存通过当前正式训练全部前置检查。

## 保存与验证

切换前 main 为 `68b9ead27f45bafc6c5a0300f4900eb79e298a6d`，已保存分支
`experiment/nq-frozen-before-corpus-20260929`。WebShop 仍为恢复后的 `235e670`
推理实现，修正后的 444 条训练数据不变；Hotpot、ALFWorld、SWE 和其它正式设置保留。
原有两个 NQ 实验工作区及未提交的研究文件均保留。

- 613 项相关回归通过，覆盖 NQ、反事实训练、检索、正式路由以及其它数据集的共享执行代码。
- WebShop 的 7 种通道、17 次 Worker 请求与恢复基线的匹配检查通过。
- 真实本地检索返回 8 段 R2D2 证据，服务身份和 1,702,133 段数量匹配；未调用语言模型。
- 全量语料、索引和编码器文件哈希与运行中服务匹配。
- 混合池的 512 条 NQ 成功转为问题输入，3,072 条其它数据集行保持原值。
- 128 题评测转换保留全部问题、ID 和参考答案。

可提交的验证记录位于 `experiment_versions/promotions/nq-r2d2-20260929/`；
本机输入备份、源码快照和完整日志位于 `state/nq-formal-promotion-20260929/`。
本次未启动正式训练或新的模型准确率评测，历史冻结版成绩不代表当前语料检索版成绩。
