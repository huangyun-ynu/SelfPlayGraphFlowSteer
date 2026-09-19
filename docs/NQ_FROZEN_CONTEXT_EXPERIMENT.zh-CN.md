# NQ 固定证据实验

这个实验把 FiD 风格的 top-k passages 在运行前写入数据的 `prompt`，并在 `metadata.context_documents` 保留同一份证据。`provided_context_inline` 模式下不提供 NQ `search` Action，也不会调用 `/retrieve`。

准备数据：

```bash
PYTHONPATH=src ../.venvs/spgfs-pats-gpu/bin/python \
  scripts/formal/prepare_nq_frozen_context.py \
  --input data/formal/eval/nq_open_official_test.jsonl \
  --output state/experiments/nq-frozen-v1/nq_open_128.jsonl \
  --top-k 8 --workers 24
```

运行正式推理：

```bash
scripts/formal/run_nq_frozen_eval.sh \
  state/experiments/nq-frozen-v1/nq_open_128.jsonl \
  state/experiments/nq-frozen-v1/run-nq-frozen
```

默认路由和并发是 Director `Qwen3.5-9B`、Worker `deepseek`、轨迹并发 24、DeepSeek 路由并发 24。密钥只从本地 `.env` 读取；运行轨迹写入 `state/`，该目录被忽略，不进入 Git。

答案提交器在固定证据模式下会把 passage 传给最终 formatter，并要求答案来自一个最短连续证据 span。普通短问答模式不会自动启用这条实验性证据选择路径。

已完成的 128 条基线运行结果（代码版本整理前）：严格 EM 50/128，39.1%；DeepSeek 语义复核 62/128，48.4%；verifier pass rate 72/128，56.25%。这些数字用于对照，不代表修改后的 span formatter 已达到目标。
