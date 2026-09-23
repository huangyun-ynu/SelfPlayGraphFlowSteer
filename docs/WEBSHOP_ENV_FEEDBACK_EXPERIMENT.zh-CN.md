# WebShop 重复搜索与返回反馈实验

2026-09-23，按用户要求适配 SkillFlow 的 `_append_webshop_neutral_env_feedback`，以正式 legacy 无 skill 的 62/128 结果为参考运行 128 题。

## 适配内容

配置项为 `[webshop].env_feedback_enabled`，默认 `false`；独立实验配置 `configs/webshop_env_feedback_eval.toml` 将其设为 `true`，其余配置与 `configs/webshop_official_eval.toml` 相同。

- 在成功执行搜索后，检查本条轨迹是否已经执行过相同查询。若近期返回搜索页，提示查询在返回后被重复使用。
- 在点击 Back to Search 后，说明刚离开的商品 ASIN，或说明刚离开的搜索结果页已不在当前观察中。
- 反馈仅描述公开状态和已执行动作，不禁止重复搜索或回访，不选择商品，不使用隐藏目标或 reward。
- 历史在同一条轨迹的 Worker 修订间保留，新任务绑定时清空；失败的请求不记作已执行动作。
- 适配差异：本项目已有结构化商品状态，返回时使用上一页面公开的 ASIN，避免从最近几次点击猜测；反馈写入独立 `env_feedback` 字段，保留在 Worker 输入中，不受页面文本截断影响。

本地参考文件：`../reference_sources/skillflow_table/SkillFlow/training/environment.py`，函数位于原核查版本第 8003 行，文件 SHA256 为 `bcdc21c310df835a969bfb693b903cad6aac6802aca6e073be9e61f0755f8ceb`。

## 48.44% 参考运行的思考模式审计

对 `state/formal-eval/webshop-legacy-page-only-reasoning-c24-20260918-172320/trajectories/` 的全部 128 条最终轨迹逐次检查：

| 调用 | 数量 | thinking_requested | thinking_effective | 思考输出 |
| --- | ---: | --- | --- | --- |
| 常规 graph_action | 721 | true | true | 全部存在 |
| 受约束的关系二选一 relation_choice | 21 | false | false | 无 |

因此 62/128 对应的最终轨迹并没有把常规 Qwen 调用中途切为无思考。21 次无思考是原有单 token `on/off` 关系策略的行为。本次使用 `--director-thinking`，并保持关系策略原样。

逐轨迹计数与哈希保存在 `state/experiments/webshop-env-feedback-20260923/baseline_thinking_audit.json`。

## 运行与复现

```bash
SPGFS_WEBSHOP_DIRECTOR_URL=http://127.0.0.1:18623/v1 \
  bash scripts/formal/run_webshop_env_feedback_eval.sh
```

入口使用独立部署的模型服务，不固定物理 GPU；可用 `SPGFS_WEBSHOP_WORKERS` 和 `SPGFS_WEBSHOP_EVAL_OUTPUT` 指定并行数和新输出目录。默认 24 路，禁止覆盖已有输出。

本次参数：同一份 `webshop_official_test_128.jsonl`，seed 0，24 路轨迹，DeepSeek Worker、thinking 开启，无 skill，legacy 页面，12+4=16 次动作预算。Qwen3.5-9B 服务端口 18623，32768 上下文，最大并发序列 24。

选择 GPU 时检查了全部显卡容量；当时 GPU 1 剩余显存最多，为 31930 MiB，因此本次使用 GPU 1，服务显存比例 0.33。服务启动后报告 KV cache 182361 tokens。原始启动日志、源码哈希、数据哈希及部署参数保存在 `state/experiments/webshop-env-feedback-20260923/`。第一次服务启动因临时 socket 路径过长退出，第一次评测启动因轻量虚拟环境缺少 transformers 退出；两次均未产生正式评测样本。随后缩短临时路径并使用项目 GPU 虚拟环境启动。

实验输出：`state/formal-eval/webshop-env-feedback-c24-20260923/`。

## 实测结果

128/128 完成，运行错误 0。

| 指标 | 正式历史参考 | 环境反馈实验 | 变化 |
| --- | ---: | ---: | ---: |
| 严格成功 | 62/128 | 58/128 | -4 题 |
| 严格成功率 | 48.4375% | 45.3125% | -3.125 个百分点 |
| 平均得分（百分制） | 69.0625 | 67.5195 | -1.5430 |
| 平均 token 消耗 | 65542.91 | 67659.40 | +2116.49 |
| 平均题目耗时（秒） | 94.67 | 108.31 | +13.63 |

逐题对照：49 题两次均成功，57 题两次均失败，9 题从失败变成功，13 题从成功变失败。配对二项精确检验（McNemar）双侧 p = 0.5235；单次结果没有显示提高，亦不足以断言反馈本身造成稳定退化。

实际反馈共 28 次，覆盖 15 题：重复查询 21 次，返回搜索 7 次。新运行的 639 次常规 Qwen 调用全部 requested/effective thinking=true；13 次关系二选一调用均为 false，与历史参考的策略一致。这里的 45.3125% 是本次 legacy+反馈实验，与归档的 retain_page_text 45.3125% 是不同运行，平均得分也不同。

正式参考继续使用 48.4375% 版本。本次实验没有替换正式配置。

比较命令：

```bash
python scripts/formal/report_webshop_env_feedback.py \
  --baseline state/formal-eval/webshop-legacy-page-only-reasoning-c24-20260918-172320 \
  --candidate state/formal-eval/webshop-env-feedback-c24-20260923 \
  --output state/experiments/webshop-env-feedback-20260923
```

比较脚本要求双方均有 128 条唯一记录且题目相同，输出成功率、平均 reward、逐题变化、配对检验、反馈触发次数和思考设置审计。

本次以当前工作区代码运行；虽然新旧实验配置只差反馈开关，48.44% 是历史源码运行的结果，不能将差值完全归因于反馈。源码版本和采样波动也可能影响成绩。

## 验证与服务保留

22 项反馈、sidecar、网络测试通过；另有 3 项 WebShop runtime 测试通过。验证覆盖跨修订记忆、跨任务隔离、关闭开关时观察一致、失败请求不计入历史，以及长页面截断后仍保留反馈。

按用户追加要求，评测后保留 Qwen 模型服务，并通过 `scripts/formal/hold_gpu_memory.py` 保留同一张卡的剩余可用显存，目标运行余量为 1024 MiB。保留进程独立于模型服务；向该进程发送 SIGTERM 仅释放额外保留空间，不停止 Qwen。实际 PID、空间和状态以实验目录的部署状态文件为准。

完成时服务地址为 `http://127.0.0.1:18623/v1`，物理 GPU 1，API server PID 37099，engine PID 39080；服务健康检查通过。显存保留进程 PID 19576 已运行，初次确认额外保留 2.25 GiB、剩余空闲约 1.04 GiB。该进程会根据运行余量调整保留量；这不是硬件级独占锁。详情见 `state/experiments/webshop-env-feedback-20260923/retained_services.json` 与 `gpu_memory_hold.json`。
