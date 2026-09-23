# WebShop LASER 页面核对提示实验

2026-09-23，用户审阅适配后授权运行 128 题评测。

## 运行设置

- 输出：`state/formal-eval/webshop-laser-checklist-c24-20260923-182224/`
- 审计：`state/experiments/webshop-laser-checklist-c24-20260923-182224/`
- 配置：`configs/webshop_laser_checklist_eval.toml`；相对正式配置，仅新增
  `[webshop].worker_guidance_policy = "laser_checklist_v1"`。
- 数据：`webshop_official_test_128.jsonl`，同一批 128 题，seed 0，24 路轨迹。
- DeepSeek Worker：`deepseek-flash`，thinking 开启，reasoning effort 为 low，
  路由并发 20，输出上限 16384 tokens。
- Qwen Director：保留的 Qwen3.5-9B 服务 `http://127.0.0.1:18623/v1`，
  常规调用 thinking 开启；沿用关系二选一的原有非思考策略。
- legacy 页面，无 skill，环境反馈关闭，12+4=16 次动作预算，单题总 token 上限 350000。

复用保留的 GPU 1 服务，其启动配置为上下文 32768、最大序列数 24、KV cache
182361 tokens。本次没有新建模型部署，也没有释放模型或额外显存保留进程。
评测入口不绑定物理 GPU，服务的位置由部署决定。

入口（本次已运行；再次执行会创建新目录）：

```bash
bash scripts/formal/run_webshop_laser_checklist_eval.sh
```

审计目录保存启动日志、退出码、配置和关键源码快照、源码 SHA256、数据集 SHA256、
当前 Git 提交及未提交源码差异。`experiment_manifest.json` 记录本次具体运行身份。

## 结果

128/128 完成，任务级运行异常 0，进程退出码 0。

| 指标 | 正式历史参考 | LASER 页面核对提示 | 变化 |
| --- | ---: | ---: | ---: |
| 严格成功 | 62/128 | 62/128 | 0 题 |
| 严格成功率 | 48.4375% | 48.4375% | 0 个百分点 |
| 平均得分（百分制） | 69.0625 | 69.5052 | +0.4427 |
| 平均 token 消耗 | 65542.91 | 69492.09 | +3949.18（+6.03%） |
| 平均题目耗时（秒） | 94.67 | 108.34 | +13.66（+14.43%） |

逐题对照：54 题两次均成功，58 题两次均失败，8 题从失败变成功，8 题从成功变失败。
配对二项精确检验（McNemar）双侧 p = 1.0。

本次没有观察到严格成功率提升，平均得分略增，但 token 和耗时均增加。
因此暂不将此配置替换为正式基线，保留为可选实验设置。

## 轨迹审计与服务保留

- 全部 172 份去重 Worker 执行记录均走 DeepSeek 路由。其中 156 份会话所有者执行
  记录了 `laser_checklist_v1` 生效，16 份无状态评审执行按设计不注入。
- 127 题实际进入了 Worker 购物流程，全部有提示生效记录。另 1 题
  `webshop/goal-00387` 在 Director 阶段未完成 Worker 配置，没有 Worker 执行记录；
  环境结果为 `missing_output_agent`，得分 0。该题在历史参考中成功，本次仍计入
  完整 128 题分母，没有剔除或重跑后替换。
- Qwen 常规 `graph_action` 共 639 次，requested/effective thinking 均为 true。
  15 次关系二选一调用均为 false，符合既有策略。
- 运行清单确认 DeepSeek thinking 开启。1222 次成功请求中，1070 次返回正数
  reasoning token 用量；152 次未报告该字段，不能将字段缺失解释为关闭思考。
  另有 1 次 DeepSeek HTTP 502 请求失败事件；这与任务级异常计数不同。
- 环境反馈出现次数为 0；实验结束时源码、配置和数据哈希均与启动快照一致。
- 结束后 Qwen `/v1/models` 和 WebShop `/health` 检查成功。GPU 1 的模型服务与
  额外显存保留进程继续运行；额外保留约 2.25 GiB，空闲约 1063 MiB。
  本次没有停止模型或释放预留显存。

审计文件：`guidance_audit.json`、`post_run_integrity.json`；配对结果：
`comparison.json`、`paired_tasks.csv`，均在本次审计目录。

## 比较方法

与用户选定的正式历史参考
`webshop-legacy-page-only-reasoning-c24-20260918-172320`（62/128）比较。
严格成功指环境完成购买且 reward 为 1，平均得分单独报告。

`scripts/formal/report_webshop_env_feedback.py` 的通用配对比较逻辑用于检查双方
任务一致性，生成 `comparison.json` 和 `paired_tasks.csv`；其中反馈统计预期为零。
`scripts/formal/audit_webshop_guidance.py` 检查所有任务的提示注入、Worker 路由、
Qwen thinking 和 DeepSeek 返回的 reasoning token 统计，不输出模型思考内容。

本次与历史成绩的差值不是在同一源码上同步重跑的控制实验。历史源码、服务状态和
模型采样波动可能参与差异，因此不能仅凭一次结果认定提示造成稳定提升或退化。

复核命令：

```bash
.venv/bin/python scripts/formal/report_webshop_env_feedback.py \
  --baseline state/formal-eval/webshop-legacy-page-only-reasoning-c24-20260918-172320 \
  --candidate state/formal-eval/webshop-laser-checklist-c24-20260923-182224 \
  --output state/experiments/webshop-laser-checklist-c24-20260923-182224
.venv/bin/python scripts/formal/audit_webshop_guidance.py \
  --run state/formal-eval/webshop-laser-checklist-c24-20260923-182224 \
  --output state/experiments/webshop-laser-checklist-c24-20260923-182224/guidance_audit.json
```
