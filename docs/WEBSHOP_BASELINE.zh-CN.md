# WebShop 正式基线

2026-09-23 按用户指定，将 `legacy` 页面模式的无 skill 评测选为正式参考版本，
将 `retain_page_text` 的 45.3125% 版本归档。

| 状态 | 页面模式 | 严格成功率 | 平均 reward | 原始运行 |
| --- | --- | --- | --- | --- |
| 正式参考 | `legacy` | 62/128 = 48.4375% | 0.690625 | `webshop-legacy-page-only-reasoning-c24-20260918-172320` |
| 历史归档 | `retain_page_text` | 58/128 = 45.3125% | 0.648359375 | `webshop-deepseek-reasoning-noskill-c24-20260918-170329` |

两次原始结果均位于 `state/formal-eval/<原始运行>/`。不移动、不覆盖这些目录，
以保留轨迹、W&B 和其它引用。严格成功要求完成购买且环境 reward 为 1；平均 reward
不是严格成功率。

## 正式设置

- 通用正式配置：`configs/formal_training.toml` 的 WebShop 页面模式为 `legacy`。
- 专用无 skill 配置：`configs/webshop_official_eval.toml`，固定 DeepSeek Worker
  路由、`deepseek-flash`、thinking 开启、Worker 路由并发 20、输出上限 16384 tokens。
- Director：Qwen3.5-9B，thinking 开启；不注入 Director skill，不使用 SkillBank。
- 数据集：`data/formal/eval/webshop_official_test_128.jsonl`，128 题，seed 0，24 个评测 workers。
- WebShop：初始 12 次动作、修订 4 次、总计 16 次；暂存提交开启；观察字符上限 0。
- 正式来源、指标及文件 SHA256：`configs/webshop_official_baseline.json`。

2026-09-23 按用户要求，`[canvas].remaining_token_admission_enabled = false`
同时关闭 WebShop 的调度预测检查、请求发送前的 token 估算拦截和收尾 token 预留。
普通执行、完整图复评和购买收尾均遵守该开关。实际 Worker token 仍累计计费，并在
执行报告返回后检查 350000 总上限；这不是请求前的精确限额，单次执行可能越过上限。
12+4=16 次动作预算保持不变。历史 48.4375% 结果未重跑，不能视为此修改后的成绩。

在既有 Director 和 WebShop sidecar 服务就绪、项目 `.env` 已配置的情况下运行：

```bash
bash scripts/formal/run_webshop_official_eval.sh
```

脚本输出到新的带时间戳目录，不覆盖选定的参考运行。可用
`SPGFS_WEBSHOP_EVAL_OUTPUT` 指定一个新输出目录，或用 `SPGFS_WEBSHOP_DIRECTOR_URL`
指定 Director 服务地址。脚本不启动训练或模型服务。

## 历史归档与成绩边界

`configs/history/webshop_retain_page_text_20260918.json` 保存从 45.3125% 原始轨迹提取的
完整 WebShop 配置、模型与评测参数、成绩、原始文件路径及 SHA256。
它是历史参数快照，不是供正式入口自动加载的配置。

48.4375% 是选定原始运行的实测成绩。本次只提升版本地位、固化配置和归档，
没有重跑评测，也没有恢复该历史时刻的完整源码或全部运行环境。
专用 TOML 基于当前正式配置构建；当前代码的新运行必须另行报告实测结果。
