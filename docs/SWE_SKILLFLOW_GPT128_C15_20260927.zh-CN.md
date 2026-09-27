# SWE SkillFlow 128 题推理运行

用户要求使用新替换的 SWE 数据集，全量运行 128 题；题目并发 15，远程正式测评并发 4，Qwen 开启 thinking，Worker 使用全部三个 GPT 路由。

本轮目录：`state/audits/swe-skillflow-gpt128-c15-gpt3-20260927/`。

- 数据 SHA256：`c56ba020f9bbf791c1e324cf71517c446fb53405a4436ed40c5398b2e17ffd38`。保持输入文件的全部 128 题与原顺序，不按结果筛选。
- Worker 路由池：`gpt`、`gpt_eco`、`gpt_student`，每条最大并发 5。前两条服务模型名为 `gpt-5.5`，第三条按现有服务使用别名 `lab-gpt-5.5-2`。
- Director：Qwen3.5-9B，thinking 开启，实际思考上限 512 token，动作预留 1024 token，总上下文上限 32768。独立服务使用 GPU 1、端口 18605。
- Director 保持 append_only 增量上下文，在线历史和训练轨迹保留 thinking；关系概率审计不进入在线上下文。仅推理、参数更新为 0、skill context 关闭。
- 每题 Director 编辑额度 24，共享 SWE 工具额度 32。正式远程测评最大并发 4，逐次记录入口等待时间和实际并发峰值。
- 源码、配置、运行脚本及输入均冻结并记录 SHA256；其他对话后续修改工作目录不会改变本轮冻结源码。

48 种公开仓库版本环境已就绪，新增 12 个版本通过真实非空公开 smoke 测试；128 个 base commit 和正式测评身份均已核对。环境准备详情见 `docs/SWE_PUBLIC_FUNCTIONAL_TESTS_20260927.zh-CN.md`。

首次双 GPT 路由尝试保存在 `state/audits/swe-skillflow-gpt128-c15-20260927/`。用户要求加入第三条路由后，该尝试在完成题数为 0 时停止；其 `superseded.json` 指向新一轮，不混入正式结果。

运行状态以本轮目录中的 `launcher.json`、`run/run_state.json` 和 `journal/` 为准。监督进程负责结束时停止本轮独立 Qwen 服务、核对远程测评实例 `STOPPED / STOP_CHARGING`，并生成 `summary.json`、`harness-queue-analysis.json`、`REPORT.zh-CN.md`。历史端口 18604 的服务和 GPU 保活进程继续保留。
