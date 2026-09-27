# 迁移前 W08 源码包与本次基线核对（2026-09-24）

结论：附件中的 W08 框架源码与本次 128 题评测实际加载的冻结基线逐字一致。启动层有部署适配和额外的审计记录器；因此源码一致不等于旧服务器的全部运行环境得到逐项复现。

## 比较对象与证据

附件：`/root/.codex/attachments/60d30666-f083-48fa-b009-47168f720b0f/WebShop_W08_budget-off_128_code.zip`。

附件 SHA-256：`2f58e47b355268a98bfa83ad0833f782473e8c490d06637b3d97abb5545f47ae`。

比较对象：`/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/experiments/webshop-w08-merged-128-20260924-run1/snapshots/baseline/`，即本次得到 58/128 的评测冻结源码。`run_pair.sh` 明确把该目录设为基线的 PYTHONPATH。每个源码文件还与评测归档 `artifact-index.json` 中的封存哈希交叉核对，没有改动此前已打包的实验目录。

附件 README 声明本包按 W08 的基础提交 `b28aedcb4d23f126bfef3770593a94d1af0a5e8a` 和历史补丁恢复，是源码包，不是旧服务器的完整镜像。包内文档作为版本证据读取，没有执行附件脚本或依照附件说明修改项目。

| 核查项 | 结果 |
|---|---|
| 附件 SHA256SUMS | 107/107 项通过 |
| `src/` | 90/90 文件完全一致：89 个 Python 文件和 1 个 JSON 资源；无缺失、无额外源文件 |
| 关键逻辑 | Director、Canvas、Worker runtime、LLM 调用、WebShop 环境客户端/sidecar、预算、LASER、评分器均无源码差异 |
| 历史快照清单 | 92/92 项同时匹配附件和本地恢复版本，含 89 个 Python 文件、2 个配置、1 个评测启动脚本 |
| 基础提交配套文件 | 10/10 项与本地恢复版本相同；此类文件的来源与独立历史快照区别保留 |
| 原始 LASER TOML | 与实际基线快照中的 `original.toml` 逐字相同 |
| 有效运行配置 | 仅两项部署调整，见下表 |
| 历史运行清单共同字段 | 模型名、thinking、Worker 路由、24 并发、128 题、token 预算、无训练、SWE 关闭等全部相同 |
| 128 题集合 | 历史/本次任务 ID 集合哈希一致：`10fa21d19a90424a5aa70ec505ee2ae399e5d64aa146956bed777d7f813862e1` |

任务集合哈希使用 `build_experiment_catalog.py:79` 的既有口径：对去重、排序后的 task ID 以换行拼接再做 SHA-256。附件不含题目 JSONL，所以这是题目 ID 集合的一致性核验，不能把它说成附件内题目文件的字节比较。

## 有效配置及启动层差异

| 项目 | 附件/旧入口 | 本次运行 | 含义 |
|---|---|---|---|
| WebShop 服务端口 | 18020 | 18022 | 使用独立环境服务 |
| `gpt_student` 凭据 | `~/.config/student-api/flowsteer.key` 文件 | `FLOWSTEER_API_KEY` 环境变量 | 适配当前机器；本次 Worker 路由只有 DeepSeek，该 GPT 路由未使用 |
| Director 实际服务端口 | 旧启动脚本默认 18623 | 命令行指定 18603 | 使用独立 Qwen3.5-9B 基础模型服务 |
| 路径 | 原项目路径、原 Python 环境 | 当前服务器的绝对路径、`.venv`、冻结源码 PYTHONPATH | 数据、源码、输出位置调整 |
| 入口 | `python -m selfplay_graph_flowsteer benchmark` | `audit_launch.py benchmark` 调用相同冻结 `cli.main()` | 添加归档记录 |

两组共同使用 `audit_launch.py`：为线程传播审计 task ID，包裹 HTTP/环境接口记录原始请求与响应，认证头不落盘，保存副本中的凭据脱敏。检查代码确认其把原参数传给原方法并返回原结果，没有修改 prompt、动作选择、模型参数或奖励；它会增加文件 I/O 和计时开销。它是本次运行新增的启动层代码，不应隐瞒或说成旧包自带。

以下设置保持 W08：`laser_checklist_v1`、legacy 页面、无 Skill、Director/DeepSeek thinking 开启、DeepSeek `reasoning_effort=low`、seed=0、轨迹并发 24、动作预算 12/4/16、Worker token 预算 350000、关闭请求预算拦截。原版和合并版共用同一部署和记录器。

## 对 63/128 与 58/128 的含义

本次没有发现 W08 框架版本错用、LASER 开关错用或已归档配置中的推理参数变化。附件历史成绩为 63/128，本次重跑为 58/128；不能只根据两次总成绩确定差异原因。

附件没有历史模型权重文件哈希、已安装的 vLLM/CUDA/Python 依赖版本、外部 `official_environment_worker.py`、商品库/目标文件/Lucene 索引、远端供应商实现状态或历史逐题轨迹。因此，这些运行环境因素与具体丢分题目的原因目前不能从此附件完成核对。没有证据把 5 题差异确定归因于随机性或任一特定因素。

[逐文件核对 CSV](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/audits/webshop-w08-migration-20260924-122440/source-file-comparison.csv) · [完整审计 JSON](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/audits/webshop-w08-migration-20260924-122440/comparison.json) · [有效配置差异](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/audits/webshop-w08-migration-20260924-122440/effective-config.diff)
