# SkillFlow 内部 Director 改造入口

本轮代码直接修改在同级仓库
[/workspace/h200-lab-7f3c/SkillFlow](/workspace/h200-lab-7f3c/SkillFlow)，
上游固定提交 `74be52bb6bd9f0e9e68dacb72636b75649197983`，分支 `codex/webshop-director`。

详见 [接入说明](/workspace/h200-lab-7f3c/SkillFlow/DIRECTOR_WEBSHOP.zh-CN.md)。
新入口为 SkillFlow 自己的 `training/webshop_director_eval.py`；
购物执行走其 `GenericTaskEnvironment.reset_react/react_step`，
Director/Canvas/图运行时复用本项目，旧 Native Worker 执行循环不参与。

本轮是推理编排接入，尚未接通完整训练周期。此前停止的 Native after2 128题测试没有重启。

用户随后要求本架构跑一轮128题真实推理，已完成：

- 实验：[webshop-director-128-20260924-run1](/workspace/h200-lab-7f3c/SkillFlow/state/experiments/webshop-director-128-20260924-run1)。
- 严格正确38/128（29.6875%），平均reward 0.516015625；93题购买、35题未购买，运行错误0。
- 同一128题，Worker DeepSeek low thinking，Director Qwen3.5-9B，Skill关闭，未训练权重；运行中冻结源码和配置。
- 相比此前无Skill合并修复版61/128、reward 0.6813802083，3题由错变对、26题由对变错；本次改造没有提升准确率。
- [完整报告](/workspace/h200-lab-7f3c/SkillFlow/state/experiments/webshop-director-128-20260924-run1/comparison/report.zh-CN.md)、[逐题指标CSV](/workspace/h200-lab-7f3c/SkillFlow/state/experiments/webshop-director-128-20260924-run1/comparison/tasks.csv)、[逐题历史对照](/workspace/h200-lab-7f3c/SkillFlow/state/experiments/webshop-director-128-20260924-run1/comparison/paired_tasks.csv)。
- 模型请求、环境动作、全部轨迹、源码快照、配置和指标完整保留。Director服务PID 130255、端口18603继续保留。

后续低准确率诊断已保存于
[诊断报告](/workspace/h200-lab-7f3c/SkillFlow/state/analyses/webshop-director-128-20260925-diagnosis/diagnosis.zh-CN.md)。
已确认：47道部分分题有选项扣分；32道未购买题被框架提前自动FINISH；4道最终零分题曾有官方评分满分的暂存候选；35次收尾报告校验失败后被替换成占位报告，且下游将其标为valid。
诊断仅分析轨迹和官方评分函数，93笔原购买分数全部复现，原实验归档SHA256核对通过；尚未实施推理代码修复。
