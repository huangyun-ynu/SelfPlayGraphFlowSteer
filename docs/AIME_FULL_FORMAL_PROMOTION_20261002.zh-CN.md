# AIME 完整版本正式训练替换记录

正式 AIME 实现已切换为 `aime-no-code-comments-full-20261002`。来源是四题真实回归 `after-run01` 实际执行的完整冻结源码；不是只迁入工具提示或截断恢复提示。

## 完整替换与训练接入

`src/formal_aime/` 保存完整 AIME 应用及依赖包，共 142 个文件，逐文件 SHA-256 与 `state/aime-no-code-comments-regression/after-run01/frozen/aime_qwen_eval/` 一致。包括 Director/Canvas、Worker 运行时、原生工具接口、候选答案保全、题目级恢复账本、求解修订、简短结果确认、来源快照、最终报告组装、提交与用量审计，以及此次禁止代码注释的提示。

正式配置 `configs/formal_training.toml` 的 `aime_actions.implementation` 明确选择完整版本。原有正式入口 `scripts/formal/run_experiment.sh` 继续调用 `selfplay_graph_flowsteer selfplay-experiment`；应用工厂对 AIME 完整委托给 `formal_aime.application`，其余数据集使用已有正式应用。正式 AIME 的主求解和反事实图重放均使用同一套新实现。

适配层 `src/selfplay_graph_flowsteer/aime_formal.py` 只负责选择完整应用及连接训练接口：共用完全相同的图、动作、异常和轨迹类型，保留一套可信提交回执权限，并共享模型供应商请求并发队列、优先级及数据集上下文。数学运行时和恢复代码没有逐段重写；冻结包的 142 个文件保持原始字节。

正式训练继续使用现有 Director 模型选择、PATS 和 SkillBank。AIME 工具数量、工具资源限制、240000 token 阈值及其他数据集的实现保留。AIME 启用该实验版本所需的 `reported_usage_threshold_v1` 题目级实际用量账本；Worker 输入、输出及恢复用量均累计，Director 不计入 Worker 额度。正式路由继续为 GPT/Grok/Gemini/DeepSeek/MiniMax，独立回归使用固定 Qwen；四题成绩不代表动态路由训练成绩。

新增三项数学反馈参数采用被测版原有默认值：摘要首部 256 字符、尾部 1024 字符、答案反馈 2000 字符。Worker thinking 禁止的规则来自完整实验实现，Director 仍采用正式配置的 thinking 设置。

训练模型清单新增 `aime_implementation_contract`，绑定完整包的每个源码哈希及适配接口。版本不同的采集记录不能作为相同执行语义直接续跑。正式启动默认输出已改为 `state/formal-training-aime-full-20261002/experiment`，提交日志改为同版本目录的 `submissions/`。

## 旧版存档

替换前正式工作区保存在 [旧版快照](../experiment_versions/checkpoints/formal-before-aime-no-code-comments-20261002T080452Z/README.zh-CN.md)。存档包含 1076 个源码、公开配置、脚本、测试和文档文件，以及当时未提交的修改；另有同级压缩包及 SHA-256 校验值。

存档不包含凭据、数据、模型权重或运行服务。旧实验轨迹和资产仍保留在原位置。恢复检视请复制到不存在的新目录，步骤见存档说明。

## 验证与证据

验证明细见 [正式接入报告](../experiment_versions/promotions/aime-no-code-comments-20261002/README.zh-CN.md)，包括冻结来源逐文件核对、旧版快照和压缩包校验、原有 AIME 测试对新正式包的回放、正式框架回归及新增训练接入测试。

新增接入测试通过正式配置和正式应用工厂实际执行 AIME：检查 Worker 请求包含新提示，完成真实运行时提交与判分，验证主训练端能读取其可信回执和奖励，再执行完整反事实图重放。模型响应使用离线夹具，没有把模拟题当作新增真实模型成绩。

既有真实四题回归为第 5、14、22、28 题：修改后 4/4 提交、3/4 正确。第 14 题仍有后续注释循环并答错；此完整版本也保留该已知限制。详细比较见 [真实回归报告](../experiment_versions/reports/aime-no-code-comments-prompt/README.zh-CN.md)。本次是正式版本替换与接入验收，没有重新训练模型或产生新的正式动态路由准确率。
