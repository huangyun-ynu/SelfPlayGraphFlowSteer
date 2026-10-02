# QA B1 正式训练接入与来源核对

正式目录 `SelfPlayGraphFlowSteer` 的 Hotpot 工作流现采用 `qa-b1-formal-20261002`。HotpotQA 与使用这套工作流的 MuSiQue 均使用可信原题载体、结果职责、内容保全修复和 B1 重放；正式训练的 HotpotQA 默认启用 `compact_factual_v1`。B2 与 `qa-b1-input-answer-v1` 仍为独立实验，没有晋升。

## 来源与外部改动

来源为 `SelfPlayGraphFlowSteer-qa-b1/state/stage-b1-20261002/frozen`，冻结清单 SHA-256 为 `8318c7c01728c418d94dc33189c6f20edb55095493e34a328cba344fe610e404`。核对冻结清单 682 个文件、B1 当前源码/配置 660 个文件，均无差异。将这轮 Compact、QA 修复和 B1 的保存补丁反向还原，127 个既有主包源码文件与最初复制清单逐文件一致；因此迁入的 18 个源码文件具有可核验的 QA 实验来源。

正式目录在迁入前已有其他工作的改动：相较最初复制时，源码、配置和脚本有 440 个新增或变化文件，其中主包有 19 个。主要涉及完整 AIME、共享实际 usage 与预测代码删除、HealthBench 答案保全以及后端/环境实现。采用验证过的原始代码作三方合并基础，保留这些正式改动；没有将 B1 的旧预算/后端文件整包覆盖回来。

合并时 `cli.py` 和 `runtime.py` 存在交叉点，保留现行实际 usage、HealthBench 恢复及 NQ 原生工具逻辑，再接入 QA 原题、schema 修复与上下文压缩。16 个外部修改而 QA 未修改的函数 AST 保持一致。执行语义指纹只追加 QA 来源、配置和源码哈希。AIME 的 142 个冻结引擎文件全部保持原始字节；其观测接口校验允许 QA `replay_trace` 更新，其他共享观测定义仍要求 AST 一致，其余共享模块仍要求字节一致。

检查期间又检测到 TriviaQA 的 application、FiD 配置和运行脚本被外部工作修改，均未由本次晋升脚本写入。哈希能够确认差异与保存的来源，不能据此确认编辑者身份。明细见 [外部改动核对](../experiment_versions/promotions/qa-b1-20261002/foreign-change-audit.json)。

## 正式训练行为

- 入口仍为 `scripts/formal/run_experiment.sh`，使用当前正式目录 `src` 与 `configs/formal_training.toml`。
- Hotpot 使用 `qa_public_task_v1`、`qa_result_integrity_v2`、`hotpot_evidence_first_v2`。原题、公开材料、Director 委派和结果职责分别传递；私有参考答案/别名/支持标签不进入请求。
- Director 保留历轮思考、动作、反馈及训练 token。压缩器只删去重复的控制信息，并引用已经发送且内容完全相同的 Worker 报告；窗口保持 32,768，训练序列上限保持 35,000。
- Director 自主生成图与选择 Worker；正式训练路由继续是 GPT/Grok/Gemini/DeepSeek/MiniMax。固定 Qwen、20/20 是历史评测设置，没有覆盖正式训练路由。
- Hotpot 实际 Worker usage 阈值保持 240,000；没有恢复时间/token 预测准入、预留或节点分账。迁入时修掉 B1 重放中两个已删除的旧预算参数。
- QA 载体/代码版本进入缓存与执行语义校验。训练采集和恢复记录观察策略与源码指纹，旧语义不能被视作同一采集续跑。默认输出改为 `state/formal-training-qa-b1-20261002/experiment`，提交目录采用同版本前缀。
- 训练数据池、评分器、PATS、SkillBank、模型权重和其余数据集的配置保持原有设置。独立 `musique_ood` 包未被这次主工作流晋升覆盖。

## 验证与限制

三个完成的独立回归集合共 **503 项通过**：QA/B1/上下文与提交 252 项；AIME、NQ、HealthBench、WebShop、共享 usage、训练 token 等相关回归 235 项；训练采集/恢复/关系反事实 16 项。另在正式应用工厂的 4 项用例中实际转换训练轨迹，核对 Director token 前缀、思考/动作和损失 mask，计入前述 252 项。此外，按 Git 暂存区导出独立目录，没有引用本机未发布的 OOD 文件，正式 QA 与 AIME 再通过 257 项，验证上传内容能独立运行。回归使用离线后端/工具夹具，未调用真实模型、未启动 GPU 训练、未更新参数。

扩大扫描曾运行到 selfplay 分布式重试等待，208 秒后中断，不计作完整通过。发现的 18 项既有失败已在晋升前快照复现：3 项 WebShop 旧行为断言、2 项应用关闭夹具、12 项将预算的 `policy` 字段误认作关系审计的断言，以及 1 项未隔离共享账本的 AIME runner 夹具。另一项串联账本失败在独立运行中通过。它们未被计入通过项，也没有作为此次 QA 晋升顺带修改。

源码、公开配置、脚本、测试与文档的迁入前快照保存在本机 `experiment_versions/promotions/qa-b1-20261002/before/`。Git 仅发布可运行的正式代码、必要依赖、定向测试与小型来源报告；凭据、模型、数据、轨迹和大快照继续留在本机。

验证清单见 [validation.json](../experiment_versions/promotions/qa-b1-20261002/validation.json)，迁入差异见 [promotion.diff](../experiment_versions/promotions/qa-b1-20261002/promotion.diff)。历史 B1 固定路由成绩并非此次正式动态路由的新成绩；本次没有重新训练模型或重新测量 EM/F1。
