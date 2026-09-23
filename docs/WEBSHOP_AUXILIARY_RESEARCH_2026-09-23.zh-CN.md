# WebShop 无 skill 辅助功能：论文与源码核查

调研日期：2026-09-23。范围：作者公开论文和仓库中的 WebShop 实现，重点考察不依赖 skill 检索、生成或注入的推理时辅助功能。源码链接固定到本次核查的 commit；当前公开代码不等同于论文实验时的精确快照。本次未运行这些仓库或重跑评测。

项目正式参考仍为 legacy、无 skill、62/128 = 48.4375% 严格成功率，平均 reward 0.690625，见 [正式基线](WEBSHOP_BASELINE.zh-CN.md)。下文的论文数字不能直接与这 128 题结果排名，也不能直接预测增益。

## 对 SkillFlow 四项功能的核查

| 功能 | 其他项目的源码先例 | 证据边界 |
| --- | --- | --- |
| 可见状态摘要 | ASH 先由 summarizer 压缩当前页面，再交给 actor；SelfSum 有历史摘要及摘要记忆输入 | ASH 是额外模型调用，还包含候选筛选和纠错提示，比确定性状态字段汇总更强。SelfSum 的完整方法涉及训练，不能把训练成绩当成给现有模型打开开关的收益 |
| 重复搜索、返回与访问反馈 | BOLAA 将已用查询写入下一次搜索提示；LASER 记录已访问商品，重复选择时要求换候选 | 确有同类机制，但未找到上述重复反馈各自的独立消融；记录“刚离开哪个商品”的具体措辞不必完全相同 |
| 合法动作重排 | LASER 和 AgentBoard 按页面构造可用动作；LASER 进一步使用结构化函数 | 这是动作筛选、约束或表示方式，不等于只改变同一动作集合的排列。本次未找到纯重排的独立增益证据 |
| 显式逐步推理提示 | ReAct 的 think/action；SelfSum WebShop 模板明确使用 `<think>`、`<action>` | 不依赖 skill，但提示存在不等于该格式单独有效；ASH 的实验也说明更好观测可能比额外显式推理更重要 |

直接源码入口：

- [ASH 摘要调用及替换观测](https://github.com/robert1003/ash-prompting/blob/e67a3b3dcd94d9bf5b2d965b0bca6593a5f2e5ff/src/chatgpt_ash.py#L260)、[摘要提示规则](https://github.com/robert1003/ash-prompting/blob/e67a3b3dcd94d9bf5b2d965b0bca6593a5f2e5ff/prompts/chatgpt_prompts.py#L110)。
- [BOLAA 已用查询反馈](https://github.com/salesforce/BOLAA/blob/f026768d262a5e3e2936321c412db2306f9b4e19/web_run/multi_agent_arch.py#L186)。
- [LASER 重复商品处理](https://github.com/Mayer123/LASER/blob/dc50dafa1f88a4b889945393456b8960144be858/laser_agent.py#L363)、[AgentBoard 页面动作集](https://github.com/hkust-nlp/AgentBoard/blob/bb7255e2daf1989069a186dad9e53f70680961db/agentboard/environment/webshop_env.py#L86)。
- [ReAct WebShop notebook](https://github.com/ysymyth/ReAct/blob/6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9/WebShop.ipynb)、[SelfSum WebShop 提示](https://github.com/BayesWatch/SelfSum/blob/36024967310344534b057f635e6a2e9c076bd373/agent_system/environments/prompts/webshop.py#L17)、[摘要环境实现](https://github.com/BayesWatch/SelfSum/blob/36024967310344534b057f635e6a2e9c076bd373/agent_system/environments/env_manager.py)。

## 可迁移的其他辅助机制

以下是源码存在且有合理作用路径的候选；没有单项消融的，不宣称已独立证明提高严格成功率。

| 机制 | 源码实际操作 | 对当前基线的启发与成本 |
| --- | --- | --- |
| 页面专用提示 | LASER 对搜索、结果、商品、规格选择分别给出当前任务和动作 | 可减少不合页面状态的操作；项目已有 JSON 工具，增量应关注阶段提示，而非重复增加格式要求 |
| 可配置选项与标题区分 | LASER 提醒标题中的颜色等可能只是默认款，打开商品后仍可修改 | 避免在结果页过早排除正确商品；属于静态提示，成本低 |
| 购买前独立规格确认 | LASER 单独选择选项并执行点击，再购买 | 应核对实际选中值与用户要求；额外模型调用和选项点击都要记账 |
| 按缺失属性查详情 | ADaPT 的 DetailMatch 打开商品及 Features，合并信息后检查要求 | 可补足标题没有给出的证据；宏动作包含多个真实点击，不能只算一步 |
| 失败触发重新规划 | ADaPT 在执行失败后分解任务，并用检查点重放已完成动作 | 可借鉴失败原因传递；完整重放和多次尝试增加预算 |
| 具体错误与恢复提示 | AgentBoard 在不能搜索的页面说明需要先返回搜索页，并提供格式纠正 | 比笼统 Invalid action 更可操作；不应泄露隐藏目标属性 |
| 搜索与点击分工 | BOLAA 将搜索、点击交给不同提示的模型角色 | 可先尝试同一 Worker 按操作切换提示；多角色架构的效果不能归因于单一提示 |
| 历史候选兜底 | LASER 在预算结束后从已浏览商品选一个候选 | 项目已有总 16 次动作与暂存提交，应先比较现有兜底语义；原实现会 reset 后重新搜索购买，预算不同 |

实现定位：[LASER 页面与购买阶段](https://github.com/Mayer123/LASER/blob/dc50dafa1f88a4b889945393456b8960144be858/laser_agent.py#L141)、[标题与可配置选项提示](https://github.com/Mayer123/LASER/blob/dc50dafa1f88a4b889945393456b8960144be858/prompt_library.py#L190)、[LASER 兜底](https://github.com/Mayer123/LASER/blob/dc50dafa1f88a4b889945393456b8960144be858/laser_agent.py#L241)、[ADaPT DetailMatch](https://github.com/archiki/ADaPT/blob/ecdc4ab0030b4be9be122622d8ea78f8c59c44c4/run_webshop.py#L588)、[ADaPT 重新规划](https://github.com/archiki/ADaPT/blob/ecdc4ab0030b4be9be122622d8ea78f8c59c44c4/run_webshop.py#L881)、[AgentBoard 恢复反馈](https://github.com/hkust-nlp/AgentBoard/blob/bb7255e2daf1989069a186dad9e53f70680961db/agentboard/environment/webshop_env.py#L136)。

## 哪些效果有实验支持

| 论文与条件 | 严格成功率 | 能支持的结论 |
| --- | --- | --- |
| ASH，500 题，code-davinci-002 | ReAct 23.4% → ASH 30.2%，+6.8 个百分点 | 页面处理与提示组合有效；不是纯字段摘要的消融 |
| ASH，500 题，gpt-3.5-turbo-0613 | ReAct 3.0% → ASH 12.6%，+9.6 个百分点 | 作者指出旧模型上下文限制导致大量基线失败，不能直接迁移幅度到现有模型 |
| LASER，500 题，GPT-4-0613 | 去掉 backup 48.4% → 完整 50.0%，+1.6 个百分点 | 该实现中的兜底有贡献，但伴随额外动作 |
| LASER，另取 200 题消融 | 去掉 function call 50.0% → 标准 52.0%，+2.0 个百分点 | 该模型下结构化函数调用优于文本 JSON 版本；不能推出动作排序有效 |

来源：[ASH 论文 Table 1](https://aclanthology.org/2023.findings-emnlp.685.pdf)、[LASER 论文 Table 1、3](https://arxiv.org/html/2309.08172v2)。这里按论文正文表格计算百分点。不同模型、样本集与预算之间不作横向排名，也不相加各项提升。

另有反例：LASER 的 200 题消融中，增加一个示例从 52.0% 降到 50.0%；因此“多加示例/提示必然更准”没有依据。ASH Figure 5 中 DIRECT+ASH 27.4% 高于 ReAct 23.4%，说明显式推理也不是唯一有效路径。来源同上。

## 不能直接当成同预算基线增强的情况

- **LATS：** WebShop 实现复制和恢复环境状态，采样多个分支，结合反思及实际终局 reward 选结果。它是允许更多模拟和反馈的搜索方案。论文报告的 75.9 是平均得分，不是 75.9% 严格成功率。[论文](https://proceedings.mlr.press/v235/zhou24r.html)、[搜索循环](https://github.com/andyz245/LanguageAgentTreeSearch/blob/853d81614607dd27433faf17c7b0a7d660f95d22/webshop/lats.py#L408)、[分支状态恢复](https://github.com/andyz245/LanguageAgentTreeSearch/blob/853d81614607dd27433faf17c7b0a7d660f95d22/webshop/lats.py#L612)。
- **Reflexion：** 核查版本把 memory 传入 EnvironmentHistory，但第 245 行模型输入由 base_prompt 和当前 prompt 拼接，未使用该带 memory 的对象。这条 WebShop 路径不能仅凭“有反思文件”就认定反思已经影响模型。[源码](https://github.com/noahshinn/reflexion/blob/218cf0ef1df84b05ce379dd4a8e47f17766733a0/webshop_runs/webshop_trial.py#L207)、[对应 issue](https://github.com/noahshinn/reflexion/issues/36)。
- **环境简化：** ReAct 的公开 notebook 有搜索结果 top-3 展示及翻页限制；其历史输入还受 6400 字符截断。不能把它描述成完整页面、无限全历史，或将简化带来的分数与正式环境直接比较。[源码](https://github.com/ysymyth/ReAct/blob/6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9/WebShop.ipynb)。
- **训练方法：** SelfSum 的摘要格式可参考；完整学习方法、微调与 RL 成绩属于另一条实验线，不能归为冻结模型的纯推理设置收益。

## 针对 48.4375% 正式参考的建议

以下为基于源码证据的实验建议，尚未在本项目验证；本次没有修改正式运行配置。

1. 优先测试页面专用提示、标题默认款与可选规格的区分、购买前要求与实际选项核对。已有 purchase_evidence 和选中状态机制应复用，先确认缺口。
2. 在现有历史记录上补充“用过的查询、商品不匹配原因、离开原因、尚未核实属性”。避免简单硬禁所有回访：回到候选商品购买是合法且可能必要的。
3. 改善恢复反馈，将失败原因和当前可执行的下一步一起展示。
4. 再做 ASH 式观测处理实验。先测确定性结构化摘要；额外 LLM 摘要另列实验，并保留完整合法目标 ID、规格及导航，防止摘要丢掉正确候选。
5. 纯动作重排、强制额外 think 标签排在后面。当前正式模型已启用 thinking，需要检查重复推理的调用和 token 成本。

对照应保持同一 128 题、模型、页面模式、seed、无 skill、16 次真实环境动作及重试口径；分别报告严格成功率、平均 reward、实际动作数、LLM 调用、tokens 和耗时。额外模型摘要可以在不增加环境动作的情况下增加推理预算，需单列成本。128 题中每题约 0.78 个百分点，先逐项消融，再组合有效项；若据此反复调参，应另留验证题，避免把同一测试集当开发集。

## 本地核查材料

源码快照：工作区 `reference_sources/webshop_auxiliary_audit_20260923/`。
其中 `verified_source_index.json` 记录实际下载文件的仓库、commit、SHA256 和源链接；各仓库 `source_manifest.json` 保存核查时的树信息。早期 `download_manifest.json` 记录过失败下载，部分随后已通过 GitHub API 补齐，应以 verified 索引为准。ReAct 的 `WebShop.code.py` 是 notebook 代码单元提取产物，不作为上游源码路径引用。
