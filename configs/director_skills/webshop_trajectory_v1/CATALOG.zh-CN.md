# WebShop Director 轨迹编排技能 v1

共 12 条，来源为合并版、身份修复版、记忆来源修复版各128条轨迹，共384条、同一组128个任务。技能供Director决定职责、依赖、修订与收尾；Worker继续按现有合法动作执行。

此版本是用户明确要求生成的 WebShop 专用 Skill 实验材料。此前无Skill基线保留。status=seed只表示可被现有加载器检索，provenance=trajectory_derived_unvalidated表示尚未通过效果验证；usage/helpful/hurt均为0，没有虚构提分结果。

## 技能目录

|ID|技能|Director的编排动作|轨迹依据与解释边界|
|---|---|---|---|
|ws-public-constraints|[公开需求保真的职责委派](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/configs/director_skills/webshop_trajectory_v1/cards/ws-public-constraints/SKILL.md)|将完整购买责任保留给会话所有者；缩小子目标时保留原始需求、数值条件与允许的替代项。|历史委派内容变化及颜色可选项歧义说明原始公开任务应保持权威；不把隐藏评分要求变成委派要求。|
|ws-session-owner|[保留单一会话所有者](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/configs/director_skills/webshop_trajectory_v1/cards/ws-session-owner/SKILL.md)|保留拥有实际会话的执行节点；辅助节点读取公开产物，反馈回同一所有者，避免把报告当成共享浏览器状态。|从三版轨迹中的 environment_owner、environment_access、output_agent 和暂存购买记录提炼成功执行的不变量；不声称已经观察到大量所有权丢失回退。|
|ws-variant-capability|[区分标题默认规格与可选变体](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/configs/director_skills/webshop_trajectory_v1/cards/ws-variant-capability/SKILL.md)|对产品类型合理但标题规格不符的候选，委派核验可选配置；未观察到的变体保持未知，不按标题直接淘汰。|喷瓶、染发剂和平板案例中，参考版已验证的可配置候选出现在新版公开输入里却未被打开；仅提炼默认标题不等于完整选项空间。|
|ws-live-options|[回访后核验当前选项](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/configs/director_skills/webshop_trajectory_v1/cards/ws-live-options/SKILL.md)|回访或换商品后，要求所有者核对当前商品和实时选项；历史选择不能替代当前已选状态。|多次离开并重开商品的轨迹显示 selected_options 会清空；选项状态不能从访问记忆或标题恢复。|
|ws-targeted-evidence|[围绕具体需求缺口核对证据](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/configs/director_skills/webshop_trajectory_v1/cards/ws-targeted-evidence/SKILL.md)|围绕具体条件组织核验，将同一商品和配置的证据标为支持、矛盾或未知；优先解决会改变选择的缺口。|年龄适用范围、颜色证据和原料质量等案例说明需要核对具体缺口；未读某页本身不等于该项失分。|
|ws-candidate-comparison|[在同一约束下比较候选](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/configs/director_skills/webshop_trajectory_v1/cards/ws-candidate-comparison/SKILL.md)|让候选按同一组公开条件比较证据、选项、价格和返回购买的成本；保留已验证的可行候选供最终决策。|多候选探索中出现已见候选被替换、最后商品缺条件及回访开销；这是待验证的比较机制，不是从成功商品倒推出固定选品规则。|
|ws-search-revision|[对无进展搜索做局部修订](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/configs/director_skills/webshop_trajectory_v1/cards/ws-search-revision/SKILL.md)|先区分缺的是搜索覆盖、可选规格还是属性证据，再围绕一个可验证缺口修订任务；不靠重复换词制造进展。|染发剂、画作和爆米花轨迹中，近似查询反复出现，末尾没有完成购买；修订应以信息缺口而非查询字面变化为目标。|
|ws-evidence-recovery|[分离访问历史与证据恢复价值](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/configs/director_skills/webshop_trajectory_v1/cards/ws-evidence-recovery/SKILL.md)|分别判断是否访问过、证据是否仍可见、是否需要返回配置或购买；只向所有者请求恢复影响决策的缺失信息。|已有轨迹中记录裁剪导致访问事实矛盾；修复后仍存在读过但文本不在输入的情况。只提炼状态与证据语义，不声称该技能必然提分。|
|ws-completion-budget|[按完整购买路径分配动作预算](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/configs/director_skills/webshop_trajectory_v1/cards/ws-completion-budget/SKILL.md)|让所有者估算从当前页返回、补齐选项、购买的完整动作成本，再决定是否增加探索；预算不足时如实说明限制。|六题耗尽16次动作未购买，其中四题最后一步返回商品页；抽象为动态完成路径预算，不把16写入技能。|
|ws-transaction-closure|[区分候选、暂存购买和环境完成](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/configs/director_skills/webshop_trajectory_v1/cards/ws-transaction-closure/SKILL.md)|根据真实交易状态区分推荐、暂存与完成；输出保留在持有有效交易的所有者上，沿用运行时提交协议。|真实成功轨迹保存暂存与提交状态；失败轨迹可能仍有总结但未购买。使用现有提交协议，不生成新的交易接口。|
|ws-bounded-review|[有证据输入的轻量评审](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/configs/director_skills/webshop_trajectory_v1/cards/ws-bounded-review/SKILL.md)|仅在独立判断能解决具体分歧时增加只读评审；提供原始需求与候选证据，将可执行反馈接回所有者。|由公开约束遗漏和比较缺口提出的编排假设，结合现有只读辅助节点接口；没有将其标记为已经实验证实的多Agent收益。|
|ws-local-revision|[保留有效状态的定向修订](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/configs/director_skills/webshop_trajectory_v1/cards/ws-local-revision/SKILL.md)|只修订新反馈涉及的职责或依赖；保留完整购买目标、会话、有效证据、实时选项与剩余预算。|历史运行存在阶段修订、候选回访、选项清空和委派内容变化；提炼局部修订与状态继承边界，不宣称委派措辞变化已被证明是降分原因。|

## 加载与上下文边界

- 原生加载文件：[snapshot.v2.json](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/configs/director_skills/webshop_trajectory_v1/snapshot.v2.json)，兼容项目 DirectorSkillBankV2。作用域是shopping，且运行时须提供WebShop搜索与点击能力；其他任务类型和缺少这些能力的执行器不匹配。
- [配置片段](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/configs/director_skills/webshop_trajectory_v1/solver_skillbank.fragment.toml)应合并到独立评测配置。保留原来的模型、low推理强度、Worker路线和预算；CLI显式使用`--skill-context on`。`--director-skill-root`属于另一种单文件静态提示加载器，不用于这个v2包。
- 现有流程只在solve开始时按原始任务检索最多3条，随后注入Director系统上下文，token上限1024。它不会在每个状态变化时重新检索，因此本包没有声称实现按阶段自动切换12条技能。后期预算/收尾技能能否被初始检索覆盖，需要单独观察。
- 技能不是固定图模板，不强制多Agent、不更换Worker路线、不规定固定动作序列，也不把历史题号、ASIN、品牌答案、隐藏选项或评分倍率写入模型可见正文。
- 逐卡SKILL.md用于阅读或其他支持该格式的工具；本项目实际读取JSON字段。目录未安装到全局Codex技能，也没有自动启用到正在使用的基线。

## 证据与验证

[provenance.json](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/configs/director_skills/webshop_trajectory_v1/provenance.json)保留源轨迹哈希、公开观察、所有权与结果关联，和模型正文分开。案例是开发材料，不是新独立测试集。轻量评审等卡片是针对已观察缺口提出的编排假设，不能说已经证明有效。

生成脚本为[scripts/formal/build_webshop_director_skills_v1.py](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/scripts/formal/build_webshop_director_skills_v1.py)。结构、实际token预算、检索与内容边界见[验证报告](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/configs/director_skills/webshop_trajectory_v1/VALIDATION.zh-CN.md)和[机器可读记录](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/configs/director_skills/webshop_trajectory_v1/validation.json)。尚未运行新的128题推理测试，也没有训练模型；Director服务保持运行。
