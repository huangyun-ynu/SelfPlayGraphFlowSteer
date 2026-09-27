# WebShop 记忆来源修复后降分：逐题及因果证据审计

本轮真实结果为 62/128→61/128，平均 reward 0.735221→0.681380。已定位的状态矛盾修好了，但这轮数据不足以证明补丁导致了全部回退。更精确的解释是：两次运行在补丁进入决策前已经广泛分歧；新版轨迹中，预算耗尽未购买、类型/标题匹配倍率下降、规格与属性匹配变差共同拉低了成绩。补丁还保留了“已访问→无新增证据”的不严谨语义，需要单独验证其影响。

## 1. 准确率和 reward 的变化不同

- 8题满分→非满分，7题非满分→满分，净少1题（-0.78125个百分点）。McNemar精确双侧p=1.0，单轮不能确认准确率稳定下降。
- reward下降23题，上升14题，持平91题。下降合计11.933333，上升合计5.041667，净减6.891667；除以128即平均reward减少0.053841。
- 23题下降按互斥规则分组如下。类型下降组中也可能有规格、属性变化，因此这些是结果分组，不是各字段的独立因果贡献。

|下降组|题数|总reward损失|对应平均100分制损失|
|---|---:|---:|---:|
|原本买到商品，新版未购买|6|3.833333|2.994792|
|完成购买，但类型/标题匹配倍率下降|8|4.266667|3.333333|
|其余属性或规格匹配变差|9|3.833333|2.994792|
|合计下降|23|11.933333|9.322917|
|其余14题改善抵消|14|+5.041667|+3.938802|

平均reward的配对bootstrap区间为[-0.102279,-0.004556]，本轮的降分不能忽略；该区间只对这128题重采样，不包含同一配置重复运行的模型波动，不能当作纯粹的补丁因果区间。

## 2. 公平配置没有保证逐步可复现

两版最初Director请求128/128正文相同，模型、seed、预算和推理强度未变。实际Director参数为temperature=0.6、top_p=0.95、top_k=20、seed=0；Worker为temperature=0、reasoning_effort=low。

首次Worker请求中，108题正文不同，逐字段解码后差异全部位于assigned_task，也就是Director给Worker的委派内容。另20题首次Worker请求正文完全相同，其中12题返回了不同工具动作。固定配置、固定seed和temperature=0在这里都没有保证实际输出完全相同；日志能证明不可逐步复现，不能确定是服务采样、数值/调度还是其他后端细节造成。

按实际环境动作对齐：17题两版完整动作一致、分数也一致；其余111题存在动作分歧，其中108题的首次分歧发生在补丁字段产生差异之前，只有3题是在补丁已经进入上下文后首次分歧。比较时已排除随机commit_id和商品列表序号，保留实际搜索词、ASIN与规格选择。

8道满分回退题全部在补丁生效前出现首次动作分歧；goal-00053、00054、00269两版整条轨迹均未触及该补丁。这个结论排除“补丁造成了这些首次分歧”，但不排除另外5题在后续不同轨迹中继续受到补丁影响。

仅有的3个“补丁先于首次动作分歧”样本为goal-00060（+0.25）、00185（-0.5）、00352（+0.666667）；三题首次Worker的assigned_task也不同，仍不是只改状态字段的严格成对实验。

## 3. 八道准确率回退题

|题号|reward变化|公开动作与评分依据|
|---|---:|---|
|[00031](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/audits/webshop-memory-source-regression-20260924/cases/goal-00031.md)|1.0000→0.7500|同一商品上，参考版选择用户指定的 himalayan pink salt coarse grind；新版选成 himalayan black rock salt coarse grind。规格得分 1→0.5。首次动作分歧在修复生效前。|
|[00053](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/audits/webshop-memory-source-regression-20260924/cases/goal-00053.md)|1.0000→0.6667|两版同一商品、同一搜索，米色改成绿色。公开任务允许 beige or green，但隐藏 goal_options 只有 beige，导致 r_option 1→0。两版全程均无补丁差异；该例不能当成补丁退化或明确的公开需求违背。|
|[00054](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/audits/webshop-memory-source-regression-20260924/cases/goal-00054.md)|1.0000→0.2500|参考版购买可选 16.9 fl oz、amber 的喷瓶。新版未打开首次结果第1位同一商品，继续搜索后购买 500ml、brown 泵瓶，属性和规格均失分。两版全程均无补丁差异。|
|[00075](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/audits/webshop-memory-source-regression-20260924/cases/goal-00075.md)|1.0000→0.0000|参考版成功商品标题是 Venus Envy 2PK，但页面可选 green envy 和 4 fl oz (pack of 3)。新版多个搜索结果均出现该商品第1位，却一直未打开，最后16次动作耗尽、未买。首次搜索与选品分歧均早于访问修复差异。|
|[00134](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/audits/webshop-memory-source-regression-20260924/cases/goal-00134.md)|1.0000→0.3333|参考版打开标题为128GB的Tab S7，再选择512GB规格。新版跳过已可见的该候选，购买标题写512GB的翻新机；官方属性与规格分项均为0。首次选品在补丁影响前。|
|[00157](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/audits/webshop-memory-source-regression-20260924/cases/goal-00157.md)|1.0000→0.7500|参考版在最初商品上完成颜色/尺码选择后购买；新版换色、换商品、返回后选项重置，最后在另一商品仅选large，颜色目标失分。搜索分歧先于补丁。|
|[00269](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/audits/webshop-memory-source-regression-20260924/cases/goal-00269.md)|1.0000→0.0000|首次 Worker 请求正文完全相同，却分别搜索 toothpaste 与 mouthwash。新版买漱口水，类型倍率1→0。两版全程均无补丁差异，证明存在与本补丁无关的决策变化。|
|[00397](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/audits/webshop-memory-source-regression-20260924/cases/goal-00397.md)|1.0000→0.5000|首次 Worker 请求正文完全相同，但检索词顺序不同；新版后续换到另一款bagel，官方 quality ingredients 属性分项1→0。首次搜索分歧早于补丁。|

三个重要候选遗漏实例可以直接复核：

- 00054：可获得满分的喷瓶已在新版首次搜索第1位，标题显示默认10OZ，但打开后可选择16.9 fl oz。新版跳过它，购买500ml棕色泵瓶。
- 00075：可获得满分的染发剂在新版8次搜索结果中都处于第1位，标题是Venus Envy 2PK，而商品页具有Green Envy三包装选项。新版没有打开它。
- 00134：可获得满分的平板在首次搜索第5位，标题写128GB，页面可选512GB。新版更偏向标题直接写512GB的其他商品。

这些候选也确实出现在Worker实际输入的合法动作列表中，不只是存在于后台结果。它们体现了标题默认规格与可选变体的混淆，不能由修正访问历史自动解决。是否因为标题而排除属于行为推断；轨迹能确定的是候选已可见、未打开，以及最终规格/属性评分更差。

## 4. 六个预算耗尽零分：本轮最清楚的可执行问题

新版6个未购买题全部耗尽16次动作，且参考版都完成了购买：00075、00105、00185、00221、00271、00457。参考版另外3个未购买题在新版完成购买，所以总体未购买数是3→6，而不是新增只有3题。

00105、00221、00271、00457在剩余2次动作时继续读详情，第16次动作返回商品页，已经没有第17次购买机会。原始Worker输入中明确给出remaining.total=2，随后为1，所以这是动作完成路径没有算好，不是预算信息缺失。00075最后一次打开商品，00185最后一次选择颜色，同样未留下购买机会。

这6题的总reward损失为3.833333；其中只有00075原本满分，另外5题原本只是部分分。它解释了为何reward明显下降，而严格成功数只少1题。这里的优化应是通用的完成路径可行性判断：从当前页经过返回、必要选项和购买还需几步，而不是强制购买或专门对这些题兜底。

## 5. 官方评分放大了商品选择波动，且有需求歧义

用原WebShop评分器对两版全部247次购买离线重算，247/247与保存reward相符，未发现本轮计分实现变化或错记。隐藏目标只用于事后诊断，没有进入模型输入、训练或题目专用代码。

源码[goal.py](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/assets/webshop/source/web_agent_site/engine/goal.py:228)将属性、规格、价格匹配先合成基础分，再乘类型倍率。类型判断还包含与目标商品标题的名词重叠；低于0.1可降到0.1倍，为0时总分变0。因此它不等同于人类对品类的宽泛判断。

例如00284、00407都从约0.6667变成0.0667，类型倍率1→0.1；00433选择的商品同样是hoodie，且已选3x-large，但官方类型倍率为0，0.75变0。00269则从牙膏换到漱口水，倍率1→0。不能把这些变化简单总结为模型少核对了一个字段。

00053是明确歧义：公开请求写beige or green，两版同一商品，分别选择beige和green；隐藏goal_options只有beige，导致1→0.6667。这个回退不应算作明确违反公开需求，更不能归因于记忆修复。此类差异应独立标注，不能用隐藏答案修正推理行为。

## 6. 此次修复还存在一个真实的设计边界

它没有只修改访问事实：由于原辅助函数将may_add_*定义为“已访问”的反值，恢复访问事实也改变了模型看到的动作价值标志。

- 在新版实际输入上，若应用旧函数，82个商品动作和145个详情动作的may_add标志会为true；当前修复后为false。
- 82个商品动作涉及24题，其中74个所在输入的product_inspections已经为空。访问过并不能证明当前上下文保留了所需规格、证据，也不能证明再次打开对返回购买没有价值。
- 145个详情动作中，142个的对应文本仍在当前checkpoint，3个对应文本已不在输入。这3次来自00015、00197，两题分数都没有下降。因此“细节文本丢失→禁止重看→本轮降分”不能作为已证实解释。
- 所有动作仍是agent_selectable=true，修复没有硬禁用重开或购买。这里只能说辅助语义可能改变探索倾向，需要独立消融，不能声称已证明其导致本次净降分。

合理的语义应分开：是否访问过、相关证据现在是否可见、是否必须返回执行操作。无法确定行动价值时保留unknown，不从seen直接推出无价值。当前六商品缓存及6000字符展示裁剪也未改变，本轮没有解决全部长期记忆问题。

## 7. 对之前分析和下一步的修正

之前定位到的状态矛盾是真实缺陷，但用“涉及57题、其中33题失败”来判断它是约50%准确率的主要瓶颈，证据不够。困难题本来就更容易搜索更多、上下文更长、发生裁剪，随后也更容易失败；这是共同出现，不能直接当成因果。此前受影响57题在新旧运行中均成功24题，也没有显示整体挽回效果。

下一步建议按三个可分离的问题验证：

1. 做受控归因：从保存的同一Worker输入或同一环境状态处分叉，固定前缀、委派内容和预算，只改变访问事实/价值字段；成对重复。完整128评测也需要同配置交错重复对照，才能估计运行波动。不要靠更高推理强度补成绩。
2. 修正通用状态语义：保留访问事实修复，单独验证删除或改为unknown的may_add价值推断。证据是否可见、操作是否需要返回与访问历史分开记录，不加入题目规则、Skill或更强提示。
3. 改善明确的解题机制：基于合法动作图计算完成购买所需最少步数；把搜索标题中的默认规格与页面的可选规格分开，未打开时将可选规格状态保留为未知。验收同时看严格成功、reward和未购买数，避免只提高兜底分数。

这轮不能宣称修复提高了准确率，也没有证据把全部退分归因于修复。当前最强的证据是模型运行的早期分歧、具体的选品/规格偏差和动作预算收尾失败；价值标志的语义缺陷属于需要下一轮隔离验证的机制。

## 复核材料

- [证据与统计汇总](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/audits/webshop-memory-source-regression-20260924/evidence_summary.json)
- [逐题补丁暴露与首次分歧](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/audits/webshop-memory-source-regression-20260924/task_attribution.json)
- [首次Worker输入逐字段差异](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/audits/webshop-memory-source-regression-20260924/initial-worker-diffs.json)
- [候选已在Worker输入中出现的证据](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/audits/webshop-memory-source-regression-20260924/visible_baseline_success_products.json)
- [247次购买的原评分器重放](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/audits/webshop-memory-source-regression-20260924/scored.json)
- [公开动作序列](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/audits/webshop-memory-source-regression-20260924/public_actions.json)
- [原实验完整对比](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/experiments/webshop-memory-source-128-20260924-run1/comparison/comparison.json)

本审计仅离线读取保留轨迹并执行纯函数/评分器，不调用模型、不重跑评测、不修改冻结代码。原归档保持不变，Director继续保留。
