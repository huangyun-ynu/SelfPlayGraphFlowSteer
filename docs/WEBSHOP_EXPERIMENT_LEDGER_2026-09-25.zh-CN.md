# WebShop 实验版本总账（更新至 2026-09-25）

当前正式训练采用 **M02**。历史成绩仍属于原评测配置；正式训练保留Director选逻辑模型、程序选接口及Qwen thinking，不等于固定DeepSeek实验复现，也没有新增训练成绩。见[当前正式版本](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/docs/WEBSHOP_BASELINE.zh-CN.md)。

本次更新整理17组完整128题结果、8组完整10题开发实验、1组提前停止的128计划，以及定向子集、撤回/失败尝试和未实施方案。仅更新记录，没有运行新推理或训练。W01–W11沿用原记录编号；其他编号是本总账索引，不是发布版本号，也不表示按表格逐行继承。

EM统一指**完全正确数÷样本数**（官方reward=1），不是文本字符串匹配。平均分为**官方reward均值×100**，保留部分得分；JSON/CSV同时保留0–1原始均值。官方WebShop没有F1。

## 1. 完整128题结果

|版本|基于/对照|主要改动|LASER/Worker提示；Director Skill|正确数 / EM|平均分 /100|
|---|---|---|---|---:|---:|
|[W01 fixed](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/experiment_versions/run_catalog.json)|目标映射修复|修正任务与环境目标映射；实际 GPT-5.5 Worker；Director thinking 关；结构化观察4000字符|无LASER；原公共指令；Skill：static Director skill|23/128，17.9688%|37.3307|
|[W02 no-skill](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/experiment_versions/run_catalog.json)|W01|移除静态 Director skill，保留 GPT-5.5 与结构化观察|无LASER；原公共指令；Skill：无|25/128，19.5312%|41.9661|
|[W03 full-index](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/experiment_versions/run_catalog.json)|W02|换完整商品索引和 DeepSeek Worker；双方 thinking 关，含多项同时变化|无LASER；原公共指令；Skill：无|53/128，41.4062%|63.3568|
|[W04 retain_page_text](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/experiment_versions/run_catalog.json)|W03|保留原页面文本、lifecycle字符上限设0；Director/Worker thinking 开；另含中间工程修复|无LASER；原公共指令；Skill：无|58/128，45.3125%|64.8359|
|[W05 legacy](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/experiment_versions/run_catalog.json)|W04|搜索观察改为 legacy，保留内部结构化状态和已选规格|无LASER；原公共指令；Skill：无|62/128，48.4375%|69.0625|
|[W06 env-feedback](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/experiment_versions/run_catalog.json)|W05|增加重复查询、离开商品页面等环境反馈|无LASER；环境反馈开；Skill：无|58/128，45.3125%|67.5195|
|[W07 LASER checklist](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/experiment_versions/run_catalog.json)|W05（不是在W06上累计）|在 legacy 上加入按页面核对需求、价格、属性、规格的 LASER；环境反馈关闭|公共指令＋独立LASER；Skill：无|62/128，48.4375%|69.5052|
|[W08 budget-off（历史）](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/experiment_versions/run_catalog.json)|W07|关闭请求token准入、执行credit与closure预留；累计token和16次动作上限仍保留|公共指令＋独立LASER，仍开启；Skill：无|63/128，49.2188%|71.0221|
|[W09 identity-fix](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/experiment_versions/run_catalog.json)|W08|恢复内部ASIN，修正已访问商品及详情证据的身份判断|公共指令＋独立LASER，仍开启；Skill：无|61/128，47.6562%|70.7552|
|[W10 detail-unlimited](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/experiment_versions/run_catalog.json)|W09|取消单段详情1400字符截断；并未取消全部商品缓存和输入裁剪上限|公共指令＋独立LASER，仍开启；Skill：无|60/128，46.8750%|67.6953|
|[W11 SkillFlow history](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/experiment_versions/run_catalog.json)|W10|完整私有Observation/Action历史，移除重复购物记忆提示；仍走原图工具协议|LASER仍开启；环境反馈关；Skill：无|60/128，46.8750%|71.8945|
|[W08-R1 W08原版重新对照](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/experiments/webshop-w08-merged-128-20260924-run1/comparison/comparison.json)|W08|用W08原版重新跑相同128题；这是新一轮采样，不能覆盖历史63/128|公共指令＋独立LASER；Skill：无|58/128，45.3125%|65.7031|
|[M01 W08提示合并](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/experiments/webshop-w08-merged-128-20260924-run1/comparison/comparison.json)|W08-R1（同期对照）|合并公共Worker指令与LASER，去重并明确核对、探索、购买、预算优先级；Director未改|merged_checklist_v1；保留清单内容，取消独立重复注入；Skill：无|66/128，51.5625%|72.3359|
|[M02 合并＋身份修复](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/experiments/webshop-merged-identity-128-20260924-run1/comparison/comparison.json)|M01|在合并版移植W09内部ASIN与访问/详情身份修复；不是直接继承W10/W11|合并清单；Skill：无|62/128，48.4375%|73.5221|
|[M03 合并＋身份＋记忆来源修复](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/experiments/webshop-memory-source-128-20260924-run1/comparison/comparison.json)|M02|访问事实改读裁剪前记录、实时动作标注和匹配商品checkpoint，区分已读未保留；may_add价值推断尚在|合并清单；Skill：无|61/128，47.6562%|68.1380|
|[S01 Director Skill v1](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/experiments/webshop-director-skills-128-20260924-run1/comparison/comparison.json)|M03|只增加12条Director编排Skill，初始检索top-3、1024 token；Worker及底座源码不变|合并清单；Skill：12条；每题初始top-3|61/128，47.6562%|70.8789|
|[F01 SkillFlow内部接入Director](/workspace/h200-lab-7f3c/SkillFlow/state/experiments/webshop-director-128-20260924-run1/comparison/comparison.json)|M03（历史对照，独立架构分支）|直接修改SkillFlow，以reset_react/react_step执行购物，接入项目Director/Canvas；新执行器、独立会话、FINISH提交；未接旧收尾|SkillFlow原始模板＋图职责/报告接口；无旧LASER/合并清单；Skill：无|38/128，29.6875%|51.6016|

原始合并M01的EM最高（66/128，51.5625%）；合并＋身份修复M02的平均分最高（73.5221/100）。这是已有单轮结果，不表示已证明稳定优于其他版本。S01使用同一128题轨迹开发Skill，不是独立泛化评测。

W08历史63/128和W08-R1重新对照58/128必须保留为不同运行。M01的同期对照是W08-R1；M02/M03/S01引用前轮封存对照。F01是直接在SkillFlow内部改造执行流程，与项目内Native适配是不同分支。W01–W04还同时改变模型、索引、thinking或页面观察，不能把差值归因于单项框架改动。

W08关闭的是请求token预算拦截，**没有关闭LASER**。W09、W10、W11也保留LASER。M01–M03、S01将LASER内容合并进公共Worker指令，取消的是重复的独立注入。N系列及F01走原生模板，不使用旧LASER/合并清单；N02/N03另有新增购物策略提示，不能称为无策略提示版。

## 2. Native完整10题开发实验

|版本|基于/对照|主要改动|LASER/Worker提示；Director Skill|正确数 / EM|平均分 /100|
|---|---|---|---|---:|---:|
|[N00 Native第一轮](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/experiment_versions/run_catalog.json)|W11|项目内适配SkillFlow原生search/click模板，每Agent独立会话、私有完整历史，FINISH提交；后续四项工程修复尚未包含|SkillFlow原始模板；无旧LASER/购物清单；Skill：无|1/10，10.0000%|37.1667|
|[N01 Native before](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/experiments/webshop-native-iteration-20260924/comparison.json)|N00后工程修复|修复跨数据集native误用、未闭合think解析、清理中断、提交回写污染；重新跑10题|原生模板；无旧LASER；Skill：无|1/10，10.0000%|63.0000|
|[N02 Native after](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/experiments/webshop-native-iteration-20260924/comparison.json)|N01|增加未购买/候选/共享额度事实反馈，并加入完成购买及职责修订提示|原生模板＋新增购物策略提示；后来撤回；Skill：无|3/10，30.0000%|63.8333|
|[N03 Native after2](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/experiments/webshop-native-iteration-20260924/comparison.json)|N02|再强调标题规格不等于已选择规格、缺属性证据需看公开详情；历史5/10版本|原生模板＋规格/详情策略提示；后来撤回；Skill：无|5/10，50.0000%|75.8333|
|[N04 Native m1](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/experiments/webshop-native-target7-20260924/comparison.json)|撤回N02/N03策略提示后|保留纯事实候选/预算反馈，给当前观察及历史补充真实selected_options|恢复原始策略模板；无旧LASER、无新增策略提示；Skill：无|2/10，20.0000%|45.8333|
|[N05 Native m2-valid](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/experiments/webshop-native-target7-20260924/comparison.json)|N04|适配官方text_rich渲染，保留按钮、选中/访问标记，修正相应动作映射|同N04；富文本观察；Skill：无|1/10，10.0000%|50.0000|
|[N06 Native m3](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/experiments/webshop-native-target7-20260924/comparison.json)|N04机制线（不是累计启用富文本）|FINISH前对无候选的选中输出Agent增加至多一次同会话收尾，使用原修订额度|同N04；输出阶段事实字段；Skill：无|2/10，20.0000%|58.8333|
|[N07 Native m4](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/experiments/webshop-native-target7-20260924/comparison.json)|N06|增加同Agent私有多轮对话历史；保持m3收尾机制，模型私有思维不进入历史|同N04；多轮消息组织；Skill：无|1/10，10.0000%|37.1667|

N00与N01都是1/10，但前者平均37.1667、后者63.0000，且N01包含四项工程修复和新的运行配置，不能合成同一轮。N03的5/10是历史提示版；其策略提示随后被撤回，后来用户指定用该冻结版扩展128题，结果单列如下。历史W11在匹配的这10题为4/10、平均70.8333，仅是W11结果切片，不是新版本或新推理轮。

## 3. 提前停止的128题计划

|版本|基于/对照|主要改动|LASER/Worker提示；Director Skill|正确数 / EM|平均分 /100|
|---|---|---|---|---:|---:|
|[N03-P128 after2扩展128（提前停止）](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/experiments/webshop-native-after2-128-20260924-run1/comparison/partial_summary.json)|N03|按用户要求恢复历史5/10冻结版本开展128题；低准确率后用户叫停|after2策略提示；无旧LASER；Skill：无|17/47，36.1702%|55.5851|

N03-P128只完成47/128，剩81题未完成；EM=17/47=36.1702%，平均55.5851/100均只针对已完成部分。不能写成17/128的完整EM，也不能与完整128题直接排名。原始请求、在途请求、已完成轨迹和停止记录保留。

## 4. 历史定向子集

|版本|基于/对照|主要改动|LASER/Worker提示；Director Skill|正确数 / EM|平均分 /100|
|---|---|---|---|---:|---:|
|[T01 closure original](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/experiment_versions/run_catalog.json)|定向子集|原购买提示；历史失败30题|见对应历史实验；不据名称推断LASER；Skill：见历史档案|0/30，0.0000%|1.2778|
|[T02 closure removed](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/experiment_versions/run_catalog.json)|定向子集|在同一30题删除购买收尾提示|见对应历史实验；不据名称推断LASER；Skill：见历史档案|1/30，3.3333%|5.5556|
|[T03 closure best_so_far](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/experiment_versions/run_catalog.json)|定向子集|在同一30题提示预算内购买已见最佳候选|见对应历史实验；不据名称推断LASER；Skill：见历史档案|2/30，6.6667%|27.4667|
|[T04 option-fixes](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/experiment_versions/run_catalog.json)|定向子集|规格和动作列表修复；21题中20题有后端失败，保留完整故障结果|见对应历史实验；不据名称推断LASER；Skill：见历史档案|1/21，4.7619%|4.7619|
|[T05 option-fixes retry](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/experiment_versions/run_catalog.json)|定向子集|重试其中20题；仍有1题含后端失败，不能只报告重试成绩|见对应历史实验；不据名称推断LASER；Skill：见历史档案|7/20，35.0000%|67.2500|
|[T06 budget-aware](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/experiment_versions/run_catalog.json)|定向子集|27题预算与购买收尾策略诊断|见对应历史实验；不据名称推断LASER；Skill：见历史档案|0/27，0.0000%|43.5494|
|[T07 price-range](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/experiment_versions/run_catalog.json)|定向子集|单题价格区间排障|见对应历史实验；不据名称推断LASER；Skill：见历史档案|0/1，0.0000%|50.0000|

上述分母各异，尤其closure选的是历史失败题，option-fixes包含大量后端故障。不能只保留重试中的较好结果。其他早期启动/单题smoke记录继续保留在跨数据集历史索引中，不作为新的完整128题版本。

## 5. 无有效新成绩的尝试与方案

|版本|基于/对照|主要改动|LASER/Worker提示；Director Skill|正确数 / EM|平均分 /100|
|---|---|---|---|---:|---:|
|[X01 r3](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/experiments/webshop-native-target7-20260924/r3/aggregate_summary.json)|—|旧提示方案的后续尝试；0题完成、3次连接失败|未作为有效新版本评测；Skill：—|—|—|
|[X02 m2](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/docs/WEBSHOP_NATIVE_MECHANISM_EXPERIMENT_2026-09-24.zh-CN.md)|—|富文本初版适配器缺少同目录导入路径，冒烟失败；没有有效完整结果|未作为有效新版本评测；Skill：—|—|—|
|[X03 m2r](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/experiments/webshop-native-target7-20260924/m2r/aggregate_summary.json)|—|服务端口未释放导致环境连接失败；0题完成、3次失败|未作为有效新版本评测；Skill：—|—|—|
|[X04 m5 high](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/experiments/webshop-native-target7-20260924/m5_WITHDRAWN.json)|—|改用high违反公平对照要求，已撤回；部分轨迹保留但不作为框架收益或有效成绩|未作为有效新版本评测；Skill：—|—|—|
|[X05 W08复测首次归档中断轮](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/experiments/webshop-w08-merged-128-20260924-run1/setup-attempt-incomplete-http-audit/reason.json)|—|模型HTTP正文记录不完整，整轮排除；修复记录器后两组重新开始|未作为有效新版本评测；Skill：—|—|—|
|[X06 早期错误目标映射](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/docs/EXPERIMENT_RECORDS.zh-CN.md)|—|prompt与环境目标映射无效，原WebShop成绩废弃，不能纳入有效版本排名|未作为有效新版本评测；Skill：—|—|—|

|版本|基于/对照|主要改动|LASER/Worker提示；Director Skill|正确数 / EM|平均分 /100|
|---|---|---|---|---:|---:|
|[P01 09-25旧版/新版修复方案](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/audits/webshop-old-repair-plan-20260925/evidence.json)|—|已完成工程诊断及修复设计；预算信息、访问价值语义、预算转交、FINISH保护尚未实施和评测|未作为有效新版本评测；Skill：—|—|—|

破折号表示没有可用于本次版本对照的成绩，不是0%或0分。m5即便留有部分轨迹，也因high推理强度不公平而排除。预算信息和访问价值字段等修复仍是方案，没有新增“修复后EM”。

## 6. 证据边界与复核

- 迁移前W01–W11、N00和早期定向子集读取保留的[指标档案](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/experiment_versions/run_catalog.json)及历史实验说明。原运行目录在当前服务器不可直接读取，因此本次没有声称重新审计这些原始轨迹。
- W08-R1、M01、M02、M03、S01、F01共6组完整128题已逐题重算正确数和reward均值，与各自汇总一致；题目ID集合相同。N01–N07共7组10题及中止轮47题也完成同样核对。
- 模型、low推理强度、动作及token预算的公平约束继续保留；F01并发与页面/执行协议不同，属于架构比较。固定配置不保证服务输出逐步一致，不能从单轮成绩作确定因果归因。
- F01存在已确认的自动提前结束、候选取消、报告错误状态和选项观测缺陷。见[诊断](/workspace/h200-lab-7f3c/SkillFlow/state/analyses/webshop-director-128-20260925-diagnosis/diagnosis.zh-CN.md)和[新旧保护核对](/workspace/h200-lab-7f3c/SkillFlow/state/analyses/webshop-merged-protections-20260925/report.zh-CN.md)。旧合并版额外核对见[预算/状态证据](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/audits/webshop-old-repair-plan-20260925/evidence.json)。分析不构成新评测成绩。
- [机器可读JSON](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/experiment_versions/reports/webshop-version-ledger-20260925.json)、[CSV](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/experiment_versions/reports/webshop-version-ledger-20260925.csv)、[重建脚本](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/scripts/formal/build_webshop_version_ledger.py)保留来源选择器、指标精度与来源文件SHA256；脚本仅读取封存产物并写总账。
