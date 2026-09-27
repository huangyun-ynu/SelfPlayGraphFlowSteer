# 合并提示版 + W09 商品身份修复：128 题复测

新版本严格成功 **62/128（48.4375%）**；已有合并版为 **66/128（51.5625%）**。准确率变化 -3.1250 个百分点，平均 reward 变化 +0.011862。

## 改动范围

以已完成 66/128 的冻结合并版为起点，移植从校验通过的 W09 源码导出的身份修复。只修改 runtime.py、webshop.py，新增 webshop_identity.py。恢复内部商品 ASIN，统一搜索访问标记、候选记忆、动作语义记忆、动作辅助判断，以及同一 ASIN 的唯一合法目标纠正。五个相关函数 AST 与 W09 一致。

身份只取已有结构化字段或当前合法 open_product 动作中的 ASIN；不查询隐藏目标、不由标题猜身份。原 legacy 展示、动作集合、商品记忆容量和合并提示保持不变，没有加入需求清单新机制、强提示或 Skill，没有参数训练。主目录已有后续实验改动，因此本次运行使用独立冻结源码。

## 完整指标

|指标|已有合并版|合并版 + 身份修复|差值|
|---|---:|---:|---:|
|严格成功数|66|62|-4|
|严格成功率|0.515625|0.484375|-0.031250|
|平均 reward（0–1）|0.723359|0.735221|0.011862|
|平均得分（100分制）|72.335938|73.522135|1.186198|
|完成购买数|123|125|2|
|未完成购买数|5|3|-2|
|部分得分任务数|56|61|5|
|零分任务数|6|5|-1|
|平均 Worker token|61,414.359375|64,824.546875|3,410.187500|
|平均 Director token|18,178.867188|18,519.468750|340.601562|
|平均合计 token|79,593.226562|83,344.015625|3,750.789062|
|Worker 总 token|7,861,038|8,297,542|436,504|
|Director 总 token|2,326,895|2,370,492|43,597|
|合计总 token|10,187,933|10,668,034|480,101|
|平均动作数|7.398438|7.453125|0.054688|
|平均题目耗时（秒）|82.760703|86.055620|3.294917|
|整组墙钟时间（秒）|602.436084|574.588527|-27.847557|

F1：不适用。官方 WebShop 评测不定义 F1，不将平均 reward 当作 F1。HTTP 请求次数、输入/输出 token、缓存与 reasoning 用量可用性、耗时分布、每题动作数、协议诊断和失败类型均保存在 comparison.json。供应商用量与执行账本分别核对，不重复相加。

## 身份修复是否生效

|审计项|已有合并版|修复后|
|---|---:|---:|
|已保留访问记录却标为未访问（次数）|136|0|
|记忆中的商品再次出现（次数）|136|130|
|重复打开同一商品（次数）|42|26|
|商品动作记忆缺失 ASIN（条）|211|0|
|保留的商品动作记忆（条）|211|205|
|搜索动作不含显式 asin 字段（条）|3030|2950|
|搜索商品动作总数（条）|3030|2950|
|存在访问状态误判的题数|32|0|

访问审计按每个 Agent 最近六个实际观察到的商品检查，不将容量淘汰后的记录缺失当成身份错误。重复打开可能是返回购买或补查信息，不能一律当成无效动作。商品动作记忆条数按保存的执行快照计数，不是独立商品数量。

## 配对结果与解释边界

两版均成功 58 题；原合并版失败、新版成功 4 题；原合并版成功、新版失败 8 题；两版均失败 58 题。McNemar 精确双侧 p=0.387695。
配对 bootstrap（10000次）成功率差值95%区间：[-8.5938, 2.3438] 个百分点；平均 reward 差值区间：[-0.03444140625, 0.05736035156249994]。

本次只新增一轮128题，参考是之前保存的合并版运行，并非本轮重新跑的同步对照。同一开发集、单个 seed，外部服务、缓存与采样可能波动；代码缺陷是否消除与准确率是否稳定提升应分开判断，不能把所有逐题变化确定归因于修复。历史 W09 61/128、W08 63/128 不参与本轮主对比。

## 公平性与产物

- 同一128题，seed 0，24路轨迹，Worker接口并发20；加载后的配置完全相同。
- Qwen3.5-9B基础 Director，thinking 开启；DeepSeek deepseek-flash Worker，thinking 开启，reasoning_effort=low。动作预算12/4/16，Worker token上限350000，请求预测预算拦截关闭。
- 两版首次 Director 请求正文相同的题数：128/128。
- 54项身份/提示/预算/环境测试通过；运行及请求审计问题：0 项。逐题 token 对账：baseline=True，candidate=True。
- 源码、配置、题集、实际模型请求/响应、环境交互、完整轨迹、评分、运行日志、离线 W&B 均保留。原参考运行不修改；baseline 是它的只读引用，独立归档使用解引用包含对应内容。
- 测试准备时缺少测试包依赖/配置夹具，补齐后54项通过；第一次评测启动因配置位置导致相对资产路径校验失败，尚未执行题目或调用模型，修正为 configs/ 下字节一致配置后从头运行，失败尝试单独保留。

## 可复核路径

- [完整对比指标](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/experiments/webshop-merged-identity-128-20260924-run1/comparison/comparison.json)
- [逐题配对CSV](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/experiments/webshop-merged-identity-128-20260924-run1/comparison/paired_tasks.csv)
- [身份访问审计](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/experiments/webshop-merged-identity-128-20260924-run1/comparison/identity_audit.json)
- [相对合并版的精确补丁](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/experiments/webshop-merged-identity-128-20260924-run1/provenance/change-from-merged.patch)
- [实际修复源码](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/experiments/webshop-merged-identity-128-20260924-run1/snapshots/candidate/src/selfplay_graph_flowsteer/runtime.py)
- [运行清单](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/experiments/webshop-merged-identity-128-20260924-run1/manifest.json)

## 后续审计补充

上文“访问状态误判为0”仅指搜索结果 inspection_status 字段。后续全量模型输入审计发现，裁剪后的记忆仍导致动作辅助判断矛盾：商品81次/22题、详情123次/48题，合计57题。不能将原结论扩大解释为整个记忆判断链路已正确；详见 [框架失败审计](WEBSHOP_FRAMEWORK_FAILURE_AUDIT_2026-09-24.zh-CN.md)。原封存运行及归档未修改。
