# WebShop 记忆数据来源修复：128 题复测

新版本严格成功 **61/128（47.6562%）**；直接参考版本为合并提示 + W09 商品身份修复，**62/128（48.4375%）**。成功率变化 -0.7812 个百分点，平均 reward 变化 -0.053841。

## 修复范围

在上次 62/128 的冻结源码上，仅修改 runtime.py 中两个函数。搜索商品辅助判断优先读取压缩前生成的 inspection_status；详情辅助判断读取与当前 ASIN 一致的 decision_checkpoint.public_sections_observed，并核对动作目标的 ASIN。缺少可靠记录时返回未知，不将压缩列表缺失解释为未访问。详情标签区分“读过且文本保留”“读过但文本已淘汰”“未读”。当前规格选择仍取最新环境状态。

商品详细缓存仍为六个，进度投影仍为 6000 字符，详情原文仍按原规则保留。未新增全历史访问索引，缓存完全淘汰后的访问事实仍不能完整恢复；may_add_* 仍沿用访问布尔值的反向标志，不是新的证据价值估计。本轮验证的是数据来源一致性，不包含下一轮记忆扩容、购买策略或提示优化。

不改变 Director/Worker 的系统提示与规则文本、模型权重、推理强度、动作集合或预算；模型输入中的状态字段按修复结果更新。不使用 Skill、训练或隐藏目标。主工作目录已有其他实验修改，因此评测使用独立冻结源码，精确补丁另存。

## 完整指标

|指标|参考：合并 + 身份修复|新增：记忆来源修复|差值|
|---|---:|---:|---:|
|严格成功数|62|61|-1|
|严格成功率|0.484375|0.476562|-0.007812|
|平均 reward（0–1）|0.735221|0.681380|-0.053841|
|平均得分（100分制）|73.522135|68.138021|-5.384115|
|完成购买数|125|122|-3|
|未完成购买数|3|6|3|
|部分得分任务数|61|57|-4|
|零分任务数|5|10|5|
|平均 Worker token|64,824.546875|65,946.195312|1,121.648438|
|平均 Director token|18,519.468750|19,788.132812|1,268.664062|
|平均合计 token|83,344.015625|85,734.328125|2,390.312500|
|Worker 总 token|8,297,542|8,441,113|143,571|
|Director 总 token|2,370,492|2,532,881|162,389|
|合计总 token|10,668,034|10,973,994|305,960|
|平均动作数|7.453125|7.648438|0.195312|
|平均题目耗时（秒）|86.055620|86.262116|0.206496|
|整组墙钟时间（秒）|574.588527|680.922775|106.334248|

F1：不适用。官方 WebShop 评测不定义 F1，不将平均 reward 作为 F1。请求次数、输入/输出 token、缓存及 reasoning 用量可用性、耗时分布、逐题动作、失败类型和协议诊断均保存在 comparison.json。供应商用量与执行账本分开核对，不重复计入。

## 状态一致性是否修好

|实际 Worker 输入审计|参考版本|修复后|
|---|---:|---:|
|被审计的 Worker 状态数|982|1018|
|商品访问状态检查数|2940|2940|
|详情访问状态检查数|1284|1314|
|商品标记与辅助判断矛盾次数|81|0|
|详情记录与辅助判断矛盾次数|123|0|
|详情保留状态标签矛盾次数|0|0|
|存在矛盾的题数|57|0|

离线验证还将修复函数应用于参考版本的 982 次已保存 Worker 输入：商品检查 2940 次、详情检查 1284 次，未发现辅助字段与已有事实矛盾，动作目标集合及输入对象保持不变。该重放不调用模型，也不模拟成绩。实际新版输入审计覆盖本轮所有 128 题；两个版本的请求数量随轨迹变化，因此矛盾次数不作为等长采样下的概率比较。审计仅覆盖现有运行时保留的事实，不宣称恢复六商品窗口之外的全部历史。

## 配对结果及解释边界

两版均成功 54 题；新增成功 7 题；原成功转失败 8 题；两版均失败 59 题。McNemar 精确双侧 p=1.000000。
配对 bootstrap 10000 次：成功率差值 95% 区间为 [-6.25, 5.4688] 个百分点，平均 reward 差值区间为 [-0.10227864583333332, -0.0045556640625000235]。

本轮新增一组 128 题，直接参考此前保留的 62/128 运行，没有同步重跑参考组。更早的纯合并版 66/128、平均 reward 0.723359375 仅作辅助参考。相同开发题和单个 seed 不能证明泛化或稳定提升；模型采样、外部服务与缓存可能波动，逐题胜负变化不能全部确定归因于此次修复。

## 公平性与验证

- 同一 128 题，seed 0，24 路轨迹，Worker 接口并发 20；配置原文和解析后的配置均相同，源代码仅上述两个函数发生变化。
- 同一个常驻 Qwen3.5-9B 基础 Director，thinking 开启；DeepSeek deepseek-flash Worker，thinking 开启，reasoning_effort=low。动作预算 12/4/16，Worker token 上限 350000，请求预测预算拦截关闭。
- 两版首次 Director 请求正文相同：128/128；两版逐题 token 对账：True / True。
- 运行与请求完整性审计问题 0 项；新版商品身份访问审计不一致题数 0。
- 62 项适用回归测试通过，包含新增 9 项；新测试中 8 项在冻结参考代码上失败、修复后通过。另一个复制来的旧 sidecar 测试依赖后续接口字段，在两份冻结源码上均失败，已明确排除并保留完整失败日志，生产 sidecar 未修改。
- 首次本地 pytest 因仓库 pythonpath 设置误加载主目录源码，纠正为显式冻结路径，并加测试会话导入路径断言后重跑；相关日志全部保存。测试准备期间没有调用模型或执行评测题目。
- 原参考目录与旧归档保持不变；本轮保存完整请求响应、环境调用、轨迹、代码快照、配置、评分、日志和离线 W&B，归档包含解引用的参考数据及逐文件 SHA256。
- Director 保持运行（PID 130255，端口 18603），按用户要求供后续继续复用。

## 本轮判断

此次修复消除了已定位的投影状态矛盾，但本轮成绩没有改善：严格成功少 1 题，平均 reward 下降约 0.05384，未完成购买从 3 题增至 6 题。不能将其认定为更强的性能基线，也不能把此前受矛盾影响的失败题等同于可直接挽回的题。

按参考轨迹预先划分：存在矛盾的 57 题，两版各成功 24 题，平均 reward 从约 0.69766 降到 0.61287；其余 71 题，成功数从 38 降到 37。完整分组与逐题变化见 affected-strata.json。分组由参考版本的动作轨迹决定，这只是描述性分析，不是因果估计。

## 复核入口

- [完整对比指标](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/experiments/webshop-memory-source-128-20260924-run1/comparison/comparison.json)
- [逐题配对 CSV](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/experiments/webshop-memory-source-128-20260924-run1/comparison/paired_tasks.csv)
- [新版状态一致性审计](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/experiments/webshop-memory-source-128-20260924-run1/comparison/candidate_memory_audit.json)
- [原版状态一致性审计](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/experiments/webshop-memory-source-128-20260924-run1/comparison/baseline_memory_audit.json)
- [受影响题目分组对比](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/experiments/webshop-memory-source-128-20260924-run1/comparison/affected-strata.json)
- [精确补丁](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/experiments/webshop-memory-source-128-20260924-run1/provenance/change-from-identity.patch)
- [实际修复源码](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/experiments/webshop-memory-source-128-20260924-run1/snapshots/candidate/src/selfplay_graph_flowsteer/runtime.py)
- [回归测试](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/experiments/webshop-memory-source-128-20260924-run1/tests/test_webshop_memory_source.py)
- [运行清单](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer/state/experiments/webshop-memory-source-128-20260924-run1/manifest.json)
