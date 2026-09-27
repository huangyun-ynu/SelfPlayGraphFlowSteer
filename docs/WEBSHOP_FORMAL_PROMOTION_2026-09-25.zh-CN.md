# WebShop M02 正式训练同步记录

用户指定M02（合并提示＋身份修复）作为正式训练版本，并要求保留Director选择模型、程序选择物理接口、Qwen thinking开启的训练架构。

## 来源与成绩

- 封存运行：`state/experiments/webshop-merged-identity-128-20260924-run1/candidate`。
- 封存源码：同目录上一级`snapshots/candidate/src`；90个Python文件SHA256保存于`state/formal-training/webshop-m02-promotion-20260925/frozen-source-sha256.json`。
- 原始128题：62题完全正确，EM 48.4375%，平均reward 0.7352213541666667，平均分73.52213541666667，125题完成购买、3题未购买。
- 来源选择器：`comparison/comparison.json → candidate.metrics`。这组成绩使用固定DeepSeek、无Skill；不代表正式训练的动态模型选择结果。

## 实际代码接入

`application.py`解析并校验`webshop.compatibility_profile`，将其传给WebShop lifecycle、普通/路由Worker执行器。`runtime.py`在初次执行、Action更新和同所有者修订中按M02恢复每段详情1400字符的边界。默认`current`实验配置仍支持后续详情扩容。

`webshop.py`在M02观察中移除后续sidecar加入的`raw_available_actions`和`raw_action`，保留M02公开动作、legacy页面与内部商品身份。Native实验仍可读取其原生动作元数据。

合并提示和商品身份解析模块与M02封存源码相同。正式配置选择graph_tools/factual_memory/merged_checklist，防止混入Native、history或其他提示策略。M03及后续讨论的记忆来源修复不在本次选择范围。

## 模型与接口的职责边界

Director只从逻辑模型候选中执行SET_MODEL；协议字段仍沿用`runtime_route`这个历史名字，其含义是逻辑模型标识。正式WebShop候选为GPT、Grok、Gemini、DeepSeek、MiniMax；`gpt_eco/gpt_student/grok45/gemini2`不是Director候选。

| Director选择 | 程序控制的接口成员 |
|---|---|
| gpt | gpt、gpt_eco、gpt_student |
| grok | grok、grok45 |
| gemini | gemini、gemini2 |
| deepseek | 当前正式配置只有deepseek一个接口 |
| minimax | 当前正式配置只有minimax一个接口 |

原程序已具备池机制；本次保留GPT池，并把已有Grok/Gemini接口加入各自的池。模型候选、接口池和最终物理调用分别记录，不能把接口轮换当作Director改选模型。

`EndpointPoolBackend.generate`通过共享计数器和文件锁轮换首选成员。成员本地并发队列最多等待0.5秒，队列饱和或可切换的接口错误会立即尝试下一成员；池内关闭单接口嵌套重试，避免在一个失败接口反复等待。配置`pool_retry_attempts=2`表示最多初始一轮加两轮重试，仍受请求和任务期限约束。

边界：0.5秒针对**本地队列**，不针对已经发出的HTTP请求。后者没有“到0.5秒就并行请求备用接口”的竞速机制；若服务端无响应，仍可能等到该请求超时。耗尽共享请求预算时，也不保证还有时间尝试后续接口。本次没有改变这一语义。

Qwen Proposer/Solver的`enable_thinking=true`沿用正式训练的当前配置；WebShop不设置thinking关闭覆盖。正式PATS、优化器和SkillBank管线保持各自训练职责，没有导入S01的12条人工编排卡。

## 验证与产物

离线测试覆盖模型候选可见性、Director真实请求的thinking标志、选中GPT/DeepSeek后的实际后端调用、M02兼容配置传递、1400字符边界、Native字段隔离、接口池轮换与排队切换，以及原有WebShop/QA/HealthBench回归。测试输出保存在`state/formal-training/webshop-m02-promotion-20260925/`，最终验证状态见该目录`verification.json`。

同目录保留变更前配置/源码、冻结源码哈希、离线Worker请求对照脚本及输出、测试日志、最终变更文件哈希和补丁。合成测试只验证实现行为；没有使用新的真实模型调用，也没有产生新的128题EM。

正式训练入口仍为`scripts/formal/run_experiment.sh`。本次完成版本和配置同步，未启动训练、更新权重、重新评测或停止Director。
