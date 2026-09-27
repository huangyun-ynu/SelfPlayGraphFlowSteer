# WebShop 正式训练版本：M02

2026-09-25 按用户指定，将 **M02（合并提示＋商品身份/访问判断修复）** 更新为正式训练采用的 WebShop 实现。它的历史128题结果是 **62/128，EM 48.4375%，平均分73.5221354/100**，平均reward为0.7352213542；WebShop没有官方F1。

这不是早期同为62/128、平均分69.0625的W05，也不是63/128的W08。此前的W08选择保存在[历史选择记录](../configs/history/webshop_w08_budget_off_20260924.json)。各版本成绩不覆盖。

## 正式训练采用的设置

- [正式配置](../configs/formal_training.toml)及两个H200训练/评测配置启用`m02_merged_identity_v1`兼容配置。
- 保留M02合并Worker提示、W09内部ASIN识别及访问判断修复、legacy观察、单会话所有者、暂存购买及SET_OUTPUT提交。
- factual memory；恢复M02每段详情1400字符、最多6个商品的原有边界。未加入后续M03记忆来源修复、W10详情扩容、W11/Native执行方式或S01人工Skill卡。
- 初始12次动作、修订4次、总16次；Worker总token预算350000；请求token预测拦截关闭。
- Qwen thinking开启。Director选择逻辑模型`gpt/grok/gemini/deepseek/minimax`；程序负责物理接口选择，Director不选择接口URL、凭据或池成员。
- GPT池含3个接口，Grok/Gemini池各2个接口；保留跨实例轮换、失败切换、0.5秒成员排队上限。已经发出的HTTP请求仍受请求超时控制，未增加并行竞速请求。
- 保留正式训练既有PATS/SkillBank学习管线；历史M02参考评测无Skill，不把该参考成绩冒充动态模型选择或训练后的成绩。

## 历史评测与正式训练的区别

历史M02使用Qwen3.5-9B Director、固定DeepSeek Worker、双方thinking开启、DeepSeek low、无Director Skill、24题并发、seed 0。封存证据在`state/experiments/webshop-merged-identity-128-20260924-run1/`。

[参考评测配置](../configs/webshop_official_eval.toml)及[评测入口](../scripts/formal/run_webshop_official_eval.sh)保留固定DeepSeek无Skill设置，用于后续对照；正式训练入口是[run_experiment.sh](../scripts/formal/run_experiment.sh)，采用上面的模型选择与接口池策略。

本次没有启动训练、重跑128题或操作Director服务。历史分数只属于原封存运行；当前正式训练组合需要另行记录实测成绩。详见[同步与验证记录](WEBSHOP_FORMAL_PROMOTION_2026-09-25.zh-CN.md)及[机器可读选择记录](../configs/webshop_official_baseline.json)。
