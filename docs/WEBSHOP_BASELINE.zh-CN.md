# WebShop 当前正式训练版本：M02＋V3 engineering-20260930

当前唯一配置为 `configs/formal_training.toml`；入口 `scripts/formal/run_experiment.sh`。采用累积工程修复、自动记忆v2、购买预留v2、bounded_research_v1，原始官方题目与评分，全题16步。保留512条原始官方训练题及正式多模型路由。

本次固定DeepSeek Flash参考评测：128题中64满分、126购买、2未购买、0未提交；平均奖励0.731641。详见 [正式升级记录](WEBSHOP_ENGINEERING_FORMAL_PROMOTION_20260930.zh-CN.md) 和 [评测报告](../experiment_versions/reports/webshop-engineering-rerun-20260930/REPORT.zh-CN.md)。

旧版本选择信息保存在 `configs/history/webshop_before_engineering_20260930.json`；当前选择记录为 `configs/webshop_official_baseline.json`，它是版本说明，不是另一份执行配置。
