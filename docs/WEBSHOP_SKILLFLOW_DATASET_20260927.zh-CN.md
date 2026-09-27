> 清理完成：本次生成的SkillFlow WebShop专用数据、ADS缓存、临时合并训练池及重建/启动脚本已删除。下文路径仅为历史记录，当前入口已恢复原始数据。评测成绩和运行日志保留。

> 已归档：用户于2026-09-27要求恢复切换前的WebShop数据。当前训练512题、测试128题及实际训练池已恢复；下文仅记录SkillFlow试验历史。

# WebShop SkillFlow数据切换

SkillFlow-Dataset版本07bb38bcc62fa8bebab6af86c39ba23b0293c97d。采用其128条IID验证目标；训练保留其500条目标，以固定SHA256排序确定12条重复，保持512条训练规模。训练/验证goal_index及归一化题面均无重叠。没有官方难度标签，不声称按难度分层。

全部12087个human_goals与本地seed233打乱后的官方清单逐位置核对一致（忽略标点并移除环境追加的价格约束），所选628条与上游seed对应的human_goals逐字一致。因此seed到goal_index映射已经验证。模型任务prompt使用真实环境的完整instruction_text，含价格约束；不是直接用SkillFlow简化文本覆盖环境目标。

测试入口仍为data/formal/eval/webshop_official_test_128.jsonl，当前是SkillFlow IID子集，不再是原官方0-499中抽取的128题。训练入口data/formal/train/webshop.jsonl，实际七数据集池state/formal-data/validated_task_pool.jsonl已同步。scripts/formal/prepare_raw_pools.py已更新，防止重建旧划分。

训练ADS特征已用Qwen3.5-9B重算；训练原始输入及上游记录在data/formal/sources/skillflow/webshop_*。旧活动数据直接替换，历史实验结果保留。其他六个数据集不变。

128题重跑位于主工作目录state/audits/webshop-skillflow-20260927/run128，当前源码冻结，采用online-full128-v1的事实记忆配置：Qwen3.5-9B Director thinking开启，DeepSeek Worker thinking关闭，40并发、16动作上限、skill off、seed0，真实WebShop sidecar服务18020。最终成绩以该目录结果报告为准。

本次重跑已完成：成功50/128（39.06%），平均得分55.74/100（全部128题，1条未提交题按0计；仅127条已评分题均分56.17/100）。详见主目录 state/audits/webshop-skillflow-20260927/REPORT.zh-CN.md。
