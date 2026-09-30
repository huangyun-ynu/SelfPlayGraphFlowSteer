# WebShop 正式训练升级：M02＋V3 engineering-20260930

已按用户确认，将真实128题重跑所用的累积修复版更新到主目录正式训练入口，并采用**原始官方题面与评分口径**。

唯一正式执行配置为 `configs/formal_training.toml`，入口为 `scripts/formal/run_experiment.sh`。本机 `configs/worker08-projects.sh main` 也指向该配置。历史评测TOML不作为额外正式训练配置；`configs/webshop_official_baseline.json`只记录当前版本来源。

## 生效内容

- 自动事实记忆 `factual_memory_v2`：完整公开事实存储、每轮6000字符自动展示；没有记忆读取工具或读取次数额度。
- 购买预留 `completion_reserve_v2`、调研调度 `bounded_research_v1`、V3提交，全题仍共享16步。
- 商品名称回退、余额/熔断恢复及此前累计修复全部保留；此次包含详情页预留误释放、Prev导航标记、错误可选计划恢复、十字符规格误识别修复。
- 核心runtime、application、记忆、预留、导航、原生生命周期和提交模块逐文件与真实评测源码一致。
- 正式训练保留Director自主选择Worker的多模型路由和PATS/SkillBank设置。本次固定DeepSeek Flash评测的路由限制不写进正式训练。

## 原始官方数据和服务

服务使用18020端口，goals为 `assets/webshop/prepared/goals.jsonl`，SHA256为 `630f68cb13f3ce9c1394a6b08c79947f973a91c1ba2dbe990334596f000f73dc`。

保留之前已完成的**512条原始官方训练题扩充**，使用 `state/formal-data/webshop-restored512-20260930-v1/validated_task_pool.jsonl`，七个数据集共3584题。训练题面和goal索引逐项核对通过，WebShop训练goal均属于训练split，与原始128题测试集goal ID无交集。质量改写前的512题ADS元数据与验证清单沿用，不重新计算或混用quality改写题面的特征。

侧边栏quality代码、数据与独立实验脚本保留。正式入口、本机项目环境变量和当前训练数据已切回原始官方版本，训练前会拒绝quality goals/任务，以及数据、端口或实现指纹不匹配的服务。新增服务实现指纹只用于健康检查，不改变购物观测或评分。

保留原始测试集和官方评分意味着此前分析的数据歧义仍然存在。本次参考评测为64/128满分、126购买、2未购买、0未提交，平均奖励0.731641；这是固定DeepSeek Flash的一次真实评测，不是正式多模型训练后的成绩。

## 验证

- **440项相关测试通过**，覆盖WebShop、runtime、统一提交、七数据集V3契约以及质量实验兼容路径。
- 正式应用创建、准备购买、阻止多余续跑、FINISH零Worker调用和训练反事实独立重放通过。旧“连续准备两个购买再选择提交”的测试显式覆盖旧调度器；新正式调度器在已有有效购买时要求提交，另有专门测试。
- 512条WebShop训练题通过公开题面/goal索引检查；3584题组合池通过ADS和验证manifest检查。
- 从本机项目切换脚本进入后，实际生效的配置、任务池和原始官方goals一致。
- 使用主目录新HTTP服务回放00358真实15步成功动作，购买成功、官方奖励1.0，与封存记录一致；没有新增模型请求，这不是新的准确率测试。探针服务已关闭。
- 升级范围外的原有861个公开文件按清单核对，未发现额外改动；quality实现和独立脚本原样保留。

结果见 [升级清单](../experiment_versions/promotions/webshop-engineering-20260930/manifest.json)、[文件保护](../experiment_versions/promotions/webshop-engineering-20260930/protection.json)、[测试日志](../experiment_versions/promotions/webshop-engineering-20260930/tests-final.log)、[真实服务验证](../experiment_versions/promotions/webshop-engineering-20260930/service-smoke.json)。

本次只更新正式版本，没有启动训练或改动模型权重。升级前文件保存在 `state/webshop-engineering-promotion-20260930-165411/before/`，本机项目切换脚本也有单独备份。最新真实评测源码和完整轨迹保存在隔离工作区的 `state/formal-eval/webshop-engineering-affected-128-20260930-164051`，详见 [评测报告](../experiment_versions/reports/webshop-engineering-rerun-20260930/REPORT.zh-CN.md)。
