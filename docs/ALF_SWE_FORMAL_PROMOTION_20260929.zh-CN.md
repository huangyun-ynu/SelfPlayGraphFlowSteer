# ALFWorld / SWE 修复版正式训练同步

2026-09-29，按用户“都要更新为正式训练版本”的要求，将 **458,339 Worker tokens**
那次真实失败轨迹验证对应的 ALFWorld 和 SWE 代码共同纳入正式 `main`。
版本标识：`alf-swe-statefix-20260929`。

## 来源与成绩口径

| 数据集 | 代码与验证来源 | 历史完整评测 | 后续真实回归 |
| --- | --- | --- | --- |
| ALFWorld | 修复 `478b38f`，验证归档 `4837fbc` | V3 102/128（79.6875%） | 2 题合法提交，1 成功、1 环境步数耗尽失败 |
| SWE | 修复 `10a144c`，验证归档 `8ac3ab6` | 65/128（50.78125%） | `django__django-13786` 官方通过 |

原始回归记录见 [真实失败轨迹验证](../experiment_versions/reports/real-failure-state-fix-20260929/README.zh-CN.md)。
458,339 是那次 3 题验证的 Worker 用量；此次同步只做离线验证，没有新增付费推理或训练。
历史 128 题成绩不能视为此次合并后的新实测成绩。

## 正式入口与策略

- 正式训练：`scripts/formal/run_experiment.sh` → `configs/formal_training.toml` → 当前 checkout 源码。
- 本机正式配置：`configs/formal_eval_worker08_main_v22.local.toml`（本机保留，不提交凭据或本机配置）。
- ALFWorld 与 SWE 均按数据集选择 `unified_task_result_v1`，Director/PATS 使用 v3；默认仍为 v2.2。
- ALFWorld 使用 `environment_reset` 的准确公开题目，Director、Worker、Skill 检索和反事实分支使用同一题目；原数据题干保留作溯源。
- 两者均使用整条题目轨迹的实际 Worker usage 账本，发送阈值 350,000；各 Agent、重试、修订共享账户。最后一个已获准请求可能使累计用量超过阈值。
- ALFWorld 开启 `finish_only_v1`：当前可信成功候选只允许提交；预算耗尽时仅允许已列出的零 Worker 清理删除。
- 已验证的最后一轮 FINISH、删除待配置模型节点后的状态清理，以及账本异常关闭修复一同生效。
- 新提交/用量目录：`state/formal-training-alf-swe-statefix-v1/submissions`，旧目录保留归档。

正式训练保留已有多模型选择、Skill、24 条滚动轨迹和训练资源配置。
ALF 评测的固定 DeepSeek、40 题并发，以及 SWE 回归的临时推理端口不写成训练默认。
SWE 正式 Worker 池仍为 GPT-student / 普通 GPT；远程环境继续采用 `STOP_CHARGING`。

## 训练分支接入

ALF 与 SWE 的 V3 覆盖同时进入混合训练协议指纹、PATS 分域审核和批次协议校验。
旧 ALF legacy 提交记录不能作为新协议的训练样本混入。

关系反事实执行使用独立环境和独立 usage 账户，不复用主轨迹账本或把额度平均分到节点。
未知/未结算用量不能生成反事实训练分；中断时关闭预创建环境、SQLite 连接和文件锁。
已获准请求跨越用量阈值、且得到可信终局时，按实际发送策略校验。

## 归档与复现

更新前正式提交为 `8ac3ab6369daa46d4b1874b647033e6ab3a07b2c`，
归档标签为 `archive/formal-before-alf-swe-statefix-20260929`。
本机更新前文件、三方合并输入和离线日志保存在：

```text
state/formal-training-promotions/alf-swe-statefix-20260929/
```

可提交的版本清单、源码指纹、测试结果位于
[同步记录](../experiment_versions/promotions/alf-swe-statefix-20260929/promotion.json)。
历史实验快照与原始轨迹不改写；当前 WebShop 独立实验副本不属于此次同步。

针对性离线回归 **421 通过、1 项排除**。排除项依赖本机缺失的旧 AIME16 原始轨迹，
未算作通过；正式训练与本机配置均通过完整加载校验。
较广检查另覆盖了训练编排：532 通过，版本号断言与旧 ALF 测试替身的 3 项失败
已修正并纳入上述通过结果；剩余 1 项即缺失的历史轨迹。

本地正式版本与远端发布是独立状态。GitHub 推送需本机可用的写入认证；
是否完成发布以实际 push 结果和远端提交核验为准。
