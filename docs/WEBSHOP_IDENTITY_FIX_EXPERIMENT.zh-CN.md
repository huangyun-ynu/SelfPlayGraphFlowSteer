# WebShop legacy 商品身份修复与 128 题实验

2026-09-23，按用户要求修复 legacy 观察移除 ASIN 后，访问状态、候选记忆和动作
辅助逻辑仍依赖该字段的问题，并与刚完成的预算拦截关闭版本进行配对比较。

**结论：身份错误已消除，但这次实验未提高准确率。** 严格成功从 63/128 变为
61/128，平均得分从 71.0221 变为 70.7552；平均 Worker token 增加 6.62%，
Worker 加 Qwen 的平均合计 token 增加 2.91%。不将本轮结果替换为历史正式基线。

## 实现与验证

新增纯函数 `webshop_identity.visible_product_asin`：优先使用已有结构化 ASIN；
legacy 动作没有该字段时，仅从当前公开的合法动作
`open_product:<页面序号>:<10 位 ASIN>` 恢复身份。身份不依赖页面序号，不从商品
标题、购物请求或商品数据库猜测。不修改原动作字典。

以下路径使用同一个身份解析函数：

1. 搜索结果的 inspection_status、visit_count 和 candidate_coverage。
2. 候选商品记忆的出现次数、已访问状态和已有商品页证据关联。
3. action_decision_support 的 already_inspected 和 may_add_product_page_evidence。
4. 最近商品动作的语义记忆，保留 ASIN，仍不保留可复用的页面局部 target_id。
5. 现有的商品动作序号纠正：仅当同一 ASIN 对应唯一当前合法动作时纠正；不创建
   新动作、不选择其他商品、歧义时不纠正。

`_legacy_webshop_subaction` 未修改。搜索动作没有重新加入 asin、title、price 字段，
页面正文、动作顺序和动作集合保持原格式。恢复的商品页证据来自此前实际查看结果。
六项商品检查记忆、十二项候选记忆的容量沿用旧值。

12 项身份专项用例覆盖大小写、页面序号变化、同名不同商品、无效 ID、持久化记忆、
已访问/新商品区分、原观察字段不恢复，以及合法目标纠正的唯一性。连同预算开关、
Worker runtime、环境反馈、LASER 提示及 sidecar 测试，共 109 项通过，lint 通过。
原审计复现脚本已将身份问题改为验证正确行为，其他四个审计问题保持原状；历史
`reproductions.json` 未覆盖，新验证结果保存于本次实验目录。

## 运行身份与控制变量

- 对照：`state/formal-eval/webshop-budget-off-laser-c24-20260923-191922/`。
- 新运行：`state/formal-eval/webshop-identity-fix-c24-20260923-202622/`。
- 审计：`state/experiments/webshop-identity-fix-c24-20260923-202622/`。
- 同一批官方 128 题，seed 0，24 路轨迹；DeepSeek Worker 服务并发 20。
- 复用 Qwen3.5-9B Director，常规 thinking 开启，关系二选一按原设置关闭 thinking。
- DeepSeek `deepseek-flash` Worker，thinking 开启、reasoning effort low、max_tokens 16384。
- legacy 页面、LASER 页面核对提示，无 skill、环境反馈关闭。
- token 预测准入和请求拦截继续关闭，实际 Worker token 总上限 350000 仍在
  执行报告返回后核验；动作预算仍为 12+4=16。

相对对照启动快照，仅 runtime.py、webshop.py 两个运行源码文件修改，新增
webshop_identity.py；配置、数据、模型设置和入口脚本哈希一致。
`change_from_baseline.patch` 保存完整源码差异。结束检查未发现运行期间源码漂移。

## 准确率、分数与 token

128/128 完成，任务级运行异常 0，退出码 0。

| 指标 | 修复前：预算关闭版 | 身份修复后 | 变化 |
| --- | ---: | ---: | ---: |
| 严格成功 | 63/128 | 61/128 | -2 题 |
| 严格成功率 | 49.2188% | 47.6563% | -1.5625 个百分点 |
| 平均得分（百分制） | 71.0221 | 70.7552 | -0.2669 |
| 平均 Worker token | 68111.69 | 72622.79 | +4511.10（+6.62%） |
| 平均 Qwen Director token | 41688.06 | 40371.02 | -1317.05（-3.16%） |
| 平均合计 token | 109799.75 | 112993.80 | +3194.05（+2.91%） |
| 128 题合计 token | 14054368 | 14463207 | +408839 |
| 平均题目耗时（秒） | 105.03 | 103.86 | -1.17 |
| 请求预算拒绝 | 0 | 0 | 0 |

逐题：55 题两次均成功、59 题两次均失败、6 题失败变成功、8 题成功变失败。
配对 McNemar 精确双侧检验 p=0.7905。不能根据一次 128 题评测认定稳定退化，
也不能将这次修复宣传为已验证的准确率提升。

Worker 输入 token 从 8031953 增至 8572104（+540151），输出从 686343 增至
723613（+37270）。新增候选记忆可能增加输入长度，但成功请求次数也由 1198
变为 1222，不能把全部 token 增量归因于新增字段。Qwen 输入从 5100248 降至
4930822，输出从 235824 变为 236668。

Worker token 按去重执行报告累计，逐题与 Canvas 账本核对；两组均无不一致，且与
各自 records.jsonl 的 token_cost 一致。Qwen 按请求 event_id 去重累计
completion_usage，包含 thinking 与动作生成；两组均无缺失用量的成功请求。

## 修复是否实际生效

| 轨迹审计项 | 修复前 | 修复后 |
| --- | ---: | ---: |
| 已保留访问记录的商品被标为未访问 | 136 次，45 题 | 0 |
| 当前记忆中的商品再次出现在搜索结果 | 136 次 | 134 次，全部正确标记 |
| 重复打开同一商品 | 44 次，28 题 | 29 次，23 题 |
| 商品动作记忆缺失 ASIN | 213/213 条 | 0/211 条 |
| 搜索动作本体不含 asin 字段 | 3270/3270 | 3070/3070 |

访问标记审计按成功观察到的商品页顺序、每个 Agent 最近六项商品记忆检查，不把
容量淘汰后的旧访问误判为身份错误。动作记忆条数按保留的执行记忆快照计数。
重复打开可能是返回购买或补查信息，不能把减少的 15 次全部视为消除无效动作。
候选记忆中带搜索出现次数的记录从 0 变为 1672 项（按执行快照计数，非独立商品数）。

本轮 goal-00082 在 Director 阶段因连续 responsibility_violation 而未进入 Worker，
得分 0；对照为 2/3。该题没有执行到商品身份修复逻辑，不能将这项变化直接归因于
本次修复。其余 127 题均进入 Worker，且 LASER 提示按预期生效。

所有 172 份 Worker 执行记录均走 DeepSeek，预算开关为 false，无活动请求额度、
无请求预算报价、无预算拒绝。635 次常规 Qwen graph_action 的 requested/effective
thinking 均为 true，13 次关系二选一为 false。未记录到 Worker 请求失败事件。

## 保留状态与产物

Qwen API server PID 37099、engine PID 39080 继续运行，模型和 WebShop 健康检查
均通过。GPU 1 的显存占位 PID 19576 继续保留约 2.25 GiB，另留约 1 GiB 运行余量。
本次未停止模型、释放占位或覆盖历史实验。

- `budget_comparison.json`、`budget_paired_tasks.csv`：通用比较脚本生成的指标及逐题对照。
- `identity_audit.json`：身份、候选记忆与重复打开审计。
- `changed_tasks.json`：严格成功状态变化的 14 题。
- `guidance_audit.json`：模型路由、提示和 thinking 审计。
- `offline_reproductions.json`、`offline_checks.log`：修复后的离线验证。
- `experiment_manifest.json`、`post_run_integrity.json`、`retained_services.json`：
  配置、源码、数据及部署保留证据。

修复代码保留在当前工作区，用户指定关闭的 token 预算拦截继续关闭。正式历史参考
仍指向原 62/128 运行，刚才 63/128 的预算关闭实验也完整保留。
