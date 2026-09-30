# WebShop 工程缺陷修复与真实轨迹回归

本轮在隔离分支 `fix/webshop-memory-v2`、已有累积提交 `520deb3` 上修复完整128题审计确认的四类工程缺陷。保留此前候选名称、自动事实记忆、余额/熔断恢复、购买预算预留和调研分工修复。侧边栏其他工作目录未修改。

验证结果：**391项相关测试全部通过；27题中的33个预留误释放点全部修复；5个错误规格值均通过真实官方环境操作验证。** 本轮没有请求模型重新解题，没有新满分率或购买率。

## 修复内容

| 缺陷 | 原行为 | 修复后行为 |
|---|---|---|
| 详情页误释放购买预留 | 预算更新先于提示注释；详情页缺少返回标记，被判为购买路径不可达 | 用当前页面和公开动作推导导航语义；预算直接处理原始状态，也能保住合法预留 |
| Prev 的语义错误 | 商品页和详情页都被标成返回当前商品 | 商品页 Prev 返回搜索结果；详情页 Prev 返回商品；搜索结果的 Prev 表示结果翻页 |
| 错误可选计划阻断动作 | 可负担的搜索、打开、选规格等动作因附加计划无效而被拒绝，反复触发修复耗尽 | 在预留v2下忽略无效、不可达或负担不起的可选 reserve 更新；原动作继续接受正常合法性和预算检查；向当前Worker显示未应用计划的反馈 |
| 十字符规格值误识别为ASIN | terracotta 等被生成虚构商品打开动作 | 已知商品选项优先按规格解析；真实搜索结果中的ASIN仍生成打开商品动作 |

导航定义集中在 `src/selfplay_graph_flowsteer/webshop_navigation.py`，由 sidecar、runtime 和购买预留共同使用，避免三个位置各自理解返回动作。预算模块先复制输入再规范化，不要求外部先生成提示，也不会修改调用方的原始轨迹。

可选计划恢复只适用于合法非购买动作，且计划明确使用 `decision=reserve`。现有有效预留不会因坏更新被覆盖；实际预算不足仍拒绝。Buy 的计划与选项校验、明确 abandon、非法动作/目标、会话检查和旧预留v1的严格行为均保留。运行时不代选商品、规格或购买动作。

## 真实轨迹回归

来源为 `webshop-memory-auto-history-128-20260930-153948` 的真实公开轨迹。测试压缩夹具保留128题公开动作/观测及33个历史误释放检查点，记录原始 `records.jsonl` 的SHA256；没有用隐藏目标或离线最优答案指导新动作。

- **预留：33/33个检查点通过，覆盖27题。** 去掉所有后加的导航标记，再把原详情页观测交给修复后的预算模块；原计划保持、返回与购买成本可计算，未再触发误释放。该集合包含17道原满分题和10道原未满分题。
- **可选计划：覆盖7道原未购买题。** 00015、00221、00328、00358、00417、00433的代表性被拒检查点，原非购买动作在可负担时通过，错误计划不生效；00163只剩2步时仍拒绝最低需3步的搜索路径。另验证已有计划不能被坏更新覆盖或挤占。
- **完整运行时链路：00433的真实错误搜索请求只执行一次，动作额度只扣一次，下一次Worker请求可见计划忽略反馈。** 后续观测来自原真实轨迹；这里使用确定性MockBackend验证执行链，不产生新的模型成绩。
- **动作解析：5个真实错误值及真实ASIN/ASIN形状规格的边界全部通过。** 原始商品身份与规格选择的对应关系保持正确。
- **相关测试：391 passed。** 覆盖全部 `test_webshop*.py`、`test_runtime.py`、`test_unified_submission.py`，见 [tests.log](tests.log)。新增工程回归56项。

测试旧快照有2项需要更新。先用本次128题的冻结源码运行同一个探针，7项结果均匹配此前历史金标；逐字段核对后，新版仅在2个检查场景增加 `return_to_search` 注释及相应字符计数。其余5项快照保持一致。原金标文件保留，新增单独覆盖文件；见 [probe-diff.json](probe-diff.json) 和 [verify_probe.py](verify_probe.py)。

## 真实官方环境验证

使用真实官方WebShop环境、同一题目和seed=0，先执行真实轨迹的成功动作前缀，然后进行明确的工程检查动作：选择问题规格 → 打开Features → 返回商品页 → 返回搜索结果。每题4个检查动作，无模型调用、无Buy调用。

| 题号 | 商品 | 修复的规格值 | 原轨迹前缀动作数 | 结果 |
|---|---|---|---:|---|
| 00015 | B09MQLDRRL | terracotta | 6 | 通过 |
| 00234 | B005GSYPPW | lieutenant | 2 | 通过 |
| 00318 | B07PHVNWN9 | cappuccino | 2 | 通过 |
| 00358 | B01HEXEHWC | cantaloupe | 9 | 通过 |
| 00365 | B07MWSX8Z7 | cottonwood | 2 | 通过 |

5题均确认：规格能实际选中且商品身份不变；进入详情页后预留保持为返回+Buy共2步；返回商品页后选择保持、预留降为1步；商品页Prev实际回到搜索结果并清空选择。所有本轮启动的环境Worker均已关闭。结果见 [real-environment.json](real-environment.json)、[运行日志](real-environment.log)，脚本为 [real_environment.py](real_environment.py)。

## 范围与复现

本次代码增量只涉及 runtime、购买预留、sidecar，以及新增共享导航模块。模型、thinking、并发、正式配置和16步动作预算未调整，记忆读取方式未调整。基于的 `520deb3` 已包含相对于128题冻结源码的两项记忆保护修正（阻止跨会话旧记忆迁移、长详情页不能误报全文已展示）；本轮原样保留，未回退。

以上验证证明工程缺陷在这些真实状态和环境路径下得到修复。对原模型失败轨迹通过某个动作检查，不代表模型会继续选择正确商品，也不代表该题已购买或满分。仍需新模型评测测量购买率和准确率；本报告不把本轮工程探针记作失败题重试成绩。

在本工作目录执行：

```bash
PYTHONPATH=src ../.venvs/spgfs-pats-gpu/bin/python -m pytest tests/test_webshop*.py tests/test_runtime.py tests/test_unified_submission.py -q --disable-warnings
PYTHONPATH=src ../.venvs/spgfs-pats-gpu/bin/python experiment_versions/reports/webshop-engineering-fixes-20260930/real_environment.py
```

重新核对旧快照：

```bash
PYTHONPATH=state/formal-eval/webshop-memory-auto-history-128-20260930-153948/evaluated_source/src:. ../.venvs/spgfs-pats-gpu/bin/python -m scripts.formal.probe_webshop_legacy > experiment_versions/reports/webshop-engineering-fixes-20260930/probe-before.json
PYTHONPATH=src:. ../.venvs/spgfs-pats-gpu/bin/python -m scripts.formal.probe_webshop_legacy > experiment_versions/reports/webshop-engineering-fixes-20260930/probe-after.json
python3 experiment_versions/reports/webshop-engineering-fixes-20260930/verify_probe.py
```

输入来源、代码和夹具哈希见 [manifest.json](manifest.json)。
