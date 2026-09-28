# HotpotQA 86.72%版本同步到正式训练项目

用户明确要求将修订版HotpotQA严格EM 111/128（86.72%）对应的版本同步到正式训练版本。本次已在正式目录 `SelfPlayGraphFlowSteer` 完成同步。

## 同步内容

- 引入 `hotpot_answer_contract.py`，版本为 `hotpot_evidence_first_v1`。先输出证据和简短确定结论，再填写最终答案；最终答案对应原始完整问题、比较方向、限定条件和要求列出的全部成员。
- 在Worker普通输出、格式恢复、输出角色提示、答案提交提示四个位置接入同一契约。没有新增独立模型审核调用，也不使用参考答案修正输出。
- 保留正式目录现有的student动作、SWE和WebShop改动，通过局部补丁合入Hotpot分支。执行器版本同时标明student动作和Hotpot契约；训练执行语义清单加入新模块的源码哈希。
- 正式默认评测文件 `data/formal/eval/hotpotqa_official_test.jsonl` 已对应 `flowsteer-hotpotqa-corrected-v1`，本次确认与86.72%那轮冻结数据逐字节一致。
- 正式训练入口 `scripts/formal/run_experiment.sh` 已使用当前正式目录的 `src`，因此后续Hotpot训练rollout会使用新契约。

## 训练与评测配置

正式PATS继续使用原有多模型选择、技能学习、训练并发和训练任务池。此次晋升的是Hotpot处理逻辑及对应评测数据，未将整套正式训练配置替换成实验评测配置。

86.72%是以下冻结条件下的既有评测成绩：Qwen3.5-9B Director thinking on、提示词v2.2；DeepSeek Flash Worker thinking off；题目并发50、路由并发50；技能上下文off、检索off。该成绩不是正式多模型训练后的实测成绩。本次没有重新运行付费推理或启动训练。

用于复现的原始配置及完整运行清单已保留在：

- `experiment_versions/reports/hotpot128-aime30-c50-20260928-204627/config.toml`
- `experiment_versions/reports/hotpot128-aime30-c50-20260928-204627/manifest.json`
- [原始评测结果](HOTPOT128_AIME30_RESULT_20260928.zh-CN.md)

这些是原运行的冻结记录，保留当时的路径和资源绑定。以后启动复现仍应根据本机空闲GPU及新输出目录生成部署配置。

## 验证

1. 新模块SHA256与评测源码清单完全一致，来源提交为 `698ef78eb000876678dc3a72d6519da2a461babd`。
2. Hotpot普通Agent及输出Agent的普通输出/恢复提示词哈希，均与86.72%那轮运行清单相同。其余数据集的这些提示词哈希与同步前相同。
3. 173项针对性离线测试通过，包括完整问题可见性、参考答案不泄漏、格式恢复、提交边界、runtime、输出契约及修订数据构建。
4. 修订数据构建器 `--check` 通过，128题SHA256为 `6e4096785ad5b869b6541c868beb25ad7f157cc64ae8968c0db3cbe2f185e27c`。
5. 复核本机正式入口实际任务池 `state/formal-data/validated_task_pool.main-235e670.webshop-index-v2.worker08.jsonl`：共3516条，其中Hotpot512条。与修订版128题按原始来源ID、规范化完整prompt检查，交集均为0。没有将评测题放入训练池。
6. 同步前记录547个已有项目文件的哈希，核对只有本次预定的5个已有源码/测试文件发生变化；新增Hotpot模块及测试文件与来源一致。

测试命令：

```bash
python -m pytest tests/test_hotpot_answer_contract.py tests/test_runtime.py tests/test_output_contract.py tests/test_hotpot_corrections.py tests/test_prepare_static_eval.py tests/test_submission_grade_boundary.py tests/test_finish_submission_contract.py -q
```

## 可追溯与回退

- `experiment_versions/promotions/hotpot-8672-20260928/validation.json`：来源、源码/提示词/数据哈希、训练池检查及测试结果。
- `experiment_versions/promotions/hotpot-8672-20260928/promotion.patch`：仅包含此次代码与测试同步的独立补丁，已通过反向应用检查。
- `state/promotions/hotpot-8672-20260928/before/`：被修改文件同步前的完整备份，包含当时正式目录已有的其他任务改动。

初次同步仅写入正式本地工作目录，没有改变正在运行的实验副本。

## 合并正式发布

用户随后要求与 SWE 65/128（50.78%）版本一起上传 Git。发布版本保留本页所述
Hotpot 模块、四处提示接入、修订评测数据与来源记录；合并后的 Hotpot Worker/
恢复提示词哈希和数据 SHA256 再次与 86.72% 那轮核对一致。
SWE 单独使用 v3，Hotpot 与其他数据集保留 v2.2；混合批次按数据集检查协议。
联合验证与正式源码指纹见 [发布清单](../experiment_versions/reports/swe-65of128-20260928/promotion.json)。
