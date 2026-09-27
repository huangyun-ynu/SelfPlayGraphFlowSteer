**SWE 工具与测试环境修复，2026-09-27**

四项修复已完成，194 项代码回归全部通过，其中新增 26 项故障回归。另使用停止批次中四道题的原始 base_commit 完成真实工具与公开测试验证。实验保持停止，没有调用模型或远程官方测评；Director thinking、增量上下文与训练目标保持原样。

修改位于 `SelfPlayGraphFlowSteer-output-contract-v1` 的现有工作区。保留该工作区原有未提交改动，本次没有提交 Git commit。修改前文件快照和仅包含本次代码改动的补丁均已保存。

| 修复 | 修改后行为 |
| --- | --- |
| 大文件读取、搜索、编辑 | 新增 `swe.max_file_bytes=10485760`，以字节限制可处理文件；`max_file_chars=200000` 保留为编辑参数大小限制。读取最多 401 行，返回内容仍受 `max_output_chars=12000` 限制，完整文件哈希分块计算。搜索逐行执行，并明确报告跳过文件及原因。编辑后的文件按 UTF-8 字节检查上限，继续验证 SHA 与唯一匹配。 |
| sklearn 缺少 six | 0.20/0.22 共用配方加入 `six==1.16.0`，两个环境均重新准备并通过公开 smoke；配方指纹变化会使旧就绪记录失效。 |
| 空语法检查被当作已测试 | `python_syntax` 未指定目标时检查本轮新增/修改且仍存在的 `.py` 文件；无目标则返回明确错误。独立检查器编译源码并报告实际文件清单，不执行源码、不写 `.pyc`。就绪的仓库环境存在时使用其 Python。 |
| 根目录空路径 | `swe_list`、`swe_search` 接受 `""`、`"."`、`"./"` 等等价根目录写法，执行与去重使用同一规范化规则。读取、编辑仍要求具体路径，真实越界仍被拒绝；目录搜索同样检查符号链接越界。 |

语法检查与公开功能测试分别标记 `test_kind=syntax/public_tests`；`test_executed` 与 `test_passed` 分别记录执行事实和结果。修改后检查证据绑定工作区版本。启动失败、空命令、语法检查超时/证据缺失不能产生已测试标记；真实执行后的语法错误或公开测试失败仍按原提交策略处理。旧记录中无文件参数却标记成功执行的 `python_syntax` 结果不再被接受为有效执行证据。

**定向回归结果**

| 检查 | 结果 |
| --- | --- |
| 代码回归 | 194 passed，0 failed，0 skipped；包含新增 26 项，以及现有 SWE workspace/public tests/candidate integrity/execution admission/terminal repairs、runtime、application 测试。 |
| xarray-6992 | 原提交的 `xarray/core/dataset.py` 为 341,113 字节；定位 DataVariables、读取局部、核验完整 SHA、局部编辑、自动语法检查、导出补丁全部通过。 |
| matplotlib-24177 | 原提交的 `lib/matplotlib/axes/_axes.py` 为 323,004 字节；定位 hist、读取局部、核验完整 SHA、局部编辑、自动语法检查、导出补丁全部通过。 |
| django-11087 | 应用原来保存的 494 字节补丁；不传 target 调用 `python_syntax`，实际检查 `django/db/models/deletion.py` 并通过。 |
| scikit-learn-9288 | 应用原来保存的 1,879 字节补丁；`sklearn/cluster/tests/test_k_means.py` 收集并实际执行 123 个测试，123 passed，返回码 0；缺少 six 的错误消失。 |
| sklearn 环境 smoke | 0.20：7 passed、1 skipped；0.22：9 passed、1 skipped。各有一个可选 pandas 测试因未安装 pandas 而跳过；目标 k-means 测试没有跳过。 |
| 根目录去重与边界 | 等价根目录写法返回相同内容；交替使用不同写法仍触发重复调用保护；绝对路径、父目录、.git 与越界符号链接保持拒绝。 |
| 检查证据反例 | 验证无变更文件、启动失败、超时、旧参数错误、错误工作区版本不能被计为有效修改后语法检查；验证真正的语法错误记录为已检查但未通过。 |
| 清理与静态检查 | 四个真实验证工作区均已清理，0 个残留；新增代码/测试的 Ruff E/F 检查与相关文件的 `git diff --check` 通过。 |

两道大文件题使用临时注释验证工具操作能力，不代表生成或验证了题目修复。Django 检查为语法检查；sklearn 的 123 项为公开测试，不能据此宣称通过官方隐藏测评。

主要实现：[swebench.py](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer-output-contract-v1/src/selfplay_graph_flowsteer/swebench.py:1293)、[语法检查器](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer-output-contract-v1/src/selfplay_graph_flowsteer/_swe_syntax_probe.py)、[运行时证据判断](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer-output-contract-v1/src/selfplay_graph_flowsteer/runtime.py:4897)、[路径规范化](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer-output-contract-v1/src/selfplay_graph_flowsteer/swe_paths.py)、[依赖配方](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer-output-contract-v1/src/selfplay_graph_flowsteer/swe_public_recipes.py:112)、[新增回归测试](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer-output-contract-v1/tests/test_swe_tool_regressions.py)。

证据：[代码测试日志](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer-output-contract-v1/state/audits/swe-tool-regressions-20260927/pytest-focused.log)、[JUnit 结果](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer-output-contract-v1/state/audits/swe-tool-regressions-20260927/pytest.xml)、[四题定向结果](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer-output-contract-v1/state/audits/swe-tool-regressions-20260927/targeted-results.json)、[sklearn 目标测试输出](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer-output-contract-v1/state/audits/swe-tool-regressions-20260927/scikit-learn__scikit-learn-9288-test.json)、[本次增量补丁](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer-output-contract-v1/state/audits/swe-tool-regressions-20260927/implementation.patch)、[真实回归脚本](/workspace/h200-lab-7f3c/SelfPlayGraphFlowSteer-output-contract-v1/state/audits/swe-tool-regressions-20260927/run_targeted.py)。

Worker 单次请求额度、图修订后的候选结果恢复、GPT 上游错误分类不属于本次四项修复的范围。
