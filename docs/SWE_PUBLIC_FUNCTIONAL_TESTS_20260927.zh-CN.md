# SWE 公开功能测试

原先 `swe_test` 只有 `python_syntax`，只能发现 Python 语法错误。现在增加按公开任务的 `repo + version` 选择 Python、依赖和测试入口的功能测试。首次补齐时的 128 道任务覆盖 11 个仓库、49 个版本组合；环境验证按每个组合选一个公开的 base commit，不代表全部 128 个 commit 或完整测试套件都已通过。

## Worker 使用

初始工作区状态和 `swe_status` 返回 `test_profiles`、`public_test_environment`，其中包含环境是否准备好、runner 和冒烟目标。工具 schema 同时提供新 profile。

| profile | 用途 | target |
| --- | --- | --- |
| `public_tests` | 运行 Worker 根据公开源码选出的相关功能测试 | 必填；支持文件、目录，pytest 支持 `文件::类::方法` |
| `public_smoke` | 运行仓库预设的轻量功能测试，检查环境是否可运行 | 不传 |
| `python_syntax` | 保留原有语法检查 | Python 文件 |

示例：

```json
{"profile":"public_tests","target":"tests/test_basic.py::test_options_work","workspace_version":1}
```

Django 使用 `tests/runtests.py --settings=test_sqlite --parallel=1`，target 支持 `utils_tests.test_datastructures` 或 `tests/utils_tests/test_datastructures.py`。SymPy 使用原生 `bin/test -C --verbose`，target 是测试文件/目录。其余仓库使用各自环境里的 pytest。具体目标必须在当前公开 checkout 中存在。

冒烟测试用于确认运行能力，不能证明当前 issue 已修复。Worker 仍应通过 `public_tests` 选择与改动相关的公开测试。

## 覆盖范围

仓库为 Django、SymPy、pytest、Sphinx、Flask、Requests、Pylint、xarray、scikit-learn、Astropy、Matplotlib。具体 Python、依赖 pin 和 smoke target 保存在 `src/selfplay_graph_flowsteer/swe_public_recipes.py`。

runner 选择参考 [SWE-bench v3.0.9 的公开安装规格](https://github.com/SWE-bench/SWE-bench/blob/da5456ec492cb591be8d7d7d79bbd20870d6332f/swebench/harness/constants/python.py)，本地依赖以实际 base checkout 验证后的配置为准。这些是本地开发环境，不是官方评分镜像。

Matplotlib 使用系统 FreeType/Qhull；数值和功能测试可运行，但涉及字体像素基准的图像比较可能受 FreeType 版本影响。数据库、网络服务、TeX 等可选集成测试仍需要对应外部设施；默认 smoke 不依赖这些设施。

## 环境准备和复用

当前配置 `configs/.swe_gpu4_c10.toml` 和 `configs/formal_training.toml` 已设置：

```toml
public_test_environment_root = "state/swe/public-test-envs"
public_test_setup_timeout_s = 600.0
```

测试本身继续使用 `local_test_timeout_s = 60.0`。源码安装/编译和同环境锁等待有独立时限。一次 `swe_test` 仍只消耗原来的一个工具调用，整题总额度仍为 32。

准备脚本只接受公开身份字段，不读取 verifier registry 或隐藏测试。输入 JSON 是对象数组，每项至少有 `repo`、`version`、`base_commit`。准备命令：

```bash
UV_PYTHON_INSTALL_DIR="$PWD/state/swe/python" PYTHONPATH=src \
  ../SelfPlayGraphFlowSteer/.venv/bin/python scripts/formal/prepare_swe_public_tests.py \
  --tasks state/audits/swe-public-tests-20260927/public-tasks.json \
  --repo-cache state/audits/unified-protocol-20260926/shared/swe/repo-cache \
  --env-root state/swe/public-test-envs \
  --uv state/swe/tools/uv \
  --micromamba state/swe/tools/micromamba \
  --jobs 3
```

迁移到新机器时需要 uv、编译器，以及 Matplotlib 的 `libfreetype6-dev` / `libqhull-dev`。旧 Django 和 Astropy 环境由 micromamba 从 conda-forge 准备 Python 3.6，其余使用 uv 管理的 Python 3.9/3.10/3.11。准备阶段允许下载公开依赖，工具调用阶段离线安装当前源码。

每个环境保存 `ready.json`、`installed.txt`、`smoke.json` 和 `provision.log`。只有非空 smoke 实际运行且成功后才产生 ready 标记；配方指纹不一致、未知版本或环境缺失会明确报错，不会套用另一版本。需要重新验证时使用 `--recheck`，可用 `--repo` / `--version` 限定范围。

## 执行与证据

每次测试在锁内复制当前 Worker 工作区，重新安装该份源码并运行测试。安装、编译和测试输出都留在环境副本中，不污染导出的补丁；同一环境的并发测试被串行化。代价是包含原生扩展的仓库会有额外编译时间。

pytest 结果区分收集、开始执行、实际运行和 fixture setup 错误。未启动、环境安装失败、收集失败、空选择或全部 fixture setup 失败，均返回 `test_executed=false`。实际断言失败如实返回非零退出码，仍作为有效的失败测试证据。

同时修复了原来“调用被拒绝也算修改后测试”的提交判断：嵌套 error、缺少退出码、明确未执行以及工作区版本不匹配，都不能满足提交所需的测试证据。合法的测试失败仍可提交给正式验证器，不强制本地测试全部通过。

本次改动不修改 Director 的 append-only 增量上下文、训练 token/mask、thinking 配置或历史 full-v5 配置与成绩。

本次 11 个仓库、49 个版本组合的公开 smoke 均成功；迁移到持久化 Python 路径后，49 个环境也都通过当前源码导入检查。项目侧共 212 项不同的回归测试通过，其中本次新增 24 项。

验证明细见 `state/audits/swe-public-tests-20260927/verification.json`。

## SkillFlow 128 题数据集追加准备

更换后的数据集覆盖 10 个仓库、48 个版本组合；新增了 12 个版本环境：Requests 2.4、pytest 4.6、Sphinx 3.2/4.0/4.2/7.1、Pylint 2.10/2.14、xarray 2022.06、scikit-learn 0.21/1.3、Astropy 3.1。新增环境均在新数据集的公开 base checkout 上完成非空 smoke 测试。原有 55 个已定义配方的指纹保持不变，目前共定义 67 个配方。

scikit-learn 0.21 的选定旧提交包含不兼容 Python 3.9 的 vendored cloudpickle，因此使用独立 Python 3.6 环境；Sphinx 7.1 单独配置 flit-core 和较新的 docutils；pytest 4.6 补充 importlib-metadata。准备脚本会显式从本地缓存 fetch 指定提交，解决新下载的无引用提交未随 clone 复制的问题，不在此步骤访问外部 Git 服务。

本轮环境证据见 `state/audits/swe-skillflow-gpt128-c15-20260927/preparation/`；准备期间的项目测试为 `tests/test_swe_public_tests.py`，24 项通过。新运行仍采用题目并发 15、远程正式测评并发 4，公开测试不替代官方评分。
