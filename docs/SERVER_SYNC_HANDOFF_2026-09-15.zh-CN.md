# 另一台服务器同步交接

更新时间：2026-09-15，Asia/Shanghai。
本文件不包含密钥，可以随 Git 分发。优先使用本文件描述的当前状态；旧迁移清单中的历史路径、提交号、开关状态和空间估算不能直接当作当前部署参数。

## 1. 同步目标与验收基线

- 仓库：https://github.com/huangyun-ynu/SelfPlayGraphFlowSteer
- 分支：`feat/pats-skill-scaffold`
- 已通过 CI 的代码基线：`8128a99897dd12fff4ee4ad8c64f4c9c5b25913d`。
- CI：https://github.com/huangyun-ynu/SelfPlayGraphFlowSteer/actions/runs/34976845359
- Python 3.11、3.12 两组 CI 均通过，包含完整测试与两个 mock 示例；3.12 为 856 passed、7 skipped。
- 源项目：`/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer`。

当前用户的下一步目标是：基础 Qwen 加各数据集的静态 Skill，做不更新参数的推理评测，先每个数据集一道题冒烟。实际可用基模型为 Qwen3.5-9B，不是先前口头描述的 8B。

**目前尚未实现“基础 Qwen + 静态 Skill”的独立推理入口。** `benchmark` 命令仍通过 AdaptiveApplication 的 Director/Worker 流程。不能将现有 `benchmark` 或正式自博弈训练误称为该基线。

真实模型七数据集端到端冒烟尚未完成；正式训练也未因本次同步启动。源服务器最近一次查询 8 张 H800 都有任务占用，目标服务器必须重新查询自身资源。

## 2. 获取代码

新目录执行：

```bash
git clone --branch feat/pats-skill-scaffold https://github.com/huangyun-ynu/SelfPlayGraphFlowSteer.git
cd SelfPlayGraphFlowSteer
git rev-parse HEAD
git merge-base --is-ancestor 8128a99897dd12fff4ee4ad8c64f4c9c5b25913d HEAD
```

已有工作区执行：

```bash
git status --short
git fetch origin feat/pats-skill-scaffold
git merge --ff-only origin/feat/pats-skill-scaffold
```

保留目标主机已有修改和运行结果。遇到分叉或未提交修改时先查看差异，不能使用 reset --hard 或覆盖目录来“同步”。GitHub 认证使用私密交接中的凭据，不能把 PAT 放进 remote URL。

## 3. 必须单独传输的文件

下表均为源服务器路径。它们不会通过 git clone 自动出现，需要用户通过 SSH/SCP/SFTP 等私密方式传到目标主机。

| 内容 | 源路径 | 目标用途 |
| --- | --- | --- |
| 私密配置备份 | `/home/bedicloud/sharestore2/iclr-users/2/SelfPlayGraphFlowSteer_PRIVATE_CONFIG_2026-09-15.md` | API keys、GitHub PAT、数据解密密钥、W&B 配置 |
| 当前环境文件 | `/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/.env` | 恢复到目标仓库 `.env`，合并目标已有配置 |
| SWE SSH 身份 | `/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/state/formal-training/private/swe/identity` | 同名目标相对路径 |
| SWE 主机信任 | `/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/state/formal-training/private/swe/known_hosts` | 同名目标相对路径 |
| SWE 注册表 | `/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/state/formal-training/private/swe/verifier-registry.json` | 恢复并校验路径 |
| Skill 全目录 | `/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/state/benchmark-skills/2026-09-15/` | 保留上游来源、适配版本、manifest 和隔离材料 |

其中可用 Skill 位于 `state/benchmark-skills/2026-09-15/adapted/project-v1`；不要把 `quarantine` 内容当作已适配技能加载。HealthBench 技能为本项目拟定，其余来源和适配情况以该目录 manifest/README 为准。

私密文件权限设置为 600，私密目录为 700。源私密配置备份在仓库之外；目标也应放在仓库外。Skill 资产在被 Git 忽略的 state 中，**本次更新 Git 并没有将它们上传**。

## 4. 环境与大体积资产

在目标机器新建 Python 3.11 环境，不要直接复制含旧 shebang 和绝对路径的 venv。

仅复现 CPU CI：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'torch>=2.6' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev,ads,tracking]'
pytest -q
```

GPU 推理/训练环境应另行按目标 CUDA、驱动与 vLLM 兼容性安装。CPU PyTorch 不能直接用于 GPU 推理。依赖组见 pyproject.toml：openai、train、serve、ads、alfworld、webshop、tracking。WebShop 建议维护独立环境。

标准资产布局：

```text
models/Qwen3.5-9B/
models/e5-base-v2/
assets/alfworld-data/json_2.1.1/
assets/webshop/source/
assets/webshop/venv/bin/python
assets/webshop/prepared/products.sqlite3
assets/webshop/prepared/goals.jsonl
assets/webshop/private-evaluation/
state/formal-training/private/swe/
```

WebShop 固定源码 revision：`64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd`，索引应位于源码的 `search_engine/indexes_100k`，另需 Java 11。源资产位置参考 MIGRATION_MANIFEST_2026-09-15.md，但目标不能保留指向另一台服务器本地盘的符号链接。

数据包在 Git 的 `data/formal/private/formal_data.tar.gz.enc`，解密密钥为私密环境中的 `SPGFS_DATA_ARCHIVE_KEY`。恢复资产与 `.env` 后：

```bash
scripts/formal/bootstrap_remote.sh
```

该脚本会向 state/formal-data 解密并写入数据，检查 3584 条七数据集 ADS 池及所需资产；如果已有同名数据，先核对用途再执行。它不会自动安装 Python 依赖、模型或环境资产。

`scripts/formal/update_remote.sh` = fetch + fast-forward merge + bootstrap，并非纯 git pull。不要把它用于覆盖正在运行中的数据目录。

## 5. 正式配置现状

权威入口：`configs/formal_training.toml`、`scripts/formal/environment.sh`、`scripts/formal/run_experiment.sh`，课程为 `configs/curriculum/formal_3500.toml`。旧 `state/pats-formal-20260912` 主要用于历史资产溯源。

- Director 看到 gpt、grok、gemini、deepseek、minimax 五个逻辑模型。
- GPT/Grok/Gemini 为双物理端点；DeepSeek/MiniMax 各单端点。
- GPT Worker 与 HealthBench Judge 共享 GPT 物理端点并发；每条 GPT 16、Grok/Gemini 每条 12、DeepSeek/MiniMax 各 30；Skill Refiner 20。
- 端点冷却 600 秒。路由 URL、模型与 reasoning 设置以正式 TOML 为准，凭据来自 `.env`。
- 主轨迹和关系反事实共享 35 槽，主轨迹优先；off/on 分别占槽。
- 图复验独立 8 槽，按数据集完成流水触发。
- 历史耗时优先级已开启；token/时间预测准入保护关闭，硬限额仍在。
- 正式入口启用 PATS、SkillBank、七数据集环境和三卡异步机制；两卡分别训练 Proposer/Solver，第三卡做下一周期异步采集。
- 正式脚本会更新模型参数，**不能直接用于当前不更新参数的基线冒烟**。

GPU 编号与 `[resources]`、SPGFS_ALLOWED_PHYSICAL_GPUS、角色 GPU 环境变量必须一致。先查询目标 GPU，不继承源机器的占用判断。

## 6. W&B 与网络

私密备份已经包含完整 W&B 恢复项，公开参数：

```dotenv
WANDB_ENTITY=yun-huang-yunnanuniversity
WANDB_PROJECT=selfplay-graph-flowsteer
WANDB_MODE=online
WANDB_BASE_URL=https://api.wandb.ai
SPGFS_WANDB_DNS_SERVER=8.8.8.8
```

源机实际上传及服务端回读通过：
https://wandb.ai/yun-huang-yunnanuniversity/selfplay-graph-flowsteer/runs/bb68aeb6c35248b9

正式入口套用 `scripts/formal/wandb_direct_exec.py`，通过私有 user/mount namespace 绑定临时 hosts，解决源主机 DNS stub 超时。目标需要 Linux unshare、mount 和允许非特权 user namespace；不保证任意目标主机都支持。该包装器只验证过 GPU 可见性与 W&B 上传，尚未完成三卡训练验证。

包装器在主程序加载 `.env` 前读取环境变量，因此自定义 WANDB_BASE_URL / SPGFS_WANDB_DNS_SERVER 要先 export。域名映射在进程启动时取得，长运行期间不会自动刷新。目标应先检查正常直连，再验证该包装方式。

模型 API 和 W&B 保持直连；代码/数据下载可以使用代理。无需把源机全局代理设置原样复制。

W&B 当前接在 selfplay-experiment；将来的独立静态 Skill 推理入口仍需接入上报器。公开指标说明见 docs/BENCHMARK_METRICS.zh-CN.md。

## 7. SWE 与评分口径

SWE 腾讯云：`sweeval@43.134.20.27:22`，实例 `ins-5n1zolfw`，新加坡，服务目录 `/data/swe-verifier`。新源 IP 必须获安全组放行。沿用已验证 known_hosts，不通过关闭主机校验来绕过错误。没有 CAM SecretId/SecretKey，因此尚不能自动开关机。

HotpotQA 按 SESA 的答案评测目标处理：不再补 supporting-facts 数据，不要求 joint 成绩。已实现的证据评分代码保留，缺金标准时为不可评分。SESA 主表是归一化 EM + 语义 Judge 的答案准确率，EM/F1 为诊断；本项目不能仅凭字段名称宣称完全复现了它的 Judge/子集/预算。

七数据集新增固定分母、已评分分母、按题平均、source bootstrap CI 和分组汇总。HealthBench 的训练奖励与官方长度修正成绩区分；当前 Judge GPT-5.5 与论文 GPT-5.4 low 不同，报告中应注明。

## 8. 交给目标服务器助手的执行指令

```text
读取本交接文档及用户单独提供的私密配置备份。
1. 检查目标工作区修改，fast-forward 同步 feat/pats-skill-scaffold，确认包含 8128a99。
2. 清点 .env、SWE SSH 文件、Skill 适配目录、Qwen/E5、ALFWorld 和 WebShop 资产。
   私密文件不得上传 Git；缺少来源文件时明确列出，不拿其他数据/模型替代。
3. 按本机 GPU/CUDA 建立环境，恢复便携路径，校验解密数据与正式配置。
4. 直连探测模型路由，重新生成新鲜报告；验证 SWE 通信、WebShop、ALFWorld、W&B。
   旧 route_report 不能代替目标网络环境的探测。
5. 实现并检查基础 Qwen3.5-9B + 数据集静态 Skill 的独立纯推理入口，
   每题一条轨迹，不更新参数。不要把 Director/多模型 Worker 自博弈当成此基线。
6. 每个数据集先固定一道题冒烟，记录实际加载 Skill、提交答案、评分、token、耗时、
   异常和 W&B 回读。HotpotQA 只要求答案评分。
7. 给出七项通过/失败清单、输出路径、缺失资产及待办。未经下一步指示不要启动长训练。
```

## 9. 同步完成标准

代码版本一致不代表运行环境一致。验收需要同时满足：凭据恢复、依赖可导入、模型/环境资产存在、正式数据校验通过、路由从目标可达、静态 Skill 已实际进入推理上下文、七数据集小样本评分完成、W&B 能回读指标。当前只完成了源机代码/CI与W&B连接验证，目标必须逐项补验。
