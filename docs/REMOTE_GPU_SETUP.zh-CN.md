# 远程 GPU 主机部署

## Git 可自动取得的内容

- 最新项目源码与测试。
- 无密钥的正式路由、PATS、并发和数据集配置。
- 加密的七数据集 3,584 条 ADS 训练池。
- 可以公开分发的测试子集和 split manifest。

HealthBench Professional 要求不要公开样本，因此正式合并池在 Git 中只
保存加密版本。API 密钥、GitHub PAT、数据解密密钥和 SWE SSH 私钥均不
进入 Git 历史。

## 远程主机操作

```bash
git clone --branch feat/pats-skill-scaffold \
  https://github.com/huangyun-ynu/SelfPlayGraphFlowSteer.git
cd SelfPlayGraphFlowSteer

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[openai,train,serve,ads,alfworld,webshop,dev]'
```

将私密配置文档中的环境变量写入仓库根目录 `.env`，将 SWE 私钥和
`known_hosts` 放入 `state/formal-training/private/swe/`，然后执行：

```bash
scripts/formal/bootstrap_remote.sh
```

脚本会校验加密包、解密正式任务池、校验 3,584 条 ADS 记录并检查 TOML。
模型、E5、ALFWorld、WebShop 运行资产仍需按输出提示放在标准目录；这些
文件体积过大，不能由普通 Git 仓库存储。

以后自动拉取最新代码、配置和加密数据包，直接运行：

```bash
scripts/formal/update_remote.sh
```

该命令只接受 fast-forward 更新，不会覆盖远端主机上的本地修改或 `.env`。

训练前必须在远程主机直连探测所有模型路由并生成 30 分钟内的新鲜
`state/formal-training/route_report.json`。模型 API 禁止使用代理。

```bash
export SPGFS_ALLOWED_PHYSICAL_GPUS=0,1,2
scripts/formal/run_experiment.sh
```

默认使用 GPU 0、1 训练 Proposer/Solver，GPU 2 使用冻结策略和维护后的
PATS 快照异步采集下一 cycle；如果目标主机使用其他编号，需要同步修改
`configs/formal_training.toml` 的 `[resources]` 和
`SPGFS_ALLOWED_PHYSICAL_GPUS`，两者必须一致。

正式入口已经开启 PATS、Director SkillBank、AIME Action、WebShop、
ALFWorld、SWE、按数据集图复验流水线、Proposer/Solver 并行训练、PATS
版本绑定的下一 cycle 异步采集和历史耗时优先级。具体并发数以所选 TOML
和命令行 `--workers` 为准，不要把历史实验中的槽位数字当成当前默认值。

当前配置的关键行为：

- `formal_training.toml` 和 GPU 4 评测配置使用 `gpt` 逻辑路由，池成员为
  `gpt` 与 `gpt_eco`，两者共用 `NEXUS_API_KEY`。
- 每个池成员最多等待本地并发槽位 0.5 秒；槽位仍满时立即切换另一个成员，
  该队列超时不计入持久路由熔断。
- WebShop 正式配置使用 `search_observation_mode = "retain_page_text"`、
  `max_observation_chars = 0`，即保留完整页面文本；`.alfworld_webshop_gpu4.toml`
  是兼容旧协议的 legacy 配置，使用前要明确选择它。
- NQ-open 和 HotpotQA 使用配置中的 Search-R1 检索服务；启动前必须验证
  Wiki-18 语料、E5/FAISS 索引和服务身份。
- Qwen3.5 Director 推理评测若遇到 reasoning 通道有内容但 action 通道为空，
  可设置 `SPGFS_QWEN_DIRECTOR_ACTION_RETRY=1` 启用一次短 JSON action 补发。
  该补发轨迹保留推理审计，但不作为连续的可训练 token 轨迹。

GPU 4 的 SWE 10 并发评测配置示例：

```bash
export SPGFS_ALLOWED_PHYSICAL_GPUS=4
export CUDA_VISIBLE_DEVICES=4
export SPGFS_QWEN_DIRECTOR_ACTION_RETRY=1

python -m selfplay_graph_flowsteer benchmark \
  --config configs/.swe_gpu4_c10.toml \
  --dataset data/formal/eval/swe_bench_verified_test_128.jsonl \
  --workers 10 \
  --director-thinking \
  --worker-route gpt \
  --skill-context off
```

命令中的 `--skill-context off` 表示无 skill 测试；不要把它与训练配置中的
`solver_skillbank.usage = "training_only"` 混为一谈。

### SWE 远程测评服务器自动开关机（可选）

SWE 的 SSH verifier 支持通过腾讯云 CVM API 自动拉起和关闭远程测评实例。该功能
使用进程内共享租约：并发评测只会启动一次，最后一个评测结束后才会按配置关闭实例。
如果实例在评测开始前已经是运行状态，本功能不会在结束时擅自关闭它。

启用前，将 CAM 的 SecretId/SecretKey 仅写入本机未纳入 Git 的 `.env`（不要写进
配置文件、日志或提交记录），并设置：

```bash
TENCENTCLOUD_SECRET_ID=...
TENCENTCLOUD_SECRET_KEY=...
SPGFS_SWE_CVM_AUTO_START=1
SPGFS_SWE_CVM_AUTO_STOP=1
```

默认实例是 `ins-5n1zolfw`，区域是 `ap-singapore`；如实例或区域不同，可在
所选配置的 `[swe]` 中修改。SSH 私钥只负责登录 verifier，不能替代腾讯云 CAM
凭据。程序会先调用 `DescribeInstancesStatus`，必要时调用
`StartInstances`/`StopInstances`，并将控制面错误记录为基础设施错误，不会伪造
评测结果。自动开关机默认关闭；只有显式打开配置或环境变量后才会启用。

### QA 结果重评分工具

完成 NQ-open 或 HotpotQA 后，可使用以下脚本做独立的答案格式化和语义重评分；
它们不会修改原始 `samples/` 和 `trajectories/`：

```bash
python scripts/formal/semantic_regrade_qa.py \
  --run-dir state/formal-eval/<run>

python scripts/formal/qa_formatter_ablation.py \
  --run-dir state/formal-eval/<run> \
  --workers 2
```

重评分结果应与原始严格 EM 分开报告，不能用本地模型 Judge 的结果替代官方指标。

## 交给远端 Codex 的指令

```text
读取 docs/REMOTE_GPU_SETUP.zh-CN.md 和我提供的私密配置文档。拉取
feat/pats-skill-scaffold 最新提交；不要把任何密钥提交 Git。根据本机 GPU、
CUDA 和磁盘调整环境，恢复 .env 与 SWE SSH 文件，运行
scripts/formal/bootstrap_remote.sh。下载或挂载缺失的 Qwen、E5、ALFWorld、
WebShop 资产，直连探测 API 并生成新鲜 route_report.json。先运行单样本
canary，确认路由池、检索服务和 SWE verifier 都能实际返回，再启动对应的
benchmark 命令；不要在未确认数据集绑定和服务身份前直接跑全量。
```
