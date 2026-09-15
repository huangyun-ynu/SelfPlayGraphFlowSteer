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
ALFWorld、SWE、35 个主轨迹/反事实共享槽、8 个异步图复验槽、按数据集
图复验流水线、Proposer/Solver 并行训练、PATS 版本绑定的下一 cycle
异步采集和历史耗时优先级。预测 token/时间 admission 保护及未部署的
retrieval 服务保持关闭。

## 交给远端 Codex 的指令

```text
读取 docs/REMOTE_GPU_SETUP.zh-CN.md 和我提供的私密配置文档。拉取
feat/pats-skill-scaffold 最新提交；不要把任何密钥提交 Git。根据本机 GPU、
CUDA 和磁盘调整环境，恢复 .env 与 SWE SSH 文件，运行
scripts/formal/bootstrap_remote.sh。下载或挂载缺失的 Qwen、E5、ALFWorld、
WebShop 资产，直连探测 API 并生成新鲜 route_report.json。先运行测试和
一轮 canary，全部通过后再启动 scripts/formal/run_experiment.sh。
```
