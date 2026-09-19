# 迁移到新服务器的运行资产清单

本文用于把项目迁移到另一台服务器时，由目标服务器自行下载或重建运行所需资产。默认不迁移历史实验产物、旧轨迹和日志。

## 1. 迁移关系

- 源服务器：worker08（`192.168.4.8`）
- 目标服务器：`192.168.4.11`
- 目标项目目录：`/workspace/test/codex-students/student02/SelfPlayGraphFlowSteer`
- Git 仓库：`https://github.com/huangyun-ynu/SelfPlayGraphFlowSteer.git`
- 当前主分支：`main`

目标服务器可以先克隆源码，再按本文下载模型、数据和依赖。模型目录不要保留指向源服务器的绝对路径软链接。

## 2. 必需模型

### Qwen3.5-9B

目标路径：

```text
models/Qwen3.5-9B/
```

当前源服务器实际模型目录约 19 GB。请从项目使用的模型仓库下载完整模型（包括 tokenizer、config、权重分片和 generation 配置），不要复制 `models/Qwen3.5-9B` 这个旧软链接本身。

### E5-base-v2

目标路径：

```text
models/e5-base-v2/
```

当前源服务器实际模型目录约 419 MB。下载完整 `intfloat/e5-base-v2`，包括 tokenizer 和配置文件。

## 3. NQ-open 的检索资产

NQ-open 使用 Search-R1 的固定 Wikipedia 2018 语料和 E5/FAISS 索引，必须成套准备；不能只下载索引或只下载语料。HotpotQA 按 FlowSteer 设置直接使用数据集提供的 distractor context，不依赖该检索服务。

发布版本：

```yaml
index_repo: PeterJinGo/wiki-18-e5-index
index_revision: a4d31160a035f30764604f4827cd8f1d0315eb86
corpus_repo: PeterJinGo/wiki-18-corpus
corpus_revision: 69c1c00ffe7c5554c68d8548355cb22e46aabc51
```

目标路径：

```text
state/formal-data/retrieval/searchr1/
```

准备内容包括：

- `e5_Flat.index`（由 `part_aa + part_ab` 合并）
- `wiki-18.jsonl`（由 `wiki-18.jsonl.gz` 解压）
- corpus offsets、manifest 和源文件校验信息

官方源文件总下载量约 69.68 GB，解压/合并后需预留至少 200 GB 磁盘空间；加载完整 FAISS 索引预计需要约 60 GiB 内存。

项目内下载脚本：

```bash
cd /workspace/test/codex-students/student02/SelfPlayGraphFlowSteer
.venv/bin/python -u scripts/formal/prepare_searchr1_retrieval.py \
  --source modelscope \
  --direct \
  --workers 1 \
  --connections 8
```

网络较慢时可按项目运行环境补充 ModelScope 的固定解析和存储 IP 参数。中断后重复同一命令即可续传；不要删除或修改 `*.partial`、`*.ranges/` 及其 manifest。下载完成后必须检查脚本生成的 `manifest.json` 和三份源文件 SHA-256。

## 4. 数据集资产

### NQ-open / HotpotQA

源码中的数据集配置和评测脚本需要对应的 JSONL/JSON 文件。目标服务器应按仓库中的数据下载脚本或配置获取官方测试集，至少确认以下文件存在并与配置路径一致：

```text
data/formal/eval/nq_open_test.jsonl
data/formal/eval/hotpotqa_official_test.jsonl
```

如果仓库当前使用不同文件名，以 `configs/` 和 `scripts/formal/` 中的实际配置为准；不要把旧服务器的评测结果目录当成数据集。

### ALFWorld（如需运行）

准备 ALFWorld 2.1.1 数据和环境配置：

```text
assets/alfworld-data/json_2.1.1/
```

当前源目录约 2.1 GB。目标服务器还需按项目依赖安装 ALFWorld，并确认 `ALFWORLD_DATA` 或项目环境脚本指向该目录。

### WebShop（如需运行）

WebShop 需要源码、商品数据库、搜索索引、prepared 数据、private evaluation worker 和独立 Python 环境。推荐从已校验的暂存包恢复，而不是从 Git 恢复大文件：

```text
assets/webshop/source/
assets/webshop/prepared/
assets/webshop/private-evaluation/
assets/webshop/indexes/
assets/webshop/venv/
```

暂存包中的源码 revision 为 `64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd`，并应使用随包提供的 `SHA256SUMS` 校验。若在目标服务器重建 `venv`，不要直接复用旧环境中的绝对路径 shebang。

## 5. Python、CUDA 和推理依赖

推荐在目标服务器重新创建项目环境，而不是复制源服务器 `.venv`：

```bash
cd /workspace/test/codex-students/student02/SelfPlayGraphFlowSteer
python3.11 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/pip install -r requirements.txt
```

当前已验证的关键版本：

```text
Python 3.11
torch 2.13.0+cu130
transformers 5.15.0
vllm 0.27.1
accelerate 1.15.0
peft 0.20.0
wandb 0.25.1
faiss-cpu 1.13.2（安装到 state/retrieval-deps）
```

服务器驱动无需修改；项目使用用户目录中的 CUDA compatibility 库时，启动前加载项目环境脚本：

```bash
source scripts/formal/environment.sh
```

然后验证：

```bash
.venv/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
.venv/bin/python -c 'import vllm; print(vllm.__version__)'
```

## 6. 配置和密钥迁移

不要把真实密钥提交到 Git，也不要把密钥写入本文。目标服务器需要由用户在本地重新配置 `.env`，变量名包括：

```text
DEEPSEEK_API_KEY
MINIMAX_API_KEY
NEXUS_API_KEY
NEXUS_PRO_API_KEY
SPGFS_DATA_ARCHIVE_KEY
SPGFS_SWE_CVM_AUTO_START
SPGFS_SWE_CVM_AUTO_STOP
SPGFS_WANDB_DNS_SERVER
TENCENTCLOUD_SECRET_ID
TENCENTCLOUD_SECRET_KEY
UUAPI_API_KEY
UUAPI_API_KEY_2
WANDB_API_KEY
WANDB_BASE_URL
WANDB_ENTITY
WANDB_MODE
WANDB_PROJECT
```

另外检查 `scripts/formal/environment.sh`、`configs/` 和本机服务配置中的绝对路径，把旧的 `/mnt/ssd/test/...` 路径改成目标目录 `/workspace/test/codex-students/student02/SelfPlayGraphFlowSteer`。

## 7. 不迁移的内容

以下内容属于历史实验、缓存或运行时状态，除非明确需要恢复某次实验，否则不要传输：

```text
state/formal-eval/
state/sota-*/
state/pats-*/
state/codex-*/
state/pytest-*/
state/formal-training/checkpoints/
state/formal-training/runtime_state/
旧 W&B debug、旧 trajectories、旧 logs、缓存目录（如 .cache、.ruff_cache）
```

如果需要保留某次实验，只单独导出该实验的配置、manifest、指标摘要和必要轨迹，不要把整个 `state/` 目录复制过去。

## 8. 推荐启动顺序

1. 克隆 `main` 分支源码并检查 Git revision。
2. 重建 `.venv`，安装依赖并验证 Torch/vLLM/CUDA。
3. 下载 Qwen3.5-9B 和 E5-base-v2 到真实目录。
4. 下载并校验 Search-R1 Wiki-18 corpus + E5/FAISS index。
5. 准备 NQ-open、HotpotQA 测试集；如需运行，再准备 ALFWorld/WebShop 资产。
6. 复制不含密钥的配置模板，手动填写目标服务器的密钥和路径。
7. 先执行单条样例链路，再启动完整评测；确认无误后再启用并发。

## 9. 迁移后的快速自检

```bash
cd /workspace/test/codex-students/student02/SelfPlayGraphFlowSteer
test -d models/Qwen3.5-9B
test -d models/e5-base-v2
test -f state/formal-data/retrieval/searchr1/e5_Flat.index
test -f state/formal-data/retrieval/searchr1/wiki-18.jsonl
.venv/bin/python -m compileall -q src scripts
git status --short
```

只有模型、检索资产、数据集和环境自检通过后，才开始正式评测。迁移完成后应在目标服务器重新生成一份本机 manifest，并记录模型 revision、数据 revision、依赖版本和 GPU/CUDA 信息。
