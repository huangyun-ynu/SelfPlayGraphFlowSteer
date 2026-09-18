# Search-R1 E5/FAISS 检索接入

## 2026-09-17 数据准备状态

ModelScope 固定版本镜像下载及发布方 SHA-256 校验已经完成。压缩语料实际为
`gzip(TAR(JSONL))`；准备脚本会先完整读取 gzip 以验证 CRC，再安全提取 TAR 中唯一的
JSONL 成员。旧版脚本仅去掉 gzip 层，会把 TAR 错放到 `wiki-18.jsonl`，造成语料行数
比 FAISS 向量数多 1；新版会自动识别并原地修复，检索服务也会提前拒绝误标的 TAR。

每段 16 MiB，检查 HTTP 206、Content-Range 和实际长度；完成段保存在 `.ranges/`。
原有 `.partial` 前缀保持不变，所有段完成后组装并校验发布方 SHA-256。
失败重启可复用已完成段，未完成段重新下载；不要修改 `.ranges/manifest.json` 或原有前缀。
同一输出目录有进程锁，重复运行会拒绝并行写入。

停止旧下载进程后，可用以下命令续传。IP 为本次直连实测结果，后续失效需重新解析，
并非永久配置；不修改系统 DNS 或 hosts。

```bash
cd /home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer
.venv/bin/python -u scripts/formal/prepare_searchr1_retrieval.py \
  --source modelscope --direct --workers 1 --connections 8 \
  --resolve modelscope.cn:443:39.99.133.195 \
  --storage-ip 47.96.203.64 --storage-ip 47.96.203.65 \
  --storage-ip 47.96.203.66 --storage-ip 47.96.203.67 \
  --storage-ip 47.96.203.68 --storage-ip 47.96.203.69 \
  --storage-ip 47.96.203.70 --storage-ip 47.96.203.71
```

此固定语料后端是正式 NQ-open 与 HotpotQA 的默认检索设置。在线 Wikipedia
后端仅保留为显式诊断选项；如需使用，必须设置
`SPGFS_RETRIEVAL_BACKEND=wikipedia`。

NQ-open 与 HotpotQA 共用 Search-R1 的固定 Wiki-18 语料及配套 E5 索引。
这个设置应标为 Search-R1 Wiki-18；它不等同于 HotpotQA 官方 2017 全维基设置，
也不等同于使用 DPR 编码器。不能仅凭恢复检索就声称与所有论文设置可比。

## 数据

- `PeterJinGo/wiki-18-e5-index`，revision `a4d31160a035f30764604f4827cd8f1d0315eb86`。
- `PeterJinGo/wiki-18-corpus`，revision `69c1c00ffe7c5554c68d8548355cb22e46aabc51`。
- 下载合计 69,682,382,633 bytes，约 69.68 GB / 64.90 GiB。
- 准备脚本保留下载文件及压缩包，另生成合并索引并从内层 TAR 提取 JSONL；建议预留 200 GB。
- 默认位置：`state/formal-data/retrieval/searchr1/`。
- 下载不读取测试问题。每个文件校验发布方 SHA-256，合并顺序固定为 `part_aa`、`part_ab`。
- 下载中断保留 `.partial`，同一命令可继续。不要把未完成文件重命名为正式文件。

```bash
cd /home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer
.venv/bin/python scripts/formal/prepare_searchr1_retrieval.py
```

## 运行依赖与服务

复用本地 E5 模型及现有 `E5SkillEmbedder`，查询前缀 `query: `，最大长度 256，
mean pooling 后 L2 归一化。编码器使用 CPU FP32，FAISS 使用 CPU inner product。
这与找到的上游服务默认 GPU FP16 不完全一致，复现实验必须记录此区别。
服务不依赖外部 MedSAM 仓库，不修改该仓库。

默认 Python 为 `../.venvs/spgfs-pats-gpu/bin/python`，模型为 `../models/e5-base-v2`。
FAISS 依赖单独安装到本项目 `state/retrieval-deps`，不修改共享训练环境：

```bash
source scripts/formal/environment.sh
"$SPGFS_RETRIEVAL_PYTHON" -m pip install --no-deps --target "$SPGFS_RETRIEVAL_DEPS" 'faiss-cpu==1.13.2'
bash scripts/formal/run_retrieval_service.sh
```

服务仅监听 `127.0.0.1:8010`，提供 `GET /health`、`POST /retrieve`。
启动会校验向量数、语料行数、模型维度和相似度类型；首次启动创建行偏移索引。
语料通过 mmap 按原始行序读取，不能按文档 ID 排序。完整 FAISS 索引约占 60.1 GiB RAM。
并发请求串行执行模型与索引操作；正式环境默认使用 4 个 FAISS/编码器 CPU
线程，并为 24 槽位排队设置 240 秒客户端超时。该服务器实测增加到 16 线程会因
内存带宽竞争显著变慢；修改线程数后必须重新完成全索引并发延迟验证。

可覆盖 `SPGFS_SEARCHR1_INDEX`、`SPGFS_SEARCHR1_CORPUS`、`SPGFS_SEARCHR1_MODEL`、
`SPGFS_RETRIEVAL_PYTHON`、`SPGFS_RETRIEVAL_THREADS`。
`SPGFS_RETRIEVAL_BACKEND=sqlite` 仅保留旧调试服务的显式入口，默认是 `faiss`。

## 验证与评测开关

以下冒烟使用三条人工文档及真实本地 E5 模型，只验证接线，不产生评测指标：

```bash
source scripts/formal/environment.sh
export PYTHONPATH="$ROOT/src:$SPGFS_RETRIEVAL_DEPS${PYTHONPATH:+:$PYTHONPATH}"
"$SPGFS_RETRIEVAL_PYTHON" scripts/formal/smoke_dense_retrieval.py \
  --model "$SPGFS_SEARCHR1_MODEL" --output state/retrieval-smoke
```

完整数据准备、真实检索及并发延迟验证通过后，才设置
`configs/formal_training.toml` 的 `retrieval.enabled=true`。
正式启动脚本使用 `SPGFS_ENABLE_LOCAL_RETRIEVAL=1` 管理服务，最长默认等待 1800 秒；
健康检查会识别 FAISS 后端，避免误把旧 SQLite 服务当作完整索引。
单独启动服务不会开启评测。W&B 维持 offline，未自动重跑 SOTA。

## 2026-09-17 全量数据修复记录

- 三个发布文件 SHA-256 通过，合并索引的两个字节区间分别与 `part_aa`、`part_ab` 一致。
- 从内层 TAR 提取真实 `wiki_dump.jsonl`，不以 TAR 容器换行数充当语料行数。
- FAISS 向量数与 JSONL 行数均为 21,015,324；数量安全检查保留。

## 2026-09-16 验证记录（历史）

- 5 项检索/数据准备测试与 2 项 QA 原问题上下文回归通过；shell 语法及修改文件 lint 通过。
- 真实本地 E5 + FAISS + HTTP + `SearchServiceTool` 冒烟通过。
  产物：`state/retrieval-smoke/report.json`，只包含三条人工文档，不用于正式评测。
- FAISS 1.13.2 的 cp310-abi3 二进制来自服务器现有安装，复制到独立的
  `state/retrieval-deps`；已在项目 Python 3.11 上实际加载并完成搜索。
- 官方大文件下载仅传输约 74 KB 后停滞；直连超时。镜像带代理分段测试约
  35-70 KB/s，8 MiB 请求在 40 秒超时，仅完成约 1.43 MB。
  本轮下载已停止，保留 `.partial`；没有后台持续下载。
- 全量数据未完成，正式配置仍为 `retrieval.enabled=false`。
  未运行正式 NQ/HotpotQA 冒烟或 24 槽位吞吐测试，不能把上述三文档冒烟视为全量服务验证。
