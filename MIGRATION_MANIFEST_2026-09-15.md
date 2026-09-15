# SelfPlayGraphFlowSteer sharestore2 迁移清单

更新时间：2026-09-15（Asia/Shanghai）

本清单记录正式训练依赖的 sharestore2 路径、外部服务和迁移后必须修改的绝对路径。密钥不写在本文件中；明文密钥只保存在权限为 `0600` 的私密备份：

`/home/bedicloud/sharestore2/iclr-users/2/SelfPlayGraphFlowSteer_PRIVATE_CONFIG_2026-09-15.md`

## 1. 当前运行结论

- 正式训练尚未启动；没有正式 `experiment/`、`checkpoints/`、`runtime_state/` 或 `route_report.json`。
- 正式任务池已就绪：7 个数据集各 512 条，共 3584 条，每个数据集 16 个 ADS 簇。
- SWE-bench Verified 已启用；本地 11 个 Git mirror 约 4.1 GB，372/372 个训练 base commit 已验证。
- 腾讯云 SWE 官方验证端已配置，但它不属于 sharestore2 文件迁移。
- 迁移后必须重新做路由探测并生成新鲜 route report，不能复制旧报告代替。

## 2. 必须迁移的本项目资产

| 资产 | 当前路径 | 体积 | 说明 |
| --- | --- | ---: | --- |
| 完整项目工作区 | `/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer` | 约 11 GB | 包含源码、`.git`、`.env`、正式状态、数据、ADS 和 SWE mirror；GitHub 仓库不包含被忽略的 `state/` 与 `.env` |
| 私密配置备份 | `/home/bedicloud/sharestore2/iclr-users/2/SelfPlayGraphFlowSteer_PRIVATE_CONFIG_2026-09-15.md` | 约 16 KB | 含 API keys 和 GitHub PAT，目标权限必须为 `0600` |
| GPU Python 环境 | `/home/bedicloud/sharestore2/iclr-users/owner/.venvs/spgfs-pats-gpu` | 约 8.3 GB | 建议在新路径重建；直接复制后需检查 shebang 和二进制路径 |
| E5 检索模型 | `/home/bedicloud/sharestore2/iclr-users/owner/models/e5-base-v2` | 约 419 MB | PATS 技能检索使用 |

完整复制上述 owner 自有资产约 19.7 GB。另有 `/home/bedicloud/sharestore2/iclr-users/owner/.cache/spgfs-pats` 约 3.8 GB，可迁移以加速启动，也可删除后重建。

项目内最重要的运行文件：

| 内容 | 路径 |
| --- | --- |
| 正式配置 | `/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/state/pats-formal-20260912/config.toml` |
| 正式环境 | `/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/state/pats-formal-20260912/environment.sh` |
| 启动脚本 | `/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/state/pats-formal-20260912/run_experiment.sh` |
| 课程配置 | `/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/configs/curriculum/formal_3500.toml` |
| API 密钥 | `/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/.env` |
| 正式 ADS 池 | `/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/state/pats-formal-20260912/private/datasets/ads/validated/validated_task_pool.jsonl` |
| 正式 ADS manifest | `/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/state/pats-formal-20260912/private/datasets/ads/validated/validated_task_pool_manifest.json` |
| 七数据集原始池 | `/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/state/pats-formal-20260912/private/datasets/raw` |
| SWE SSH 私钥 | `/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/state/pats-formal-20260912/private/swe/identity` |
| SWE known_hosts | `/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/state/pats-formal-20260912/private/swe/known_hosts` |
| SWE 验证器注册表 | `/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/state/pats-formal-20260912/private/swe/verifier-registry.json` |
| SWE 私有 Verified manifest | `/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/state/pats-formal-20260912/private/swe/verified-500-manifests.json` |
| SWE 本地源码 mirror | `/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/state/pats-formal-20260912/swe/repo-cache` |

## 3. 数据集产物

| 数据 | 路径 | 状态 |
| --- | --- | --- |
| HealthBench Professional 固定版本 | `/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/state/datasets/healthbench-professional/349962fd46dd02343a0d8a606491baf59154ea1a` | 官方数据、128 测试、512 训练及划分 manifest |
| SWE-bench Verified 固定版本 | `/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/state/datasets/swe-bench-verified/78f471bf655a3137b2e8a75af1501690ec009ec3` | 128 公开测试、512 原始训练及验证器注册表 |
| 正式七数据集 ADS | `/home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer/state/pats-formal-20260912/private/datasets/ads` | 已验证，合并池 SHA256 为 `7153b49ea9ecf1b5ed5427bc063c42961578864dd70c49f5793b71ead3a3b47b` |
| Git 加密正式数据包 | `data/formal/private/formal_data.tar.gz.enc` | 约 6.8 MiB；由私密配置中的 `SPGFS_DATA_ARCHIVE_KEY` 解密 |
| WebShop 官方 test 子集 | `data/formal/eval/webshop_official_test_128.jsonl` | 128 条，全部位于官方 `goal_index 0–499` |

正式数据目录合计约 208 MB；`state/datasets` 约 19 MB。旧普通 SWE train 和旧 SWE ADS 已删除，不应迁移或恢复。

## 4. 其他账号上的共享依赖

若新账号仍可稳定读取这些路径，可继续引用；若旧账号会删除或权限会撤销，则必须复制到新账号并修改配置。

| 用途 | 当前路径 | 体积 |
| --- | --- | ---: |
| Qwen3.5-9B 基模型 | `/home/bedicloud/sharestore2/iclr-users/3/gpf/models/Qwen3.5-9B` | 约 19 GB |
| ALFWorld 数据 | `/home/bedicloud/sharestore2/iclr-users/1/skillev-87e8171/SKILLEV/skillflow-resources/alfworld-data/json_2.1.1` | 约 2.1 GB |
| WebShop 固定源码 | `/home/bedicloud/sharestore2/iclr-users/1/skillev-87e8171/SKILLEV/protocol10-data/sources/webshop-git-64fa2a5` | 约 11 GB |
| WebShop Python 环境 | `/home/bedicloud/sharestore2/iclr-users/1/skillev-87e8171/SKILLEV/protocol10-data/envs/webshop` | 约 1.6 GB |
| WebShop 商品、目标与索引 | `/home/bedicloud/sharestore2/iclr-users/1/skillev-87e8171/SKILLEV/protocol10-data/prepared/webshop-full-streaming-v1` | 约 7.2 GB |
| WebShop 私有环境 Worker | `/home/bedicloud/sharestore2/iclr-users/1/skillev-87e8171/SKILLEV/formal-protocol10-669ce2b/packages/private-evaluation` | 约 3.8 MB |
| Protocol v10 原始物化池 | `/home/bedicloud/sharestore2/iclr-users/1/skillev-87e8171/SKILLEV/protocol10-data/materialized/protocol-v10-v6` | 约 1.7 GB；仅重新生成 AIME/HotpotQA/ALFWorld/WebShop 原始池时需要 |

完全复制这些共享依赖约 40.9 GB。加上 owner 自有必需资产，完全脱离旧路径的迁移规模约 60.6 GB；若同时复制可重建缓存则约 64.4 GB。

上面的 40.9 GB 按正式运行依赖计算，不包含 1.7 GB 的 Protocol v10 再生成源；正式原始池和 ADS 已在项目内，正常迁移无需复制该源。

WebShop 还依赖 sharestore2 之外的 Java：

`/home/bedicloud/localstore/test/openjdk11-pyserini-20260819/root/usr/lib/jvm/java-11-openjdk-amd64`（约 246 MB）

此路径不会随 sharestore2 账号迁移，必须确认新运行节点可访问或重新安装。

## 5. 项目内历史与可选资产

| 路径 | 体积 | 建议 |
| --- | ---: | --- |
| `state/codex-pats-validation` | 约 5.0 GB | 测试日志和临时结果，审计需要则复制，否则可省略 |
| `state/pats-real-architecture-20260912` | 约 946 MB | 历史三数据集实验、检查点和诊断，建议保留 |
| `state/codex-pats-experiment` | 约 225 MB | 历史实验，可选 |
| `state/codex-inspection` | 约 154 MB | 历史检查，可选 |
| `state/codex-pats-formal-preflight` | 约 116 MB | 旧 96 题预处理审计，可选，不得作为正式池 |
| `state/codex-pats-verified` | 约 76 MB | 历史验证，可选 |
| `state/codex-pats-smoke` | 约 72 MB | 历史 smoke，可选 |

Codex 附件位于 `/home/bedicloud/sharestore2/iclr-users/2/.codex/attachments`，合计约 1.6 MB。运行不依赖它们；其中原始 `sweverifier09152.pem` 已复制为正式 `private/swe/identity`，PATS PDF 仅供阅读。

## 6. 当前路由和调度摘要

- Director Worker 只看到 5 个逻辑模型：`gpt`、`grok`、`gemini`、`deepseek`、`minimax`。
- GPT、Grok、Gemini、HealthBench Judge 各有双物理端点池；DeepSeek 和 MiniMax 当前各一条路由。
- GPT Worker 与 HealthBench Judge 按物理 endpoint、credential 和并发上限共享 gate。
- 每条 GPT 路由 16 并发；每条 Grok/Gemini 路由 12；DeepSeek/MiniMax 各 30；Skill Refiner 20。
- 物理端点冷却 600 秒；池内有限重试 2 次，回退间隔 1 秒。
- 所有外部模型路由均为 `network_path = "direct"`，模型 API 不得使用代理。
- 主轨迹和关系反事实共享 35 个轨迹槽，主轨迹优先；每个反事实 off/on 是两个可独立运行的槽位工作项。
- 图复验为独立 8 槽，按数据集完成即流水触发；历史耗时优先级开关已实现但当前关闭。
- 七数据集 rollout 和槽位硬上限均为 900 秒；无进展阈值 120 秒。

完整路由 URL、模型、API surface、reasoning 模式及明文密钥见私密配置备份。

## 7. 外部服务，不随文件迁移

腾讯云 SWE 验证服务器：

- 公网 IP：`43.134.20.27`
- SSH：`sweeval@43.134.20.27:22`
- 实例：`swe-verifier`，实例 ID `ins-5n1zolfw`
- 区域：新加坡二区
- 服务目录：`/data/swe-verifier`
- 远端已有约 502 个 SWE 镜像；本地 sharestore2 只存源码 mirror 和 SSH 客户端凭据。
- 当前没有腾讯云 CAM SecretId/SecretKey，代码不能自动开关机；迁移后仍需手工开机或补 CAM 凭据。

## 8. 迁移后必须替换的路径

至少检查并修改以下文件中的旧绝对路径：

1. `state/pats-formal-20260912/config.toml`
2. `state/pats-formal-20260912/environment.sh`
3. `state/pats-formal-20260912/run_experiment.sh`
4. `state/pats-formal-20260912/run_ads_when_gpu_free.sh`
5. `state/pats-formal-20260912/run_seven_dataset_ads.py`
6. `state/pats-formal-20260912/prepare_raw_pools.py`
7. `state/pats-formal-20260912/prepare_swe_verified_split.py`

迁移后扫描残留旧路径：

```bash
rg -n '/home/bedicloud/sharestore2/iclr-users/(owner|1|3)|/home/bedicloud/localstore/test' \
  state/pats-formal-20260912 configs PROJECT_STATUS_2026-09-14.md \
  MIGRATION_MANIFEST_2026-09-15.md
```

## 9. 推荐复制与验证顺序

1. 停止训练、WebShop sidecar 和本地模型服务，确认没有写入中的状态文件。
2. 使用 `rsync -a --no-owner --no-group --info=progress2` 复制项目和私密备份；不要只做 `git clone`。
3. 按需复制 E5、Qwen、ALFWorld、WebShop 和运行缓存；在新路径重建 GPU venv 更稳妥。
4. 替换绝对路径，将 `.env`、私密备份、SWE identity 和 known_hosts 权限设为 `0600`。
5. 运行配置校验、七数据集池审计、SWE 372 个 commit 检查及本地 workspace clone/checkout probe。
6. 直连探测 API 路由，生成新的 `route_report.json`；不要迁移过期报告。
7. 开启腾讯云实例，核对 Host Key 后做空 patch 验证。
8. 运行一轮 canary；确认通过后再启动正式长训练。

目标账号尚未提供，因此本文件不写具体目标路径。迁移时用新账号绝对路径替换 `<NEW_ROOT>`，并保留源目录直到全部验证完成。

## 10. 迁移前关键 SHA256

```text
c2e665de2702aa776fdaab4de8977428acdee860449e9df605aa551720502776  state/pats-formal-20260912/config.toml
155f055e810393b6fbff03cf57aa84f5572e0949c128771230cbec702da6f89b  state/pats-formal-20260912/environment.sh
7153b49ea9ecf1b5ed5427bc063c42961578864dd70c49f5793b71ead3a3b47b  state/pats-formal-20260912/private/datasets/ads/validated/validated_task_pool.jsonl
746af77fb86fbf1b2d4b04132cd6d235a2a558f56124027b1034339b3cdfac32  state/pats-formal-20260912/private/swe/identity
bf98379e1f2af05478b4e7fa835ff6d6f818a701a04ba5d671ce196a30860a82  state/pats-formal-20260912/private/swe/known_hosts
```

这些哈希用于确认迁移过程中关键配置、正式池和 SSH 信任材料未被截断或改写；迁移并替换绝对路径后，`config.toml` 与 `environment.sh` 的新哈希应当变化。
