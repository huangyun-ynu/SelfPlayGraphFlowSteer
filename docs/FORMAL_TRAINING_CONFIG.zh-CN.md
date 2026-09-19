# Formal Training 配置与凭据说明

本文档根据 `configs/formal_training.toml`、`scripts/formal/*.sh` 和当前项目目录整理，
用于在一台新主机上恢复正式训练/评测环境。文档只记录变量名、路由和路径，**不记录真实
API key、私钥或腾讯云 SecretKey**。真实值必须写入仓库根目录的 `.env`，该文件不得提交 Git。

## 1. 当前配置的范围

正式入口是：

```text
configs/formal_training.toml
scripts/formal/bootstrap_remote.sh
scripts/formal/run_experiment.sh
```

本配置包含：本地 Qwen3.5-9B Proposer/Solver、GPT 路由池、Skill Refiner、Grok、Gemini、
DeepSeek、MiniMax、Search-R1 检索、WebShop sidecar、ALFWorld，以及远程 SWE verifier。
正式启动还要求新鲜的 `state/formal-training/route_report.json`；报告超过 1,800 秒会被拒绝。

本工作区当前已恢复的私密配置状态如下（只记录状态，不在文档中复制密钥正文）：

- 根目录 `.env` 已存在且权限为 `0600`，路由 key、数据解密 key、W&B 变量和 CVM 变量均已设置。
- `state/formal-training/private/swe/identity`、`known_hosts`、`verifier-registry.json` 均已存在且权限为 `0600`。
- `state/formal-training/route_report.json` 已由当前 `.env` 对 11 个配置路由执行认证 `/models` 探测后生成。

如需查看或替换真实值，直接编辑 `.env`；不要把密钥复制到 TOML 或普通公共文档。现有私密
配置备忘位于 `docs/PRIVATE_CONFIG.local.md`，其内容同样只能留在受控本机环境。

## 2. `.env` 模板（允许填写 API key）

在项目根目录创建或编辑 `.env`。下列模板中的 `填入...` 必须替换为真实值；不要加引号，
不要把值写入 TOML：

```dotenv
# Model/API routes
NEXUS_API_KEY=填入_nexus_codex_key
NEXUS_PRO_API_KEY=填入_nexus_codex_pro_key
UUAPI_API_KEY=填入_uuapi_key
UUAPI_API_KEY_2=填入_uuapi_secondary_key
DEEPSEEK_API_KEY=填入_deepseek_key
MINIMAX_API_KEY=填入_minimax_key

# Encrypted formal data (scripts/formal/bootstrap_remote.sh 必需)
SPGFS_DATA_ARCHIVE_KEY=填入_formal_data_decryption_key

# Optional experiment tracking
WANDB_API_KEY=填入_wandb_key
WANDB_PROJECT=selfplay-graph-flowsteer
WANDB_ENTITY=填入_wandb_entity
WANDB_MODE=online
WANDB_BASE_URL=https://api.wandb.ai
SPGFS_WANDB_DNS_SERVER=8.8.8.8

# Optional Tencent Cloud CVM lifecycle; keep both flags 0/absent unless needed
# TENCENTCLOUD_SECRET_ID=填入_cam_secret_id
# TENCENTCLOUD_SECRET_KEY=填入_cam_secret_key
# SPGFS_SWE_CVM_AUTO_START=1
# SPGFS_SWE_CVM_AUTO_STOP=1
```

写入后立即限制权限并确认不会被 Git 跟踪：

```bash
chmod 600 .env
git check-ignore -q .env && echo '.env is ignored'
```

安全检查只输出变量名和是否非空，绝不要运行 `env`、`printenv` 或 `echo "$NEXUS_API_KEY"`。

## 3. 路由、模型和密钥变量

以下表格与 `configs/formal_training.toml` 一致。`gpt`/`gpt_eco` 是正式 Worker 池，
`gpt_judge`/`gpt_judge_eco` 是 HealthBench Judge 池；`skill_refiner` 仅用于周期之间的
Skill Distiller，不属于 Worker 选择池。

| 逻辑路由 | Endpoint | served model | `.env` 变量 | 并发 |
|---|---|---|---|---:|
| `gpt` | `https://nexus.itssx.com/api/codex/codex/v1` | `gpt-5.5` | `NEXUS_API_KEY` | 10 |
| `gpt_eco` | `https://nexus.itssx.com/api/codex_eco/v1` | `gpt-5.5` | `NEXUS_API_KEY` | 10 |
| `gpt_judge` | `https://nexus.itssx.com/api/codex/codex/v1` | `gpt-5.5` | `NEXUS_API_KEY` | 16 |
| `gpt_judge_eco` | `https://nexus.itssx.com/api/codex_eco/v1` | `gpt-5.5` | `NEXUS_API_KEY` | 16 |
| `skill_refiner` | `https://nexus.itssx.com/api/codex/codex_pro/v1` | `gpt-6-astra` | `NEXUS_PRO_API_KEY` | 20 |
| `grok` | `https://nexus.itssx.com/api/grok/v1` | `grok-4.5` | `NEXUS_API_KEY` | 12 |
| `grok45` | `https://nexus.itssx.com/api/grok45/v1` | `grok-4.5` | `NEXUS_API_KEY` | 12 |
| `gemini` | `https://uuapi.io/v1` | `gemini-3.6-flash` | `UUAPI_API_KEY` | 12 |
| `gemini2` | `https://uuapi.io/v1` | `gemini-3.6-flash` | `UUAPI_API_KEY_2` | 12 |
| `deepseek` | `https://api.deepseek.com/v1` | `deepseek-flash` | `DEEPSEEK_API_KEY` | 20 |
| `minimax` | `https://api.minimaxi.com/v1` | `MiniMax-M2.7` | `MINIMAX_API_KEY` | 30 |

当前 `runtime_routing.worker_routes = ["gpt"]`，池为 `gpt,gpt_eco`；正式脚本用
`--minimum-selected-routes 1`，因此至少要有一个 Worker 池成员可用，但 route report 仍须
覆盖所有配置候选路由和池成员。HealthBench 启用时还必须成功探测 Judge 路由。

## 4. 本地模型与服务资产

这些不是 API key，需在项目根目录按固定路径准备：

| 资源 | 固定路径/地址 |
|---|---|
| Proposer 模型 | `models/Qwen3.5-9B`，本地服务 `127.0.0.1:18601/v1` |
| Solver 模型 | `models/Qwen3.5-9B`，本地服务 `127.0.0.1:18602/v1` |
| 冻结 Runtime | `127.0.0.1:18603/v1` |
| SkillBank embedding | `models/e5-base-v2` |
| ALFWorld | `assets/alfworld-data/json_2.1.1` |
| WebShop 源码 | `assets/webshop/source` |
| WebShop Python | `assets/webshop/venv/bin/python` |
| WebShop store/goals | `assets/webshop/prepared/products.sqlite3`、`goals.jsonl` |
| 加密数据解密后任务池 | `state/formal-data/validated_task_pool.jsonl` |

GPU 默认由 `scripts/formal/environment.sh` 使用 `0,1,2`；Proposer/Solver/异步 rollout
分别占用三个不同 GPU。若主机编号不同，同时调整 `SPGFS_ALLOWED_PHYSICAL_GPUS` 和
`[resources]`，不可只改其中一处。

## 5. 初始化顺序与验证

1. 安装项目依赖并填写 `.env`。
2. 准备模型、E5、ALFWorld、WebShop 和 SWE 文件。
3. 执行解密和资产检查：

   ```bash
   scripts/formal/bootstrap_remote.sh
   ```

   该脚本会校验加密包 SHA256、用 `SPGFS_DATA_ARCHIVE_KEY` 解密正式数据，验证 3,584 条
   任务，并检查模型/服务/SWE 文件。失败时先按报错补齐资源，不要直接启动训练。

4. 检查 key 是否已设置（不打印值）：

   ```bash
   python - <<'PY'
   import os
   names = ["NEXUS_API_KEY", "NEXUS_PRO_API_KEY", "UUAPI_API_KEY",
            "UUAPI_API_KEY_2", "DEEPSEEK_API_KEY", "MINIMAX_API_KEY",
            "SPGFS_DATA_ARCHIVE_KEY"]
   for name in names:
       print(f"{name}: {'set' if os.environ.get(name) else 'MISSING'}")
   PY
   ```

5. 在目标主机的直连网络中探测全部路由，生成：

   ```text
   state/formal-training/route_report.json
   ```

   当前仓库中的报告记录了 11 个路由的认证 `GET /models` HTTP 200 结果。报告至少要包含
   `routes_requested`、`usable_routes` 以及每个路由的成功/失败证据；
   `routes_requested` 必须覆盖上表所有配置候选路由和 `gpt` endpoint-pool 成员。报告必须
   在正式启动前 30 分钟内生成，不能复用其他主机或旧实验报告。若更换 `.env`、网络或目标
   主机，必须重新探测并覆盖该文件。`/models` 成功只证明认证和端点可达，正式训练前仍应
   运行单样本 canary 验证实际 chat/tool roundtrip。

6. 运行正式入口（脚本会再次检查 route report、检索服务和 WebShop sidecar）：

   ```bash
   scripts/formal/run_experiment.sh
   ```

## 6. SWE 远程 verifier

`configs/formal_training.toml` 已固定如下身份：

```text
host: 43.134.20.27
port: 22
user: sweeval
dataset_revision: 78f471bf655a3137b2e8a75af1501690ec009ec3
```

必须存在以下本地私密文件（均建议 `0600`）：

```text
state/formal-training/private/swe/identity
state/formal-training/private/swe/known_hosts
state/formal-training/private/swe/verifier-registry.json
```

`identity` 是 SSH 私钥，`known_hosts` 必须来自可信渠道并启用严格主机校验；不能用
`StrictHostKeyChecking=no` 绕过错误。连接检查：

```bash
chmod 600 state/formal-training/private/swe/identity \
           state/formal-training/private/swe/known_hosts \
           state/formal-training/private/swe/verifier-registry.json
ssh -i state/formal-training/private/swe/identity \
  -o BatchMode=yes -o StrictHostKeyChecking=yes \
  -o UserKnownHostsFile=state/formal-training/private/swe/known_hosts \
  -o ConnectTimeout=10 sweeval@43.134.20.27 hostname
```

SSH 成功后再做远程 workspace clone/checkout、空 patch 验证和单个 SWE canary；网络超时
只能说明 verifier 尚未可达，不能把它标记为评测成功。

## 7. 腾讯云 CVM 自动开关机（可选）

默认关闭：`cvm_auto_start = false`、`cvm_auto_stop = false`。只有在确认 CAM 权限、实例
和区域后才启用，并把以下值仅放在 `.env`：

```dotenv
TENCENTCLOUD_SECRET_ID=填入_cam_secret_id
TENCENTCLOUD_SECRET_KEY=填入_cam_secret_key
SPGFS_SWE_CVM_AUTO_START=1
SPGFS_SWE_CVM_AUTO_STOP=1
```

当前配置的实例为 `ins-5n1zolfw`，区域为 `ap-singapore`，API endpoint 为
`cvm.tencentcloudapi.com`。SSH 私钥和 CVM CAM 密钥是两套独立凭据；缺一不可。

注意：应用配置会把 `SPGFS_SWE_CVM_AUTO_START=1` 或
`SPGFS_SWE_CVM_AUTO_STOP=1` 视为显式启用，即使 TOML 里的两个布尔值仍为 `false`。
因此修改或复制 `.env` 时必须同时检查这两个环境变量；不需要自动生命周期时应删除它们或
设为 `0`。

## 8. 完成判定清单

- [ ] `.env` 已填写所需 key，权限 `0600`，且未被 Git 跟踪。
- [ ] `SPGFS_DATA_ARCHIVE_KEY` 可解密正式数据，任务池为 3,584 条。
- [ ] Qwen、E5、ALFWorld、WebShop 资产在固定路径。
- [ ] route report 新鲜（不超过 1,800 秒）并覆盖所有要求的路由。
- [ ] SWE 三个文件存在、权限正确、registry 的 dataset revision 匹配。
- [ ] SSH 严格校验连接成功，并完成空 patch/canary 验证。
- [ ] 检索服务身份和 WebShop sidecar health 检查成功。
- [ ] 以上全部满足后，才运行 `scripts/formal/run_experiment.sh`。

若 API key 曾被提交、粘贴到公开日志或发送到不受信任位置，应先在供应商控制台撤销并重新
生成，再更新 `.env`；不要试图在文档中“遮罩后继续使用”旧凭据。
