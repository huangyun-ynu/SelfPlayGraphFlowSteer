# W&B 配置

项目已有数值指标上报器 `WandbTracker`。本机凭据保存在项目根目录的 `.env`，该文件受 Git 忽略且权限为 `600`，不要上传、打印或打包分享。

## 本次验证状态（2026-09-15）

- 官方 GraphQL 身份查询通过，使用直连而非代理；默认团队为 `yun-huang-yunnanuniversity`。
- 项目环境安装 W&B SDK 0.25.1，修正上报器不受支持的 `finish_timeout` 参数。
- 27 项针对遥测、开关优先级及 SDK 参数兼容性的测试通过。
- **在线数值同步已验证成功**：测试运行创建成功，指标历史上传与结束同步均收到 W&B 服务端 `200 OK`，运行地址为 [direct-upload-check-20260915](https://wandb.ai/yun-huang-yunnanuniversity/selfplay-graph-flowsteer/runs/bb68aeb6c35248b9)。
- 当前宿主机的 `127.0.0.53` DNS stub 对静态编译的 `wandb-core` 不可靠。正式启动脚本现在通过 `scripts/formal/wandb_direct_exec.py` 获取实时 IPv4 地址，并只在该进程自己的 mount namespace 中绑定临时 hosts 文件。它不会修改宿主机 `/etc/hosts`，也不会使用代理或关闭 TLS 验证。
- 本机 `.env` 使用 `WANDB_MODE=online`。连接检查没有启动模型或训练，检查进程已经结束。

## 配置与依赖

以下只是配置结构，不含真实密钥：

```dotenv
WANDB_API_KEY=<private-key>
WANDB_ENTITY=<your-team>
WANDB_PROJECT=selfplay-graph-flowsteer
WANDB_BASE_URL=https://api.wandb.ai
WANDB_MODE=online
SPGFS_WANDB_DNS_SERVER=8.8.8.8
```

可选依赖为 `tracking`；迁移到其他 Python 环境后安装：

```bash
python -m pip install '.[tracking]'
```

当前项目 `.venv` 已安装 SDK。其他训练/推理虚拟环境需要独立安装，不能假设共用依赖。

## 开关与范围

`selfplay-experiment` 开始时读取配置文件所在项目最近的 `.env`。优先级为：显式 `--wandb-mode` > 已导出的 `WANDB_MODE` > `.env` > `disabled`。允许值为 `online`、`offline`、`disabled`；无效值会在创建上报器之前报错。

- `online`：上传数值指标。
- `offline`：保留离线 W&B 记录与本地指标，暂不上传。
- `disabled`：不初始化 W&B，仍保留项目原有本地遥测。

已有正式启动脚本调用 `selfplay-experiment`，因此自动继承该设置；也可以通过脚本末尾透传 `--wandb-mode disabled` 显式关闭。**这不是启动训练的指令；只进行纯推理评测时，不要为了开启 W&B 而运行正式训练脚本。**

正式启动脚本还会自动套用直连 DNS 包装器。包装器优先通过 `systemd-resolved` 获取当前地址，必要时向 `SPGFS_WANDB_DNS_SERVER` 重试查询，然后在隔离 namespace 内启动实验。GPU 在该 namespace 内保持可见，已用 `nvidia-smi` 验证。单独运行其他 Python 入口且需要 W&B 时，使用：

```bash
python scripts/formal/wandb_direct_exec.py -- python -m <module> [args...]
```

目前上报器接在 `selfplay-experiment` 路径。尚未实现的“基础 Qwen + 静态 Skill”独立推理入口仍需接入同一指标上报器；配置好 W&B 不等于已经接通那条评测流水线。已经运行的进程也不会自动读取新设置，需要在下次启动时生效。

## 数据与网络边界

项目仅显式上报数值统计和经过允许列表筛选的配置，不上传题目、答案、完整轨迹、密钥、源码或模型权重；SDK 的代码、Git、控制台日志与部分元数据收集已关闭。实际运行可记录所分配 GPU 的系统统计，连接检查使用合成标记关闭系统统计。

W&B 网络请求按现有上报器规则直连官方服务，不走 VPN 代理。项目 `.env` 使用环境变量认证，不需要把密钥另外写进共享的全局 `.netrc`。官方认证说明：[环境变量](https://docs.wandb.ai/models/track/environment-variables)、[登录与验证](https://docs.wandb.ai/models/ref/python/functions/login)。

上报失败不会改变优化器或任务结果：数值缓冲保留在运行目录的 `wandb_outbox/`，错误信息仅记录阶段和异常类型。已有去重/恢复机制以运行目录内的 `wandb_tracking.json` 为准；SDK 接受队列不等于服务器已持久保存，连接检查另行读取服务器记录核实。
