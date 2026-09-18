# 在线 Wikipedia 检索

当前正式配置为在线英文 Wikipedia，供 NQ-open 和 HotpotQA 的 Worker 按需调用。
不下载固定语料、不预先读取测试问题、不使用本地 E5/FAISS。
这与 Wiki-18 固定语料评测不同；结果必须注明 online Wikipedia，不能混入原固定语料结果。

`scripts/formal/run_retrieval_service.sh` 默认启动在线服务，监听 `127.0.0.1:8010`。
正式训练和 SOTA 启动脚本默认管理该服务；修改脚本不会自动启动评测。
SOTA 默认结果目录已改为 `qwen35-9b-director-wikipedia-online-v1`，W&B 保持 offline。

```bash
cd /home/bedicloud/sharestore2/iclr-users/owner/SelfPlayGraphFlowSteer
bash scripts/formal/run_retrieval_service.sh
```

请求 `POST /retrieve` 保持原来的 queries/topk/return_scores 协议。
每次查询先调用 MediaWiki 搜索，再分别提取各篇全文，保持搜索排名。
返回每篇最多 6000 字符的导语和与查询相关的段落；不生成答案、不伪造相似度分数。
HotpotQA 的多跳由 Worker 发起后续工具调用完成。

本地缓存默认在 `state/retrieval/wikipedia-v1/`，可通过 `SPGFS_WIKIPEDIA_CACHE` 设置。
每条缓存包含原始问题、检索时间、文章 URL、版本 ID、完整 API 响应和实际返回的摘录。
相同查询和 top_k 在相同缓存目录内复用结果；需要全新在线实验时使用新的缓存目录。
缓存仅来自实际工具调用；缓存会影响结果的时间一致性，复现实验应保留并记录缓存目录。
上游异常返回 HTTP 502，不会缓存为无结果；真实搜索无命中可以缓存。

联系信息由 `SPGFS_WIKIPEDIA_CONTACT` 或本地 Git 忽略的
`state/private/wikipedia-contact.txt` 提供，自动加入 User-Agent；缺少时拒绝外网请求。
这只是客户端联系标识，不代表账户认证或已被分配某一额度。

外网抓取使用服务进程的代理环境，最多 3 个并发外部请求、请求启动间隔至少 0.4 秒
（持续速率约 150 次/分钟，包含重试），
每个请求超时 15 秒、最多尝试 2 次。HTTP 429 会按 Retry-After 暂停新的上游请求，
没有该响应头时默认冷却 60 秒，不立即重试。模型 API 的连接设置不受此服务影响。
`GET /health` 只表示本地服务运行且后端类型正确，不证明 Wikipedia 上游当时可达。
24 槽位的端到端吞吐还需单独验证；此轮不自动重跑正式评测。

旧后端仍可显式选择 `SPGFS_RETRIEVAL_BACKEND=faiss` 或 `sqlite`，不能混淆实验设置。

## 本次验证

2026-09-16：7 项检索相关测试、shell 语法与新代码 lint 通过。
通过真实 `SearchServiceTool` 请求在线服务，`Hamlet author` 与
`Christopher Nolan birthplace` 均返回 5 篇有正文的文章，分别约 27.37 秒和 11.10 秒。
前者再次调用命中缓存，约 0.005 秒；证据包含 Shakespeare。
第二个查询结果相关性较差，说明需由 Agent 改写查询，不能将接口成功视为答案正确。
原始响应和文章版本信息已保存在默认缓存目录。这些是人工查询接线测试，不是测试集评测。

## 24 槽位验证：未通过

使用 `scripts/formal/benchmark_wikipedia_concurrency.py`，24 个不同人工查询同时发起，
每个 top_k=5、客户端超时 120 秒，上游保持最多 4 个并发请求。

- 第一轮冷缓存 0/24 成功，并出现本地连接重置。监听 backlog 从 5 提升到 128。
- 修复后第二轮冷缓存仍 0/24 成功：10 个 HTTP 502，14 个超时，最大约 120.11 秒。
  日志确认上游 HTTP 429 和 URLError；完整客户端结果已保存后，终止尚未结束的上游请求。
- 独立缓存测试使用此前两条真实查询重复组成 24 个并发请求，禁止外网访问：24/24 成功，
  P50 0.106 秒，P95 0.180 秒，最大 0.185 秒。这不能代表 24 个新查询的在线能力。
- 新增 429 冷却处理，8 项相关测试通过；未再次压测处于限流中的上游。
- 未调用模型、未重跑 SOTA、未上传 W&B。当前不应据此启动正式 24 槽位在线评测。

本地结果：
- `state/retrieval-loadtest-24-20260916-v2/cold.json`
- `state/retrieval-loadtest-24-20260916-v2.log`
- `state/retrieval-loadtest-24-20260916-cached/report.json`

第一轮报告的 warm 字段实际仍是缓存未命中的重试，不能作为热缓存成绩；
压测脚本现已在缓存不足 24 条时跳过该阶段。上述缓存成绩来自独立 cached-only 测试。

## 联系标识与限速调整后的复测

本地联系信息已加入 User-Agent，上游并发 3、启动间隔 0.4 秒、24 个不同查询，
独立空缓存、top_k=5、客户端超时 120 秒。

- 客户端成功 0/24：23 个超时，1 个 HTTP 502。
- 客户端 P50 120.104 秒、P95 120.108 秒；整轮约 120.242 秒。
- 等服务器处理完成后，20 个查询已有缓存，但未在客户端等待时间内返回，不能计为成功。
- 热缓存阶段跳过，没有用部分完成的缓存宣称通过。
- 本轮仍未通过；不能仅凭添加 User-Agent 判断额度已提升，也不能把全部失败都归因为限流。
- 进程已退出，无后台压测；未调用模型、未跑 SOTA 或上传 W&B。

报告：`state/retrieval-loadtest-24-contact-v3/report.json`。
日志：`state/retrieval-loadtest-24-contact-v3.log`。

## 账户认证后的 24 槽位测试

使用 Bot Password 登录并通过 userinfo 确认会话，给压测服务传入认证 Cookie opener。
此次仅为测试注入会话，常规服务启动脚本尚未自动接入账户登录。
保持上游并发 3、间隔 0.4 秒、top_k=5、120 秒超时及独立空缓存。

- 客户端 0/24 成功，24 个全部超时，P95 120.108 秒。
- 停止前已记录的 75 次上游请求都携带 Cookie：59 个 HTTP 200 响应头，
  13 个 URLError，3 个 TimeoutError，未记录到 HTTP 429。
  HTTP 200 不表示完整正文一定读取成功，也不能据此声称限流已解除。
- 所有客户端结果保存后停止残余后台请求；无查询缓存完成，未测热缓存。
- 认证有效，但此次未解决 24 槽位冷缓存的端到端超时。没有调整等待时间来改变判定标准。
- 密码、令牌及 Cookie 内容未写入报告或日志。未调用模型、未跑 SOTA 或上传 W&B。

报告：`state/retrieval-loadtest-24-auth-v1/report.json`。
每次上游请求的脱敏状态记录：`state/retrieval-loadtest-24-auth-v1/upstream.jsonl`。
