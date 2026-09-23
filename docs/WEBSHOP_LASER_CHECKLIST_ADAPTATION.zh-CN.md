# WebShop：LASER 页面核对提示适配（待审阅）

2026-09-23。已完成可选实现和离线验证，尚未采用为正式基线。
用户随后授权的 128 题评测已完成：62/128（48.4375%），与正式参考持平，平均 token
增加 6.03%。详细记录见 `WEBSHOP_LASER_CHECKLIST_EXPERIMENT.zh-CN.md`。
正式参考仍是 legacy、无 skill、62/128（48.4375%）。

## 配置与作用位置

新增 `[webshop].worker_guidance_policy`：

- `baseline`：默认值，沿用当前 Worker 提示。
- `laser_checklist_v1`：为拥有 WebShop 会话的 Worker 增加页面核对提示。

独立候选配置为 `configs/webshop_laser_checklist_eval.toml`。解析 TOML 后，它与
`configs/webshop_official_eval.toml` 仅相差上述一个配置项。正式配置未切换；
`env_feedback_enabled` 仍取默认 `false`，没有叠加上次环境反馈实验。

实现入口：

- `src/selfplay_graph_flowsteer/webshop_guidance.py`：完整英文提示与策略名称。
- `src/selfplay_graph_flowsteer/application.py`：配置加载、校验，向各 Worker 执行器传递策略。
- `src/selfplay_graph_flowsteer/runtime.py`：在会话所有者的系统提示末尾追加核对清单，
  常规执行与修订都生效；DeepSeek 路由执行器同样传递该设置。

本适配把四类页面规则一次性放入系统提示，由模型依据最新 `page_type` 选择适用部分，
并没有在每步由 Python 切换提示模板。运行时更新页面上下文时保留这段系统提示。
无状态规划/评审 Worker 不注入，其他数据集也不注入。

环境 `page_text`、结构化观察、合法动作列表均未增加字段或重新排序。
开启时，最终运行记录增加 `artifact.webshop_progress.worker_guidance`，记录策略和
是否实际注入；这是事后审计字段，不是发送给模型的新观察。配置清单也会记录所选策略。

## 新增提示具体要求

| 当前阶段 | 模型核对内容 |
| --- | --- |
| `search` | 从公开请求识别商品类别、区分属性、价格上限和选项；构造搜索词，结合可见搜索历史明确改写查询要解决什么问题。 |
| `search_results` | 对照商品类别、属性与价格筛选；标题中的默认颜色/尺寸不代表全部可选款式，需要查看相关候选的实际选项；避免没有信息增量的重复查看。 |
| `product` | 逐项核对属性、价格、可用选项和已选选项；区分“有该选项”与“已经选中”；决定补充详情、选项选择、换商品或购买。 |
| `product_section` | 从描述、特征或评论解决尚未确认的需求；缺失内容保留为未知；必要时返回商品页，并计入返回所需动作。 |

购买前遵守项目已有的 `purchase_evidence` 和预算规则。已经暂存购买或终止时遵守
现有完成协议，不再开始购物。规则不要求增加 `Rationale` 文本响应、XML 标签或
单独的 `think` 动作，继续使用现有原生 thinking 和 JSON Action/最终输出协议。

现有基线提示本来就包含价格、选项、预算、部分匹配和暂存购买规则。本次增量是
按页面组织检查内容及明确各阶段信息需求，不能将它描述成从“完全没有核对”变成“有核对”。

## 与 LASER 原实现的区别

核对源码：`Mayer123/LASER`，固定提交
`dc50dafa1f88a4b889945393456b8960144be858`。

- [原始提示词](https://github.com/Mayer123/LASER/blob/dc50dafa1f88a4b889945393456b8960144be858/prompt_library.py)
- [原始执行器](https://github.com/Mayer123/LASER/blob/dc50dafa1f88a4b889945393456b8960144be858/laser_agent.py)

| 维度 | LASER 所核查实现 | 本次适配 |
| --- | --- | --- |
| 页面提示选择 | 程序按搜索、选商品、商品核验等阶段切换专用模板；商品核验使用独立流程。 | 在一个 Worker 系统提示中放入条件清单，由模型按最新页面应用；沿用现有执行循环。 |
| 理由与动作 | 页面模型先生成下一步理由；当未得到函数调用时，`auxilary_get_action` 再调用模型，把理由映射到函数。 | 在当前思考模式中完成核对，直接生成现有 JSON 动作；不新增理由转动作调用。 |
| 输出格式 | 部分页面指定 `Rationale:` 格式，动作映射使用函数调用。 | 不要求单独输出理由或新标签，不改变 `webshop_search` / `webshop_click` 参数。 |
| 选项处理 | 有额外的定制项提取及购买参数构造流程。 | 读取已有公开选项与 `selected_options`，通过现有逐次选项点击操作。 |
| 重访/重复详情 | 提示不再选择已查看商品；执行器对已查看候选另作处理，并在详情读取后移除相应可用函数。 | 只提醒避免无信息增量的重复操作，保留为补充信息或最终购买而重访的自由；不屏蔽合法动作。 |
| 购买 | 最终执行环境购买；另有利用已查看商品的备用购买流程。 | 沿用本项目证据声明、暂存 Buy Now、Canvas 选择输出后提交的机制；未移植备用购买流程。 |
| 模型与历史 | 原代码包含 GPT-4 调用方式以及自己的理由、动作和商品历史管理。 | DeepSeek Worker、Qwen Director、已有公开历史与会话修订机制均沿用；不注入 skill。 |

因此这是 **LASER 页面核对提示的局部适配**，不是完整 LASER 算法复现，也不能直接用
LASER 论文的成绩预估本版本准确率。

## 保留的实验条件与实际成本

候选配置保留 legacy 页面模式、无 skill、DeepSeek Worker thinking 开启，以及
12 次初始动作 + 4 次修订动作 = 16 次总预算。Director thinking 由评测入口控制，
本适配没有修改该开关，也没有改变既有关系二选一策略。未来评测应继续对齐正式基线。

代码不强制新增模型调用、思考动作或检查动作；模型实际决策和轨迹长度仍可能变化。
额外系统提示会增加输入 token，也可能改变思考长度。在固定总 token 预算下，其开销
仍会计入预算；不能声称没有成本或已经提升准确率。

## 验证

运行：

```bash
.venv/bin/python -m pytest tests/test_webshop_guidance.py \
  tests/test_webshop_env_feedback.py tests/test_webshop_sidecar.py \
  tests/test_webshop_network.py tests/test_runtime.py tests/test_application.py -q
```

结果：123 项通过，其中新增 6 项针对性用例。使用相同模拟模型输出时，开启和关闭
设置的用户上下文、工具 schema、调用参数、工具调用和动作预算一致，唯有所有者的
系统提示多出清单；同时覆盖路由传递、修订延续、无状态评审隔离、其他数据集隔离、
配置加载及拼写错误校验。新增文件 lint 和 `git diff --check` 通过。

上述结果是适配阶段的离线功能与回归检查，不是准确率评测。适配阶段没有调用部署模型、
修改正式入口或调整 GPU/模型服务。后续授权的 128 题实验另行记录。
