# WebShop：SkillFlow 完整交互历史适配（历史记录，已移除）

2026-09-27：根据用户要求，已从当前代码删除 `skillflow_history_v1` 模式、候选配置、启动脚本和专用审计脚本；当前只支持 `factual_memory_v1`。以下描述的是当时的实现与命令，不能作为当前使用说明。已完成实验的冻结源码、输入和结果保留，最新128题结果见 [完整历史复跑报告](../state/audits/webshop-engineering-20260927/online-history128-v2/REPORT.zh-CN.md)。

实现参考本地 SkillFlow 提交 `74be52bb6bd9f0e9e68dacb72636b75649197983` 的
`training/environment.py`：`_react_step` 将动作前的观察与动作追加到 `_react_history`，
`_build_react_prompt` 在后续 WebShop 请求中遍历本题历史。这里移植的是这套历史输入机制。

## 配置与范围

新增 `[webshop].worker_memory_policy`：

- `factual_memory_v1`：默认值，沿用现有商品记忆与事实摘要。
- `skillflow_history_v1`：由同一 WebShop 会话的 Worker 保存并读取完整页面/动作历史。

候选配置：`configs/webshop_skillflow_history_eval.toml`。与上一轮使用的
`configs/webshop_laser_checklist_eval.toml` 相比，TOML 仅增加这一行：

```toml
[webshop]
# 其余已有配置保留
worker_memory_policy = "skillflow_history_v1"
```

候选入口：

```bash
bash scripts/formal/run_webshop_skillflow_history_eval.sh
```

入口使用同一官方 128 题、seed 0、24 路轨迹、DeepSeek Worker、保留中的 Qwen3.5-9B
Director（常规 thinking 开启）、无 skill、LASER checklist 和关闭的预算准入。
环境反馈仍关闭，动作预算仍为 12+4=16，实际 token 和运行时间限制沿用。
入口复用现有服务，不启停模型、不操作 GPU 占位。

此适配阶段完成离线测试和模拟请求检查，尚未运行新模式的 128 题准确率实验。
上一轮 60/128 属于“取消单段详情上限”的旧记忆模式，不能算作本模式成绩。

## 历史如何进入模型

模型可见位置：`action_environment.react_history`。示意：

```json
{
  "step": 2,
  "observation": "上一动作之后、这次点击之前的完整页面正文",
  "page_type": "search_results",
  "action": {
    "name": "webshop_click",
    "arguments": {"target_id": "open_product:0:B000000001", "state_version": 1}
  },
  "status": "ok"
}
```

`observation` 是动作前的页面。该动作的结果是下一条历史的页面，或者当前最新页面。
记录失败尝试时附加 `error`，不会将失败查询描述为成功执行。一次模型响应中多余的
状态动作仍按原协议丢弃，丢弃的动作只留在原审计轨迹中，不加入已选择执行的历史步骤。
不保存模型思考文本，不读取隐藏商品目标或评分数据。

每次动作后重新生成 Worker 请求时，带上完整历史与当前页面。历史中旧的 target ID、
state_version 只表示过去的操作；执行下一步仍须使用当前合法动作和当前版本。
当前 selected_options 保持权威，历史选项点击不代表现在仍选中。

历史存储在任务绑定的会话所有者 journal 中，同一个 Worker 修订时恢复；不同任务重置，
无状态规划/评审 Worker 不获得这份会话历史。即使动作预算用完，最终报告与报告格式修复
请求也携带完整历史和完整当前页面；无需新建或替换环境会话。

原有 `react_trace` 审计继续保留。新的执行产物仅增加 `worker_memory` 的策略、是否应用、
历史条数和正文字符数等审计信息，不把完整历史额外复制进 Director 的执行产物。

## 原来的商品记忆如何处理

新模式在发送给 Worker 的副本中移除旧商品记忆及依赖它的辅助字段：

- `product_inspections`、`candidate_ledger`、查询/访问/近期动作摘要、原决策检查点及状态提示；
- 搜索页的 candidate_coverage、search_decision_state；
- 动作上的 inspection_status、visit_count、candidate_evidence、observed_option_groups、
  evidence_status、action_semantics、navigation_effect；
- 由这些摘要推导的 action_decision_support、public_constraint_matrix。

这样，旧六商品记忆发生淘汰后生成的“未访问”标签不会与完整历史一起送给 Worker。
当前环境提供的动作 target ID、label、选项、selected_options、购买和终止状态继续提供。
现有购买协议需要的 policy_contract、purchase_evidence_checkpoint 继续提供。

内部商品记忆仍按原规则维护，供运行时兼容和轨迹审计使用；其六商品限制不会截断
`react_history`。原有购买提交、状态版本校验、动作预算、重复/无进展控制逻辑沿用。
这不是完整 SkillFlow 控制器的复现。

## 长度限制

新模式的历史不按商品数量、历史条数或正文字符数裁剪：

| 原机制 | 新模式中的处理 |
|---|---|
| 商品记忆最多 6 个商品 | 内部审计结构沿用；完整历史不受它限制 |
| 每商品最多两个详情正文 | 内部结构沿用；历史保留所有实际观察到的详情页面 |
| 单段详情 1,400 字符 | 之前已经取消；完整历史直接使用页面正文 |
| 整体 progress 6,000 字符淘汰 | 完整历史独立于该摘要，不参与其淘汰 |
| 当前页面在 Worker 提示中保留 8,000 字符 | 新模式绕过此裁剪，保留当前完整页面 |
| 最终报告的当前页面 1,600 字符 | 新模式保留完整页面，omitted_chars 记为 0 |

配置校验要求 `search_observation_mode` 为 `legacy` 或 `retain_page_text`，且
`max_observation_chars = 0`，防止页面在进入历史前已被截断或丢弃。候选配置满足这些条件。
模型上下文窗口及现有执行总预算仍然存在；完整历史会增加每步输入长度，实际成本和
准确率变化需要新实验确认。

## 与 SkillFlow 原实现的区别

| 方面 | SkillFlow | 本次适配 |
|---|---|---|
| 基本内容 | 动作前观察、动作的时间序列 | 相同；附加 page_type 和成功/错误状态 |
| 提示格式 | 把 Observation/Action 拼成文本 | 作为现有 Worker JSON 上下文中的列表 |
| 动作协议 | search[...] / click[...] | 现有 webshop_search / webshop_click、target_id、state_version |
| 生存周期 | 单题环境的 `_react_history` | 单题、同一会话所有者，跨 Worker 修订和报告收尾 |
| 可选状态反馈 | 原项目额外功能开关控制 | 继续使用本项目设置；候选环境反馈关闭 |
| 控制器和终止 | SkillFlow 的动作循环 | 本项目的 DeepSeek Worker、Qwen Director、暂存购买和 Canvas 提交 |
| 模型提示内容 | 原项目模板 | 原 Worker 协议和 LASER 清单，加历史使用说明 |

## 验证与代码位置

相关回归检查 **281 项通过**（其中本模式专项用例 12 项），lint、启动脚本语法检查和
修改文件的 `git diff --check` 通过。验证包括 WebShop、Worker runtime、配置、benchmark
runtime 和 selfplay runtime；没有向部署模型发起推理请求。

`tests/test_webshop_history.py` 覆盖：直接与路由执行器、七商品与超过 8,000 字符页面、
合法动作和预算保持、旧访问标记移除、同所有者修订、新任务清空、评审隔离、零动作报告、
失败尝试、批次中未执行动作排除、其他数据集隔离、配置传递及不兼容配置拒绝。

- `webshop_history.py`：历史记录格式、模型输入投影、使用说明。
- `runtime.py`：每步记录、跨修订恢复、Worker 请求与最终报告接线。
- `application.py`：配置加载、校验、执行器传递。
- `webshop.py`：只读取得当前所有者 journal，用于不新开会话的报告分支。

默认模式的输入行为保留，可直接切换配置开展后续配对评测。

离线审计目录：`state/experiments/webshop-skillflow-history-adaptation-20260923-213009/`。
其中 `candidate_mock_worker_request.json` 是明确标记为模拟数据的完整请求样例，包含
14 次实际执行的模拟动作，以及第一个商品超过 8,000 字符的完整历史页面；
`baseline_mock_worker_request.json` 保存相同操作下的旧模式请求，二者实际工具调用及动作
预算一致。`adaptation_manifest.json`、源码快照和差异文件用于核对本次实现。
