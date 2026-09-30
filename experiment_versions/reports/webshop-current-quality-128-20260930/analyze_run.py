"""Validate complete coverage and summarize the corrected 128-task evaluation."""
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent
RUN = Path((REPORT / 'run-path.txt').read_text().strip())
sys.path.insert(0, str(REPORT.parent / 'webshop-engineering-rerun-20260930'))
from audit_common import artifacts

manifest = json.loads((RUN / 'manifest.json').read_text())
records = [json.loads(line) for line in (RUN / 'results/records.jsonl').open()]
assert len(records) == len({r['task_id'] for r in records}) == 128
assert {r['task_id'] for r in records} == set(manifest['ids'])
rescore = json.loads((RUN / 'rescore.json').read_text())
rescored = {r['task_id']: r for r in rescore['rows']}
changed = set(manifest['prompt_changed_ids'])
rows = []
events = {}
for record in records:
    env = record['trajectory']['task']['metadata'].get('webshop_environment_result') or {}
    purchased = bool(env.get('purchased'))
    receipt = record['trajectory'].get('submission_receipt') or {}
    arts = artifacts(record)
    used = max([a.get('webshop_progress', {}).get('action_budget', {}).get('total_used', 0) for a in arts] or [0])
    assert 0 <= used <= 16
    if purchased:
        assert env['scoring']['scorer_version'] == manifest['scorer_version']
        assert record['task_id'] in rescored
    if record['passed']:
        assert purchased and record['score'] == 1
    if record['submission_status'] == 'submitted':
        assert receipt['version'] == 'unified_submission_v1'
        assert receipt['payload']['purchased'] == purchased
    checks = env.get('scoring', {}).get('quality_checks', [])
    for event in [e for a in arts for e in a.get('backend_request_events', [])] + record['trajectory']['task']['metadata'].get('backend_request_events', []):
        if event.get('request_role') == 'worker':
            events[event['event_id']] = event
    row = {'task_id': record['task_id'], 'score': record['score'], 'passed': record['passed'],
           'purchased': purchased, 'submission_status': record['submission_status'],
           'outcome_status': record['outcome_status'], 'actions_used': used,
           'prompt_changed': record['task_id'] in changed, 'token_cost': record['token_cost'],
           'unmet_checks': [c for c in checks if c.get('matched') is False],
           'stop_reasons': sorted({a.get('webshop_progress', {}).get('stop_reason', '') for a in arts})}
    if purchased:
        row.update({k: v for k, v in rescored[record['task_id']].items() if k in ('asin', 'selected_options', 'official_reward_same_purchase')})
    rows.append(row)
rows.sort(key=lambda r: r['task_id'])
full = sum(r['passed'] is True for r in rows)
purchased = sum(r['purchased'] for r in rows)
summary = {'run': str(RUN), 'source_commit': manifest['source_commit'], 'scorer_version': manifest['scorer_version'],
           'tasks': 128, 'full': full, 'accuracy': full / 128, 'purchased': purchased,
           'not_purchased': 128 - purchased, 'purchased_not_full': purchased - full,
           'submitted': sum(r['submission_status'] == 'submitted' for r in rows),
           'unsubmitted': sum(r['submission_status'] != 'submitted' for r in rows),
           'mean_reward': sum(r['score'] or 0 for r in rows) / 128,
           'unknown_score': sum(r['score'] is None for r in rows),
           'execution_errors': len(list((RUN / 'results/errors').glob('*.json'))),
           'worker_requests': len(events), 'worker_request_statuses': dict(Counter(e.get('event') for e in events.values())),
           'worker_tokens': sum(r['token_cost'] for r in rows),
           'not_purchased_ids': [r['task_id'] for r in rows if not r['purchased']],
           'purchased_not_full_ids': [r['task_id'] for r in rows if r['purchased'] and not r['passed']],
           'unmet_check_counts': dict(Counter(c.get('kind') for r in rows for c in r['unmet_checks'])),
           'prompt_changed_tasks': len(changed),
           'full_by_prompt_group': {'changed': sum(r['passed'] is True for r in rows if r['prompt_changed']),
                                   'unchanged': sum(r['passed'] is True for r in rows if not r['prompt_changed'])},
           'purchased_official_label_full_same_purchase': sum(r.get('official_reward_same_purchase') == 1 for r in rows),
           'official_label_mean_reward_same_purchase': sum(r.get('official_reward_same_purchase', 0) for r in rows) / 128,
           'rescore': {k: v for k, v in rescore.items() if k != 'rows'},
           'source_integrity': json.loads((RUN / 'source_integrity.json').read_text()),
           'cleanup': json.loads((RUN / 'cleanup.json').read_text())}
for name, value in [('summary.json', summary), ('per-task.json', rows)]:
    text = json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    (RUN / name).write_text(text)
    (REPORT / name).write_text(text)
child = json.loads((RUN / 'child.json').read_text())
started = datetime.fromtimestamp((RUN / 'child.json').stat().st_mtime, ZoneInfo('America/Tijuana'))
report = f'''# 当前代码＋侧边栏修订测试集与测评器：128 题

当前主目录提交 `{manifest['source_commit'][:7]}`，累计修复、自动记忆 v2、完成动作预算预留 v2、调研调度与工程修复全部保留。评测从正式配置生成，冻结运行源码、goals 和测试集；侧边栏评分版本 `{manifest['scorer_version']}`。这是一套项目修订评分规则，源自原官方环境，未称为官方发布的新版本。

本次是 128 题全量真实模型购物，每题一次新尝试；没有按成绩筛选重试。128 个 ID 与原测试集一致，28 条题面修订。开始时间 {started.isoformat()}（America/Tijuana）。

| 指标 | 本次结果 |
|---|---:|
| 满分 | {full}/128（{full/128:.2%}） |
| 已购买 | {purchased}/128 |
| 未购买 | {128-purchased} |
| 已购买未满分 | {purchased-full} |
| 未提交 | {summary['unsubmitted']} |
| 平均奖励 | {summary['mean_reward']:.6f} |
| 运行异常 | {summary['execution_errors']} |

固定 DeepSeek Flash，thinking=false，请求并发 50；题目并发 40；Qwen3.5-9B Director thinking=true，32768 上下文；seed=0；技能上下文关闭；每题 16 个环境动作。现有 Director 服务复用；本次独立 CPU WebShop 服务已关闭。

购买终态独立重算：{rescore['purchases']} 条修订分数与环境结果一致，原标签分数也一致。原标签下同一批购买的满分数为 {summary['purchased_official_label_full_same_purchase']}，平均奖励 {summary['official_label_mean_reward_same_purchase']:.6f}；这些是同一次购买的附加记录，28 条题面已变化，不能当作原题重新评测的准确率。

未购买：{', '.join(summary['not_purchased_ids']) or '无'}。已购买未满分的逐项评分检查和商品选项见 `per-task.json`、`rescore.json`。失败检查按规则类型计数：{json.dumps(summary['unmet_check_counts'], ensure_ascii=False)}；一题可以有多个失败检查。

运行目录：`{RUN}`。冻结源码与数据完整性：{json.dumps(summary['source_integrity'], ensure_ascii=False)}。

全局正式训练配置未被本次评测覆盖。本次配置、任务及原始轨迹见运行目录中的 `config.toml`、`tasks.jsonl` 与 `results/`；本报告目录保存启动器与独立复核脚本。
'''
(REPORT / 'REPORT.zh-CN.md').write_text(report)
(RUN / 'REPORT.zh-CN.md').write_text(report)
print(json.dumps(summary, ensure_ascii=False, indent=2))
