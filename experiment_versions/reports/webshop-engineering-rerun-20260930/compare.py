"""Paired task outcomes and defect-cohort summaries; no fresh model requests."""
import csv
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
RUN = Path((OUT / 'run-path.txt').read_text().strip())
BASE = RUN.parent / 'webshop-memory-auto-history-128-20260930-153948'


def metrics(rows):
    return {'tasks': len(rows), 'full': sum(r['passed'] for r in rows),
        'purchased': sum(r['purchased'] for r in rows),
        'not_purchased': sum(not r['purchased'] for r in rows),
        'purchased_partial': sum(r['purchased'] and not r['passed'] for r in rows),
        'unsubmitted': sum(r['submission_status'] != 'submitted' for r in rows),
        'mean_reward': sum(r['score'] or 0 for r in rows) / len(rows)}


def main():
    before = {r['task_id']: r for r in json.loads((BASE / 'per-task.json').read_text())}
    after = {r['task_id']: r for r in json.loads((RUN / 'per-task.json').read_text())}
    selection = json.loads((OUT / 'selection.json').read_text())
    assert set(before) == set(after) == set(selection['ids'])
    groups = {'all_affected': set(after), 'direct_issue_union': set(selection['direct_issue_ids'])}
    groups.update({k: {'webshop/goal-' + n for n in v} for k, v in selection['groups'].items()})
    paired = []
    for task in sorted(after):
        a, b = before[task], after[task]
        paired.append({'task_id': task, 'before_score': a['score'], 'after_score': b['score'],
            'delta': (b['score'] or 0) - (a['score'] or 0),
            'before_purchased': a['purchased'], 'after_purchased': b['purchased'],
            'before_actions': a['action_budget_used'], 'after_actions': b['action_budget_used'],
            'before_stops': a['stops'], 'after_stops': b['stops'],
            'direct_issue': task in groups['direct_issue_union']})
    report = {'baseline': str(BASE), 'run': str(RUN),
        'cohorts': {k: {'before': metrics([before[t] for t in sorted(ids)]),
                        'after': metrics([after[t] for t in sorted(ids)])} for k, ids in groups.items()},
        'improved': [r['task_id'] for r in paired if r['delta'] > 1e-9],
        'worsened': [r['task_id'] for r in paired if r['delta'] < -1e-9],
        'unchanged': [r['task_id'] for r in paired if abs(r['delta']) <= 1e-9],
        'new_full': [t for t in sorted(after) if after[t]['passed'] and not before[t]['passed']],
        'lost_full': [t for t in sorted(after) if before[t]['passed'] and not after[t]['passed']],
        'recovered_purchase': [t for t in sorted(after) if after[t]['purchased'] and not before[t]['purchased']],
        'lost_purchase': [t for t in sorted(after) if before[t]['purchased'] and not after[t]['purchased']],
        'still_not_purchased': [t for t in sorted(after) if not before[t]['purchased'] and not after[t]['purchased']],
        'paired': paired,
        'interpretation': 'One fresh attempt per task; outcome changes alone do not identify deterministic causal gains.'}
    (OUT / 'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    with (OUT / 'comparison.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(paired)
    print(json.dumps({k: v for k, v in report.items() if k != 'paired'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
