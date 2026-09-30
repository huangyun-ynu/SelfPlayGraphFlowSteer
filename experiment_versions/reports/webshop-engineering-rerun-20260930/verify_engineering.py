"""Audit actual new traces and replay their recorded reservation bookkeeping."""
import copy
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

from audit_common import artifacts

OUT = Path(__file__).resolve().parent
RUN = Path((OUT / 'run-path.txt').read_text().strip())
sys.path.insert(0, str(RUN / 'evaluated_source/src'))
from selfplay_graph_flowsteer.webshop_action_reserve import PurchaseReservation, FLEXIBLE_POLICY, quote


def main():
    rows = []
    advertised_asins = set()
    allowed_errors = {'purchase_budget_reserved', 'invalid_completion_plan', 'purchase_plan_abandoned',
        'purchase_plan_incomplete', 'completion_plan_required'}
    for line in (RUN / 'results/records.jsonl').read_text().splitlines():
        record = json.loads(line)
        arts = artifacts(record)
        manager = PurchaseReservation(policy=FLEXIBLE_POLICY)
        states = {}
        remaining = 16
        counts = Counter()
        mismatch = []
        bad_navigation = []
        fake_options = []
        sections = []
        released_paths = []
        latest_events = []
        error_actions = []
        ignored_feedbacks = []
        for art in arts:
            owner = art['agent_id']
            progress = art.get('webshop_progress', {})
            scope = progress.get('scheduling', {}).get('nodes', {}).get(owner, {}).get('scope', 'task_result')
            events = progress.get('purchase_reservation_events', [])
            if len(events) >= len(latest_events):
                latest_events = events
            for trace in art['react_trace']:
                action, observation = trace['action'], trace['observation']
                state = states.get(owner, {})
                choice = next((a for a in state.get('valid_subactions', [])
                    if a.get('target_id') == action['arguments'].get('target_id')), {})
                code = observation.get('error', {}).get('code')
                if observation['status'] == 'error':
                    counts['error:' + str(code)] += 1
                    error_actions.append({'action': action, 'code': code,
                        'remaining': trace['remaining_budget']['total'], 'kind': choice.get('kind')})
                if observation['status'] == 'error' and code not in allowed_errors:
                    continue  # Tool/schema admission happens before the budget manager.
                error, _ = manager.preflight(owner=owner, binding=('audit', owner), state=state,
                    name=action['name'], arguments=action['arguments'], remaining=remaining,
                    task_result=scope == 'task_result', task=record['trajectory']['task']['prompt'])
                if bool(error) != (observation['status'] == 'error') or (error and error['code'] != code):
                    mismatch.append({'action': action, 'recorded_error': code,
                        'replayed_error': error and error['code'], 'remaining': remaining, 'scope': scope})
                feedback = manager.plan_feedback.get(owner)
                if feedback:
                    ignored_feedbacks.append({'action': action, 'feedback': copy.deepcopy(feedback),
                        'action_executed': observation['status'] == 'ok', 'remaining_before': remaining})
                if observation['status'] == 'ok':
                    current = copy.deepcopy(observation['output'])
                    page = current.get('page_type')
                    if action['name'] == 'webshop_click' and choice.get('target_id', '').startswith('previous_page:'):
                        counts['executed_prev_from:' + str(state.get('page_type'))] += 1
                    for a in current.get('valid_subactions', []):
                        if a.get('kind') == 'open_product':
                            advertised_asins.add(a['target_id'].rsplit(':', 1)[-1].upper())
                        if a.get('target_id', '').startswith('previous_page:'):
                            counts['prev_labels_checked'] += 1
                            expected = {'product': 'return_to_search_results',
                                'product_section': 'return_to_current_product_page',
                                'search_results': 'previous_results_page'}.get(page)
                            if a.get('navigation_effect') != expected:
                                bad_navigation.append({'page': page, 'action': a})
                        value = a.get('target_id', '').rsplit(':', 1)[-1].upper()
                        if value in {'TERRACOTTA', 'LIEUTENANT', 'CAPPUCCINO', 'CANTALOUPE', 'COTTONWOOD'}:
                            if a['kind'] == 'open_product':
                                fake_options.append(a)
                            if a['kind'] == 'select_option':
                                counts['corrected_option_observations'] += 1
                                counts['corrected_option:' + value.lower()] += 1
                    before = copy.deepcopy(manager.plan)
                    for a in current.get('valid_subactions', []):
                        a.pop('navigation_effect', None)
                    manager.observe(owner, ('audit', owner), current)
                    if before and manager.plan is None:
                        released_paths.append({'action': action, 'page': page,
                            'reserved_asin': before['asin'], 'public_path_cost': quote(before, current)})
                    if before and before['owner'] == owner and page == 'product_section' and quote(before, current) is not None:
                        sections.append({'owner': owner, 'remaining': trace['remaining_budget']['total'],
                            'retained': manager.plan == before, 'cost': quote(before, current)})
                    states[owner] = current
                remaining = trace['remaining_budget']['total']
                manager.snapshot(remaining, owner)
        actual_counts = Counter(e['event'] for e in latest_events)
        replay_counts = Counter(e['event'] for e in manager.events)
        counts.update({'actual_event:' + k: v for k, v in actual_counts.items()})
        row = {'task_id': record['task_id'], 'counts': counts, 'bad_navigation': bad_navigation,
            'fake_option_products': fake_options, 'reserved_section_transitions': sections,
            'released_unreachable_paths': released_paths,
            'preflight_mismatches': mismatch, 'actual_event_counts': actual_counts,
            'replay_event_counts': replay_counts, 'error_actions': error_actions,
            'ignored_plan_feedback': ignored_feedbacks}
        rows.append(row)
    with sqlite3.connect(f'file:{OUT.parents[2]}/assets/webshop/prepared/products.sqlite3?mode=ro', uri=True) as db:
        missing_asins = [asin for asin in sorted(advertised_asins)
            if db.execute('select 1 from products where asin=?', (asin,)).fetchone() is None]
    summary = {'tasks': len(rows), 'counts': dict(sum((Counter(r['counts']) for r in rows), Counter())),
        'advertised_product_ids_checked': len(advertised_asins), 'nonexistent_product_ids': missing_asins,
        'wrong_navigation': sum(len(r['bad_navigation']) for r in rows),
        'fake_option_products': sum(len(r['fake_option_products']) for r in rows),
        'reserved_section_transitions': sum(len(r['reserved_section_transitions']) for r in rows),
        'section_reservations_lost': sum(not s['retained'] for r in rows for s in r['reserved_section_transitions']),
        'preflight_mismatches': sum(len(r['preflight_mismatches']) for r in rows),
        'release_count_mismatches': [r['task_id'] for r in rows
            if r['actual_event_counts'].get('released', 0) != r['replay_event_counts'].get('released', 0)],
        'ignored_count_mismatches': [r['task_id'] for r in rows
            if r['actual_event_counts'].get('plan_update_ignored', 0) != r['replay_event_counts'].get('plan_update_ignored', 0)]}
    result = {'summary': summary, 'tasks': rows}
    (OUT / 'engineering-verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    assert summary['wrong_navigation'] == summary['fake_option_products'] == summary['section_reservations_lost'] == 0
    assert not missing_asins
    assert summary['preflight_mismatches'] == 0, 'Review replay boundaries before interpreting reservation counts'
    assert not summary['release_count_mismatches'] and not summary['ignored_count_mismatches']


if __name__ == '__main__':
    main()
