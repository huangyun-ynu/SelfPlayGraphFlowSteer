"""Select affected tasks from captured public traces, independently of reward."""
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path

REPORT = Path(__file__).resolve().parent
ROOT = REPORT.parents[2]
FIXTURE = ROOT / 'tests/fixtures/webshop_engineering_128_real_trajectories.json.gz'


def main():
    data = json.loads(gzip.decompress(FIXTURE.read_bytes()))
    groups = defaultdict(set)
    navigation_observations = 0
    for number, case in data['cases'].items():
        if case['section_plan_releases']:
            groups['section_reservation_released'].add(number)
        for trace in case['trace']:
            observation = trace['observation']
            state = observation.get('output', {})
            code = observation.get('error', {}).get('code')
            if code == 'invalid_completion_plan':
                groups['invalid_optional_plan'].add(number)
            if (code == 'purchase_budget_reserved'
                    and trace['action']['arguments'].get('completion_plan', {}).get('decision') == 'reserve'):
                groups['unaffordable_or_unreachable_plan'].add(number)
            if state.get('page_type') == 'product' and any(
                a.get('target_id', '').startswith('previous_page:')
                and a.get('navigation_effect') == 'return_to_current_product_page'
                for a in state.get('valid_subactions', [])
            ):
                groups['wrong_product_previous_label'].add(number)
                navigation_observations += 1
            for action in state.get('valid_subactions', []):
                if action.get('kind') == 'open_product' and action.get('target_id', '').rsplit(':', 1)[-1].upper() in {
                    'TERRACOTTA', 'LIEUTENANT', 'CAPPUCCINO', 'CANTALOUPE', 'COTTONWOOD'
                }:
                    groups['option_parsed_as_product'].add(number)
    direct = set().union(*(v for k, v in groups.items() if k != 'wrong_product_previous_label'))
    affected = set().union(*groups.values())
    assert len(affected) == 128 and len(direct) == 59
    assert {k: len(v) for k, v in groups.items()} == {
        'section_reservation_released': 27, 'invalid_optional_plan': 37,
        'unaffordable_or_unreachable_plan': 4, 'wrong_product_previous_label': 128,
        'option_parsed_as_product': 5,
    }
    result = {
        'source_run': data['source_run'], 'source_records_sha256': data['source_records_sha256'],
        'fixture_sha256': hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
        'rule': 'Union of observed public engineering-defect exposures; includes successful tasks. All 128 saw the wrong product Prev label.',
        'groups': {k: sorted(v) for k, v in sorted(groups.items())},
        'group_counts': {k: len(v) for k, v in sorted(groups.items())},
        'direct_issue_ids': ['webshop/goal-' + n for n in sorted(direct)],
        'ids': ['webshop/goal-' + n for n in sorted(affected)], 'tasks': len(affected),
        'wrong_navigation_observations': navigation_observations,
        'no_hidden_labels_used': True,
        'note': 'Exposure is not proof that a defect caused a task to fail; groups overlap.',
    }
    (REPORT / 'selection.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'tasks': len(affected), 'direct_issue_tasks': len(direct), 'groups': result['group_counts']}))


if __name__ == '__main__':
    main()
