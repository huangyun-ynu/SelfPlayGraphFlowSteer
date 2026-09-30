"""Check the reviewed navigation-only changes against the frozen real-run source.

Run probe_webshop_legacy with each source tree first; this script reads those
outputs and checks both historical hashes and an explicit change allowlist.
"""
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def differences(old, new, path=()):
    if isinstance(old, dict) and isinstance(new, dict):
        for key in sorted(set(old) | set(new)):
            if key not in old or key not in new:
                yield {'path': list(path + (key,)), 'before': old.get(key), 'after': new.get(key)}
            else:
                yield from differences(old[key], new[key], path + (key,))
    elif isinstance(old, list) and isinstance(new, list):
        assert len(old) == len(new), path
        for index, (a, b) in enumerate(zip(old, new)):
            yield from differences(a, b, path + (index,))
    elif old != new:
        if isinstance(old, str) and isinstance(new, str):
            try:
                a, b = json.loads(old), json.loads(new)
            except ValueError:
                pass
            else:
                yield from differences(a, b, path + ('<json>',))
                return
        yield {'path': list(path), 'before': old, 'after': new}


def main():
    before = json.loads((OUT / 'probe-before.json').read_text())
    after = json.loads((OUT / 'probe-after.json').read_text())
    fixture = ROOT / 'tests/fixtures'
    expected = json.loads((fixture / 'webshop_235e670_behavior_sha256.json').read_text())
    expected.update(json.loads((fixture / 'webshop_candidate_title_behavior_sha256.json').read_text()))
    assert set(before) == set(after) == set(expected)
    assert all(digest(before[k]) == expected[k] for k in expected)
    changes = list(differences(before, after))
    allowed = []
    for name in ('inspection_routed_False', 'inspection_routed_True'):
        allowed += [
            {'path': [name, 'initial_progress', 'prompt_projection', 'prompt_context_chars_sum'], 'before': 13732, 'after': 13810},
            {'path': [name, 'initial_progress', 'prompt_projection', 'raw_context_chars_sum'], 'before': 11932, 'after': 11971},
        ]
        for request in (0, 1):
            for field in ('action_decision_support', 'valid_subactions'):
                allowed.append({'path': [name, 'requests', request, 'messages', 1, 'content', '<json>',
                    'action_environment', 'state', field, 0, 'navigation_effect'],
                    'before': None, 'after': 'return_to_search'})
    assert changes == allowed, changes
    overlay = json.loads((fixture / 'webshop_engineering_behavior_sha256.json').read_text())
    assert overlay == {k: digest(after[k]) for k in after if before[k] != after[k]}
    report = {'frozen_source_matches_prior_goldens': True, 'snapshot_count': len(before),
        'changed_snapshots': sorted(overlay), 'changes': changes,
        'unchanged_snapshots': sorted(set(before) - set(overlay))}
    (OUT / 'probe-diff.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'snapshots_checked': len(before), 'reviewed_changes': len(changes)}))


if __name__ == '__main__':
    main()
