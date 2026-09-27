"""Regressions for ten-v4 SWE failures: obsolete freeze and lost tool errors."""
import pytest

from selfplay_graph_flowsteer.outcome_admission import terminal_policy_failure
from selfplay_graph_flowsteer.runtime import _tool_failure_code
from .test_unified_submission import make_canvas, add, step


@pytest.mark.parametrize('dataset', ['swe_bench', 'aime', 'alfworld', 'webshop', 'hotpotqa', 'nq_open', 'healthbench_professional'])
def test_rejections_do_not_permanently_disable_rebuild_in_unified_mode(tmp_path, dataset):
    c = make_canvas(tmp_path, dataset, director_budget_policy='edits_v1', max_director_edits=24)
    add(c, scope='subtask')
    for _ in range(3):
        assert not step(c, dict(action='delete_agent', target='missing')).accepted
    assert not c.topology_edits_frozen
    assert step(c, dict(action='delete_agent', target='solver')).accepted
    assert 'add_agent' in c.control_snapshot()['allowed_actions']
    add(c, 'replacement')
    assert c.director_edits_used == 5


@pytest.mark.parametrize('code,known', [('stale_file_sha', True), ('transport_failure', False), ('tool_reported_error', False)])
def test_nested_workspace_error_preserves_attribution(code, known):
    # astropy-7166 emitted swe_edit with expected_sha256='PLACEHOLDER'.
    turn = {'action': {'name': 'swe_edit'}, 'observation': {
        'status': 'ok', 'output': {'status': 'error', 'error': {'code': code}}}}
    extracted = _tool_failure_code(turn)
    assert extracted == code
    args = dict(dataset='swe_bench', terminal=True,
        rejection_codes=['director_edit_budget_dead_end'], artifacts={}, output_agent=None,
        rounds=30, max_rounds=0, worker_tokens=335680, worker_token_limit=350000,
        director_edits=24, director_edit_limit=24,
        historical_artifacts={'old': {'runtime_tool_evidence': {
            'failed_count': 1, 'failure_codes': [extracted]}}})
    result = terminal_policy_failure(**args)
    assert (result is not None) == known
    assert terminal_policy_failure(**args, infrastructure_failure=True) is None
    if known:
        assert result['code'] == 'director_edit_budget_exhausted'
        assert terminal_policy_failure(**{**args, 'director_edits': 23}) is None
