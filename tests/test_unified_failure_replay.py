"""Adapt captured Worker outputs to the new protocol; never rewrite historical scores."""
import copy
import json
from pathlib import Path

from selfplay_graph_flowsteer.contracts import AgentArtifact
from .test_unified_submission import make_canvas, add, step, prompt

FIXTURE=json.loads((Path(__file__).parent/'fixtures/unified_aime_failure_artifacts.json').read_text())


def captured(number,round_index):
    return copy.deepcopy(FIXTURE['cases'][str(number)]['rounds'][str(round_index)])


def executor(canvas,artifacts):
    original=canvas.runtime.executor.execute
    def execute(**kwargs):
        base=original(**kwargs)
        record=artifacts.pop(0)
        for key,value in record.items():
            setattr(base,key,copy.deepcopy(value))
        return base
    canvas.runtime.executor.execute=execute


def test_aime22_complete_candidate_finish_never_consumes_failed_next_output(tmp_path):
    c=make_canvas(tmp_path)
    artifacts=[captured(22,3),captured(22,10)]
    executor(c,artifacts);add(c)
    assert c.submission_assessment('solver')['submit_ready']
    before=len(c.runtime.executor.calls)
    assert step(c,dict(action='finish',target='solver'),True).accepted
    assert len(c.runtime.executor.calls)==before==1
    assert len(artifacts)==1
    assert c.submission_receipt.raw_answer_snapshot==captured(22,3)['answer']


def test_aime15_failed_state_exposes_bounded_repair_without_legacy_freeze(tmp_path):
    c=make_canvas(tmp_path)
    executor(c,[captured(15,5),captured(15,7),captured(15,7)])
    add(c)
    for _ in range(3):
        step(c,dict(action='set_output',target='solver'))
    assert not c.topology_edits_frozen
    snapshot=c.control_snapshot()
    assert 'finish' not in snapshot['allowed_actions']
    assert 'run_agent' in snapshot['allowed_actions']
    assert 'set_prompt' in snapshot['allowed_actions']
    for _ in range(2):
        assert step(c,dict(action='run_agent',target='solver')).accepted
    snapshot=c.control_snapshot()
    assert 'run_agent' not in snapshot['allowed_actions']
    assert 'finish' not in snapshot['allowed_actions']
    assert c.submission_receipt is None
    assert c._unified_recovery_used==2
    assert len(c.runtime.executor.calls)==3


def test_restart_with_unknown_commit_never_executes_another_worker(tmp_path):
    from selfplay_graph_flowsteer.unified_submission import SubmissionJournal
    journal=SubmissionJournal(str(tmp_path),'same-run')
    journal.write({'state':'committing','target':'old','run_id':'same-run'})
    c=make_canvas(tmp_path);c.run_id='same-run'
    assert not step(c,dict(action='add_agent')).accepted
    assert c.control_snapshot()['allowed_actions']==[]
    assert not c.runtime.executor.calls
def test_recorded_illegal_action_exhaustion_is_policy_failure_only_with_clean_runtime():
    from selfplay_graph_flowsteer.outcome_admission import terminal_policy_failure
    # The one-v8 AIME attempt repeated actions absent from the current menu;
    # its explicit runtime terminal event must not become an unknown answer.
    args = dict(dataset='aime', terminal=True,
        rejection_codes=['director_action_not_allowed'] * 3 + ['director_no_progress_exhausted'],
        artifacts={}, output_agent=None, rounds=11, max_rounds=24,
        worker_tokens=100, worker_token_limit=240000)
    assert terminal_policy_failure(**args)['code'] == 'director_repeated_illegal_actions'
    assert terminal_policy_failure(**args, infrastructure_failure=True) is None
    args['rejection_codes'] = ['director_no_progress_exhausted']
    assert terminal_policy_failure(**args) is None


def test_empty_graph_without_enough_configuration_rounds_is_typed_budget_failure(tmp_path):
    from dataclasses import replace
    from selfplay_graph_flowsteer.outcome_admission import terminal_policy_failure
    c = make_canvas(tmp_path, 'swe_bench')
    c.config = replace(c.config, max_rounds=24)
    c.runtime_routes = ('deepseek',)
    c.round_index = 22
    assert c.control_snapshot()['allowed_actions'] == []
    end = c.terminate_director_stall('director_no_legal_continuation')
    assert end.rejection_code == 'director_round_budget_dead_end'
    args = dict(dataset='swe_bench', terminal=True, rejection_codes=[end.rejection_code],
        artifacts={}, output_agent=None, rounds=22, max_rounds=24,
        worker_tokens=339395, worker_token_limit=350000)
    assert terminal_policy_failure(**args)['code'] == 'director_round_budget_dead_end'
    assert terminal_policy_failure(**args, infrastructure_failure=True) is None
    # Deletion must not hide an earlier route or unclassified environment fault.
    for historic in ({'backend_failure': True}, {'runtime_tool_evidence':{
            'failure_codes':['action_execution_failed'], 'failed_count':1}}):
        assert terminal_policy_failure(**args, historical_artifacts={'deleted':historic}) is None
    assert terminal_policy_failure(**{**args, 'rounds': 10}) is None


def test_empty_graph_commit_uncertainty_is_never_relabelled_as_budget_failure(tmp_path):
    from dataclasses import replace
    c = make_canvas(tmp_path, 'swe_bench')
    c.config = replace(c.config, max_rounds=24)
    c.runtime_routes = ('deepseek',)
    c.round_index = 22
    c._unified_transaction = {'state':'commit_unknown'}
    assert c.terminate_director_stall('director_no_legal_continuation').rejection_code == 'director_no_legal_continuation'


def test_replay_actual_swe_deleted_graph_failure_without_worker_or_score_rewrite(tmp_path):
    from dataclasses import replace
    from selfplay_graph_flowsteer.outcome_admission import terminal_policy_failure
    source = Path(__file__).parent/'fixtures/unified_swe_empty_graph_failure.json'
    raw = source.read_bytes()
    captured = json.loads(raw)
    snapshot = captured['control_snapshot']
    c = make_canvas(tmp_path, 'swe_bench')
    c.config = replace(c.config, max_rounds=snapshot['round_budget']['max'],
        max_total_tokens=snapshot['token_budget']['max'])
    c.runtime_routes = tuple(captured['final_graph']['runtime_routes'])
    c.round_index = snapshot['round_budget']['used']
    c.total_tokens = snapshot['token_budget']['used']
    assert captured['final_graph']['nodes'] == []
    assert c.control_snapshot()['allowed_actions'] == snapshot['allowed_actions'] == []
    end = c.terminate_director_stall('director_no_legal_continuation')
    outcome = terminal_policy_failure(dataset='swe_bench', terminal=not c.active,
        rejection_codes=[end.rejection_code], artifacts={}, output_agent=None,
        rounds=c.round_index, max_rounds=c.config.max_rounds,
        worker_tokens=c.total_tokens, worker_token_limit=c.config.max_total_tokens,
        historical_artifacts=captured['historical_artifacts'])
    assert outcome['code'] == 'director_round_budget_dead_end'
    assert not c.runtime.executor.calls
    assert source.read_bytes() == raw
