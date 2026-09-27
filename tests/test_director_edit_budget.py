"""Director decisions, committed edits and Worker calls are separate accounts."""
from dataclasses import replace

import pytest

from selfplay_graph_flowsteer.config import CanvasConfig
from selfplay_graph_flowsteer.director import GraphDirector
from selfplay_graph_flowsteer.llm import MockBackend
from .test_unified_submission import add, make_canvas, prompt, step


def test_ten_edits_not_ten_decisions_and_finish_at_exact_limit(tmp_path):
    c=make_canvas(tmp_path,max_director_edits=10)
    add(c)
    assert c.director_edit_budget()['used']==2
    # Invalid and unchanged actions spend decisions, never edits or Workers.
    calls=len(c.runtime.executor.calls)
    assert not step(c,dict(action='delete_agent',target='missing')).accepted
    assert not step(c,prompt()).accepted
    assert c.director_edit_budget()['used']==2
    assert len(c.runtime.executor.calls)==calls
    for index in range(8):
        action=prompt();action['objective']=f'Check independent requirement {index}'
        assert step(c,action).accepted
    assert c.round_index==12 and c.director_edits_used==10
    snapshot=c.control_snapshot()
    assert snapshot['director_edit_budget']['remaining']==0
    assert 'finish' in snapshot['allowed_actions']
    assert not set(snapshot['allowed_actions']) & {'add_agent','set_prompt','delete_agent','set_layer'}
    before=len(c.runtime.executor.calls)
    blocked=step(c,dict(action='delete_agent',target='solver'))
    assert blocked.rejection_code=='director_edit_budget_exhausted'
    assert 'solver' in c.graph.nodes and c.director_edits_used==10
    assert step(c,dict(action='finish',target='solver'),True).accepted
    assert len(c.runtime.executor.calls)==before


def test_node_recreation_cannot_refresh_edit_account_but_new_question_can(tmp_path):
    c=make_canvas(tmp_path,max_director_edits=5)
    add(c)
    assert step(c,dict(action='delete_agent',target='solver')).accepted
    assert c.director_edits_used==3
    add(c)
    assert c.director_edits_used==5
    c.reset()
    assert c.director_edits_used==0
    add(c)
    assert c.director_edits_used==2


def test_add_agent_reserves_prompt_and_required_model_edits(tmp_path):
    c=make_canvas(tmp_path,max_director_edits=1)
    assert 'add_agent' not in c.control_snapshot()['allowed_actions']
    blocked=step(c,dict(action='add_agent',agent_id='unfinished'))
    assert blocked.rejection_code=='director_edit_budget_exhausted'
    assert not c.graph.nodes and c.director_edits_used==0
    c.runtime_routes=('deepseek',)
    c.config=replace(c.config,max_director_edits=2)
    assert not step(c,dict(action='add_agent',agent_id='unfinished')).accepted
    assert not c.graph.nodes


def test_relation_proposal_and_unchanged_off_do_not_spend_edit(tmp_path):
    c=make_canvas(tmp_path,max_director_edits=6)
    c.binary_relation_policy=True
    add(c,'a');add(c,'b')
    assert c.director_edits_used==4
    assert step(c,dict(action='consider_relation',source='a',target='b')).accepted
    assert c.director_edits_used==4
    assert c.resolve_relation_choice('off').accepted
    assert c.director_edits_used==4 and not c.graph.bidirectional_edges
    # New prompt creates a fresh relation-decision version.
    action=prompt('a');action['objective']='Review the public evidence'
    assert step(c,action).accepted
    assert step(c,dict(action='consider_relation',source='a',target='b')).accepted
    assert c.resolve_relation_choice('on').accepted
    assert c.director_edits_used==6 and c.graph.bidirectional_edges=={('a','b')}


def test_failure_recovery_executes_without_spending_director_edit(tmp_path):
    from selfplay_graph_flowsteer.canvas import GraphCanvas
    from selfplay_graph_flowsteer.runtime import ModelAgentExecutor, MultiAgentRuntime
    backend=MockBackend(['{"answer":"WORKER_PROTOCOL_FAILURE"}', '{"answer":"001"}'])
    c=GraphCanvas(task='Compute integer',dataset='aime',
        runtime=MultiAgentRuntime(ModelAgentExecutor(backend)),
        config=CanvasConfig(submission_protocol='unified_task_result_v1',
            max_director_edits=2,submission_journal_dir=str(tmp_path),max_total_tokens=350000))
    add(c)
    assert 'run_agent' in c.control_snapshot()['allowed_actions']
    before=len(backend.calls)
    assert step(c,dict(action='run_agent',target='solver')).accepted
    assert c.director_edits_used==2 and len(backend.calls)==before+1


def test_new_budget_counters_do_not_hide_four_consecutive_stalls(tmp_path):
    import json
    c=make_canvas(tmp_path,max_director_edits=10)
    add(c)
    backend=MockBackend([json.dumps(dict(action='delete_agent',target='missing'))]*30)
    run=GraphDirector(canvas=c,backend=backend).run()
    assert any(s.rejection_code=='director_no_progress_exhausted' for s in c.history)
    assert len(backend.calls)<=6
    assert c.director_edits_used==2


def test_edit_exhaustion_without_final_result_has_typed_evidence_not_invented_answer(tmp_path):
    from selfplay_graph_flowsteer.outcome_admission import terminal_policy_failure
    c=make_canvas(tmp_path,max_director_edits=2)
    add(c,scope='subtask')
    assert c.control_snapshot()['allowed_actions']==[]
    end=c.terminate_director_stall('director_no_legal_continuation')
    assert end.rejection_code=='director_edit_budget_dead_end'
    assert c.submission_receipt is None
    args=dict(dataset='aime',terminal=True,rejection_codes=[end.rejection_code],
        artifacts={},output_agent=None,rounds=2,max_rounds=40,worker_tokens=0,
        worker_token_limit=350000,director_edits=2,director_edit_limit=2)
    assert terminal_policy_failure(**args)['code']=='director_edit_budget_exhausted'
    assert terminal_policy_failure(**args,infrastructure_failure=True) is None
    assert terminal_policy_failure(**{**args,'director_edits':1}) is None
    assert terminal_policy_failure(**{**args,'director_edits':3}) is None


def test_captured_aime22_candidate_submits_with_zero_edits_left(tmp_path):
    from .test_unified_failure_replay import captured, executor
    c=make_canvas(tmp_path,max_director_edits=2)
    artifacts=[captured(22,3),captured(22,10)]
    executor(c,artifacts);add(c)
    assert c.director_edit_budget()['remaining']==0
    assert step(c,dict(action='finish',target='solver'),True).accepted
    assert len(artifacts)==1 and c.submission_receipt.raw_answer_snapshot==captured(22,3)['answer']


@pytest.mark.parametrize('limit',[0,10,24,None])
def test_dataset_edit_limit_is_independent_of_decisions_and_tools(limit):
    c=CanvasConfig(max_director_edits=limit,max_director_edits_by_dataset={'aime':10})
    assert c.director_edit_limit('aime')==10
    assert c.director_edit_limit('webshop')==limit
    assert c.max_rounds==20


@pytest.mark.parametrize('dataset',['aime','hotpotqa','nq_open','healthbench_professional','alfworld','webshop','swe_bench'])
def test_two_account_mode_has_no_hidden_decision_cap_for_any_dataset(tmp_path,dataset):
    c=make_canvas(tmp_path,dataset,max_director_edits=24,
        max_director_edits_by_dataset={'aime':10},director_budget_policy='edits_v1')
    c.round_index=100  # Exceed both the old 24 and the legacy default.
    assert c.active
    assert c.terminate_round_limit_without_graph_repair() is None
    snapshot=c.control_snapshot()
    assert 'round_budget' not in snapshot
    assert snapshot['decision_statistics']=={'turns':100,'limited':False}
    assert 'add_agent' in snapshot['allowed_actions']
    add(c)
    assert c.active and c.round_index==102 and c.director_edits_used==2
    assert 'statistics only' in c.history[-1].feedback


def test_two_account_mode_allows_relation_choice_and_finish_after_old_round_cap(tmp_path):
    c=make_canvas(tmp_path,max_director_edits=10,director_budget_policy='edits_v1')
    c.binary_relation_policy=True
    add(c,'a');add(c,'b')
    c.round_index=100
    assert step(c,dict(action='consider_relation',source='a',target='b')).accepted
    assert c.resolve_relation_choice('on').accepted
    assert c.director_edits_used==5
    assert step(c,dict(action='finish',target='a'),True).accepted
    assert c.round_index==103


@pytest.mark.parametrize('bad',[-1,True,2.5,'10'])
def test_invalid_edit_caps_are_rejected(bad):
    with pytest.raises(ValueError,match='edit limits'):
        CanvasConfig(max_director_edits=bad)
    with pytest.raises(ValueError,match='edit limits'):
        CanvasConfig(max_director_edits_by_dataset={'aime':bad})
