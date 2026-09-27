import json
from uuid import uuid4

import pytest

from selfplay_graph_flowsteer.actions import ActionParser
from selfplay_graph_flowsteer.canvas import GraphCanvas
from selfplay_graph_flowsteer.config import CanvasConfig
from selfplay_graph_flowsteer.director import GraphDirector, director_prompt_components
from selfplay_graph_flowsteer.llm import MockBackend
from selfplay_graph_flowsteer.runtime import MultiAgentRuntime
from selfplay_graph_flowsteer.submission_contract import _director_call_context, receipt_error
from .helpers import NumericRecordingExecutor


def make_canvas(tmp_path, dataset='aime', **options):
    return GraphCanvas(task='Find the requested result', dataset=dataset,
        runtime=MultiAgentRuntime(NumericRecordingExecutor()),
        config=CanvasConfig(submission_protocol='unified_task_result_v1',
            submission_journal_dir=str(tmp_path), max_rounds=40, **options))


def prompt(target='solver', result_scope='task_result', **kwargs):
    return dict(action='set_prompt', target=target, role='Analyst',
        objective='Solve the assigned task', scope='Reason independently',
        expected_output='The requested result', result_scope=result_scope, **kwargs)


def step(canvas, action, finish=False):
    canvas.run_id = canvas.run_id or uuid4().hex
    call_id=f'test:{len(canvas.history)}'
    canvas.observe_submission_candidates(call_id)
    return canvas.step(json.dumps(action), director_context=(
        _director_call_context(canvas.run_id, call_id) if finish else None))


def add(canvas, name='solver', scope='task_result'):
    assert step(canvas, dict(action='add_agent', agent_id=name)).accepted
    result=step(canvas, prompt(name, scope))
    assert result.accepted, result.feedback


@pytest.mark.parametrize('dataset', ['aime','nq_open','hotpotqa','healthbench_professional'])
def test_real_director_submits_one_generation_and_durable_receipt(tmp_path,dataset):
    canvas=make_canvas(tmp_path,dataset)
    actions=[dict(action='add_agent',agent_id='solver'),prompt(),dict(action='finish',target='solver')]
    run=GraphDirector(canvas=canvas,backend=MockBackend([json.dumps(a) for a in actions]),call_namespace=uuid4().hex).run()
    assert run.finished, [t.feedback for t in run.turns]
    assert len(canvas.runtime.executor.calls)==1
    assert canvas.history[-1].executed_agents==[]
    assert run.submission_receipt.version=='unified_submission_v1'
    assert receipt_error(run.submission_receipt,run=run,events=canvas.history,run_id=canvas.run_id,dataset=dataset) is None
    journal=json.loads(next(tmp_path.glob('*.json')).read_text())
    assert journal['state']=='committed'
    assert journal['receipt']['output_agent_id']=='solver'


def test_scope_revision_reexecutes_and_demotion_cannot_submit(tmp_path):
    c=make_canvas(tmp_path);add(c,scope='subtask')
    assert c.submission_assessment('solver')['blockers']==['local_result_only']
    assert step(c,prompt()).accepted
    assert len(c.runtime.executor.calls)==2
    assert c.submission_assessment('solver')['submit_ready']
    assert step(c,prompt(result_scope='subtask')).accepted
    # The revision consumes its actual previous result, so these are new inputs.
    assert len(c.runtime.executor.calls)==3
    assert not step(c,dict(action='finish',target='solver'),True).accepted
    assert not step(c,dict(action='set_output',target='solver')).accepted


def test_delete_recreate_changes_incarnation_and_no_relabelled_cache(tmp_path):
    c=make_canvas(tmp_path);add(c)
    incarnation=c.graph.nodes['solver'].metadata['incarnation_id']
    assert step(c,dict(action='delete_agent',target='solver')).accepted
    add(c)
    assert c.graph.nodes['solver'].metadata['incarnation_id']!=incarnation
    assert len(c.runtime.executor.calls)==2


def test_optional_relations_do_not_connect_graph(tmp_path):
    c=make_canvas(tmp_path);add(c,'a');add(c,'b')
    assert not c.graph.directed_edges and not c.graph.bidirectional_edges
    assert c.control_snapshot()['legal_action_parameters']['consider_relation']['relations']
    assert not c.submission_assessment('a')['submit_ready']
    assert not c.submission_assessment('b')['submit_ready']
    assert step(c,dict(action='delete_agent',target='b')).accepted
    assert c.submission_assessment('a')['submit_ready']


def test_ready_candidate_has_no_legacy_missing_output_graph_blocker(tmp_path):
    c = make_canvas(tmp_path)
    add(c)
    snapshot = c.control_snapshot()
    assert snapshot['result_assessments']['solver']['submit_ready']
    assert snapshot['graph_state']['submission_target'] is None
    assert snapshot['graph_state']['reachability_by_target']['solver'] == {
        'unreachable_nodes': [], 'graph_blockers': []}
    assert 'finish_validation_errors' not in snapshot['graph_state']


def test_finish_cannot_execute_dirty_or_missing_worker(tmp_path):
    c=make_canvas(tmp_path);add(c)
    c.dirty_agents.add('solver');before=len(c.runtime.executor.calls)
    assert not step(c,dict(action='finish',target='solver'),True).accepted
    assert len(c.runtime.executor.calls)==before
    assert step(c,dict(action='run_agent',target='solver')).accepted
    assert len(c.runtime.executor.calls)==before+1


def test_pending_nodes_can_be_deleted(tmp_path):
    c=make_canvas(tmp_path)
    assert step(c,dict(action='add_agent',agent_id='a')).accepted
    assert step(c,dict(action='delete_agent',target='a')).accepted
    assert not c.graph.nodes and c.pending_agent_id is None


def test_protocol_requires_scope_and_finish_target():
    p=ActionParser(unified=True)
    for action in [dict(action='finish'), dict(action='set_output',target='a'), {k:v for k,v in prompt().items() if k!='result_scope'}]:
        assert not p.parse_policy_output(json.dumps(action)).action.valid
    assert p.parse_policy_output(json.dumps(prompt())).action.valid
    assert p.parse_policy_output('{"action":"finish","target":"a"}').action.valid
    base,hints=director_prompt_components('v3')
    assert 'FINISH' in base and 'result_scope' in base


def test_same_input_failed_attempt_keeps_valid_candidate_and_attempt_audit(tmp_path):
    c=make_canvas(tmp_path);add(c)
    previous=c.runtime.artifacts['solver'].artifact_id
    original=c.runtime.executor.execute
    def fail(**kwargs):
        artifact=original(**kwargs)
        artifact.answer='WORKER_PROTOCOL_FAILURE'
        artifact.integrity_risks=['terminal_protocol_failure']
        return artifact
    c.runtime.executor.execute=fail
    c.dirty_agents.add('solver')
    result=step(c,dict(action='run_agent',target='solver'))
    assert result.accepted
    assert c.runtime.artifacts['solver'].artifact_id==previous
    assert result.execution.attempt_artifacts[-1].answer=='WORKER_PROTOCOL_FAILURE'
    assert result.execution.execution_events[-1]['preserved_candidate_id']==previous
    assert c.submission_assessment('solver')['submit_ready']
    assert c.total_tokens==10


def test_changed_prompt_failed_attempt_cannot_restore_old_candidate(tmp_path):
    c=make_canvas(tmp_path);add(c)
    original=c.runtime.executor.execute
    def fail(**kwargs):
        artifact=original(**kwargs)
        artifact.answer='WORKER_PROTOCOL_FAILURE'
        artifact.integrity_risks=['terminal_protocol_failure']
        return artifact
    c.runtime.executor.execute=fail
    action=prompt();action['scope']='Verify independently'
    assert step(c,action).accepted
    assert not c.submission_assessment('solver')['submit_ready']
    assert not step(c,dict(action='finish',target='solver'),True).accepted


def test_v3_semantic_manifest_and_pats_contract_are_version_isolated():
    from selfplay_graph_flowsteer.execution_contract import execution_semantics
    from selfplay_graph_flowsteer.pats_semantics import contract_hash, director_design_reference
    semantics=execution_semantics('v3')
    assert semantics['submission_contract_version']=='unified_submission_v1'
    assert semantics['director_action_protocol_version']=='director_action_json_v3'
    assert contract_hash('v3')!=contract_hash('v2.1')
    assert 'result_scope' in director_design_reference('v3')


@pytest.mark.parametrize('target', [1, '1', ' 1 '])
@pytest.mark.parametrize('answer', ['35', '1027'])
def test_accepted_target_normalization_survives_receipt_validation_and_scoring(tmp_path, target, answer):
    from selfplay_graph_flowsteer.adaptive import AdaptiveWorkflowSolver
    from selfplay_graph_flowsteer.observability import NumericVerifier, TaskSpec
    from .test_submission_grade_boundary import RawAnswerExecutor
    actions = [dict(action='add_agent', agent_id='1'), prompt(target),
               dict(action='finish', target=target)]
    executor = RawAnswerExecutor(answer)
    solver = AdaptiveWorkflowSolver(
        director_backend=MockBackend([json.dumps(a) for a in actions]),
        runtime=MultiAgentRuntime(executor), verifier=NumericVerifier(),
        director_prompt_variant='v3',
        canvas_config=CanvasConfig(submission_protocol='unified_task_result_v1',
            submission_journal_dir=str(tmp_path), max_rounds=10),
    )
    task = TaskSpec('numeric-id', 'Find the result', reference='35', metadata={'dataset':'aime'})
    result = solver.solve(task, run_id=uuid4().hex)
    assert result.outcome_decision.status == 'scored'
    assert result.verification.score == float(answer == '35')
    assert task.metadata['submission_status'] == 'submitted'
    assert result.director_run.submission_receipt.output_agent_id == '1'
    assert result.answer_submission.submitted_answer == answer
    assert len(executor.calls) == 1


def test_restored_ready_candidate_gets_one_submission_chance_after_graph_repair(tmp_path):
    c=make_canvas(tmp_path)
    actions=[dict(action='add_agent',agent_id='a'),prompt('a'),
        dict(action='add_agent',agent_id='b'),prompt('b'),
        *[dict(action='set_output',target='a') for _ in range(3)],
        dict(action='delete_agent',target='b'),dict(action='finish',target='a')]
    result=GraphDirector(canvas=c,backend=MockBackend([json.dumps(a) for a in actions]),
        call_namespace=uuid4().hex).run()
    assert result.finished,[(t.round_index,t.feedback[:100]) for t in result.turns]
    assert len(c.runtime.executor.calls)==2
    assert len(c._unified_submission_recoveries)==1
    assert result.submission_receipt.output_agent_id=='a'


def test_submission_grace_is_not_renewed_for_repeating_the_same_ready_state(tmp_path):
    c=make_canvas(tmp_path);add(c,'a')
    signature=c.director_progress_signature()
    c._unified_previous_submit_ready=False
    c._record_unified_progress(accepted=True)
    assert c._unified_submission_recoveries=={signature}
    c._unified_previous_submit_ready=False
    c._unified_no_progress=3
    c._record_unified_progress(accepted=True)
    assert c._unified_no_progress==4
    assert len(c._unified_submission_recoveries)==1
