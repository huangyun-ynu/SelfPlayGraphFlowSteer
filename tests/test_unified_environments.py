import json
from types import SimpleNamespace

import pytest

from selfplay_graph_flowsteer.canvas import GraphCanvas
from selfplay_graph_flowsteer.config import CanvasConfig
from selfplay_graph_flowsteer.contracts import CodeArtifactRef
from selfplay_graph_flowsteer.runtime import MultiAgentRuntime
from .test_unified_submission import add, step, prompt, make_canvas
from .test_webshop_native import build, ASIN1, report


def shopping(tmp_path, responses):
    executor,backend,life,client,registry=build(responses)
    canvas=GraphCanvas(task='Buy a product',dataset='webshop',
        runtime=MultiAgentRuntime(executor),action_adapter=registry.get('webshop'),
        config=CanvasConfig(submission_protocol='unified_task_result_v1', submission_journal_dir=str(tmp_path), max_rounds=40, max_total_tokens=1000000))
    return canvas,backend,life,client


def stage():
    return ['search[product]',f'click[{ASIN1}]','click[buy now]',report()]


def test_staged_purchase_commits_at_finish_without_model_or_worker(tmp_path):
    c,b,life,client=shopping(tmp_path,stage());add(c)
    assert not client.commits
    assert c.submission_assessment('solver')['submit_ready']
    calls=len(b.calls)
    result=step(c,dict(action='finish',target='solver'),True)
    assert result.accepted, result.feedback
    assert len(b.calls)==calls and len(client.commits)==1
    assert c.submission_receipt.payload['purchased']
    assert c.submission_receipt.payload_kind=='purchase'
    assert not step(c,dict(action='finish',target='solver'),True).accepted
    assert len(client.commits)==1


def test_lost_commit_response_locks_transaction_and_preserves_unknown(tmp_path,monkeypatch):
    c,b,life,client=shopping(tmp_path,stage());add(c)
    original=client.commit
    def lost(*args,**kwargs):
        original(*args,**kwargs)
        raise ConnectionError('purchase response lost')
    monkeypatch.setattr(client,'commit',lost)
    result=step(c,dict(action='finish',target='solver'),True)
    assert not result.accepted and result.rejection_code=='submission_commit_unknown'
    assert len(client.commits)==1 and c.submission_receipt is None
    assert not step(c,dict(action='delete_agent',target='solver')).accepted
    assert json.loads(next(tmp_path.glob('*.json')).read_text())['state']=='commit_unknown'


def test_delete_prepared_candidate_closes_session_without_purchase(tmp_path):
    c,b,life,client=shopping(tmp_path,stage()+stage());add(c)
    old=list(client.sessions)
    assert step(c,dict(action='delete_agent',target='solver')).accepted
    assert all(s in client.closed for s in old)
    assert not life.commit_ready_agents() and not client.commits
    add(c)
    assert len(client.sessions)==2 and not client.commits
    assert step(c,dict(action='finish',target='solver'),True).accepted
    assert len(client.commits)==1


def test_live_environment_change_invalidates_candidate(tmp_path):
    c,b,life,client=shopping(tmp_path,stage());add(c)
    life._episodes['solver']._results['solver']['state_version']='changed-outside-canvas'
    assert 'stale_artifact' in c.submission_assessment('solver')['blockers']
    assert not step(c,dict(action='finish',target='solver'),True).accepted
    assert not client.commits


def test_report_without_purchase_cannot_finish_live_environment(tmp_path):
    c,b,life,client=shopping(tmp_path,[report()]);add(c)
    outcome=c.submission_assessment('solver')
    assert not outcome['submit_ready'],outcome
    assert 'purchase_not_prepared' in outcome['blockers']
    assert not step(c,dict(action='finish',target='solver'),True).accepted
    assert c._unified_can_run('solver')
    assert not client.commits and len(b.calls)==1


def test_exhausted_environment_budget_removes_explicit_continuation(tmp_path):
    from selfplay_graph_flowsteer.alfworld import ALFWorldSessionLifecycle
    from .test_alfworld_sessions import Client
    life = ALFWorldSessionLifecycle(Client(), tmp_path, max_rollout_steps=1)
    c = make_canvas(tmp_path / 'journal', 'alfworld')
    add(c)
    c.runtime.executor.tools = {'alfworld_step': SimpleNamespace(lifecycle=life)}
    life._rollout_steps = 1
    c.dirty_agents.add('solver')
    assert not c._unified_can_run('solver')
    assert not step(c, dict(action='run_agent', target='solver')).accepted


@pytest.mark.parametrize('tested,failed,ready',[(False,False,False),(True,False,True),(True,True,True)])
def test_swe_requires_post_edit_test_but_accepts_real_failed_tests(tmp_path,tested,failed,ready):
    c=make_canvas(tmp_path,'swe_bench')
    original=c.runtime.executor.execute
    def execute(**kwargs):
        artifact=original(**kwargs)
        artifact.code_artifact_ref=CodeArtifactRef('a'*64,'instance','repo','base',10,('code.py',))
        artifact.swe_progress={'trusted':True,'commit_ready':tested,'test_after_latest_edit':tested,
            'test_failure_count':int(failed)}
        return artifact
    c.runtime.executor.execute=execute
    add(c)
    assert c.submission_assessment('solver')['submit_ready'] is ready
    result=step(c,dict(action='finish',target='solver'),True)
    assert result.accepted is ready
    assert len(c.runtime.executor.calls)==1

@pytest.mark.parametrize('broken_report',[False,True])
def test_alfworld_continuation_keeps_episode_and_finish_trusts_environment(tmp_path,broken_report):
    from selfplay_graph_flowsteer.alfworld import ALFWorldSessionLifecycle
    from selfplay_graph_flowsteer.observability import TaskSpec
    from .test_alfworld_sessions import Client
    game=tmp_path/'game.tw-pddl';game.write_text('fixture')
    life=ALFWorldSessionLifecycle(Client(),tmp_path)
    life.bind_task(TaskSpec('task','Move mug',metadata={'game_path':str(game)}))
    c=make_canvas(tmp_path/'journal','alfworld')
    original=c.runtime.executor.execute
    c.runtime.executor.tools={'alfworld_step':SimpleNamespace(lifecycle=life)}
    def execute(**kwargs):
        artifact=original(**kwargs)
        state=life.begin_execution(agent_id=kwargs['node'].agent_id,seed=0,revision=kwargs['revision'])
        if not state['done']:
            life.step(state['admissible_actions'][0]['action_id'])
        life.end_execution()
        artifact.environment_result=life.result_for(artifact.agent_id)
        if broken_report:
            artifact.answer='WORKER_PROTOCOL_FAILURE'
            artifact.integrity_risks=['terminal_protocol_failure']
        return artifact
    c.runtime.executor.execute=execute
    add(c)
    assert life.client.created==1 and not life.result_for('solver')['won']
    result=step(c,dict(action='run_agent',target='solver'))
    assert result.accepted,result.feedback
    assert life.client.created==1 and life.result_for('solver')['won']
    assert c.submission_assessment('solver')['submit_ready']
    result=step(c,dict(action='finish',target='solver'),True)
    assert result.accepted,result.feedback
    assert len(c.runtime.executor.calls)==2
    assert c.submission_receipt.payload['won']
    life.close_all()


def test_alfworld_delete_does_not_refund_episode_budget(tmp_path):
    from selfplay_graph_flowsteer.alfworld import ALFWorldSessionLifecycle
    from selfplay_graph_flowsteer.observability import TaskSpec
    from .test_alfworld_sessions import Client
    game=tmp_path/'game.tw-pddl';game.write_text('fixture')
    life=ALFWorldSessionLifecycle(Client(),tmp_path,max_rollout_steps=1)
    life.bind_task(TaskSpec('task','Move mug',metadata={'game_path':str(game)}))
    state=life.begin_execution(agent_id='a',seed=0,revision=False)
    life.step(state['admissible_actions'][0]['action_id']);life.end_execution()
    life.discard_pending('a')
    assert life.client.closed==['1']
    with pytest.raises(RuntimeError,match='budget is exhausted'):
        life.begin_execution(agent_id='b',seed=0,revision=False)


def test_lost_navigation_response_invalidates_state_and_cannot_resume(tmp_path,monkeypatch):
    c,b,life,client=shopping(tmp_path,[report()]);add(c)
    def lost(*args,**kwargs):
        raise ConnectionError('navigation outcome unavailable')
    monkeypatch.setattr(client,'search',lost)
    life.begin_execution(agent_id='solver',seed=0,revision=True)
    with pytest.raises(ConnectionError):
        life.search('product')
    life.end_execution()
    assessment=c.submission_assessment('solver')
    assert 'environment_state_unknown' in assessment['blockers']
    assert not c._unified_can_run('solver')
    assert not step(c,dict(action='finish',target='solver'),True).accepted
    with pytest.raises(RuntimeError,match='unknown'):
        life.begin_execution(agent_id='solver',seed=0,revision=True)
    assert not client.commits


def test_generic_shopping_sessions_resume_and_are_not_first_owner_locked(tmp_path):
    c,b,life,client=shopping(tmp_path,[])
    life.require_native_actions=False
    life.set_submission_protocol('unified_task_result_v1')
    first=life.begin_execution(agent_id='a',seed=0,revision=False)
    life.search('product');life.end_execution()
    life.begin_execution(agent_id='b',seed=0,revision=False);life.end_execution()
    resumed=life.begin_execution(agent_id='a',seed=0,revision=True)
    assert resumed['page_type']=='search_results' and resumed['session_reused']
    assert life.owner_agent is None and life.owner_agents()==('a','b')
    life.end_execution();life.discard_pending('a')
    assert life.owner_agents()==('b',)
    assert not client.commits
    life.close_all()


def test_generic_executor_preflights_child_session_and_stages_purchase(tmp_path):
    from selfplay_graph_flowsteer.runtime import ModelAgentExecutor
    from .test_webshop_native import node
    def action(name, **arguments):
        return json.dumps({'action_call': {'name': name, 'arguments': arguments}})
    responses = [
        action('webshop_search', query='product'),
        action('webshop_click', target_id=f'open_product:0:{ASIN1}', state_version=1),
        action('webshop_click', target_id=f'purchase:{ASIN1}', state_version=2,
               purchase_evidence={'verified_requirements':['Product page matches the requested product'],
                                  'unresolved_constraints':[]}),
        report(),
    ]
    c,b,life,client = shopping(tmp_path, responses)
    life.require_native_actions = False
    native = c.runtime.executor
    executor = ModelAgentExecutor(b, tools=native.tools, action_registry=native.action_registry)
    n = node()
    n.metadata.update(submission_protocol='unified_task_result_v1', result_scope='task_result', task_dataset='webshop')
    artifact = executor.execute(task='Buy a product', node=n, upstream=[], peers=[], revision=False, seed=0)
    assert len(client.calls) == 2, artifact.to_dict()
    assert life.commit_ready_agents() == ('a',)
    assert not client.commits
    assert life.commit_pending('a')['purchased']
    assert len(client.commits) == 1
