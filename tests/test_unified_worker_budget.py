"""Local results, bounded environment recovery and real gateway token admission."""
import json

import pytest

from selfplay_graph_flowsteer import llm
from selfplay_graph_flowsteer.canvas import GraphCanvas
from selfplay_graph_flowsteer.config import CanvasConfig
from selfplay_graph_flowsteer.contracts import AgentNode
from selfplay_graph_flowsteer.dataset_actions import default_dataset_action_registry
from selfplay_graph_flowsteer.runtime import ModelAgentExecutor, MultiAgentRuntime
from .test_runtime import FakeNamedTool
from .test_unified_submission import add, step


def alf_executor(responses):
    tool = FakeNamedTool('alfworld_step')
    # A real action can make progress without completing the full episode.
    def execute(arguments):
        tool.calls.append(arguments)
        return json.dumps({'status': 'ok', 'done': False, 'observation': 'CD acquired'})
    tool.execute = execute
    tools = {tool.name: tool}
    backend = llm.MockBackend(responses)
    return ModelAgentExecutor(backend, tools=tools, action_registry=default_dataset_action_registry(tools)), backend, tool


def alf_node(scope):
    return AgentNode('worker', 'Retrieve the CD', allowed_tools=('alfworld_step',),
        operation_policy_configured=True, initial_tool_budget=64, revision_tool_budget=16,
        total_tool_budget=80, metadata={'action_adapter': 'alfworld', 'task_dataset': 'alfworld',
        'submission_protocol': 'unified_task_result_v1', 'result_scope': scope})


def test_alfworld_local_result_stops_after_delegated_work_without_completing_episode():
    ex,b,t = alf_executor([
        '{"action_call":{"name":"alfworld_step","arguments":{"action_id":"public-action"}}}',
        '{"answer":"CD acquired","summary":"Lamp work remains"}',
    ])
    a=ex.execute(task='Examine CD under lamp',node=alf_node('subtask'),upstream=[],peers=[],revision=False,seed=0)
    assert len(b.calls)==2 and len(t.calls)==1
    assert a.answer=='CD acquired'
    assert a.alfworld_progress['state']=='incomplete'
    assert not any(d['stage']=='environment_action_required' for d in a.protocol_diagnostics)
    assert 'full episode may still be active' in b.calls[0]['messages'][0]['content']


@pytest.mark.parametrize('scope,expected_calls', [('subtask',2),('task_result',3)])
def test_bare_action_id_has_bounded_recovery_and_cannot_establish_completion(scope,expected_calls):
    ex,b,t=alf_executor(['{"action_id":"s0015:a016"}']*100)
    a=ex.execute(task='Examine CD under lamp',node=alf_node(scope),upstream=[],peers=[],revision=False,seed=0)
    # Two finalization attempts follow at most one action repair.
    assert len(b.calls)<=expected_calls+1
    assert not t.calls
    assert a.answer=='WORKER_PROTOCOL_FAILURE'
    assert a.alfworld_progress['state']=='incomplete'


class CreditedBackend:
    """Use gateway admission and charge actual serialized input; no paid API."""
    def __init__(self):
        self.credits=[]
        self.dispatched=[]
    def generate(self,messages,*,role,actions=(),max_tokens=None,**kwargs):
        credit=llm._TOKEN_CREDIT.get()
        assert credit is not None
        self.credits.append(credit.limit)
        request={'messages':messages,'max_tokens':max_tokens or 300000}
        quote=llm._request_credit_admission(request,credit)
        self.dispatched.append((credit.limit,quote))
        return llm.LLMResponse(text='{"answer":"Local evidence"}',model='fixture',
            token_in=quote['input_bound'],token_out=min(1000,request['max_tokens']))


@pytest.mark.parametrize('dataset',['alfworld','healthbench_professional','unknown_dataset'])
def test_unified_worker_requests_and_recovery_share_remaining_question_budget(tmp_path,dataset):
    b=CreditedBackend()
    c=GraphCanvas(task='Public task',dataset=dataset,runtime=MultiAgentRuntime(ModelAgentExecutor(b)),
        config=CanvasConfig(submission_protocol='unified_task_result_v1',
            submission_journal_dir=str(tmp_path),max_total_tokens=60000,max_rounds=30))
    add(c,'first','subtask')
    spent=c.total_tokens
    assert 0<spent<=60000
    add(c,'second','subtask')
    assert c.total_tokens<=60000
    assert c.graph.nodes['second'].metadata['_runtime_token_credit']==60000-spent
    assert all(q['required_tokens']<=limit for limit,q in b.dispatched)
    assert any(x<=60000-spent for x in b.credits)
    # A local result, including a failed bounded report, cannot be submitted.
    assert not step(c,dict(action='finish',target='second'),True).accepted


def test_failed_worker_is_not_implicitly_retried_by_an_unrelated_agent(tmp_path):
    b=llm.MockBackend(['{"answer":"WORKER_PROTOCOL_FAILURE"}','{"answer":"independent result"}'])
    c=GraphCanvas(task='Question',dataset='aime',runtime=MultiAgentRuntime(ModelAgentExecutor(b)),
        config=CanvasConfig(submission_protocol='unified_task_result_v1',
            submission_journal_dir=str(tmp_path),max_total_tokens=350000,max_rounds=30))
    add(c,'failed','subtask')
    first=c.runtime.artifacts['failed'].artifact_id
    assert not c.runtime.artifact_matches_current_input_signature('failed',task=c.worker_task,graph=c.graph)
    add(c,'independent','subtask')
    assert len(b.calls)==2
    assert c.runtime.artifacts['failed'].artifact_id==first
    assert c.runtime.artifacts['independent'].answer=='independent result'
    assert not c.submission_assessment('failed')['submit_ready']
    assert step(c,dict(action='run_agent',target='failed')).accepted
    assert len(b.calls)==3


def test_shared_credit_covers_additional_stale_node_discovered_after_estimation(tmp_path):
    b=CreditedBackend()
    runtime=MultiAgentRuntime(ModelAgentExecutor(b))
    c=GraphCanvas(task='Question',dataset='healthbench_professional',runtime=runtime,
        config=CanvasConfig(submission_protocol='unified_task_result_v1',
            submission_journal_dir=str(tmp_path),max_total_tokens=350000,max_rounds=30))
    add(c,'a','subtask');add(c,'b','subtask')
    # Canvas sees only b dirty; runtime also discovers stale a, like the captured
    # SWE trajectory. Force a small shared balance to exercise its hard boundary.
    runtime._stale_artifacts.add('a')
    for n in c.graph.nodes.values():
        n.metadata['_runtime_token_credit']=20000
    before=len(b.dispatched)
    report=runtime.execute(task=c.worker_task,graph=c.graph,dirty_agents={'b'},token_credit=20000)
    assert report.token_in+report.token_out<=20000
    assert b.dispatched[before][0]<=20000
    for limit,quote in b.dispatched[before+1:]:
        assert limit<20000
    assert 'a' in report.scheduled_agents


def test_captured_alfworld_failure_reports_are_local_results_not_navigation_requests():
    from pathlib import Path
    fixture=json.loads((Path(__file__).parent/'fixtures/unified_alfworld_local_result_failure.json').read_text())
    assert fixture['original_worker_tokens']>fixture['worker_limit']
    assert fixture['environment_action_required_count']>60
    for response in fixture['local_reports_rejected']:
        ex,b,t=alf_executor([response])
        a=ex.execute(task='Examine CD under lamp',node=alf_node('subtask'),upstream=[],peers=[],revision=False,seed=0)
        assert len(b.calls)==1 and not t.calls
        assert a.answer==json.loads(response)['answer']
        assert a.alfworld_progress['state']=='incomplete'
    ex,b,t=alf_executor([fixture['repeated_bare_action']]*80)
    a=ex.execute(task='Examine CD under lamp',node=alf_node('task_result'),upstream=[],peers=[],revision=False,seed=0)
    assert len(b.calls)==4 and not t.calls
    assert a.answer=='WORKER_PROTOCOL_FAILURE'


def test_explicit_continuation_does_not_implicitly_repeat_failed_peers():
    from selfplay_graph_flowsteer.graph import MultiAgentGraph
    from .helpers import NumericRecordingExecutor
    executor=NumericRecordingExecutor()
    original=executor.execute
    def fail(**kwargs):
        artifact=original(**kwargs)
        artifact.answer='WORKER_PROTOCOL_FAILURE'
        return artifact
    executor.execute=fail
    graph=MultiAgentGraph()
    for aid in ('a','b'):
        graph.add_agent(aid);graph.set_prompt(aid,'Inspect evidence '+aid)
        graph.nodes[aid].metadata.update(submission_protocol='unified_task_result_v1',result_scope='subtask')
    graph.set_relation('a','b','bidirectional')
    runtime=MultiAgentRuntime(executor)
    runtime.execute(task='Fix repository',graph=graph)
    before=len(executor.calls)
    report=runtime.execute(task='Fix repository',graph=graph,dirty_agents={'a'},
        invalidation_reasons={'a':{'explicit_continuation'}})
    assert len(executor.calls)==before+1
    assert report.executed_agents==['a']
    assert report.reused_agents==['b']
