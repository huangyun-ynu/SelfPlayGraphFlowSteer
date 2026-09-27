"""Shared task tool budgets across phases, graph edits and runtime routes."""
import json

import pytest

from selfplay_graph_flowsteer.canvas import GraphCanvas
from selfplay_graph_flowsteer.config import CanvasConfig
from selfplay_graph_flowsteer.contracts import AgentNode
from selfplay_graph_flowsteer.dataset_actions import default_dataset_action_registry
from selfplay_graph_flowsteer.llm import MockBackend
from selfplay_graph_flowsteer.runtime import ActionBudgetLedger, MultiAgentRuntime, RoutedModelAgentExecutor
from .test_runtime import FakePythonTool, FakeNamedTool
from .test_unified_submission import prompt, step


@pytest.mark.parametrize('total,initial,revision', [(10,3,1),(16,12,4),(100,50,50),(32,24,8)])
@pytest.mark.parametrize('first_phase', [False,True])
def test_all_capacity_is_usable_in_either_phase_and_cannot_overdraw(total,initial,revision,first_phase):
    node=AgentNode('a','Work',operation_policy_configured=True,
        initial_tool_budget=initial,revision_tool_budget=revision,total_tool_budget=total,
        metadata={'dataset_capability_policy':{'action_budget_policy':'shared_total_v1'}})
    ledger=ActionBudgetLedger()
    for i in range(total):
        phase=first_phase if i<total//2 else not first_phase
        assert ledger.remaining(node,revision=phase)['phase']==total-i
        assert ledger.consume(node,revision=phase)==(True,None)
    for phase in (False,True):
        assert ledger.remaining(node,revision=phase)['phase']==0
        assert ledger.consume(node,revision=phase)==(False,'total_action_budget_exhausted')
    assert ledger.usage['a'].total_used==total


def test_legacy_phase_caps_remain_opt_in_compatible():
    node=AgentNode('a','Work',initial_tool_budget=3,revision_tool_budget=1,total_tool_budget=4)
    ledger=ActionBudgetLedger()
    assert ledger.consume(node,revision=True)[0]
    assert ledger.consume(node,revision=True)==(False,'revision_action_budget_exhausted')
    assert ledger.remaining(node,revision=False)['total']==3


def make_canvas(tmp_path,responses,tool_limit=10):
    tool=FakePythonTool()
    tools={tool.name:tool, **{name:FakeNamedTool(name) for name in ('symbolic_compute','finite_search')}}
    registry=default_dataset_action_registry(tools,aime_budgets=(3,1,tool_limit),
        action_budget_policy='shared_total_v1')
    backend=MockBackend(responses)
    executor=RoutedModelAgentExecutor({'first':backend,'second':backend},('first','second'),
        tools=tools,action_registry=registry)
    canvas=GraphCanvas(task='Compute the integer',dataset='aime',runtime=MultiAgentRuntime(executor),
        action_adapter=registry.get('aime'),runtime_routes=('first','second'),config=CanvasConfig(
            submission_protocol='unified_task_result_v1',action_budget_policy='shared_total_v1',
            submission_journal_dir=str(tmp_path),max_rounds=80,max_total_tokens=350000))
    return canvas,backend,tool


CALL=json.dumps({'action_call':{'name':'python_exec','arguments':{'code':'print(1)'}}})
REPORT=json.dumps({'answer':'001','summary':'Computed'})


def create(canvas,name,scope='subtask',route='first'):
    assert step(canvas,dict(action='add_agent',agent_id=name)).accepted
    assert step(canvas,prompt(name,scope)).accepted
    result=step(canvas,dict(action='set_model',target=name,runtime_route=route))
    assert result.accepted,result.feedback


def test_aime_ten_calls_in_first_execution_and_finish_never_executes(tmp_path):
    c,b,t=make_canvas(tmp_path,[CALL]*10+[REPORT])
    create(c,'a','task_result')
    assert len(t.calls)==10
    assert c.runtime.shared_tool_budget_status(10)['remaining']==0
    assert c.runtime.artifacts['a'].answer=='001'
    before=len(b.calls)
    assert step(c,dict(action='finish',target='a'),True).accepted
    assert len(b.calls)==before and len(t.calls)==10


def test_approved_aime_six_tools_and_ten_edits_are_independent(tmp_path):
    from dataclasses import replace
    c,b,t=make_canvas(tmp_path,[CALL]*6+[REPORT],tool_limit=6)
    c.config=replace(c.config,director_budget_policy='edits_v1',max_director_edits=10)
    c.round_index=100
    create(c,'a','task_result')
    assert len(t.calls)==6
    assert c.director_edits_used==3
    assert c.control_snapshot()['action_budget']['remaining']==0
    assert 'round_budget' not in c.control_snapshot()
    assert step(c,dict(action='finish',target='a'),True).accepted
    assert len(t.calls)==6 and c.director_edits_used==3


def test_prompt_scope_route_new_node_and_recreation_share_one_account(tmp_path):
    c,b,t=make_canvas(tmp_path,[CALL,REPORT]*10+[REPORT]*10)
    create(c,'a')
    for scope in ('task_result','subtask'):
        assert step(c,prompt('a',scope)).accepted
    assert step(c,dict(action='set_model',target='a',runtime_route='second')).accepted
    assert len(t.calls)==4
    create(c,'b',route='second')
    assert len(t.calls)==5
    assert step(c,dict(action='delete_agent',target='a')).accepted
    create(c,'a')
    assert len(t.calls)==6
    # More than the previous one-call AIME revision limit remains available.
    for i in range(4):
        action=prompt('a','subtask')
        action['objective']=f'Check calculation {i}'
        assert step(c,action).accepted
    assert len(t.calls)==10
    assert step(c,dict(action='delete_agent',target='a')).accepted
    create(c,'a')
    assert len(t.calls)==10
    assert c.control_snapshot()['action_budget']['remaining']==0
    # Only a new question resets the account.
    c.reset()
    assert c.runtime.shared_tool_budget_status(10)['used']==0


@pytest.mark.parametrize('routed',[False,True])
@pytest.mark.parametrize('revision',[False,True])
def test_native_webshop_can_spend_all_sixteen_in_one_phase(routed,revision):
    from .test_webshop_native import build,execute,node,report
    ex,backend,life,client,_=build(['click[invented]']*16+[report()],routed=routed)
    n=node()
    n.metadata['dataset_capability_policy']={'action_budget_policy':'shared_total_v1'}
    a=execute(ex,n,revision=revision)
    assert len(a.react_trace)==16
    assert a.webshop_progress['action_budget']['total_used']==16
    assert a.webshop_progress['action_budget']['total_remaining']==0
    # Failed environment actions are charged, but never silently purchased.
    assert not client.commits


def test_shared_closure_transfers_only_remaining_total_once():
    node=AgentNode('a','Work',initial_tool_budget=12,revision_tool_budget=4,total_tool_budget=16,
        metadata={'dataset_capability_policy':{'action_budget_policy':'shared_total_v1'}})
    ledger=ActionBudgetLedger()
    for _ in range(7):
        assert ledger.consume(node,revision=True)[0]
    assert ledger.begin_webshop_closure(node,session_id='session',official_remaining_steps=100)
    assert ledger.remaining(node,revision=True,closure_session='session')['phase']==9
    assert not ledger.begin_webshop_closure(node,session_id='another',official_remaining_steps=100)
    for _ in range(9):
        assert ledger.consume(node,revision=True,closure_session='session')[0]
    assert not ledger.consume(node,revision=False,closure_session='session')[0]
