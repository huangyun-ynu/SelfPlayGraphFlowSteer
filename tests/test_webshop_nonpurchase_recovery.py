"""Actual public observations/failed requests, replayed against completion v2.

New Buy calls are controlled staging probes, never fabricated official scores.
"""
import copy
import json
from pathlib import Path
from dataclasses import replace

import pytest

from selfplay_graph_flowsteer.webshop_action_reserve import PurchaseReservation, FLEXIBLE_POLICY, POLICY
from selfplay_graph_flowsteer.runtime import ModelAgentExecutor, MultiAgentRuntime, AgentActionUsage
from selfplay_graph_flowsteer.canvas import GraphCanvas
from selfplay_graph_flowsteer.config import CanvasConfig
from selfplay_graph_flowsteer.dataset_actions import default_dataset_action_registry
from selfplay_graph_flowsteer.llm import MockBackend
from selfplay_graph_flowsteer.observability import TaskSpec
from selfplay_graph_flowsteer.webshop import WebShopClickTool, WebShopSearchTool
from selfplay_graph_flowsteer.webshop_native import NativeWebShopLifecycle
from .test_unified_submission import step, prompt
from .test_webshop_action_reserve import CheckpointClient
from .test_webshop_real_trajectory_regression import CapturedClient

DATA = json.loads((Path(__file__).parent/'fixtures/webshop_nonpurchase_real_trajectories.json').read_text())
CASES = DATA['cases']


def turns(number):
    return [t for a in CASES[number]['artifacts'] for t in a['react_trace']]


def successful(number):
    return [t for t in turns(number) if t['observation']['status'] == 'ok']


def replay_to_error(number, code, policy=FLEXIBLE_POLICY):
    manager = PurchaseReservation(policy=policy)
    current = {}; used = 0
    for t in turns(number):
        if t['observation']['status'] == 'ok':
            current = copy.deepcopy(t['observation']['output']); used += 1
            manager.observe('a', 'captured-session', current)
        elif t['observation']['error']['code'] == code:
            return manager, current, t['action'], 16-used
    raise AssertionError((number, code))


def preflight(m, number, state, action, remaining):
    return m.preflight(owner='a', binding='captured-session', state=state,
        name=action['name'], arguments=action['arguments'], remaining=remaining,
        task_result=True, task=CASES[number]['task'])


@pytest.mark.parametrize('number', ['00024','00090','00107','00157','00433','00440'])
def test_actual_uncertainty_rejections_now_allow_legal_progress(number):
    m, s, action, remaining = replay_to_error(number, 'invalid_completion_plan')
    assert action['arguments']['completion_plan']['unresolved_constraints']
    error, _ = preflight(m, number, s, action, remaining)
    assert error is None
    assert m.reserved() <= remaining
    # The recorded uncertainty remains visible; no evidence is silently cleared.
    assert m.plan['unresolved_constraints'] == action['arguments']['completion_plan']['unresolved_constraints']


@pytest.mark.parametrize('number', ['00015','00105','00107','00256','00453','00471'])
def test_actual_missing_plan_requests_can_still_inspect_navigate_or_select(number):
    m, s, action, remaining = replay_to_error(number, 'completion_plan_required')
    assert remaining == 4 and 'completion_plan' not in action['arguments']
    error, _ = preflight(m, number, s, action, remaining)
    assert error is None and m.plan is None  # No automatic candidate choice.


def test_invalid_optional_updates_never_establish_unseen_candidates_or_options():
    old, s, action, r = replay_to_error('00440','invalid_completion_plan', POLICY)
    assert preflight(old,'00440',s,action,r)[0]['code'] == 'invalid_completion_plan'
    new, s, action, r = replay_to_error('00440','invalid_completion_plan')
    a = copy.deepcopy(action); a['arguments']['completion_plan']['asin'] = 'UNSEEN'
    assert preflight(new,'00440',s,a,r)[0] is None
    assert new.plan is None and new.plan_feedback['a']['code']=='invalid_completion_plan'
    a = copy.deepcopy(action); a['arguments']['completion_plan']['options']['color'] = 'INVENTED'
    assert preflight(new,'00440',s,a,r)[0] is None
    assert new.plan is None and new.plan_feedback['a']['code']=='invalid_completion_plan'
    assert any(v['target_id']==a['arguments']['target_id'] for v in s['valid_subactions'])


@pytest.mark.parametrize(('kind','remaining','allowed'), [
    ('purchase',1,True),('select_option',1,False),('select_option',2,True),
    ('view_section',2,False),('view_section',3,True),('search',2,False),('search',3,True)])
def test_no_plan_preserves_a_minimum_purchase_path(kind,remaining,allowed):
    s = copy.deepcopy(successful('00440')[8]['observation']['output'])
    if kind == 'search': action = {'name':'webshop_search','arguments':{'query':'public alternative'}}
    else:
        target = next(a for a in s['valid_subactions'] if a['kind']==kind)
        action = {'name':'webshop_click','arguments':{'target_id':target['target_id']}}
    m = PurchaseReservation(policy=FLEXIBLE_POLICY)
    error, _ = preflight(m,'00440',s,action,remaining)
    assert (error is None) == allowed
    assert m.plan is None
    if error:
        assert error['code']=='purchase_budget_reserved' and not error['details']['plan_required']


def make_canvas(tmp_path, number, client, responses, *, charged=0, research_used=0, scope='task_result'):
    life = NativeWebShopLifecycle(client, require_native_actions=False, stage_purchases=True)
    life.bind_task(TaskSpec('webshop/goal-'+number, CASES[number]['task'], metadata={'goal_id':'goal-'+number}))
    tools = {'webshop_search':WebShopSearchTool(life),'webshop_click':WebShopClickTool(life)}
    registry = default_dataset_action_registry(tools,webshop_commit_on_finish=True,action_budget_policy='shared_total_v1')
    backend = MockBackend(list(responses) + [CASES[number]['artifacts'][-1]['raw_response']]*16)
    executor = ModelAgentExecutor(backend,tools=tools,action_registry=registry,
        webshop_worker_guidance_policy='merged_checklist_v1',webshop_purchase_budget_policy=FLEXIBLE_POLICY,
        webshop_scheduling_policy='bounded_research_v1')
    c = GraphCanvas(task=CASES[number]['task'],dataset='webshop',runtime=MultiAgentRuntime(executor),
        action_adapter=registry.get('webshop'),config=CanvasConfig(submission_protocol='unified_task_result_v1',
        submission_journal_dir=str(tmp_path),action_budget_policy='shared_total_v1',max_rounds=40,
        max_total_tokens=1000000,remaining_token_admission_enabled=False))
    assert step(c,dict(action='add_agent',agent_id='a')).accepted
    executor.budget_ledger.usage['tool-rollout:whole-graph'] = AgentActionUsage(initial_used=charged,total_used=charged,research_used=research_used)
    result = step(c,prompt('a',scope))
    assert result.accepted,result.feedback
    return c,life,backend


def test_real_00440_prefix_handoff_and_previously_rejected_selection_can_reach_buy(tmp_path):
    number='00440'; prefix=copy.deepcopy(successful(number)[:9])
    client=CapturedClient({'artifacts':[{'react_trace':prefix}]})
    c,life,backend=make_canvas(tmp_path,number,client,
        [json.dumps({'action_call':t['action']}) for t in prefix[:8]],scope='subtask')
    assert len(client.calls)==8 and c.runtime.webshop_scheduling_status(16)['research_used']==8
    sid=life.session_binding('a')
    _,_,rejected,_=replay_to_error(number,'invalid_completion_plan')
    final_state=prefix[8]['observation']['output']
    buy={'name':'webshop_click','arguments':{'target_id':'purchase:'+final_state['product']['asin'],
        'state_version':final_state['state_version'],
        'purchase_evidence':{'verified_requirements':['Observed quick-release band, teal selected, displayed range below $40'],
            'unresolved_constraints':['Exact variant price not individually shown; entire range is below the ceiling']}}}
    backend.responses.clear(); backend.responses.extend([
        json.dumps({'action_call':rejected}),json.dumps({'action_call':buy}),CASES[number]['artifacts'][-1]['raw_response']])
    assert step(c,prompt('a')).accepted
    assert len(client.calls)==9 and not client.observations
    assert life.session_binding('a')==sid and life.commit_ready_agents()==('a',)
    assert c.runtime.shared_tool_budget_status(16)['remaining']==6
    assert c.submission_assessment('a')['submit_ready'] and c.submission_receipt is None
    assert c.runtime.artifacts['a'].webshop_progress['purchase_reservation']['policy']==FLEXIBLE_POLICY


def test_actual_00417_report_gets_one_truthful_scope_reminder_then_bounded_failure(tmp_path):
    number='00417'; s=successful(number)[-1]['observation']['output']
    client=CheckpointClient(s)
    c,life,backend=make_canvas(tmp_path,number,client,[],charged=9,research_used=8)
    assert not client.calls and c.runtime.shared_tool_budget_status(16)['remaining']==7
    context=json.loads(backend.calls[0]['messages'][1]['content'])
    schedule=context['action_environment']['scheduling']
    assert schedule['current_scope']=='task_result' and schedule['current_action_allowance']==7
    assert schedule['research_remaining']==0 and not schedule['local_research_limit_applies']
    art=c.runtime.artifacts['a']
    assert sum(d['stage']=='webshop_completion_scope_reminder' for d in art.protocol_diagnostics)==1
    assert c._unified_can_run('a')
    assert step(c,dict(action='run_agent',target='a')).accepted
    assert not any(d['stage']=='webshop_completion_scope_reminder' for d in c.runtime.artifacts['a'].protocol_diagnostics)
    assert c.submission_assessment('a')['submit_ready'] and not c._unified_can_run('a')


def test_flexible_policy_config_manifest(monkeypatch):
    from .test_webshop_formal_promotion import load_formal
    config=load_formal('formal_training.toml',monkeypatch)
    config=replace(config,webshop=replace(config.webshop,purchase_budget_policy=FLEXIBLE_POLICY))
    config.validate()
    assert config.model_manifest()['webshop']['purchase_budget_policy']==FLEXIBLE_POLICY


CORRECTIONS = json.loads((Path(__file__).parent/'fixtures/webshop_nonpurchase_correction_real_trajectories.json').read_text())['cases']


@pytest.mark.parametrize('number', ['00457'])
def test_real_late_rejection_delivers_actionable_next_request_correction(tmp_path, monkeypatch, number):
    captured = CORRECTIONS[number]
    monkeypatch.setitem(CASES, number, captured)
    ts = [t for a in captured['artifacts'] for t in a['react_trace']]
    legal = [t for t in ts if t['observation']['status']=='ok']
    bad = next(t for t in ts if t['observation']['status']=='error')
    state = legal[-1]['observation']['output']
    client = CheckpointClient(state)
    c,life,backend = make_canvas(tmp_path,number,client,
        [json.dumps({'action_call':bad['action']})],charged=len(legal))
    feedback = json.loads(backend.calls[1]['messages'][-1]['content'])['webshop_action_correction']
    assert feedback['last_error_code']==bad['observation']['error']['code']
    assert feedback['remaining_actions']==16-len(legal)
    assert not client.calls and c.runtime.shared_tool_budget_status(16)['used']==len(legal)
    choices=feedback['public_completion_choices']
    assert any(x['kind']=='purchase' for x in choices)
    assert not any(x['kind']=='view_section' for x in choices)


def test_real_00105_bad_optional_plan_allows_the_unchanged_public_open_action():
    captured=CORRECTIONS['00105']
    traces=[t for a in captured['artifacts'] for t in a['react_trace']]
    legal=[t for t in traces if t['observation']['status']=='ok']
    bad=next(t for t in traces if t['observation']['status']=='error')
    state=copy.deepcopy(legal[-1]['observation']['output'])
    target=bad['action']['arguments']['target_id']
    assert any(a['target_id']==target and a['kind']=='open_product' for a in state['valid_subactions'])
    manager=PurchaseReservation(policy=FLEXIBLE_POLICY)
    error,_=manager.preflight(owner='a',binding='live',state=state,name=bad['action']['name'],
        arguments=bad['action']['arguments'],remaining=16-len(legal),task_result=True,task=captured['task'])
    assert error is None and manager.plan is None
    assert manager.snapshot(16-len(legal),'a')['plan_update_feedback']['code']=='invalid_completion_plan'


def test_real_repeated_search_gets_strategy_feedback_before_fuse(tmp_path):
    from selfplay_graph_flowsteer.webshop_action_reserve import recovery_feedback
    ts = [t for a in CORRECTIONS['00271']['artifacts'] for t in a['react_trace'] if t['observation']['status']=='ok']
    state = copy.deepcopy(ts[-1]['observation']['output']); before=copy.deepcopy(state)
    correction=recovery_feedback(state,8,no_progress=2,searches=2)
    assert correction and any('Change strategy now' in s for s in correction['instructions'])
    assert correction['public_completion_choices'] and state==before


def test_policy_is_correct_before_first_worker_and_after_reset():
    executor=ModelAgentExecutor(MockBackend([]),webshop_purchase_budget_policy=FLEXIBLE_POLICY)
    runtime=MultiAgentRuntime(executor)
    for _ in range(2):
        assert runtime.shared_tool_budget_status(16)['purchase_reservation']['policy']==FLEXIBLE_POLICY
        executor.reset()
