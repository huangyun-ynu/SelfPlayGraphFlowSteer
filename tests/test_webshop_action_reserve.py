"""Public failure checkpoints + real runtime admission; no invented successful retry.

Saved actions/states are original captures. New completion_plan declarations below
are controlled regression inputs, not claims about historical model choices.
"""
import copy
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pytest

from selfplay_graph_flowsteer.webshop_action_reserve import PurchaseReservation, quote, POLICY
from selfplay_graph_flowsteer.runtime import AgentActionUsage, ModelAgentExecutor, MultiAgentRuntime
from selfplay_graph_flowsteer.contracts import AgentNode
from selfplay_graph_flowsteer.webshop_native import NativeWebShopLifecycle
from selfplay_graph_flowsteer.webshop import WebShopClickTool, WebShopSearchTool
from selfplay_graph_flowsteer.observability import TaskSpec
from selfplay_graph_flowsteer.dataset_actions import default_dataset_action_registry
from selfplay_graph_flowsteer.llm import MockBackend
from selfplay_graph_flowsteer.canvas import GraphCanvas
from selfplay_graph_flowsteer.config import CanvasConfig
from .test_unified_submission import step, prompt

DATA = json.loads((Path(__file__).parent/'fixtures/webshop_purchase_reserve_real_trajectories.json').read_text())
CASES = DATA['cases']


def turns(number):
    return [t for a in CASES[number]['artifacts'] for t in a['react_trace'] if t['observation']['status'] == 'ok']


def state(number, step):
    return copy.deepcopy(turns(number)[step-1]['observation']['output'])


def plan(number, s, options=None):
    # Only observed public state and the literal public task are used here.
    return dict(decision='reserve', asin=s['product']['asin'],
        options=copy.deepcopy(s.get('selected_options', {}) if options is None else options),
        requirement_quotes=[CASES[number]['task']], verified_requirements=[s['product'].get('title') or 'Public product inspected'],
        unresolved_constraints=[], accept_partial=False)


def admit(m, number, s, action, r, p=None, owner='a', task_result=True, binding='live-a'):
    args=copy.deepcopy(action['arguments'])
    if p is not None: args['completion_plan']=p
    return m.preflight(owner=owner,binding=binding,state=s,name=action['name'],arguments=args,
                       remaining=r,task_result=task_result,task=CASES[number]['task'])


@pytest.mark.parametrize(('number','checkpoint'), [('00428',14),('00040',12),('00060',12),('00107',12)])
def test_real_failure_exploration_rejected_before_charge(number,checkpoint):
    s=state(number,checkpoint); m=PurchaseReservation()
    # Declare the publicly available options the model selected in the real tail.
    final=state(number,16)
    p=plan(number,s,final.get('selected_options',{}))
    inspected=next(state(number,i) for i in range(checkpoint,0,-1)
                   if state(number,i).get('page_type')=='product' and
                   state(number,i).get('product',{}).get('asin')==s['product']['asin'])
    m.observe('a','live-a',inspected)
    err,_=admit(m,number,s,turns(number)[checkpoint]['action'],16-checkpoint,p)
    assert err['code']=='purchase_budget_reserved'
    assert m.reserved() <= 16-checkpoint
    assert m.plan['phase']=='complete' if 16-checkpoint <= m.reserved()+1 else m.plan['phase']=='explore'


def test_unrequested_scent_does_not_spend_an_option_step():
    s=state('00299',15);m=PurchaseReservation()
    assert 'scent' in s['unselected_option_groups']
    p=plan('00299',s,{'size':'3 pack'})
    purchase={'name':'webshop_click','arguments':{'target_id':'purchase:'+s['product']['asin']}}
    err,progress=admit(m,'00299',s,purchase,1,p)
    assert err is None and progress and m.reserved()==1
    # A wrong selected required value DOES require one more action.
    wrong=copy.deepcopy(s);wrong['selected_options']['size']='1 pack'
    assert quote(m.plan,wrong)==2


def test_search_reopen_discards_previous_selections_and_unknown_routes_are_unknown():
    s=state('00040',12);m=PurchaseReservation();p=plan('00040',s)
    m.observe('a','live-a',state('00040',11))
    admit(m,'00040',s,turns('00040')[12]['action'],4,p)
    assert quote(m.plan,state('00040',13))==4
    missing=copy.deepcopy(state('00040',13));missing['valid_subactions']=[]
    assert quote(m.plan,missing) is None


@pytest.mark.parametrize('number',['00064','00225'])
def test_local_owner_pauses_and_only_explicit_promotion_resumes(number):
    s=next(state(number,i) for i in range(12,0,-1) if state(number,i).get('page_type')=='product')
    m=PurchaseReservation();p=plan(number,s,{})
    action={'name':'webshop_click','arguments':{'target_id':'purchase:'+s['product']['asin']}}
    err,_=admit(m,number,s,action,2,p,task_result=False)
    assert err['details']['stop_reason']=='purchase_reserve_promote_owner'
    assert m.blocker('a',False,2)=='purchase_reserve_promote_owner'
    assert m.blocker('a',True,2) is None
    assert m.blocker('b',True,1)=='purchase_reserve_other_owner'
    assert 'completion_plan' not in m.snapshot(2,'b')
    assert m.reserved()==1
    err,progress=admit(m,number,s,action,2,task_result=True)
    assert err is None and progress


def test_unknown_candidate_invalid_options_and_new_session_cannot_keep_stale_plan():
    s=state('00428',14);action=turns('00428')[14]['action'];m=PurchaseReservation()
    p=plan('00428',s);p['asin']='FAKE_ASIN'
    assert admit(m,'00428',s,action,2,p)[0]['code']=='invalid_completion_plan'
    p=plan('00428',s);p['options']['color']='not a visible color'
    assert admit(m,'00428',s,action,2,p)[0]['code']=='invalid_completion_plan'
    assert m.plan is None
    m=PurchaseReservation();admit(m,'00428',s,action,2,plan('00428',s))
    m.observe('a','new-session',{'page_type':'search_results','valid_subactions':[]})
    assert m.plan is None and not m.seen.get('a')


class CheckpointClient:
    """Resume a captured checkpoint in the offline lifecycle; never contact/scorer-buy."""
    def __init__(self,s):self.initial=copy.deepcopy(s);self.calls=[]
    def create_session(self,goal_id,seed):return dict(self.initial,session_id='captured-checkpoint')
    def search(self,sid,query):self.calls.append(query);raise AssertionError('Rejected search reached environment')
    def click(self,sid,target):self.calls.append(target);raise AssertionError('Rejected click reached environment')
    def close_session(self,sid):pass
    def commit(self,*args,**kwargs):raise AssertionError('Offline regression cannot invent an official result')


def checkpoint_canvas(tmp_path,number,n,action_responses,scope='task_result'):
    s=state(number,n);client=CheckpointClient(s)
    life=NativeWebShopLifecycle(client,require_native_actions=False,stage_purchases=True)
    life.bind_task(TaskSpec('webshop/goal-'+number,CASES[number]['task'],metadata={'goal_id':'goal-'+number}))
    tools={'webshop_search':WebShopSearchTool(life),'webshop_click':WebShopClickTool(life)}
    registry=default_dataset_action_registry(tools,webshop_commit_on_finish=True,action_budget_policy='shared_total_v1')
    # Saved real model final response follows controlled admission probes.
    final=CASES[number]['artifacts'][-1]['raw_response']
    backend=MockBackend([json.dumps({'action_call':a}) for a in action_responses]+[final]*8)
    executor=ModelAgentExecutor(backend,tools=tools,action_registry=registry,
        webshop_worker_guidance_policy='merged_checklist_v1',webshop_purchase_budget_policy=POLICY)
    c=GraphCanvas(task=CASES[number]['task'],dataset='webshop',runtime=MultiAgentRuntime(executor),
        action_adapter=registry.get('webshop'),config=CanvasConfig(submission_protocol='unified_task_result_v1',
        submission_journal_dir=str(tmp_path),action_budget_policy='shared_total_v1',max_rounds=40,
        max_total_tokens=1000000,remaining_token_admission_enabled=False))
    executor.budget_ledger.usage['tool-rollout:whole-graph']=AgentActionUsage(initial_used=n,total_used=n)
    # The checkpoint is charged for ALL its actual preceding actions, with no refund.
    assert step(c,dict(action='add_agent',agent_id='a')).accepted
    result=step(c,prompt('a',scope))
    assert result.accepted,result.feedback
    return c,life,client,backend


def test_actual_runtime_rejects_boundary_action_without_environment_or_ledger_charge(tmp_path):
    number='00428';s=state(number,14)
    action=copy.deepcopy(turns(number)[14]['action']);action['arguments']['completion_plan']=plan(number,s)
    c,life,client,backend=checkpoint_canvas(tmp_path,number,14,[action]*3)
    art=c.runtime.artifacts['a']
    assert not client.calls and c.runtime.shared_tool_budget_status(16)['remaining']==2
    assert len(art.react_trace)==3
    assert all(t['observation']['error']['code']=='purchase_budget_reserved' for t in art.react_trace)
    assert art.webshop_progress['stop_reason']=='purchase_plan_repair_exhausted'
    assert not c._unified_can_run('a') and not life.commit_ready_agents()
    assert c.submission_assessment('a')['payload_kind']=='environment_failure'
    assert 'purchase_reservation' in json.loads(backend.calls[0]['messages'][1]['content'])['action_environment']
    assert 'completion_plan' in backend.calls[0]['actions'][0]['parameters']['properties']


def test_recorded_final_buy_stages_with_one_action_and_bad_evidence_is_free(tmp_path):
    number='00387';s=state(number,15)
    raw=[t for a in CASES[number]['artifacts'] for t in a['react_trace']]
    buys=[copy.deepcopy(t['action']) for t in raw if t['action']['arguments'].get('target_id','').startswith('purchase:')]
    assert len(buys)==2
    # First captured request lacked purchase evidence; second is the actual corrected Buy.
    buys[1]['arguments']['completion_plan']=plan(number,s)
    c,life,client,backend=checkpoint_canvas(tmp_path,number,15,buys)
    assert not client.calls  # Existing staged protocol contacts official scorer only at FINISH.
    assert c.runtime.shared_tool_budget_status(16)['remaining']==0
    assert life.commit_ready_agents()==('a',)
    assert c.submission_assessment('a')['submit_ready']
    assert c.runtime.artifacts['a'].react_trace[0]['observation']['status']=='error'
    assert c.runtime.artifacts['a'].react_trace[1]['observation']['status']=='ok'


def test_canvas_pause_is_local_delete_is_guarded_and_promote_preserves_session(tmp_path):
    number='00428';s=state(number,14)
    action=copy.deepcopy(turns(number)[14]['action']);action['arguments']['completion_plan']=plan(number,s)
    c,life,client,backend=checkpoint_canvas(tmp_path,number,14,[action],scope='subtask')
    assert c.runtime.artifacts['a'].webshop_progress['state']=='purchase_reserve_paused'
    assert 'local_result_only' in c.submission_assessment('a')['blockers']
    assert not c.runtime.artifacts['a'].webshop_progress['policy_failure']
    assert 'integrity:terminal_tool_failure' not in c.submission_assessment('a')['blockers']
    from selfplay_graph_flowsteer.runtime import _terminal_tool_failure
    assert not _terminal_tool_failure(c.runtime.artifacts['a'])
    assert not c._unified_can_run('a')
    before=len(backend.calls);sid=life.session_binding('a')
    assert not step(c,dict(action='run_agent',target='a')).accepted
    deletion=step(c,dict(action='delete_agent',target='a'))
    assert not deletion.accepted and deletion.rejection_code=='purchase_budget_reserved'
    assert len(backend.calls)==before
    buy={'name':'webshop_click','arguments':{'target_id':'purchase:'+s['product']['asin'],
        'state_version':s['state_version'],'purchase_evidence':{'verified_requirements':['Recorded public product'], 'unresolved_constraints':[]}}}
    backend.responses.clear();backend.responses.extend([json.dumps({'action_call':buy}),CASES[number]['artifacts'][-1]['raw_response']]*2)
    assert step(c,prompt('a')).accepted
    assert life.session_binding('a')==sid and life.commit_ready_agents()==('a',)
    assert c.runtime.shared_tool_budget_status(16)['remaining']==1
    # This added stage is a controlled positive protocol check, NOT a real task success.
    assert c.submission_receipt is None


def test_explicit_abandon_does_not_execute_attached_action_or_claim_exhaustion(tmp_path):
    a=copy.deepcopy(turns('00428')[14]['action']);a['arguments']['completion_plan']={'decision':'abandon','reason':'Candidate has unresolved public requirements; decline partial purchase.'}
    c,life,client,_=checkpoint_canvas(tmp_path,'00428',14,[a])
    assert not client.calls and not life.commit_ready_agents()
    assert c.runtime.shared_tool_budget_status(16)['remaining']==2
    assert c.runtime.artifacts['a'].webshop_progress['policy_failure']['code']=='purchase_plan_abandoned'


def test_rollout_lock_serializes_competing_shared_budget_admission():
    # Same lock held by real ModelAgentExecutor around lifecycle + preflight + consume.
    from selfplay_graph_flowsteer.runtime import ActionBudgetLedger
    ledger=ActionBudgetLedger();m=ledger.purchase_reserve
    s=state('00428',14);a=copy.deepcopy(turns('00428')[14]['action'])
    admit(m,'00428',s,a,2,plan('00428',s))  # Protect one final Buy.
    n=AgentNode(agent_id='b',total_tool_budget=16,initial_tool_budget=12,revision_tool_budget=4)
    n.metadata['dataset_capability_policy']={'action_budget_policy':'shared_total_v1'}
    ledger.usage['all']=AgentActionUsage(total_used=14)
    def attempt(_):
        with m.lock:
            r=ledger.remaining(n,revision=False,scope='all')['total']
            error,_=admit(m,'00428',{'page_type':'search_results','valid_subactions':[]},
                         {'name':'webshop_search','arguments':{'query':'public'}},r,owner='b',binding='live-b')
            return False if error else ledger.consume(n,revision=False,scope='all')[0]
    with ThreadPoolExecutor(max_workers=2) as pool: result=list(pool.map(attempt,range(2)))
    assert sum(result)==1 and ledger.remaining(n,revision=False,scope='all')['total']==1


def test_config_switch_parses_and_preserves_shared_formal_contract(monkeypatch):
    from dataclasses import replace
    from .test_webshop_formal_promotion import load_formal
    config=load_formal('formal_training.toml',monkeypatch)
    assert config.webshop.purchase_budget_policy=='completion_reserve_v2'
    config=replace(config,webshop=replace(config.webshop,purchase_budget_policy=POLICY))
    config.validate()
    assert config.model_manifest()['webshop']['purchase_budget_policy']==POLICY
    with pytest.raises(ValueError,match='purchase_budget_policy'):
        replace(config.webshop,purchase_budget_policy='unknown').validate()


def test_unknown_environment_mutation_invalidates_reservation_without_refund():
    s=state('00428',14);m=PurchaseReservation()
    admit(m,'00428',s,turns('00428')[14]['action'],2,plan('00428',s))
    m.observe('a','live-a',dict(s,resource_status='unknown',termination_reason='environment_step_failed'))
    assert m.plan is None and m.events[-1]['reason']=='environment_state_unknown'


def test_no_candidate_does_not_automatically_select_or_purchase():
    m=PurchaseReservation()
    err,progress=admit(m,'00064',{'page_type':'search_results','valid_subactions':[]},
        {'name':'webshop_search','arguments':{'query':'public task'}},1)
    assert err is None and not progress and m.plan is None


def test_owner_reexecution_keeps_reservation_and_cannot_refund_steps():
    s=state('00428',14);m=PurchaseReservation()
    admit(m,'00428',s,turns('00428')[14]['action'],2,plan('00428',s))
    version=m.version
    m.observe('a','live-a',s)
    assert m.version==version and m.reserved()==1
    assert m.snapshot(1)['remaining']==1
    assert m.blocker('other',True,1)=='purchase_reserve_other_owner'


def test_full_real_00040_prefix_replays_and_search_tail_is_rejected(tmp_path):
    """Replay 12 actual transitions, add plan metadata, reject original step 13."""
    from .test_webshop_real_trajectory_regression import CapturedClient
    number='00040';case=CASES[number];prefix=copy.deepcopy(turns(number)[:12])
    client=CapturedClient({'artifacts':[{'react_trace':prefix}]})
    p=plan(number,state(number,11))
    prefix[10]['action']['arguments']['completion_plan']=p
    blocked=copy.deepcopy(turns(number)[12]['action'])
    backend=MockBackend([json.dumps({'action_call':t['action']}) for t in prefix]
        +[json.dumps({'action_call':blocked})]*3+[case['artifacts'][-1]['raw_response']]*8)
    life=NativeWebShopLifecycle(client,require_native_actions=False,stage_purchases=True)
    life.bind_task(TaskSpec('webshop/goal-'+number,case['task'],metadata={'goal_id':'goal-'+number}))
    tools={'webshop_search':WebShopSearchTool(life),'webshop_click':WebShopClickTool(life)}
    registry=default_dataset_action_registry(tools,webshop_commit_on_finish=True,action_budget_policy='shared_total_v1')
    executor=ModelAgentExecutor(backend,tools=tools,action_registry=registry,
        webshop_worker_guidance_policy='merged_checklist_v1',webshop_purchase_budget_policy=POLICY)
    c=GraphCanvas(task=case['task'],dataset='webshop',runtime=MultiAgentRuntime(executor),
        action_adapter=registry.get('webshop'),config=CanvasConfig(submission_protocol='unified_task_result_v1',
        submission_journal_dir=str(tmp_path),action_budget_policy='shared_total_v1',max_rounds=40,
        max_total_tokens=1000000,remaining_token_admission_enabled=False))
    assert step(c,dict(action='add_agent',agent_id='a')).accepted
    assert step(c,prompt('a')).accepted
    assert not client.observations and len(client.calls)==12
    assert c.runtime.shared_tool_budget_status(16)['remaining']==4
    a=c.runtime.artifacts['a']
    assert a.webshop_progress['stop_reason']=='purchase_plan_repair_exhausted'
    assert len(a.react_trace)==15
    assert all(t['observation']['status']=='ok' for t in a.react_trace[:12])
    assert all(t['observation']['error']['code']=='purchase_budget_reserved' for t in a.react_trace[12:])
    assert not life.commit_ready_agents() and c.submission_receipt is None
