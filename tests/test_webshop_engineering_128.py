"""Engineering regressions on the latest 128 real public trajectories.

No new model outcome or reward is inferred from these deterministic replays.
"""
import copy
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from selfplay_graph_flowsteer.webshop_action_reserve import (
    FLEXIBLE_POLICY, POLICY, PurchaseReservation, minimum_completion_cost, quote,
)
from selfplay_graph_flowsteer.webshop_navigation import annotate_navigation
from selfplay_graph_flowsteer.webshop_sidecar import WebShopSession

DATA = json.loads(gzip.decompress((Path(__file__).parent/'fixtures/webshop_engineering_128_real_trajectories.json.gz').read_bytes()))
CASES = DATA['cases']
RELEASES = [(number, p) for number, case in CASES.items() for p in case['section_plan_releases']]


def raw_state(state):
    state = copy.deepcopy(state)
    for action in state.get('valid_subactions', []):
        action.pop('navigation_effect', None)
    return state


@pytest.mark.parametrize(('number', 'release'), RELEASES,
                         ids=[f'{n}-{p["index"]}' for n,p in RELEASES])
def test_all_33_real_section_transitions_keep_the_existing_reservation(number, release):
    traces = CASES[number]['trace']
    section = raw_state(traces[release['index']]['observation']['output'])
    before = next(raw_state(t['observation']['output']) for t in reversed(traces[:release['index']])
                  if t['observation']['status']=='ok')
    manager = PurchaseReservation(policy=FLEXIBLE_POLICY)
    manager.observe('agent_1', 'live', before)
    manager.plan = copy.deepcopy(release['plan'])
    manager.plan['binding'] = 'live'
    plan = copy.deepcopy(manager.plan)
    assert section['page_type']=='product_section'
    cost = quote(plan, section)
    assert cost is not None and cost >= 2
    manager.observe('agent_1', 'live', section)
    assert manager.plan == plan and manager.reserved()==cost
    assert not any(e['event']=='released' for e in manager.events)
    # Caller data need not carry prompt annotations for the budget to work.
    assert all('navigation_effect' not in a for a in section['valid_subactions'])


@pytest.mark.parametrize(('page','effect','cost'), [
    ('product_section','return_to_current_product_page',2),
    ('product','return_to_search_results',3),
    ('search_results','previous_results_page',3),
])
def test_navigation_uses_page_semantics_even_if_old_journal_label_is_wrong(page,effect,cost):
    state={'page_type':page,'valid_subactions':[{'kind':'navigate','target_id':'previous_page:4',
        'navigation_effect':'return_to_current_product_page'}]}
    assert minimum_completion_cost(state,'webshop_click',{'target_id':'previous_page:4'})==cost
    annotate_navigation(state)
    assert state['valid_subactions'][0]['navigation_effect']==effect
    from selfplay_graph_flowsteer.runtime import _annotate_webshop_product_state
    _annotate_webshop_product_state(state,product_inspections={})
    assert state['valid_subactions'][0]['navigation_effect']==effect


def checkpoint(number, code, *, last=False):
    manager=PurchaseReservation(policy=FLEXIBLE_POLICY); state={}; found=[]
    for trace in CASES[number]['trace']:
        if trace['observation']['status']=='ok':
            state=raw_state(trace['observation']['output'])
            manager.observe('agent_1','live',state)
        elif trace['observation']['error']['code']==code:
            found.append((copy.deepcopy(state),trace))
            if not last:
                return manager,state,trace
    state,trace=found[-1]
    return manager,state,trace


def admit(manager, number, state, trace, **kwargs):
    return manager.preflight(owner='agent_1',binding='live',state=state,
        name=trace['action']['name'],arguments=trace['action']['arguments'],
        remaining=kwargs.get('remaining',trace['remaining_budget']['total']),
        task_result=True,task=CASES[number]['task'])


@pytest.mark.parametrize(('number','code'), [('00433','invalid_completion_plan'),
    ('00417','invalid_completion_plan'),('00221','invalid_completion_plan'),
    ('00015','purchase_budget_reserved'),('00328','purchase_budget_reserved'),
    ('00358','purchase_budget_reserved')])
def test_bad_optional_plan_does_not_block_real_affordable_nonpurchase_action(number,code):
    manager,state,trace=checkpoint(number,code)
    original=copy.deepcopy(trace)
    for _ in range(3):
        error,_=admit(manager,number,state,trace)
        assert error is None and manager.plan is None
        assert manager.snapshot(trace['remaining_budget']['total'],'agent_1')['plan_update_feedback']['status']=='ignored'
        assert manager.rejects.get('agent_1',0)==0 and not manager.blocked
    assert trace==original  # Requested action/options are never silently rewritten.


def test_00163_remains_blocked_when_actual_completion_needs_more_actions():
    manager,state,trace=checkpoint('00163','purchase_budget_reserved',last=True)
    assert trace['remaining_budget']['total']==2
    error,_=admit(manager,'00163',state,trace)
    assert error['code']=='purchase_budget_reserved' and error['details']['minimum_required_actions']==3
    assert manager.plan is None


def test_invalid_update_preserves_existing_plan_and_cannot_spend_its_reserve():
    number,release=RELEASES[0]
    trace=CASES[number]['trace'][release['index']]
    state=raw_state(trace['observation']['output'])
    manager=PurchaseReservation(policy=FLEXIBLE_POLICY)
    manager.observe('agent_1','live',state)
    manager.plan=copy.deepcopy(release['plan']); manager.plan['binding']='live'
    remaining=manager.reserved()
    original=copy.deepcopy(manager.plan)
    bad={'action':{'name':'webshop_search','arguments':{'query':'different product',
         'completion_plan':{'decision':'reserve','asin':'UNOBSERVED','options':{}}}},
         'remaining_budget':{'total':remaining}}
    error,_=admit(manager,number,state,bad)
    assert error['code']=='purchase_budget_reserved'
    assert manager.plan['asin']==original['asin'] and manager.plan['options']==original['options']
    assert manager.reserved()==remaining
    assert manager.snapshot(remaining,'agent_1')['plan_update_feedback']['existing_plan_retained']
    assert 'plan_update_feedback' not in manager.snapshot(remaining,'other_agent')


@pytest.mark.parametrize('kind',['bad_buy_plan','bad_buy_options','abandon','invalid_abandon','legacy'])
def test_recovery_does_not_bypass_purchase_intent_or_explicit_stop(kind):
    state=next(raw_state(t['observation']['output']) for t in CASES['00232']['trace']
               if t['observation'].get('output',{}).get('page_type')=='product')
    manager=PurchaseReservation(policy=POLICY if kind=='legacy' else FLEXIBLE_POLICY)
    buy=next(a for a in state['valid_subactions']if a['kind']=='purchase')
    args={'target_id':buy['target_id'],'completion_plan':{'decision':'reserve','asin':'UNOBSERVED','options':{}}}
    if kind=='bad_buy_options':
        args['completion_plan'].update(asin=state['product']['asin'],options={'color':'INVENTED'})
    if 'abandon' in kind:
        args['completion_plan']={'decision':'abandon','reason':'Stop without buying' if kind=='abandon' else ''}
    name='webshop_click'
    if kind=='legacy':name='webshop_search';args['query']='public search'
    error,_=manager.preflight(owner='agent_1',binding='live',state=state,name=name,arguments=args,
        remaining=8,task_result=True,task=CASES['00232']['task'])
    assert error and not manager.plan_feedback and manager.plan is None
    assert error['code']==('purchase_plan_abandoned' if kind=='abandon' else 'invalid_completion_plan')


@pytest.mark.parametrize(('number','word'), [('00015','terracotta'),('00234','lieutenant'),
    ('00318','cappuccino'),('00358','cantaloupe'),('00365','cottonwood')])
def test_actual_ten_character_options_are_not_product_ids(number,word):
    state=next(t['observation']['output'] for t in CASES[number]['trace']
        if any(a.get('target_id','').endswith(':'+word.upper()) for a in t['observation'].get('output',{}).get('valid_subactions',[])))
    assert word in state['page_text'].lower()
    options={}
    for a in state['valid_subactions']:
        if a.get('kind')=='select_option':options.setdefault(a['option_name'],[]).append(a['option_value'])
    # The captured page explicitly lists these values in its color group.
    options.setdefault('color',[]).append(word)
    session=WebShopSession('live',SimpleNamespace(),SimpleNamespace(product=lambda asin:{'options':options}),
        number,current_asin=state['product']['asin'])
    actions,_=session._subactions(['click[buy now]',f'click[{word}]','click[< prev]'],'','product')
    chosen=actions[1]
    assert chosen['kind']=='select_option' and chosen['option_value']==word
    assert chosen['target_id']==f'select_option:color:{word}'
    assert actions[2]['navigation_effect']=='return_to_search_results'
    session._track_click(chosen['kind'],word)
    assert session.selected_options=={'color':word}
    assert session.current_asin==state['product']['asin']


def test_search_asin_remains_openable_while_asin_shaped_option_is_selectable():
    session=WebShopSession('live',SimpleNamespace(),SimpleNamespace(product=lambda asin:{'options':{'code':['B012345678']}}),'task')
    search,_=session._subactions(['click[B012345678]'],'B012345678 [SEP] Product [SEP] $10','search_results')
    assert search[0]['kind']=='open_product'
    session.current_asin='B099999999'
    product,_=session._subactions(['click[B012345678]'],'','product')
    assert product[0]['kind']=='select_option'


def test_real_invalid_plan_executes_original_search_once_and_exposes_feedback(tmp_path,monkeypatch):
    from selfplay_graph_flowsteer.contracts import AgentNode
    from selfplay_graph_flowsteer.dataset_actions import default_dataset_action_registry
    from selfplay_graph_flowsteer.llm import MockBackend
    from selfplay_graph_flowsteer.observability import TaskSpec
    from selfplay_graph_flowsteer.runtime import ModelAgentExecutor, AgentActionUsage
    from selfplay_graph_flowsteer.webshop import WebShopClickTool, WebShopSearchTool
    from selfplay_graph_flowsteer.webshop_native import NativeWebShopLifecycle
    from .test_webshop_memory_v2 import final
    _,state,rejected=checkpoint('00433','invalid_completion_plan')
    expected=next(t for t in CASES['00433']['trace'] if t['observation']['status']=='ok'
        and t['action']['arguments'].get('query')==rejected['action']['arguments']['query'])
    calls=[]
    def search(sid,query):
        calls.append(query)
        assert query==rejected['action']['arguments']['query']
        return raw_state(expected['observation']['output'])
    def no_click(*args):raise AssertionError('No click or purchase was requested')
    client=SimpleNamespace(create_session=lambda *args,**kwargs:dict(state,session_id='real-checkpoint'),
        search=search,click=no_click,commit=no_click,close_session=lambda *args:None)
    life=NativeWebShopLifecycle(client,require_native_actions=False,stage_purchases=True,max_observation_chars=0)
    life.bind_task(TaskSpec('webshop/goal-00433',CASES['00433']['task'],metadata={'goal_id':'goal-00433'}))
    tools={'webshop_search':WebShopSearchTool(life),'webshop_click':WebShopClickTool(life)}
    registry=default_dataset_action_registry(tools,webshop_commit_on_finish=True,action_budget_policy='shared_total_v1')
    backend=MockBackend([json.dumps({'action_call':rejected['action']})]+[final()]*8)
    monkeypatch.setenv('SPGFS_WEBSHOP_MEMORY_DIR',str(tmp_path/'memory'))
    executor=ModelAgentExecutor(backend,tools=tools,action_registry=registry,webshop_worker_memory_policy='factual_memory_v2',
        webshop_worker_guidance_policy='merged_checklist_v1',webshop_purchase_budget_policy=FLEXIBLE_POLICY)
    data=json.loads((Path(__file__).parent/'fixtures/webshop_memory_real_delegation.json').read_text())
    data['allowed_tools']=tuple(data['allowed_tools']);node=AgentNode(**data)
    used=16-rejected['remaining_budget']['total']
    executor.budget_scope='tool-rollout:whole-graph'
    executor.budget_ledger.usage['tool-rollout:whole-graph']=AgentActionUsage(initial_used=used,total_used=used)
    try:
        artifact=executor.execute(task=CASES['00433']['task'],node=node,upstream=[],peers=[],revision=False,seed=0)
        assert calls==[rejected['action']['arguments']['query']]
        assert len(artifact.react_trace)==1 and artifact.react_trace[0]['observation']['status']=='ok'
        assert artifact.webshop_progress['action_budget']['total_used']==used+1
        context=json.loads(backend.calls[1]['messages'][1]['content'])
        feedback=context['action_environment']['purchase_reservation']['plan_update_feedback']
        assert feedback['status']=='ignored' and not feedback['existing_plan_retained']
        assert not life.commit_ready_agents()
    finally:
        life.close_all()
