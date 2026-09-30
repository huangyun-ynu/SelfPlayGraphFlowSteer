"""Lossless public memory, bounded projection and actual runtime integration."""
import copy
import json
from pathlib import Path
from dataclasses import replace

import pytest

from selfplay_graph_flowsteer.webshop_memory import WebShopMemory, JOURNAL_KEY, clean_section, encoded
from selfplay_graph_flowsteer.webshop_memory_projection import project_memory, action_decision_support
from selfplay_graph_flowsteer.runtime import ModelAgentExecutor, AgentActionUsage
from selfplay_graph_flowsteer.contracts import AgentNode
from selfplay_graph_flowsteer.dataset_actions import default_dataset_action_registry
from selfplay_graph_flowsteer.llm import MockBackend
from selfplay_graph_flowsteer.observability import TaskSpec
from selfplay_graph_flowsteer.webshop import WebShopSearchTool, WebShopClickTool
from selfplay_graph_flowsteer.webshop_native import NativeWebShopLifecycle

CAPTURE=json.loads((Path(__file__).parent/'fixtures/webshop_candidate_title_real_trajectory.json').read_text())


def store_at(tmp_path):
    return WebShopMemory({},owner='a',task=CAPTURE['task'],session='session',root=tmp_path)


def apply(store,trace):
    state=copy.deepcopy(trace['observation']['output'])
    store.observe(state,action=trace['action']['name'],arguments=trace['action']['arguments'])
    store.annotate(state)
    return state


def test_actual_search_open_features_keep_ten_names_and_fact_status(tmp_path):
    store=store_at(tmp_path); expected=None
    for trace in CAPTURE['react_trace']:
        state=apply(store,trace)
        projection=project_memory(store,state=state)
        names={c['asin']:c['title'] for c in projection['candidate_ledger']}
        expected=expected or names
        assert set(names)==set(expected) and len(names)==10
        assert all(names.values())
        assert len(encoded(projection))<=6000
    asin=state['product']['asin'].lower()
    assert store.fact(asin,'features')['observation_status']=='observed'
    reopened=copy.deepcopy(CAPTURE['react_trace'][1]['observation']['output'])
    reopened['selected_options']={}
    store.annotate(reopened)
    support=action_decision_support(reopened)
    feature=next(a for a in support if a['target_id'].startswith('view_features:'))
    assert feature['observation_status']=='observed'
    assert 'already_observed' not in feature and 'may_add_section_evidence' not in feature
    assert reopened['selected_options']=={}


def test_complete_store_paging_recovery_and_owner_isolation(tmp_path):
    store=store_at(tmp_path)
    for i in range(20):
        state={'page_type':'product_section','product':{'asin':f'B{i:09d}','title':'Name'*300},
               'page_text':'Instruction: [SEP] '+CAPTURE['task']+' [SEP] Back to Search [SEP] < Prev [SEP] '+('Not plastic, 45.7 mm. '*200),
               'state_version':i,'valid_subactions':[{'kind':'select_option','option_name':'size','option_value':str(n)} for n in range(130)]}
        for name in ('description','features','reviews'):
            store.observe(state,action='webshop_click',arguments={'target_id':f'view_{name}:B{i:09d}'})
            # Distinct public section identity must be part of the observation source.
            state['state_version']+=100
    assert len(store.data['products'])==20
    asin='b000000000'
    assert len(store.data['products'][asin]['sections'])==3
    assert len(store._get(store.data['products'][asin]['options'][-1]['ref'])['size'])==130
    body=store._get(store.data['products'][asin]['sections']['features'][-1]['cleaned'])
    assert body.endswith('Not plastic, 45.7 mm. ') and len(body)>1400
    chunks=[];cursor=0
    while True:
        read=store.read({'kind':'section','asin':asin,'section':'features','cursor':cursor})
        chunks.append(read['text'])
        assert len(encoded(read))<=1800
        if read['complete']:break
        cursor=read['next_cursor']
    assert ''.join(chunks)==body
    handle=store.persist()
    restored=WebShopMemory({JOURNAL_KEY:handle},owner='a',task=CAPTURE['task'],session='session',root=tmp_path)
    assert restored.data==store.data
    other=WebShopMemory({JOURNAL_KEY:handle},owner='b',task=CAPTURE['task'],session='session',root=tmp_path)
    assert other.read({'kind':'title','asin':asin})['error']=='not_observed_in_this_session'
    newtask=WebShopMemory({JOURNAL_KEY:handle},owner='a',task='New request',session='session',root=tmp_path)
    assert not newtask.data['products']


def test_legacy_partial_and_projection_never_rewrite_facts(tmp_path):
    journal={'schema_version':1,'product_inspections':[{'asin':'b012345678','sections_viewed':['features'],
             'section_evidence':{'features':'partial body'}}]}
    store=WebShopMemory(journal,owner='a',task='Task',session='s',root=tmp_path)
    assert store.fact('b012345678','features')['evidence_in_store']=='partial'
    assert store.fact('b000000000')['observation_status']=='unknown'
    before=copy.deepcopy(store.data)
    for cap in (900,1200,6000,9000):
        assert len(encoded(project_memory(store,state={},max_chars=cap)))<=cap
        assert store.data==before
    store.observe({'status':'error','page_type':'product','product':{'asin':'b000000000'}})
    assert store.data==before


@pytest.mark.parametrize('separator',[' [SEP] ','\n'])
def test_cleaner_only_removes_verified_prefix_and_keeps_negation_navigation_words(separator):
    raw=separator.join(['Instruction:','Buy blue item','Back to Search','< Prev','Not red; 10 mm. Back to Search is printed on the label.'])
    result=clean_section(raw,'Buy blue item')
    assert result['text']=='Not red; 10 mm. Back to Search is printed on the label.'
    assert raw[slice(*result['raw_range'])]==result['text']
    assert clean_section(raw,'Different task')['text']==raw


def test_json_escaping_and_huge_names_are_bounded_with_explicit_recovery(tmp_path):
    store=store_at(tmp_path)
    store.observe({'page_type':'search_results','valid_subactions':[
        {'kind':'open_product','asin':f'B{i:09d}','label':'"\\\n𝕏'*2000} for i in range(30)]})
    before=copy.deepcopy(store.data)
    for cap in (900,1500,6000,9000):
        p=project_memory(store,state={},max_chars=cap)
        assert len(encoded(p))<=cap and p['memory']['omitted_candidates']>0
        assert p['memory']['delivery']=='automatic'
        assert store.data==before
    assert all(len(store.title(k))>240 for k in store.data['products'])


class ReplayClient:
    def __init__(self):self.calls=[]
    def create_session(self,goal_id,seed):
        return {'session_id':'memory-regression','page_type':'search','page_text':'Search',
                'state_version':1,'valid_subactions':[],'purchased':False,'done':False,'steps':0}
    def take(self,name,arg):
        trace=CAPTURE['react_trace'][len(self.calls)]
        key='query' if name=='webshop_search' else 'target_id'
        assert (name,arg)==(trace['action']['name'],trace['action']['arguments'][key])
        self.calls.append((name,arg))
        return copy.deepcopy(trace['observation']['output'])
    def search(self,sid,query):return self.take('webshop_search',query)
    def click(self,sid,target):return self.take('webshop_click',target)
    def close_session(self,sid):pass


def runtime(tmp_path,monkeypatch,handler):
    monkeypatch.setenv('SPGFS_WEBSHOP_MEMORY_DIR',str(tmp_path/'memory'))
    client=ReplayClient()
    life=NativeWebShopLifecycle(client,require_native_actions=False,stage_purchases=True,max_observation_chars=0,
                               search_observation_mode='legacy',compatibility_profile='m02_merged_identity_v1')
    life.bind_task(TaskSpec(CAPTURE['task_id'],CAPTURE['task'],metadata={'goal_id':'goal-00060'}))
    tools={'webshop_search':WebShopSearchTool(life),'webshop_click':WebShopClickTool(life)}
    registry=default_dataset_action_registry(tools,webshop_commit_on_finish=True,action_budget_policy='shared_total_v1')
    backend=MockBackend(handler=handler)
    executor=ModelAgentExecutor(backend,tools=tools,action_registry=registry,webshop_worker_memory_policy='factual_memory_v2',
        webshop_worker_guidance_policy='merged_checklist_v1',webshop_compatibility_profile='m02_merged_identity_v1',
        webshop_purchase_budget_policy='completion_reserve_v2',webshop_scheduling_policy='bounded_research_v1')
    data=json.loads((Path(__file__).parent/'fixtures/webshop_memory_real_delegation.json').read_text())
    data['allowed_tools']=tuple(data['allowed_tools'])
    node=AgentNode(**data)
    return executor,node,life,client,backend


def final():
    return json.dumps({'answer':'Public evidence inspected; no purchase yet.','summary':'Public evidence inspected.',
        'confidence':0.5,'evidence':['Observed public features'],'unresolved_issues':['Not purchased']})



def test_automatic_memory_real_requests_and_revision_without_extra_calls(tmp_path,monkeypatch):
    captured=[];sequence=[t['action'] for t in CAPTURE['react_trace']]
    def respond(messages,role):
        ctx=json.loads(messages[1]['content']);captured.append(ctx)
        assert len(encoded(ctx['action_environment']['webshop_progress']))<=6000
        assert set(ctx['action_environment']['visible_actions'])=={'webshop_search','webshop_click'}
        assert 'webshop_memory_read' not in messages[1]['content']
        i=len(captured)-1
        return json.dumps({'action_call':sequence[i]}) if i<len(sequence) else final()
    executor,node,life,client,backend=runtime(tmp_path,monkeypatch,respond)
    art=executor.execute(task=CAPTURE['task'],node=node,upstream=[],peers=[],revision=False,seed=0)
    assert len(client.calls)==3 and len(art.react_trace)==3
    first_execution_calls=len(backend.calls)
    baseline_responses=iter([json.dumps({'action_call':a}) for a in sequence])
    def baseline_respond(messages,role):
        return next(baseline_responses,final())
    old,old_node,old_life,old_client,old_backend=runtime(tmp_path/'baseline',monkeypatch,baseline_respond)
    old.webshop_worker_memory_policy='factual_memory_v1'
    old.execute(task=CAPTURE['task'],node=old_node,upstream=[],peers=[],revision=False,seed=0)
    assert len(old_backend.calls)==first_execution_calls
    old_life.close_all()
    monkeypatch.setenv('SPGFS_WEBSHOP_MEMORY_DIR',str(tmp_path/'memory'))
    assert art.webshop_progress['action_budget']['total_used']==3
    assert 'memory_read_trace' not in art.webshop_progress
    assert 'memory_reads_used' not in art.webshop_progress
    memory=captured[-1]['action_environment']['webshop_progress']
    assert len(memory['candidate_ledger'])==10
    assert memory['memory']['delivery']=='automatic'
    prior=copy.deepcopy(art.webshop_progress['memory'])
    journal=life.runtime_transaction_journal_for(node.agent_id)
    journal['semantic_no_progress_count']=2
    journal['semantic_no_progress_streak']=2
    art2=executor.execute(task=CAPTURE['task'],node=node,upstream=[],peers=[],revision=True,seed=0)
    assert art2.webshop_progress['memory']['store']==prior['store']
    assert art2.webshop_progress['memory']['observations']==prior['observations']
    assert art2.webshop_progress['semantic_no_progress_count']==2
    assert art2.webshop_progress['semantic_no_progress_streak']==2
    assert len(backend.calls)>first_execution_calls and len(client.calls)==3
    life.close_all()


def test_exhausted_session_does_not_reactivate_for_memory(tmp_path,monkeypatch):
    executor,node,life,client,backend=runtime(tmp_path,monkeypatch,lambda messages,role:final())
    executor.budget_ledger.usage[node.agent_id]=AgentActionUsage(initial_used=12,total_used=16)
    art=executor.execute(task=CAPTURE['task'],node=node,upstream=[],peers=[],revision=True,seed=0)
    assert not client.calls and not art.react_trace
    assert not art.webshop_progress.get('memory_read_trace')
    life.close_all()


def test_automatic_long_evidence_includes_public_requirement_tail(tmp_path):
    store=WebShopMemory({},owner='a',task='Find washable cotton with diameter 45.7 mm',session='s',root=tmp_path)
    raw='Introduction. '*140+'Not plastic. Washable cotton; diameter 45.7 mm.'
    section={'page_type':'product_section','product':{'asin':'b012345678','title':'Public product'},
        'page_text':raw,'action_effect':{'kind':'view_features'},'valid_subactions':[]}
    store.observe(section,arguments={'target_id':'view_features:b012345678'})
    page={'page_type':'product','product':section['product'],'valid_subactions':[]}
    projected=project_memory(store,state=page,max_chars=2500)
    assert '45.7 mm' in encoded(projected)
    for evidence in projected['evidence']:
        for excerpt in evidence['excerpts']:
            assert raw[slice(*excerpt['range'])]==excerpt['text']
    assert len(encoded(projected))<=2500
    assert store._get(store.data['products']['b012345678']['sections']['features'][-1]['raw'])==raw


def test_real_00040_observed_price_and_variants_survive_return_to_search(tmp_path):
    fixture=json.loads((Path(__file__).parent/'fixtures/webshop_automatic_history_real_trajectory.json').read_text())
    store=WebShopMemory({},owner='a',task=fixture['task'],session='s',root=tmp_path)
    for trace in fixture['react_trace']:
        state=apply(store,trace)
    assert state['page_type']=='search_results'
    projected=project_memory(store,state=state)
    item=next(p for p in projected['observed_products'] if p['asin']=='b005fkmuhq')
    assert item['price_at_observation']['price']==100.0
    assert 'navy linen' in item['option_values']['color']
    assert 'futon and chaise set' in item['option_values']['style']
    assert item['evidence_in_prompt'] in {'full','excerpt'}
    assert len(encoded(projected))<=6000
    assert len(projected['candidate_ledger'])==10
    assert 'selected_options' not in item


def test_other_product_features_cannot_mark_current_features_as_displayed(tmp_path):
    from selfplay_graph_flowsteer.runtime import _webshop_context_for_prompt
    store=store_at(tmp_path)
    store.observe({'page_type':'product_section','product':{'asin':'b000000001'},'page_text':'Observed A features',
        'action_effect':{'kind':'view_features'}},arguments={'target_id':'view_features:b000000001'})
    state={'page_type':'product','product':{'asin':'b000000002'},
        'valid_subactions':[{'kind':'view_section','target_id':'view_features:b000000002','section':'features'}]}
    store.observe(state);store.annotate(state)
    context={'action_environment':{'state':state,'webshop_progress':project_memory(store,state=state)}}
    projected=_webshop_context_for_prompt(context)
    fact=projected['action_environment']['state']['valid_subactions'][0]['memory_fact']
    assert fact['observation_status']=='unobserved'
    assert fact['evidence_in_prompt']=='omitted'


def test_long_current_page_does_not_claim_full_prompt_evidence(tmp_path):
    from selfplay_graph_flowsteer.runtime import _webshop_context_for_prompt
    store=store_at(tmp_path)
    state={'page_type':'product_section','product':{'asin':'b012345678'},
        'page_text':'Not plastic. '*900, 'valid_subactions':[], 'action_effect':{'kind':'view_features'}}
    store.observe(state,arguments={'target_id':'view_features:b012345678'})
    projection=project_memory(store,state=state)
    context=_webshop_context_for_prompt({'action_environment':{'state':state,'webshop_progress':projection}})
    assert context['action_environment']['state']['page_text_projection']['original_chars']>8000
    evidence=context['action_environment']['webshop_progress']['evidence'][0]
    assert evidence['evidence_in_store']=='full'
    assert evidence['evidence_in_prompt'] in {'excerpt','omitted'}
    assert len(encoded(context['action_environment']['webshop_progress']))<=6000
    assert len(store._get(store.data['products']['b012345678']['sections']['features'][-1]['raw']))>8000


def test_archived_store_relocates_and_detects_corrupted_evidence(tmp_path):
    import shutil
    store=store_at(tmp_path/'original')
    for trace in CAPTURE['react_trace']:
        state=apply(store,trace)
    handle=store.persist()
    shutil.copytree(tmp_path/'original',tmp_path/'archive')
    restored=WebShopMemory({JOURNAL_KEY:handle},owner='a',task=CAPTURE['task'],session='session',root=tmp_path/'archive')
    assert restored.data==store.data
    asin=state['product']['asin'].lower()
    entry=restored.data['products'][asin]['sections']['features'][-1]
    body=restored._get(entry['cleaned'])
    assert restored.read({'kind':'section','asin':asin,'section':'features'})['text']==body
    (restored.directory/(entry['cleaned']+'.json')).write_text(json.dumps('Corrupted body'))
    with pytest.raises(ValueError,match='hash mismatch'):
        restored.read({'kind':'section','asin':asin,'section':'features'})


def test_new_session_does_not_import_old_v2_journals_legacy_mirrors(tmp_path):
    store=store_at(tmp_path)
    state=apply(store,CAPTURE['react_trace'][1])
    asin=state['product']['asin'].lower()
    journal={'schema_version':1,'owner_agent':'a',JOURNAL_KEY:store.persist(),
             'product_inspections':[{'asin':asin,'sections_viewed':['features']}],
             'candidate_ledger':[{'asin':asin,'preview_title':'Old session evidence'}]}
    fresh=WebShopMemory(journal,owner='a',task=CAPTURE['task'],session='new-session',root=tmp_path)
    assert fresh.data['products']=={}
    assert fresh.fact(asin)['observation_status']=='unobserved'
    different_owner=WebShopMemory({'schema_version':1,'owner_agent':'other',
        'product_inspections':journal['product_inspections']},owner='a',task=CAPTURE['task'],session='session',root=tmp_path)
    assert different_owner.data['products']=={}
