"""State/transaction regressions independent of model quality or hidden reward."""

from .webshop_native_fixture import native_response
from dataclasses import replace
import json
import pytest
from .test_unified_environments import shopping, stage
from .test_unified_submission import add, step, prompt
from .test_webshop_native import report, ASIN1, node
from selfplay_graph_flowsteer.runtime import ModelAgentExecutor

@pytest.mark.parametrize('scope',['task_result','subtask'])
def test_revised_purchase_requires_real_restaging(tmp_path,scope):
    c,b,life,client=shopping(tmp_path,stage()+[report('Previously staged')]+['click[buy now]',report()])
    add(c,scope=scope)
    assert step(c,dict(prompt(), objective='Check the current requirements again')).accepted
    result=life.result_for('solver')
    assert not result['commit_ready'] and result['termination_reason']=='active'
    assert not c.submission_assessment('solver')['submit_ready']
    assert c._unified_can_run('solver')
    assert step(c,dict(action='run_agent',target='solver')).accepted
    assert step(c,dict(action='finish',target='solver'),True).accepted
    assert len(client.commits)==1

@pytest.mark.parametrize('edits_left',[0,10])
def test_isolated_draft_does_not_block_valid_purchase(tmp_path,edits_left):
    c,b,life,client=shopping(tmp_path,stage());add(c)
    assert step(c,dict(action='add_agent',agent_id='draft')).accepted
    c.config=replace(c.config,director_budget_policy='edits_v1',max_director_edits=c.director_edits_used+edits_left)
    before=len(b.calls)
    assert 'finish' in c.control_snapshot()['allowed_actions']
    assert step(c,dict(action='finish',target='solver'),True).accepted
    assert 'draft' not in c.graph.nodes and c.pending_agent_id is None
    assert len(b.calls)==before and len(client.commits)==1

def test_connected_pending_node_is_not_ignored(tmp_path):
    c,b,life,client=shopping(tmp_path,stage());add(c)
    step(c,dict(action='add_agent',agent_id='draft'))
    c.graph.directed_edges.add(('solver','draft'))
    assert not c.submission_assessment('solver')['submit_ready']
    assert not step(c,dict(action='finish',target='solver'),True).accepted
    assert not client.commits

def test_malformed_action_keeps_budget_and_exposes_bounded_recovery(tmp_path):
    c,b,life,client=shopping(tmp_path,['{"action_call":', '{"action_call":',report('budget exhausted')])
    life.require_native_actions=False
    native=c.runtime.executor
    c.runtime.executor=ModelAgentExecutor(b,tools=native.tools,action_registry=native.action_registry)
    add(c)
    a=c.runtime.artifacts['solver']
    assert a.webshop_progress['state']=='needs_recovery'
    assert a.webshop_progress['stop_reason']=='protocol_recovery_exhausted'
    assert a.webshop_progress['action_budget']['total_used']==0
    assert c._unified_can_run('solver')
    c._unified_recovery_used=c.config.max_recovery_executions
    assert not c._unified_can_run('solver')
    assert not c.submission_assessment('solver')['submit_ready']
    assert not client.calls and not client.commits


def test_format_failure_can_resume_to_real_staging_without_refund(tmp_path):
    def action(name, **arguments):
        return native_response({'name':name,'arguments':arguments})
    responses=[action('webshop_search',query='product'),
        action('webshop_click',target_id=f'open_product:0:{ASIN1}',state_version=1),
        '{"action_call":','{"action_call":',report('Unable to emit action'),
        action('webshop_click',target_id=f'purchase:{ASIN1}',state_version=2,
            purchase_evidence={'verified_requirements':['Product matches requested product'],
                               'unresolved_constraints':[]}),report()]
    c,b,life,client=shopping(tmp_path,responses)
    life.require_native_actions=False
    native=c.runtime.executor
    c.runtime.executor=ModelAgentExecutor(b,tools=native.tools,action_registry=native.action_registry)
    add(c)
    assert c.runtime.artifacts['solver'].webshop_progress['state']=='needs_recovery'
    assert step(c,dict(action='run_agent',target='solver')).accepted
    assert c._unified_recovery_used==1
    assert c.runtime.artifacts['solver'].webshop_progress['action_budget']['total_used']==3
    assert len(client.sessions)==1
    before=len(b.calls)
    assert step(c,dict(action='finish',target='solver'),True).accepted
    assert len(b.calls)==before and len(client.commits)==1


def test_navigation_versions_are_not_new_evidence_but_options_and_content_are():
    import copy
    from selfplay_graph_flowsteer.runtime import _webshop_evidence_signature
    state={'page_type':'product','page_text':'Product description',
        'product':{'asin':ASIN1},'selected_options':{'size':'8'},
        'valid_subactions':[{'kind':'navigate','target_id':'back_to_search:5'},
                            {'kind':'navigate','target_id':'previous_page:5'}]}
    other=copy.deepcopy(state)
    other['state_version']=100
    other['valid_subactions'][0]['target_id']='back_to_search:100'
    other['valid_subactions'][1]['target_id']='previous_page:100'
    assert _webshop_evidence_signature(state)==_webshop_evidence_signature(other)
    other['selected_options']['size']='9'
    assert _webshop_evidence_signature(state)!=_webshop_evidence_signature(other)
    other=copy.deepcopy(state);other['page_text']='New public attribute'
    assert _webshop_evidence_signature(state)!=_webshop_evidence_signature(other)
    state['page_type']='search_results';other=copy.deepcopy(state)
    other['valid_subactions'][0]['target_id']='back_to_search:100'
    assert _webshop_evidence_signature(state)==_webshop_evidence_signature(other)
