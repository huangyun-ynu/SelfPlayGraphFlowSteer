"""Replay frozen public tool observations and Worker outputs through the real runtime.

No model/server calls, inferred purchases, hidden reward or rescoring. Historical
Director actions run until the repaired boundary; an explicit test FINISH probes
candidate availability when the historical Director got stuck configuring a draft.
"""
import copy
import json
from pathlib import Path
import pytest
from selfplay_graph_flowsteer.canvas import GraphCanvas
from selfplay_graph_flowsteer.config import CanvasConfig
from selfplay_graph_flowsteer.dataset_actions import default_dataset_action_registry
from selfplay_graph_flowsteer.llm import MockBackend
from selfplay_graph_flowsteer.observability import TaskSpec
from selfplay_graph_flowsteer.runtime import MultiAgentRuntime, RoutedModelAgentExecutor
from selfplay_graph_flowsteer.webshop import WebShopSearchTool, WebShopClickTool
from selfplay_graph_flowsteer.webshop_native import NativeWebShopLifecycle
from .test_unified_submission import step

CASES=json.loads((Path(__file__).parent/'fixtures/webshop_engineering_real_trajectories.json').read_text())

class CapturedClient:
    def __init__(self,case):
        self.observations=[]
        self.calls=[]
        for a in case['artifacts']:
            for t in a['react_trace']:
                action=t['action']
                if action['name']=='webshop_click' and action['arguments']['target_id'].startswith('purchase:'):
                    continue
                assert t['observation']['status']=='ok'
                self.observations.append((action,copy.deepcopy(t['observation']['output'])))
    def create_session(self,goal_id,seed):
        return {'session_id':'captured-session','page_type':'search','page_text':'Search',
                'state_version':1,'valid_subactions':[], 'purchased':False,'done':False,
                'steps':0,'termination_reason':'active'}
    def take(self,name,arg):
        action,output=self.observations.pop(0)
        expected=action['arguments']['query' if name=='webshop_search' else 'target_id']
        assert (name,arg)==(action['name'],expected)
        self.calls.append((name,arg))
        return output
    def search(self,sid,query):return self.take('webshop_search',query)
    def click(self,sid,target):return self.take('webshop_click',target)
    def close_session(self,sid):pass
    def commit(self,*args,**kwargs):raise AssertionError('Real replay must not fabricate a purchase outcome')


def replay(tmp_path,number):
    case=CASES[number];responses=[]
    for a in case['artifacts']:
        responses.extend(json.dumps({'action_call':t['action']}) for t in a['react_trace'])
        responses.extend(d['raw_response'] for d in a['protocol_diagnostics']
                         if d['stage'] in {'webshop_active_protocol_recovery','initial_nonfinal'})
        responses.append(a['raw_response'])
    backend=MockBackend(responses);client=CapturedClient(case)
    life=NativeWebShopLifecycle(client,require_native_actions=False,stage_purchases=True)
    life.bind_task(TaskSpec('webshop/goal-'+number,case['task'],metadata={'goal_id':'goal-'+number}))
    tools={'webshop_search':WebShopSearchTool(life),'webshop_click':WebShopClickTool(life)}
    registry=default_dataset_action_registry(tools,webshop_commit_on_finish=True,action_budget_policy='shared_total_v1')
    executor=RoutedModelAgentExecutor({'deepseek':backend},('deepseek',),tools=tools,action_registry=registry,
        webshop_worker_guidance_policy='merged_checklist_v1')
    c=GraphCanvas(task=case['task'],dataset='webshop',runtime=MultiAgentRuntime(executor),
        action_adapter=registry.get('webshop'),runtime_routes=('deepseek',),config=CanvasConfig(
            submission_protocol='unified_task_result_v1',submission_journal_dir=str(tmp_path),
            action_budget_policy='shared_total_v1',director_budget_policy='edits_v1',max_director_edits=24,
            max_rounds=40,max_total_tokens=1000000,remaining_token_admission_enabled=False))
    for record in case['actions']:
        action=json.loads(record['raw'])
        # Rejected historical edits are retained in the fixture but cannot mutate state.
        if not record['accepted']:continue
        if action['action']=='finish':break
        result=step(c,action)
        assert result.accepted,result.feedback
    if client.observations:
        assert number in {"00299", "00225"}
        assert c.runtime.artifacts["agent_1"].webshop_progress["stop_reason"] == "webshop_semantic_no_progress_fuse"
    return c,life,client,backend

@pytest.mark.parametrize('number',['00004','00269','00319'])
def test_real_staged_then_revised_report_is_not_terminal(tmp_path,number):
    c,life,client,b=replay(tmp_path,number)
    assert life.result_for('agent_1')['termination_reason']=='active'
    assert not life.commit_ready_agents()
    assert not c.submission_assessment('agent_1')['submit_ready']
    assert c._unified_can_run('agent_1')
    assert not step(c,dict(action='finish',target='agent_1'),True).accepted


def test_real_malformed_purchase_retains_twelve_actions(tmp_path):
    c,life,client,b=replay(tmp_path,'00400')
    a=c.runtime.artifacts['agent_1']
    assert a.webshop_progress['state']=='needs_recovery'
    assert a.webshop_progress['stop_reason']=='protocol_recovery_exhausted'
    assert c.runtime.shared_tool_budget_status(16)['remaining']==12
    assert len(client.calls)==4
    assert 'used 4/16; remaining 12' in a.summary
    assert c._unified_can_run('agent_1')
    assert not step(c,dict(action='finish',target='agent_1'),True).accepted


def test_real_valid_candidate_survives_unconfigured_extension(tmp_path):
    c,life,client,b=replay(tmp_path,'00010')
    assert c.pending_agent_id is not None
    assert life.commit_ready_agents()==('agent_1',)
    assert c.submission_assessment('agent_1')['submit_ready']
    assert 'finish' in c.control_snapshot()['allowed_actions']
    # This checks availability, not a new Director choice or an invented reward.
    assert c.control_snapshot()['legal_action_parameters']['finish']['targets']==['agent_1']

@pytest.mark.parametrize('number',['00369','00075'])
def test_real_exhausted_search_remains_failure_without_budget_refund(tmp_path,number):
    c,life,client,b=replay(tmp_path,number)
    assert c.runtime.shared_tool_budget_status(16)['remaining']==0
    assert not c._unified_can_run('agent_1')
    outcome=c.submission_assessment('agent_1')
    assert outcome['submit_ready'] and outcome['payload_kind']=='environment_failure'
    before=len(b.calls)
    assert step(c,dict(action='finish',target='agent_1'),True).accepted
    assert len(b.calls)==before


@pytest.mark.parametrize('number',['00299','00225'])
def test_real_duplicate_evidence_now_triggers_existing_bounded_fuse(tmp_path,number):
    c,life,client,b=replay(tmp_path,number)
    progress=c.runtime.artifacts['agent_1'].webshop_progress
    assert progress['semantic_no_progress_count'] > 0
    assert progress['stop_reason']=='webshop_semantic_no_progress_fuse'
    budget=c.runtime.shared_tool_budget_status(16)
    assert 0 < budget['remaining'] < 16
    assert budget['used']==len(client.calls)
    assert not life.commit_ready_agents() and c.submission_receipt is None
    assert client.observations  # The fixed loop detector stops before the old tail.
