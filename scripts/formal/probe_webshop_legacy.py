"""Offline observable-behavior probe, runnable against either source checkout."""
from dataclasses import asdict, is_dataclass
import json

from selfplay_graph_flowsteer.canvas import GraphCanvas
from selfplay_graph_flowsteer.config import CanvasConfig
from selfplay_graph_flowsteer.director import director_prompt_components
from selfplay_graph_flowsteer.llm import MockBackend
from selfplay_graph_flowsteer.runtime import ModelAgentExecutor, MultiAgentRuntime
from selfplay_graph_flowsteer.webshop import WebShopClickTool, WebShopSearchTool, WebShopSessionLifecycle
from tests.test_webshop_guidance import setup_worker, execute
from tests.test_webshop_native import NativeClient, node, report, ASIN1
from selfplay_graph_flowsteer.dataset_actions import default_dataset_action_registry
from selfplay_graph_flowsteer.observability import TaskSpec


def probe():
    out={}
    for routed in (False,True):
        e,b,n,client=setup_worker('merged_checklist_v1',routed=routed)
        artifact=execute(e,n)
        b.responses.append(report('Continue inspecting'))
        revision=execute(e,n,revision=True)
        out['inspection_routed_'+str(routed)]={'requests':b.calls,'environment_calls':client.calls,
            'initial_progress':artifact.webshop_progress,'revision_progress':revision.webshop_progress}
    for malformed in (False,True):
        client=NativeClient()
        life=WebShopSessionLifecycle(client,compatibility_profile='m02_merged_identity_v1',search_observation_mode='legacy')
        life.bind_task(TaskSpec('fixture','Buy a product',metadata={'goal_id':'goal-1'}))
        tools={'webshop_search':WebShopSearchTool(life),'webshop_click':WebShopClickTool(life)}
        def action(name,**arguments):return json.dumps({'action_call':{'name':name,'arguments':arguments}})
        responses=[action('webshop_search',query='product'),action('webshop_click',target_id='open_product:0:'+ASIN1)]
        if malformed:responses.append(action('webshop_click',target_id='purchase:'+ASIN1,purchase_evidence={'verified_requirements':['x'*500],'unresolved_constraints':[]}))
        responses += [action('webshop_click',target_id='purchase:'+ASIN1,purchase_evidence={'verified_requirements':['Public product'],'unresolved_constraints':[]}),report()]
        backend=MockBackend(responses)
        registry=default_dataset_action_registry(tools,webshop_commit_on_finish=True)
        executor=ModelAgentExecutor(backend,tools=tools,action_registry=registry,webshop_worker_guidance_policy='merged_checklist_v1')
        artifact=executor.execute(task='Buy a product',node=node(),upstream=[],peers=[],revision=False,seed=0)
        out['purchase_malformed_'+str(malformed)]={'requests':backend.calls,'trace':artifact.react_trace,
            'progress':artifact.webshop_progress,'environment_calls':client.calls,'staged':life.commit_ready_agents()}
        life.close_all()
    canvas=GraphCanvas(task='Buy a product',dataset='webshop',config=CanvasConfig(),
        runtime=MultiAgentRuntime(ModelAgentExecutor(MockBackend([]))))
    out['empty_canvas']=canvas.control_snapshot()
    out['director_v22']=director_prompt_components('v2.2')
    # The later ALF/SWE pending-model deletion fix must not silently alter
    # the legacy WebShop graph state machine during this selective restore.
    canvas = GraphCanvas(
        task='Buy a product', dataset='webshop', config=CanvasConfig(),
        runtime=MultiAgentRuntime(ModelAgentExecutor(MockBackend([]))),
        runtime_routes=('deepseek',),
    )
    for action in (
        {'action': 'add_agent', 'agent_id': 'shopper'},
        {'action': 'set_prompt', 'target': 'shopper', 'role': 'Shopper',
         'objective': 'Buy a product', 'scope': 'Inspect public products',
         'expected_output': 'Report the observed outcome'},
    ):
        step = canvas.step(json.dumps(action))
        assert step.accepted, step.feedback
    canvas.topology_edits_frozen = True
    step = canvas.step(json.dumps({'action': 'delete_agent', 'target': 'shopper'}))
    out['pending_model_deletion'] = {
        'accepted': step.accepted, 'rejection_code': step.rejection_code,
        'snapshot': canvas.control_snapshot(),
    }
    return out


if __name__=='__main__':
    print(json.dumps(probe(),sort_keys=True,ensure_ascii=False,
        default=lambda value:asdict(value) if is_dataclass(value) else str(value)))
