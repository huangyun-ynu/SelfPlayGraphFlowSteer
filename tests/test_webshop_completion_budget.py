"""Replay current-run public trajectories; never invent a purchase or reward."""
import copy
import json
from pathlib import Path

import pytest

from selfplay_graph_flowsteer.canvas import GraphCanvas
from selfplay_graph_flowsteer.config import CanvasConfig
from selfplay_graph_flowsteer.dataset_actions import default_dataset_action_registry
from selfplay_graph_flowsteer.llm import MockBackend
from selfplay_graph_flowsteer.observability import TaskSpec
from selfplay_graph_flowsteer.runtime import (
    MultiAgentRuntime,
    RoutedModelAgentExecutor,
    _webshop_completion_steps,
    _webshop_has_feasible_completion_path,
)
from selfplay_graph_flowsteer.webshop import WebShopClickTool, WebShopSearchTool
from selfplay_graph_flowsteer.webshop_native import NativeWebShopLifecycle
from .test_unified_submission import step
from .test_webshop_real_trajectory_regression import CapturedClient

FIXTURE = json.loads((Path(__file__).parent / 'fixtures/webshop_v3_completion_real_trajectories.json').read_text())
CASES = FIXTURE['cases']


def replay(tmp_path, number):
    case = CASES[number]
    client = CapturedClient(case)
    life = NativeWebShopLifecycle(client, require_native_actions=False, stage_purchases=True)
    life.bind_task(TaskSpec('webshop/goal-' + number, case['task'], metadata={'goal_id': 'goal-' + number}))
    tools = {'webshop_search': WebShopSearchTool(life), 'webshop_click': WebShopClickTool(life)}
    registry = default_dataset_action_registry(tools, webshop_commit_on_finish=True,
                                              action_budget_policy='shared_total_v1')
    backend = MockBackend([])
    executor = RoutedModelAgentExecutor({'deepseek': backend}, ('deepseek',), tools=tools,
        action_registry=registry, webshop_worker_guidance_policy='merged_checklist_v1',
        webshop_compatibility_profile='m02_merged_identity_v1')
    canvas = GraphCanvas(task=case['task'], dataset='webshop', runtime=MultiAgentRuntime(executor),
        action_adapter=registry.get('webshop'), runtime_routes=('deepseek',), config=CanvasConfig(
            submission_protocol='unified_task_result_v1', submission_journal_dir=str(tmp_path),
            action_budget_policy='shared_total_v1', max_rounds=40, max_total_tokens=1000000,
            remaining_token_admission_enabled=False))
    boundaries = []
    for index, artifact in enumerate(case['artifacts']):
        backend.responses.clear()
        backend.responses.extend(json.dumps({'action_call': t['action']}) for t in artifact['react_trace'])
        # Only saved model text follows the recorded actions. Repetition allows
        # the runtime's format-repair calls; no new shopping action is invented.
        backend.responses.extend([artifact['raw_response']] * 8)
        before = len(backend.calls)
        if index == 0:
            for action in case['actions']:
                assert action['accepted']
                result = step(canvas, json.loads(action['raw']))
                assert result.accepted, result.feedback
        else:
            result = step(canvas, {'action': 'run_agent', 'target': 'agent_1'})
            assert result.accepted, result.feedback
        boundaries.append(backend.calls[before + len(artifact['react_trace'])])
    return canvas, life, client, backend, boundaries


@pytest.mark.parametrize(('number', 'remaining'), [('00031', 7), ('00411', 2), ('00428', 4)])
def test_real_selected_product_can_continue_before_buy(tmp_path, number, remaining):
    canvas, life, client, backend, boundaries = replay(tmp_path, number)
    assert not client.observations
    assert canvas.runtime.shared_tool_budget_status(16)['remaining'] == remaining
    assert len(client.calls) == 16 - remaining
    # The exact boundary that previously forced a final report now still offers
    # real action tools. This is availability, not an assertion of model success.
    boundary = boundaries[-1]
    assert {a['name'] for a in boundary['actions']} == {'webshop_search', 'webshop_click'}
    context = json.loads(boundary['messages'][1]['content'])
    budget = context['action_environment']['webshop_progress']['completion_budget']
    assert budget['remaining_action_steps'] == remaining
    assert budget['remaining_environment_steps'] is None
    assert budget['budget_source'] == 'runtime_action_ledger'
    assert 'remaining_steps' not in context['action_environment']['state']
    progress = canvas.runtime.artifacts['agent_1'].webshop_progress
    assert progress['completion_path_fuse_deferrals'] == 1
    assert progress['stop_reason'] != 'webshop_semantic_no_progress_fuse'
    if number != '00031':
        assert not boundaries[0]['actions']  # Original section-page fuse still applies.
        assert progress['journal']['restored_for_this_execution']
    assert not life.commit_ready_agents()
    assert canvas.submission_receipt is None


def test_real_mixed_search_trace_does_not_invent_a_purchase(tmp_path):
    canvas, life, client, backend, boundaries = replay(tmp_path, '00340')
    progress = canvas.runtime.artifacts['agent_1'].webshop_progress
    # A product page earlier in this mixed trace now legitimately gets grace;
    # replaying the old search tail is not evidence that the model will buy.
    assert progress['completion_path_fuse_deferrals'] == 1
    assert canvas.runtime.shared_tool_budget_status(16)['remaining'] == 4
    assert not client.observations and not life.commit_ready_agents()
    assert canvas.submission_receipt is None


def test_resume_does_not_renew_the_one_time_completion_deferral(tmp_path):
    canvas, life, client, backend, _ = replay(tmp_path, '00031')
    turn = copy.deepcopy(CASES['00031']['artifacts'][0]['react_trace'][-1])
    # Controlled negative extension: repeat an actual recorded public selection.
    # This is a loop regression, not an invented successful historical outcome.
    turn['action']['arguments']['state_version'] = turn['observation']['output']['state_version']
    client.observations.extend((turn['action'], copy.deepcopy(turn['observation']['output'])) for _ in range(4))
    backend.responses.clear()
    backend.responses.extend([json.dumps({'action_call': turn['action']})] * 4)
    backend.responses.extend([CASES['00031']['artifacts'][0]['raw_response']] * 8)
    result = step(canvas, {'action': 'run_agent', 'target': 'agent_1'})
    assert result.accepted, result.feedback
    progress = canvas.runtime.artifacts['agent_1'].webshop_progress
    assert progress['journal']['restored_for_this_execution']
    assert progress['completion_path_fuse_deferrals'] == 1
    assert progress['stop_reason'] == 'webshop_semantic_no_progress_fuse'
    assert 0 < canvas.runtime.shared_tool_budget_status(16)['remaining'] <= 7
    assert not life.commit_ready_agents() and canvas.submission_receipt is None


@pytest.mark.parametrize(('official', 'ledger', 'expected'), [
    (None, {'total': 7, 'phase': 7}, 7),
    (2, {'total': 7, 'phase': 7}, 2),
    (7, {'total': 0, 'phase': 0}, 0),
    (7, {'total': 7, 'phase': 1}, 1),
    (7, {'total': 7, 'phase': 7, 'environment': 0}, 0),
    (0, {'total': 7, 'phase': 7}, 0),
    (True, {'total': 4, 'phase': 4}, 4),
    (None, None, None),
])
def test_completion_uses_strictest_known_limit(official, ledger, expected):
    state = {'remaining_steps': official}
    assert _webshop_completion_steps(state, remaining_budget=ledger) == expected


def test_no_completion_credit_for_terminal_or_exhausted_page():
    state = copy.deepcopy(CASES['00031']['artifacts'][0]['react_trace'][-1]['observation']['output'])
    assert not _webshop_has_feasible_completion_path(state)  # Original missing-field symptom.
    assert _webshop_has_feasible_completion_path(state, remaining_budget={'total': 7, 'phase': 7})
    assert not _webshop_has_feasible_completion_path(state, remaining_budget={'total': 0, 'phase': 0})
    state['purchase_visible'] = False
    assert not _webshop_has_feasible_completion_path(state, remaining_budget={'total': 7, 'phase': 7})
