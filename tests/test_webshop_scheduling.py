"""Replay public failed trajectories; controlled graph edits exercise new boundaries.

No hidden reward is used. Staging checks never claim an official purchase score.
"""
import copy
import json
from dataclasses import replace

import pytest

from selfplay_graph_flowsteer.canvas import GraphCanvas
from selfplay_graph_flowsteer.config import CanvasConfig
from selfplay_graph_flowsteer.dataset_actions import default_dataset_action_registry
from selfplay_graph_flowsteer.llm import MockBackend
from selfplay_graph_flowsteer.observability import TaskSpec
from selfplay_graph_flowsteer.runtime import AgentActionUsage, ModelAgentExecutor, MultiAgentRuntime
from selfplay_graph_flowsteer.webshop import WebShopClickTool, WebShopSearchTool
from selfplay_graph_flowsteer.webshop_native import NativeWebShopLifecycle
from selfplay_graph_flowsteer.webshop_scheduling import POLICY, SchedulingState, input_keys
from .test_unified_submission import prompt, step
from .test_webshop_action_reserve import CASES, CheckpointClient, plan, state, turns
from .test_webshop_real_trajectory_regression import CapturedClient


def canvas(tmp_path, number, *, checkpoint=None, research_used=0, responses=(), scope='task_result'):
    client = CapturedClient(CASES[number]) if checkpoint is None else CheckpointClient(state(number, checkpoint))
    life = NativeWebShopLifecycle(client, require_native_actions=False, stage_purchases=True)
    life.bind_task(TaskSpec('webshop/goal-' + number, CASES[number]['task'], metadata={'goal_id': 'goal-' + number}))
    tools = {'webshop_search': WebShopSearchTool(life), 'webshop_click': WebShopClickTool(life)}
    registry = default_dataset_action_registry(tools, webshop_commit_on_finish=True, action_budget_policy='shared_total_v1')
    raw = CASES[number]['artifacts'][-1]['raw_response']
    backend = MockBackend([json.dumps({'action_call': a}) for a in responses] + [raw] * 40)
    executor = ModelAgentExecutor(backend, tools=tools, action_registry=registry,
        webshop_worker_guidance_policy='merged_checklist_v1',
        webshop_purchase_budget_policy='completion_reserve_v1', webshop_scheduling_policy=POLICY)
    c = GraphCanvas(task=CASES[number]['task'], dataset='webshop', runtime=MultiAgentRuntime(executor),
        action_adapter=registry.get('webshop'), config=CanvasConfig(submission_protocol='unified_task_result_v1',
        submission_journal_dir=str(tmp_path), action_budget_policy='shared_total_v1', max_rounds=40,
        max_total_tokens=1000000, remaining_token_admission_enabled=False))
    assert step(c, dict(action='add_agent', agent_id='a')).accepted
    if checkpoint:
        # Restore the captured prefix's ledger before executing the configured node.
        executor.budget_ledger.usage['tool-rollout:whole-graph'] = AgentActionUsage(
            initial_used=checkpoint, total_used=checkpoint, research_used=research_used)
    result = step(c, prompt('a', scope))
    assert result.accepted, result.feedback
    return c, life, client, backend


def buy(number, s):
    return {'name': 'webshop_click', 'arguments': {
        'target_id': 'purchase:' + s['product']['asin'], 'state_version': s['state_version'],
        'purchase_evidence': {'verified_requirements': [s['product'].get('title', 'Public product')], 'unresolved_constraints': []},
        'completion_plan': plan(number, s)}}


def set_responses(backend, number, actions=()):
    backend.responses.clear()
    backend.responses.extend([json.dumps({'action_call': a}) for a in actions] + [CASES[number]['artifacts'][-1]['raw_response']] * 40)


def test_actual_00064_research_stops_at_eight_and_promotion_preserves_session(tmp_path):
    number = '00064'
    prefix = [copy.deepcopy(t['action']) for t in turns(number)[:8]]
    c, life, client, backend = canvas(tmp_path, number, responses=prefix, scope='subtask')
    assert len(client.calls) == 8
    assert c.runtime.shared_tool_budget_status(16)['remaining'] == 8
    art = c.runtime.artifacts['a']
    assert art.webshop_progress['stop_reason'] == 'webshop_research_budget_handoff'
    assert art.webshop_progress['scheduling']['research_used'] == 8
    assert not c._unified_can_run('a')
    assert 'local_result_only' in c.submission_assessment('a')['blockers']
    calls = len(backend.calls)
    assert not step(c, dict(action='run_agent', target='a')).accepted
    assert len(backend.calls) == calls
    sid = life.session_binding('a')
    current = life.result_for('a')
    assert current['page_type'] == 'product'
    set_responses(backend, number, [buy(number, current)])
    result = step(c, prompt('a'))
    assert result.accepted, result.feedback
    assert life.session_binding('a') == sid
    assert life.commit_ready_agents() == ('a',)
    assert c.runtime.shared_tool_budget_status(16)['remaining'] == 7
    assert c.runtime.webshop_scheduling_status(16)['research_used'] == 8
    assert c.control_snapshot()['allowed_actions'] == ['finish']
    assert not step(c, prompt('a')).accepted
    assert c.submission_receipt is None  # Offline staging is not an official score.


def test_early_local_report_rejects_identical_resume_but_accepts_new_assignment(tmp_path):
    c, life, client, backend = canvas(tmp_path, '00064', checkpoint=2, scope='subtask')
    assert not c._unified_can_run('a')
    before = len(backend.calls)
    assert not step(c, dict(action='run_agent', target='a')).accepted
    assert not step(c, prompt('a', 'subtask')).accepted  # Existing no-op edit admission.
    assert len(backend.calls) == before  # Includes automatic graph execution admission.
    changed = prompt('a', 'subtask'); changed['objective'] = 'Assess the observed product size evidence'
    assert step(c, changed).accepted
    assert len(backend.calls) > before
    assert not client.calls


def test_full_task_keeps_one_retry_then_graph_edits_cannot_bypass_no_work_limit(tmp_path):
    c, life, client, backend = canvas(tmp_path, '00428', checkpoint=10)
    assert c._unified_can_run('a')
    assert step(c, dict(action='run_agent', target='a')).accepted
    assert not c._unified_can_run('a')
    assert c.submission_assessment('a')['submit_ready']
    assert c.submission_assessment('a')['payload_kind'] == 'environment_failure'
    before = len(backend.calls)
    changed = prompt('a'); changed['objective'] = 'Continue the same task now'
    result = step(c, changed)
    assert result.accepted, result.feedback
    assert len(backend.calls) == before and not client.calls
    art = c.runtime.artifacts['a']
    assert art.model == 'runtime-webshop-scheduler'
    assert art.webshop_progress['stop_reason'] == 'webshop_no_work_recovery_exhausted'
    assert art.webshop_progress['runtime_only'] and not art.react_trace
    assert result.execution.worker_model_calls_total == 0
    assert c.submission_assessment('a')['payload_kind'] == 'environment_failure'
    assert c.runtime.shared_tool_budget_status(16)['remaining'] == 6


def test_actual_00387_final_buy_is_still_allowed_on_the_one_remaining_retry(tmp_path):
    c, life, client, backend = canvas(tmp_path, '00387', checkpoint=15)
    assert c._unified_can_run('a')
    set_responses(backend, '00387', [buy('00387', life.result_for('a'))])
    assert step(c, dict(action='run_agent', target='a')).accepted
    assert life.commit_ready_agents() == ('a',)
    assert c.runtime.shared_tool_budget_status(16)['remaining'] == 0
    assert c.control_snapshot()['allowed_actions'] == ['finish']


def test_local_buy_hands_off_without_spending_or_terminal_tool_failure(tmp_path):
    number = '00428'; s = state(number, 10)
    c, life, client, backend = canvas(tmp_path, number, checkpoint=10, scope='subtask', responses=[buy(number, s)])
    art = c.runtime.artifacts['a']
    assert art.webshop_progress['stop_reason'] == 'webshop_research_budget_handoff'
    assert not client.calls and not life.commit_ready_agents()
    assert c.runtime.shared_tool_budget_status(16)['remaining'] == 6
    assert 'integrity:terminal_tool_failure' not in c.submission_assessment('a')['blockers']


def test_zero_actions_promotes_real_session_to_truthful_failure_without_worker(tmp_path):
    c, life, client, backend = canvas(tmp_path, '00064', checkpoint=15, scope='subtask')
    # Restore the actual final public observation and all sixteen charged actions.
    life._episodes['a']._results['a'] = state('00064', 16)
    c.runtime.executor.budget_ledger.usage['tool-rollout:whole-graph'].total_used = 16
    old_artifact = c.runtime.artifacts['a']
    before = len(backend.calls); sid = life.session_binding('a')
    assert c._webshop_schedule_control()['promotion_targets'] == ['a']
    result = step(c, prompt('a'))
    assert result.accepted, result.feedback
    assert len(backend.calls) == before and not client.calls
    assert life.session_binding('a') == sid
    art = c.runtime.artifacts['a']
    assert art.answer != old_artifact.answer and art.artifact_id != old_artifact.artifact_id
    assert art.webshop_progress['runtime_only']
    assert art.environment_result['purchased'] is False
    assert c.submission_assessment('a')['submit_ready']
    assert step(c, dict(action='finish', target='a'), True).accepted
    assert c.submission_receipt is not None
    assert len(backend.calls) == before


def test_research_account_survives_new_nodes_and_report_edits(tmp_path):
    prefix = [copy.deepcopy(t['action']) for t in turns('00064')[:8]]
    c, life, client, backend = canvas(tmp_path, '00064', responses=prefix, scope='subtask')
    calls = len(backend.calls)
    assert c.control_snapshot()['allowed_actions'] == ['set_prompt']
    assert not step(c, dict(action='add_agent', agent_id='b')).accepted
    changed = prompt('a', 'subtask'); changed['objective'] = 'Research one more thing'
    assert not step(c, changed).accepted
    assert len(backend.calls) == calls and len(client.calls) == 8
    assert life.result_for('b')['termination_reason'] == 'agent_never_executed'
    assert c.runtime.webshop_scheduling_status(16)['research_used'] == 8
    other = copy.deepcopy(c.graph.nodes['a']); other.agent_id = 'b'
    assert c.runtime.webshop_scheduling_blocker(other) == 'webshop_research_budget_handoff'


def test_staged_purchase_with_unrelated_research_node_requires_safe_cleanup(tmp_path):
    c, life, client, backend = canvas(tmp_path, '00428', checkpoint=10, scope='subtask')
    assert step(c, dict(action='add_agent', agent_id='b')).accepted
    set_responses(backend, '00428', [buy('00428', state('00428', 10))])
    assert step(c, prompt('b')).accepted
    calls = len(backend.calls); sid = life.session_binding('b')
    snapshot = c.control_snapshot()
    assert snapshot['allowed_actions'] == ['delete_agent']
    assert snapshot['legal_action_parameters']['delete_agent']['targets'] == ['a']
    assert not step(c, dict(action='delete_agent', target='b')).accepted
    assert not step(c, prompt('b')).accepted
    assert step(c, dict(action='delete_agent', target='a')).accepted
    assert len(backend.calls) == calls and not client.calls
    assert life.session_binding('b') == sid and life.commit_ready_agents() == ('b',)
    assert c.submission_assessment('b')['submit_ready']
    assert c.control_snapshot()['allowed_actions'] == ['finish']


@pytest.mark.parametrize('unknown', [False, True])
def test_zero_action_graph_cleanup_never_dispatches_worker_or_invents_unknown_state(tmp_path, unknown):
    c, life, client, backend = canvas(tmp_path, '00064', checkpoint=10, scope='subtask')
    assert step(c, dict(action='add_agent', agent_id='b')).accepted
    assert step(c, prompt('b', 'subtask')).accepted
    c.runtime.executor.budget_ledger.usage['tool-rollout:whole-graph'].total_used = 16
    if unknown:
        for child in life._episodes.values():
            for resource in child._results.values():
                resource.update(resource_status='unknown', termination_reason='environment_step_failed')
    calls = len(backend.calls)
    assert not step(c, dict(action='add_agent', agent_id='c')).accepted
    if unknown:
        assert not c._webshop_schedule_control()['promotion_targets']
        assert not step(c, prompt('a')).accepted
        assert not step(c, dict(action='finish', target='a'), True).accepted
        assert c.submission_receipt is None
    else:
        assert step(c, dict(action='delete_agent', target='b')).accepted
        assert step(c, prompt('a')).accepted
        assert c.submission_assessment('a')['submit_ready']
        assert step(c, dict(action='finish', target='a'), True).accepted
    assert len(backend.calls) == calls and not client.calls


def test_rollout_reset_is_the_only_research_budget_reset(tmp_path):
    prefix = [copy.deepcopy(t['action']) for t in turns('00064')[:8]]
    c, _, _, _ = canvas(tmp_path, '00064', responses=prefix, scope='subtask')
    assert c.runtime.webshop_scheduling_status(16)['research_remaining'] == 0
    c.runtime.executor.budget_ledger.reset()
    assert c.runtime.webshop_scheduling_status(16)['research_remaining'] == 8
    assert not c.runtime.executor.budget_ledger.webshop_scheduling.entries


def test_config_switch_and_provenance(monkeypatch):
    from .test_webshop_formal_promotion import load_formal
    config = load_formal('formal_training.toml', monkeypatch)
    assert config.webshop.scheduling_policy == 'bounded_research_v1'
    config = replace(config, webshop=replace(config.webshop, scheduling_policy=POLICY))
    config.validate()
    assert config.model_manifest()['webshop']['scheduling_policy'] == POLICY
    with pytest.raises(ValueError, match='scheduling_policy'):
        replace(config.webshop, scheduling_policy='unknown').validate()


def test_evidence_keys_ignore_report_rephrasing_but_change_on_new_public_evidence():
    from selfplay_graph_flowsteer.contracts import AgentNode, AgentArtifact
    node = AgentNode(agent_id='a', total_tool_budget=16)
    a = AgentArtifact.from_model_text(text='First wording', artifact_id='1', agent_id='b', model='mock')
    b = AgentArtifact.from_model_text(text='Different wording', artifact_id='2', agent_id='b', model='mock')
    s = state('00428', 10)
    assert input_keys(node, s, [a]) == input_keys(node, s, [b])
    assert input_keys(node, s, []) == input_keys(node, s, [a, b])
    b.webshop_progress['product_inspections'] = [{'asin': s['product']['asin']}]
    assert input_keys(node, s, [a]) != input_keys(node, s, [b])
    assert input_keys(node, s, [b]) == input_keys(node, s, [b, b, a])
