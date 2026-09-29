"""ALFWorld V3: public reset task, physical usage boundary and trusted submission."""
import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from selfplay_graph_flowsteer import llm
from selfplay_graph_flowsteer.adaptive import AdaptiveWorkflowSolver
from selfplay_graph_flowsteer.alfworld import ALFWorldSessionLifecycle, ALFWorldStepTool
from selfplay_graph_flowsteer.canvas import GraphCanvas
from selfplay_graph_flowsteer.config import CanvasConfig
from selfplay_graph_flowsteer.dataset_actions import default_dataset_action_registry
from selfplay_graph_flowsteer.director import GraphDirector
from selfplay_graph_flowsteer.llm import LLMResponse, MockBackend
from selfplay_graph_flowsteer.observability import TaskSpec
from selfplay_graph_flowsteer.runtime import ModelAgentExecutor, MultiAgentRuntime
from selfplay_graph_flowsteer.submission_contract import receipt_error
from selfplay_graph_flowsteer.worker_usage_ledger import WorkerUsageLedger
from .test_alfworld_sessions import Client
from .test_unified_submission import add, prompt, step

GOAL = "put some peppershaker on drawer."
ORIGINAL = "Put a salt shaker in a drawer"


class GoalClient(Client):
    def state(self, session):
        state = super().state(session)
        if self.sessions[session] == 0:
            state['observation'] = 'A room. Your task is to: ' + GOAL
        return state


def lifecycle(tmp_path, client=None):
    game = tmp_path / 'game.tw-pddl'
    game.write_text('fixture')
    task = TaskSpec('goal-conflict', ORIGINAL, metadata={'dataset': 'alfworld', 'game_path': str(game)})
    life = ALFWorldSessionLifecycle(client or GoalClient(), tmp_path, task_prompt_source='environment_reset')
    return life, task


def test_preparation_reuses_first_episode_and_preserves_original(tmp_path):
    life, task = lifecycle(tmp_path)
    life.bind_task(task)
    assert task.prompt == ORIGINAL
    assert task.metadata['alfworld_task_binding']['effective_task'] == GOAL
    assert life.client.created == 1 and life._rollout_steps == 0 and life._attempt_index == 0
    state = life.begin_execution(agent_id='a', seed=0, revision=False)
    assert state['public_task_statement'] == GOAL
    assert life.client.created == 1 and life.result_for('a')['attempt_index'] == 1
    assert life.result_for('a')['session_id'] == task.metadata['alfworld_task_binding']['prepared_session_id']
    life.step(state['admissible_actions'][0]['action_id']); life.end_execution()
    assert life.begin_execution(agent_id='a', seed=0, revision=True)['step'] == 1
    life.end_execution()
    assert life.begin_execution(agent_id='b', seed=0, revision=False)['public_task_statement'] == GOAL
    assert life.client.created == 2
    life.close_all()
    assert sorted(life.client.closed) == ['1', '2']


def test_prepared_but_unused_episode_is_closed(tmp_path):
    life, task = lifecycle(tmp_path)
    life.bind_task(task); life.close_all()
    assert life.client.closed == ['1']


def test_missing_goal_fails_and_closes_instead_of_using_dataset_prompt(tmp_path):
    life, task = lifecycle(tmp_path, Client())
    with pytest.raises(RuntimeError, match='no public task'):
        life.bind_task(task)
    assert life.client.closed == ['1']


def test_other_agent_reset_cannot_change_effective_task(tmp_path, monkeypatch):
    life, task = lifecycle(tmp_path); life.bind_task(task)
    life.begin_execution(agent_id='a', seed=0, revision=False); life.end_execution()
    original = life.client.state
    monkeypatch.setattr(life.client, 'state', lambda session: {
        **original(session), 'observation': 'Your task is to: wrong goal'})
    with pytest.raises(RuntimeError, match='differs from the prepared'):
        life.begin_execution(agent_id='b', seed=0, revision=False)
    assert life.client.closed == ['2']
    assert life.effective_task == GOAL
    life.close_all()


class PhysicalBackend:
    """Exercise the real gateway accounting boundary without an external API."""
    def __init__(self, monkeypatch, life):
        self.calls = []; self.messages = []; self.life = life
        @contextmanager
        def slot(*args, **kwargs):
            yield SimpleNamespace(route='deepseek', timeout_s=10, request_started_monotonic=0,
                request_budget_s=10, queue_wait_s=0, priority='primary')
        monkeypatch.setattr(llm, '_request_slot', slot)

    def generate(self, messages, **kwargs):
        self.messages.append(messages)
        def create(**request):
            self.calls.append(request)
            state = self.life._active_state
            content = json.dumps({'action_call': {'name': 'alfworld_step', 'arguments': {
                'action_id': state['admissible_actions'][0]['action_id']}}})
            return SimpleNamespace(id=f'response-{len(self.calls)}', text=content,
                usage=SimpleNamespace(prompt_tokens=8, completion_tokens=2))
        response = llm._openai_completion_attempt(
            SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
            SimpleNamespace(route_name='deepseek', stream=False),
            {'model': 'deepseek-flash', 'messages': messages}, None,
            attempt=1, request_budget_cap_s=10)
        return LLMResponse(text=response.text, token_in=8, token_out=2, model='fixture')


def build(tmp_path, monkeypatch, threshold=15):
    life, task = lifecycle(tmp_path)
    backend = PhysicalBackend(monkeypatch, life)
    tools = {'alfworld_step': ALFWorldStepTool(life)}
    registry = default_dataset_action_registry(tools)
    runtime = MultiAgentRuntime(ModelAgentExecutor(backend, tools=tools, action_registry=registry))
    config = CanvasConfig(submission_protocol='unified_task_result_v1',
        submission_journal_dir=str(tmp_path / 'journal'), max_rounds=24,
        max_total_tokens=threshold, max_total_tokens_by_dataset={'alfworld': threshold},
        alfworld_terminal_candidate_policy='finish_only_v1',
        worker_token_budget_by_dataset={'alfworld': {'policy': 'reported_usage_threshold_v1', 'start_threshold': threshold}},
        remaining_time_admission_enabled=False, remaining_token_admission_enabled=False)
    return life, task, backend, registry, runtime, config


def test_cross_threshold_success_keeps_actions_and_zero_worker_finish(tmp_path, monkeypatch):
    life, task, backend, registry, runtime, config = build(tmp_path, monkeypatch)
    life.bind_task(task)
    runtime.worker_usage_ledger = WorkerUsageLedger(tmp_path / 'usage.sqlite3', question_attempt_id='test', threshold=15)
    c = GraphCanvas(task=GOAL, dataset='alfworld', runtime=runtime, config=config, action_adapter=registry.get('alfworld'))
    actions = [dict(action='add_agent', agent_id='solver'), prompt(), dict(action='finish', target='solver')]
    run = GraphDirector(canvas=c, backend=MockBackend([json.dumps(a) for a in actions])).run()
    assert run.finished, [t.feedback for t in run.turns]
    assert len(backend.calls) == 2 and c.total_tokens == 20
    assert len(runtime.artifacts['solver'].react_trace) == 2
    assert runtime.artifacts['solver'].model == 'runtime-alfworld-usage-boundary'
    assert c.submission_receipt.payload['won'] and c.submission_receipt.worker_dispatch_valid
    assert c.submission_receipt.worker_tokens_used == 20
    assert receipt_error(c.submission_receipt, run=run, events=c.history, run_id=c.run_id, dataset='alfworld') is None
    assert c.history[-1].executed_agents == []
    life.close_all(); runtime.reset()


def test_nonterminal_stop_keeps_truth_and_blocks_new_agent(tmp_path, monkeypatch):
    life, task, backend, registry, runtime, config = build(tmp_path, monkeypatch, threshold=5)
    life.bind_task(task)
    runtime.worker_usage_ledger = WorkerUsageLedger(tmp_path / 'usage.sqlite3', question_attempt_id='test', threshold=5)
    c = GraphCanvas(task=GOAL, dataset='alfworld', runtime=runtime, config=config, action_adapter=registry.get('alfworld'))
    add(c)
    assert len(backend.calls) == 1
    assert len(runtime.artifacts['solver'].react_trace) == 1
    assert not runtime.artifacts['solver'].environment_result['won']
    assert not step(c, dict(action='add_agent', agent_id='retry')).accepted
    assert not step(c, dict(action='run_agent', target='solver')).accepted
    assert step(c, dict(action='finish', target='solver'), True).accepted
    assert not c.submission_receipt.payload['won'] and len(backend.calls) == 1
    life.close_all(); runtime.reset()


def test_goal_reaches_first_director_and_worker_and_scores_true_episode(tmp_path, monkeypatch):
    life, task, backend, registry, runtime, config = build(tmp_path, monkeypatch)
    director = MockBackend([json.dumps(a) for a in [
        dict(action='add_agent', agent_id='solver'), prompt(), dict(action='finish', target='solver')]])
    solver = AdaptiveWorkflowSolver(director_backend=director, runtime=runtime,
        action_registry=registry, canvas_config=config, director_prompt_variant='v3')
    result = solver.solve(task, run_id='goal-binding-test')
    c = solver.active_canvas
    assert c.task == c.worker_task == c.director_task == GOAL
    assert task.prompt == ORIGINAL
    assert GOAL in json.dumps(director.calls[0]) and ORIGINAL not in json.dumps(director.calls[0])
    assert GOAL in json.dumps(backend.messages[0]) and ORIGINAL not in json.dumps(backend.messages[0])
    assert task.metadata['alfworld_environment_result']['won']
    assert 'submission_binding_error' not in task.metadata
    assert life.client.created == 1 and life.client.closed == ['1']
    runtime.reset()


def test_ready_terminal_candidate_rejects_edits_without_restoring_history(tmp_path, monkeypatch):
    life, task, backend, registry, runtime, config = build(tmp_path, monkeypatch)
    life.bind_task(task)
    runtime.worker_usage_ledger = WorkerUsageLedger(tmp_path / 'usage.sqlite3', question_attempt_id='test', threshold=15)
    c = GraphCanvas(task=GOAL, dataset='alfworld', runtime=runtime, config=config, action_adapter=registry.get('alfworld'))
    add(c)
    before = runtime.artifacts['solver'].artifact_id
    assert c.control_snapshot()['allowed_actions'] == ['finish']
    assert step(c, dict(action='delete_agent', target='solver')).rejection_code == 'alfworld_terminal_candidate_protected'
    assert step(c, {**prompt(), 'objective': 'Different objective'}).rejection_code == 'alfworld_terminal_candidate_protected'
    assert runtime.artifacts['solver'].artifact_id == before and len(backend.calls) == 2
    # Out-of-band input mutation must invalidate the candidate; protection cannot relabel history.
    c.graph.nodes['solver'].prompt = 'Changed behind the canvas'
    assert not c._protected_alfworld_candidates()
    assert not step(c, dict(action='finish', target='solver'), True).accepted
    life.close_all(); runtime.reset()


def test_prepared_episode_closed_when_initialization_fails(tmp_path, monkeypatch):
    life, task, backend, registry, runtime, config = build(tmp_path, monkeypatch)
    class BrokenSkills:
        def retrieve(self, *args, **kwargs):
            raise RuntimeError('skill initialization failed')
    solver = AdaptiveWorkflowSolver(director_backend=MockBackend([]), runtime=runtime,
        action_registry=registry, canvas_config=config, skillbank=BrokenSkills())
    with pytest.raises(RuntimeError, match='skill initialization'):
        solver.solve(task, run_id='failed-preparation')
    assert life.client.created == 1 and life.client.closed == ['1']


def test_config_and_manifest_pin_alfworld_protocol(tmp_path):
    from dataclasses import replace
    from selfplay_graph_flowsteer.application import load_adaptive_config
    from .test_application import write_config
    path = write_config(tmp_path)
    with path.open('a') as f:
        f.write('\n[alfworld]\ntask_prompt_source = "environment_reset"\n')
    config = load_adaptive_config(path)
    reset = config.model_manifest()['execution_semantics']
    legacy = replace(config, alfworld=replace(config.alfworld, task_prompt_source='dataset')).model_manifest()['execution_semantics']
    assert reset != legacy
    assert 'worker_usage_ledger' in reset['contract_source_sha256']
    with pytest.raises(ValueError, match='requires unified'):
        CanvasConfig(worker_token_budget_by_dataset={'alfworld': {'policy': 'reported_usage_threshold_v1'}})


def test_export_uses_physical_usage_even_when_response_tokens_not_in_artifact(tmp_path, monkeypatch):
    from dataclasses import replace
    from selfplay_graph_flowsteer.application import AdaptiveApplicationResult
    life, task, backend, registry, runtime, config = build(tmp_path, monkeypatch)
    original_generate = backend.generate
    def missing_artifact_usage(*args, **kwargs):
        return replace(original_generate(*args, **kwargs), token_in=0, token_out=0)
    monkeypatch.setattr(backend, 'generate', missing_artifact_usage)
    director = MockBackend([json.dumps(a) for a in [
        dict(action='add_agent', agent_id='solver'), prompt(), dict(action='finish', target='solver')]])
    solver = AdaptiveWorkflowSolver(director_backend=director, runtime=runtime,
        action_registry=registry, canvas_config=config, director_prompt_variant='v3')
    result = solver.solve(task, run_id='export-usage-test')
    assert runtime.artifacts['solver'].token_in == 0
    exported = AdaptiveApplicationResult(run_id='export-usage-test', task=task,
        solver_result=result, skills_used=(), mace_statistics_path=None, trace_store_path='').to_dict()
    assert (exported['token_in'], exported['token_out']) == (16, 4)
    runtime.reset()
