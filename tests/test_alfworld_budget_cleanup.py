"""Closing a current ALFWorld result after physical Worker usage stops."""
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import json

import pytest

from selfplay_graph_flowsteer.alfworld import ALFWorldStepTool
from selfplay_graph_flowsteer.canvas import GraphCanvas
from selfplay_graph_flowsteer.config import CanvasConfig
from selfplay_graph_flowsteer.dataset_actions import default_dataset_action_registry
from selfplay_graph_flowsteer.director import GraphDirector
from selfplay_graph_flowsteer.llm import LLMResponse, MockBackend, WorkerUsageDispatchStopped
from selfplay_graph_flowsteer.runtime import ModelAgentExecutor, MultiAgentRuntime
from selfplay_graph_flowsteer.submission_contract import receipt_error
from selfplay_graph_flowsteer.worker_usage_ledger import WorkerUsageLedger, UsageDispatchStopped, active_worker_usage
from .test_alfworld_v3_usage import lifecycle, GOAL
from .test_unified_submission import add, prompt, step


@contextmanager
def blocked_canvas(tmp_path, *, planners=1):
    life, task = lifecycle(tmp_path)
    life.bind_task(task)

    class Backend:
        calls = 0

        def generate(self, messages, **kwargs):
            ledger, agent_id, execution_id = active_worker_usage()
            try:
                attempt = ledger.begin(route='fixture', agent_id=agent_id,
                    execution_id=execution_id, request={'messages': messages})
            except UsageDispatchStopped as stopped:
                raise WorkerUsageDispatchStopped(stopped)
            ledger.settle(attempt, input_tokens=8, output_tokens=2)
            self.calls += 1
            if agent_id.startswith('planner'):
                text = json.dumps({'answer': 'Local plan', 'summary': 'Inspect then move object', 'confidence': .5})
            else:
                state = life._active_state
                text = json.dumps({'action_call': {'name': 'alfworld_step', 'arguments': {
                    'action_id': state['admissible_actions'][0]['action_id']}}})
            return LLMResponse(text=text, token_in=8, token_out=2, model='fixture')

    backend = Backend()
    tools = {'alfworld_step': ALFWorldStepTool(life)}
    registry = default_dataset_action_registry(tools)
    runtime = MultiAgentRuntime(ModelAgentExecutor(backend, tools=tools, action_registry=registry))
    ledger = WorkerUsageLedger(tmp_path / 'usage.sqlite3', question_attempt_id='cleanup',
        threshold=planners * 10 + 15)
    runtime.worker_usage_ledger = ledger
    c = GraphCanvas(task=GOAL, dataset='alfworld', runtime=runtime, action_adapter=registry.get('alfworld'),
        config=CanvasConfig(submission_protocol='unified_task_result_v1',
            submission_journal_dir=str(tmp_path / 'journal'), max_rounds=40,
            alfworld_terminal_candidate_policy='finish_only_v1', max_total_tokens=planners * 10 + 15,
            remaining_time_admission_enabled=False, remaining_token_admission_enabled=False))
    try:
        for i in range(planners):
            add(c, f'planner{i}', scope='subtask')
        add(c, 'solver')
        assert life.result_for('solver')['won']
        assert not c.submission_assessment('solver')['submit_ready']
        yield c, backend, life, ledger
    finally:
        life.close_all()
        runtime.close_worker_usage_ledger()


@pytest.mark.parametrize('admission_enabled', [False, True])
@pytest.mark.parametrize('finish_on_last_round', [False, True])
def test_real_director_cleans_and_submits_same_current_result_without_worker(tmp_path, monkeypatch, admission_enabled, finish_on_last_round):
    with blocked_canvas(tmp_path) as (c, backend, life, ledger):
        c.config.remaining_token_admission_enabled = admission_enabled
        c.config.remaining_time_admission_enabled = admission_enabled
        if finish_on_last_round:
            c.config.max_rounds = c.round_index + 2
        artifact = c.runtime.artifacts['solver']
        binding = c.runtime.artifact_input_binding('solver')
        usage = ledger.status(); digest = ledger.digest()
        calls = backend.calls
        snapshot = c.control_snapshot()
        assert snapshot['allowed_actions'] == ['delete_agent']
        assert snapshot['legal_action_parameters']['delete_agent']['targets'] == ['planner0']
        # Cleanup must not even enter the runtime scheduler, including a dry sync.
        def forbidden(*args, **kwargs):
            raise AssertionError('Closing must not schedule a Worker')
        monkeypatch.setattr(c.runtime, 'execute', forbidden)
        run = GraphDirector(canvas=c, backend=MockBackend([
            json.dumps({'action': 'delete_agent', 'target': 'planner0'}),
            json.dumps({'action': 'finish', 'target': 'solver'}),
        ]), call_namespace='closing').run()
        assert run.finished, [t.feedback for t in run.turns]
        assert c.runtime.artifacts == {'solver': artifact}
        assert c.runtime.artifact_input_binding('solver') == binding
        assert ledger.status() == usage and ledger.digest() == digest and backend.calls == calls
        assert c.submission_receipt.payload['won']
        assert c.submission_receipt.artifact_id == artifact.artifact_id
        assert all(not e.executed_agents and not e.scheduled_agents for e in c.history[-2:])
        assert c.history[-2].token_admission['budget_cleanup'] == 'current_candidate_zero_worker_v1'
        assert receipt_error(c.submission_receipt, run=run, events=c.history,
            run_id=c.run_id, dataset='alfworld') is None


def test_multiple_isolated_nodes_can_be_deleted_in_bounded_sequence(tmp_path):
    with blocked_canvas(tmp_path, planners=2) as (c, backend, life, ledger):
        calls = backend.calls
        assert c._alfworld_budget_cleanup_targets() == ['planner0', 'planner1']
        for name in ['planner0', 'planner1']:
            assert step(c, {'action': 'delete_agent', 'target': name}).accepted
        assert c.control_snapshot()['allowed_actions'] == ['finish']
        assert step(c, {'action': 'finish', 'target': 'solver'}, True).accepted
        assert backend.calls == calls and c.submission_receipt.payload['won']


@pytest.mark.parametrize('change', ['dirty', 'prompt', 'resource', 'missing', 'subtask', 'not_won'])
def test_stale_or_ineligible_result_cannot_enable_cleanup(tmp_path, change):
    with blocked_canvas(tmp_path) as (c, backend, life, ledger):
        if change == 'dirty':
            c.dirty_agents.add('solver')
        elif change == 'prompt':
            c.graph.nodes['solver'].prompt += ' Different task'
        elif change == 'resource':
            life._results['solver']['session_id'] = 'another-session'
        elif change == 'missing':
            c.runtime.artifacts.pop('solver')
        elif change == 'subtask':
            c.graph.nodes['solver'].metadata['result_scope'] = 'subtask'
        else:
            life._results['solver']['won'] = False
        calls = backend.calls
        assert not c._alfworld_budget_cleanup_targets()
        assert not step(c, {'action': 'delete_agent', 'target': 'planner0'}).accepted
        assert not step(c, {'action': 'finish', 'target': 'solver'}, True).accepted
        assert backend.calls == calls and 'planner0' in c.graph.nodes


def test_cleanup_cannot_delete_winner_or_change_inputs_and_budget_stays_stopped(tmp_path):
    with blocked_canvas(tmp_path) as (c, backend, life, ledger):
        calls = backend.calls
        for action in [
            {'action': 'delete_agent', 'target': 'solver'},
            {'action': 'run_agent', 'target': 'solver'},
            {'action': 'add_agent', 'agent_id': 'new_budget'},
            {**prompt('solver'), 'objective': 'New task'},
        ]:
            assert not step(c, action).accepted
        assert backend.calls == calls
        assert step(c, {'action': 'delete_agent', 'target': 'planner0'}).accepted
        assert not step(c, {'action': 'run_agent', 'target': 'solver'}).accepted
        with pytest.raises(UsageDispatchStopped):
            ledger.begin(route='fixture', agent_id='new_agent', execution_id='new', request={})


def test_retained_stale_node_prevents_cleanup_scheduler_bypass(tmp_path):
    with blocked_canvas(tmp_path, planners=2) as (c, backend, life, ledger):
        c.dirty_agents.add('planner1')
        assert c._alfworld_budget_cleanup_targets() == ['planner1']
        assert not step(c, {'action': 'delete_agent', 'target': 'planner0'}).accepted
        assert step(c, {'action': 'delete_agent', 'target': 'planner1'}).accepted
        assert c._alfworld_budget_cleanup_targets() == ['planner0']


def test_deletion_of_dependency_cannot_be_mistaken_for_zero_worker_cleanup(tmp_path):
    with blocked_canvas(tmp_path) as (c, backend, life, ledger):
        # A concurrent graph edit changes the candidate's input binding.
        c.graph.set_layer('solver', 1)
        c.graph.set_relation('planner0', 'solver', 'directed')
        assert not c._alfworld_budget_cleanup_targets()
        assert not step(c, {'action': 'delete_agent', 'target': 'planner0'}).accepted


def test_forty_parallel_questions_keep_usage_and_cleanup_isolated(tmp_path):
    def run_one(index):
        directory = tmp_path / str(index); directory.mkdir()
        with blocked_canvas(directory) as (c, backend, life, ledger):
            assert step(c, {'action': 'delete_agent', 'target': 'planner0'}).accepted
            assert step(c, {'action': 'finish', 'target': 'solver'}, True).accepted
            return backend.calls, ledger.status()['confirmed_used'], ledger.digest()
    with ThreadPoolExecutor(max_workers=40) as pool:
        results = list(pool.map(run_one, range(40)))
    assert all(calls == 3 and tokens == 30 for calls, tokens, _ in results)
    assert len({digest for _, _, digest in results}) == 40
