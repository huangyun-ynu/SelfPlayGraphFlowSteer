"""SWE question credit is shared by real requests, including revisions/retries."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from selfplay_graph_flowsteer import llm
from selfplay_graph_flowsteer.canvas import GraphCanvas
from selfplay_graph_flowsteer.config import CanvasConfig
from selfplay_graph_flowsteer.contracts import AgentNode
from selfplay_graph_flowsteer.dataset_actions import default_dataset_action_registry
from selfplay_graph_flowsteer.endpoint_pool import EndpointPoolBackend
from selfplay_graph_flowsteer.runtime import (
    SWE_SHARED_TOKEN_BUDGET, ModelAgentExecutor, MultiAgentRuntime, RoutedModelAgentExecutor,
)

from .helpers import RecordingExecutor
from .test_runtime import FakeNamedTool
from .test_skill_refiner_json_backend import completion, fake_backend
from .test_unified_submission import add, step


class CreditRecordingExecutor(RecordingExecutor):
    def execute(self, **kwargs):
        credit = kwargs['node'].metadata['_runtime_token_credit']
        artifact = super().execute(**kwargs)
        self.calls[-1]['credit'] = credit
        artifact.token_in, artifact.token_out = 15000, 5000
        assert credit >= 20000
        return artifact


def canvas(tmp_path, executor, *, admission=False):
    return GraphCanvas(
        task='Fix repository', dataset='swe_bench', runtime=MultiAgentRuntime(executor),
        config=CanvasConfig(
            submission_protocol='unified_task_result_v1', max_rounds=60,
            submission_journal_dir=str(tmp_path), max_total_tokens=350000,
            remaining_token_admission_enabled=admission,
            # An obsolete configured reserve must have no effect on SWE.
            finalization_token_reserve=100000, worker_token_cold_start=300000,
        ),
    )


@pytest.mark.parametrize('admission', [False, True])
def test_bidirectional_cache_hits_do_not_divide_or_charge_shared_balance(tmp_path, admission):
    executor = CreditRecordingExecutor()
    c = canvas(tmp_path, executor, admission=admission)
    add(c, 'a', 'subtask')
    add(c, 'b', 'subtask')
    c.total_tokens = 223529  # Captured django-13569 question balance: 126471.
    for node in c.graph.nodes.values():
        node.metadata.update(
            _runtime_budget_kind='swe_primary_request_credit_v1',
            _runtime_token_credit=31617, _runtime_finalization_output_reserve=2048,
        )
    # The old estimator includes two cached initial passes plus two revisions.
    candidate = c.graph.clone()
    candidate.set_relation('a', 'b', 'bidirectional')
    assert c.runtime.estimate_execution_tokens(candidate, {'a', 'b'}, quantile=0.8,
        minimum_samples=3, cold_start_tokens=4096)['call_count'] == 4
    before = len(executor.calls)
    result = step(c, dict(action='set_relation', source='a', target='b', relation='bidirectional'))
    assert result.accepted, result.feedback
    assert result.execution.cache_hits == 2
    assert [call['credit'] for call in executor.calls[before:]] == [126471, 106471]
    assert all(call['revision'] for call in executor.calls[before:])
    assert result.execution.token_in + result.execution.token_out == 40000
    assert c.total_tokens == 263529
    assert c._token_admission_event['allocation'] == 'shared_remaining'
    assert c._token_admission_event['reserved_closure_tokens'] == 0
    assert 'per_execution_credit' not in c._token_admission_event
    assert all('_runtime_finalization_output_reserve' not in n.metadata for n in c.graph.nodes.values())


def test_explicit_recovery_and_recreation_keep_question_spending(tmp_path):
    executor = CreditRecordingExecutor()
    c = canvas(tmp_path, executor)
    add(c, 'a', 'subtask')
    c.dirty_agents.add('a')
    assert step(c, dict(action='run_agent', target='a')).accepted
    assert step(c, dict(action='delete_agent', target='a')).accepted
    add(c, 'a', 'subtask')
    assert [call['credit'] for call in executor.calls] == [350000, 330000, 310000]
    assert c.total_tokens == 60000


def test_force_and_discovered_stale_node_receive_live_balance(tmp_path):
    executor = CreditRecordingExecutor()
    c = canvas(tmp_path, executor)
    add(c, 'a', 'subtask')
    add(c, 'b', 'subtask')
    c.total_tokens = 250000
    c.runtime.cache.clear()
    c.runtime._stale_artifacts.update({'a', 'b'})
    c.dirty_agents.clear()
    for node in c.graph.nodes.values():
        node.metadata['_runtime_token_credit'] = 1
    before = len(executor.calls)
    report = c._execute_dirty(force=True)
    assert set(report.executed_agents) == {'a', 'b'}
    assert [call['credit'] for call in executor.calls[before:]] == [100000, 80000]
    assert c.total_tokens == 290000


def swe_node(credit=126471, *, kind=SWE_SHARED_TOKEN_BUDGET):
    return AgentNode('a', 'Inspect the assigned module', metadata={
        'action_adapter': 'swe_bench', 'submission_protocol': 'unified_task_result_v1',
        'result_scope': 'subtask', '_runtime_budget_kind': kind,
        '_runtime_token_credit': credit, '_runtime_finalization_output_reserve': 100000,
    })


def run_worker(backend, node, *, tools=None):
    tools = {name: (tools or {}).get(name, FakeNamedTool(name)) for name in
             ('swe_list', 'swe_search', 'swe_read', 'swe_edit', 'swe_apply_artifact', 'swe_test', 'swe_status')}
    node.allowed_tools = tuple(tools)
    return ModelAgentExecutor(
        backend, tools=tools, action_registry=default_dataset_action_registry(tools),
    ).execute(task='Fix repository', node=node, upstream=[], peers=[], revision=False, seed=0)


def gateway(*replies):
    backend, requests = fake_backend(*replies)
    backend.config.request_profile = 'openai'
    backend.config.roles['worker'] = replace(backend.config.roles['worker'], max_tokens=16384)
    return backend, requests


def reply(text='{"answer":"Located the assigned module"}', *, token_in=700, token_out=100, finish='stop'):
    result = completion(text, finish_reason=finish)
    result.usage = SimpleNamespace(prompt_tokens=token_in, completion_tokens=token_out)
    return result


@pytest.mark.parametrize('kind', [SWE_SHARED_TOKEN_BUDGET, 'swe_primary_request_credit_v1'])
def test_large_request_uses_whole_balance_without_submission_reserve(kind):
    backend, requests = gateway(reply())
    node = swe_node(kind=kind)
    node.prompt += 'x' * 57000
    artifact = run_worker(backend, node)
    assert len(requests) == 1
    event = next(e for e in artifact.backend_request_events if 'request_token_budget' in e)
    quote = event['request_token_budget']
    assert 31617 < quote['input_bound'] < quote['required_tokens'] <= 126471
    assert requests[0]['max_tokens'] == 16384
    assert artifact.token_in + artifact.token_out == 800
    assert not any(d['stage'] == 'swe_submission_reserve' for d in artifact.protocol_diagnostics)


@pytest.mark.parametrize('tool_limit', [1, 2])
def test_tool_loop_and_finalization_subtract_actual_usage(tool_limit):
    backend, requests = gateway(
        reply('{"action_call":{"name":"swe_read","arguments":{"path":"sample.py"}}}',
              token_in=2000, token_out=100),
        reply(token_in=1000, token_out=200),
    )
    node = swe_node(40000)
    node.allowed_tools = ('swe_read',)
    node.operation_policy_configured = True
    node.initial_tool_budget = node.total_tool_budget = tool_limit
    tool = FakeNamedTool('swe_read')
    artifact = run_worker(backend, node, tools={tool.name: tool})
    assert len(tool.calls) == 1 and len(requests) == 2
    quotes = [e['request_token_budget'] for e in artifact.backend_request_events if 'request_token_budget' in e]
    assert [q['credit'] for q in quotes] == [40000, 37900]
    assert requests[1]['max_tokens'] == (4096 if tool_limit == 1 else 16384)
    assert artifact.token_in == 3000 and artifact.token_out == 300


@pytest.mark.parametrize('credit', [0, 100])
def test_exhausted_shared_balance_dispatches_no_requests(credit):
    backend, requests = gateway(reply())
    artifact = run_worker(backend, swe_node(credit))
    assert not requests
    assert artifact.answer == 'WORKER_PROTOCOL_FAILURE'
    assert artifact.token_in + artifact.token_out == 0
    assert any(d['stage'] == 'swe_request_token_credit_exhausted' for d in artifact.protocol_diagnostics)


def test_output_cap_and_length_retry_cannot_spend_past_remaining_balance():
    backend, requests = gateway()
    credit = 12000

    def consume_remaining(**request):
        requests.append(request)
        quote = llm.request_budget_quote(request)
        assert 128 <= request['max_tokens'] < 16384
        assert quote['required_tokens'] == credit
        return reply('unfinished', token_in=quote['input_bound'],
                     token_out=request['max_tokens'], finish='length')

    backend.client.chat.completions.create = consume_remaining
    artifact = run_worker(backend, swe_node(credit))
    assert len(requests) == 1  # Retry and compact finalization are both denied before sending.
    assert artifact.token_in + artifact.token_out == credit
    assert artifact.answer == 'WORKER_PROTOCOL_FAILURE'


def test_successful_length_retry_usage_is_charged_once():
    backend, requests = gateway(reply('unfinished', token_in=700, token_out=100, finish='length'),
                                reply(token_in=900, token_out=50))
    artifact = run_worker(backend, swe_node(40000))
    assert len(requests) == 2
    assert artifact.token_in == 1600 and artifact.token_out == 150
    quotes = [e['request_token_budget'] for e in artifact.backend_request_events if 'request_token_budget' in e]
    assert [q['spent'] for q in quotes] == [0, 800]


def test_failed_length_repair_then_endpoint_failover_keeps_known_usage(tmp_path):
    error = RuntimeError('invalid api key')
    error.status_code = 401
    bad, first_requests = gateway(
        reply('unfinished', token_in=700, token_out=100, finish='length'),
        error,
    )
    good, second_requests = gateway(reply(token_in=900, token_out=50))
    pool = EndpointPoolBackend('gpt', {'gpt': bad, 'gpt_eco': good}, tmp_path)
    artifact = run_worker(pool, swe_node(40000))
    assert len(first_requests) == 2 and len(second_requests) == 1
    assert artifact.token_in == 1600 and artifact.token_out == 150


@pytest.mark.parametrize('tool_limit', [1, 2])
def test_backend_failure_preserves_prior_tool_and_recovery_usage(tmp_path, tool_limit):
    error = RuntimeError('invalid api key')
    error.status_code = 401
    backend, requests = gateway(
        reply('{"action_call":{"name":"swe_read","arguments":{"path":"sample.py"}}}',
              token_in=2000, token_out=100),
        reply('unfinished', token_in=700, token_out=100, finish='length'),
        error,
    )
    tools = {name: FakeNamedTool(name) for name in
             ('swe_list', 'swe_search', 'swe_read', 'swe_edit', 'swe_apply_artifact', 'swe_test', 'swe_status')}
    registry = default_dataset_action_registry(tools, swe_budgets=(tool_limit, 0, tool_limit))
    executor = RoutedModelAgentExecutor({'gpt': backend}, ('gpt',), tools=tools, action_registry=registry)
    c = GraphCanvas(task='Fix repository', dataset='swe_bench', runtime=MultiAgentRuntime(executor),
                    action_adapter=registry.get('swe_bench'),
                    config=CanvasConfig(submission_protocol='unified_task_result_v1',
                                        submission_journal_dir=str(tmp_path), max_total_tokens=350000))
    add(c, 'a', 'subtask')
    artifact = c.runtime.artifacts['a']
    assert artifact.answer == 'WORKER_BACKEND_FAILURE'
    assert len(requests) == 3
    assert artifact.token_in == 2700 and artifact.token_out == 200
    assert c.total_tokens == 2900
    assert c.runtime._execution_token_remaining == 350000 - 2900


@pytest.mark.parametrize('dataset,stage', [('hotpotqa', 'qa_submission_reserve'),
                                          ('healthbench_professional', 'worker_submission_reserve')])
def test_other_datasets_keep_existing_reserve(tmp_path, dataset, stage):
    backend, requests = gateway(reply('{"answer":"35"}'))
    c = GraphCanvas(task='Question', dataset=dataset, runtime=MultiAgentRuntime(ModelAgentExecutor(backend)),
                    config=CanvasConfig(submission_protocol='unified_task_result_v1',
                                        submission_journal_dir=str(tmp_path), max_total_tokens=350000))
    add(c, 'a', 'subtask')
    assert len(requests) == 1
    assert any(d['stage'] == stage for d in c.runtime.artifacts['a'].protocol_diagnostics)
