"""Failover must retain unknown physical attempts in the final request audit."""
import json
from types import SimpleNamespace

import pytest

from selfplay_graph_flowsteer import llm
from selfplay_graph_flowsteer.backend_failures import BackendRequestError
from selfplay_graph_flowsteer.endpoint_pool import EndpointPoolBackend, _merge_request_events
from selfplay_graph_flowsteer.worker_usage_ledger import worker_usage_scope
from .test_hotpot_usage_budget import gateway_slot, ledger


class PhysicalGateway:
    def __init__(self, name, *, fails=False):
        self.config = SimpleNamespace(base_url='https://' + name, timeout_s=30,
                                      route_name=name, stream=False)
        self.fails = fails

    def generate(self, messages, *, role):
        events = []
        with llm._capture_request_events(events, role=role):
            if self.fails:
                def fail(**kwargs):
                    error = RuntimeError('upstream returned HTTP 502')
                    error.status_code = 502
                    raise error
                llm._openai_response_attempt(
                    SimpleNamespace(responses=SimpleNamespace(create=fail)), self.config,
                    dict(model='student', input=messages), None,
                    attempt=1, request_budget_cap_s=30)
            else:
                response = SimpleNamespace(id='physical-success', usage=SimpleNamespace(
                    prompt_tokens=50, completion_tokens=10))
                llm._openai_completion_attempt(
                    SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: response))),
                    self.config, dict(model='fallback', messages=messages), None,
                    attempt=1, request_budget_cap_s=30)
        return llm.LLMResponse(text='Paris', model='fallback', metadata={'backend_request_events': events})


@pytest.mark.parametrize('fallback_succeeds', [True, False])
def test_real_gateway_failover_keeps_failed_attempt_id_in_final_audit(tmp_path, gateway_slot, fallback_succeeds):
    account = ledger(tmp_path)
    pool = EndpointPoolBackend('gpt', {'student': PhysicalGateway('student', fails=True),
                                     'fallback': PhysicalGateway('fallback', fails=not fallback_succeeds)}, tmp_path)
    with worker_usage_scope(account, agent_id='solver', execution_id='one'):
        if fallback_succeeds:
            result = pool.generate([{'role':'user', 'content':'public task'}], role='worker')
            events = result.metadata['backend_request_events']
        else:
            with pytest.raises(BackendRequestError) as error:
                pool.generate([{'role':'user', 'content':'public task'}], role='worker')
            events = error.value.request_events
    rows = account._db.execute('SELECT attempt_id,state FROM attempts ORDER BY started_at').fetchall()
    observed = [e['worker_usage_attempt_id'] for e in events if 'worker_usage_attempt_id' in e]
    assert observed == [r[0] for r in rows]
    assert rows[0][1] == 'unknown' and rows[1][1] == ('complete' if fallback_succeeds else 'unknown')
    assert account.status()['confirmed_used'] == (60 if fallback_succeeds else 0)
    audit = [json.loads(s) for s in pool.counter_path.with_suffix('.events.jsonl').read_text().splitlines()]
    assert audit[0]['backend_request_events'][0]['worker_usage_attempt_id'] == rows[0][0]
    assert all(e.get('request_role') == 'worker' for e in events if 'worker_usage_attempt_id' in e)
    account.close()


def test_merge_only_deduplicates_the_same_event_id():
    first = {'event_id':'one', 'worker_usage_attempt_id':'attempt-one'}
    second = {'event_id':'two', 'worker_usage_attempt_id':'attempt-two'}
    assert _merge_request_events([first], [first, second]) == [first, second]


def test_unknown_limit_stop_keeps_both_previous_physical_failures(tmp_path, gateway_slot):
    account = ledger(tmp_path)
    pool = EndpointPoolBackend('gpt', {'one':PhysicalGateway('one', fails=True),
                                     'two':PhysicalGateway('two', fails=True),
                                     'three':PhysicalGateway('three')}, tmp_path)
    with worker_usage_scope(account, agent_id='solver', execution_id='one'):
        with pytest.raises(llm.WorkerUsageDispatchStopped) as error:
            pool.generate([{'role':'user','content':'public task'}], role='worker')
    actual = {r[0] for r in account._db.execute('SELECT attempt_id FROM attempts')}
    assert len(actual) == 2
    assert {e['worker_usage_attempt_id'] for e in error.value.request_events} == actual
    assert account.status()['unsettled_attempt_count'] == 2
    account.close()
