"""Encode recorded action fixtures as provider-native responses, never text calls."""
from selfplay_graph_flowsteer.action_protocol import ActionCall
from selfplay_graph_flowsteer.llm import LLMResponse


def native_response(value):
    calls = value if isinstance(value, list) else [value]
    return LLMResponse(text='', model='fixture-native', action_calls=[
        ActionCall(call.get('call_id', f'native-fixture-{i}'), call['name'], call['arguments'])
        for i, call in enumerate(calls)
    ])
