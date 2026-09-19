from types import SimpleNamespace

from selfplay_graph_flowsteer.qwen_compat import response_content, response_policy_parts


def test_qwen_reasoning_alias_is_preserved_with_visible_action():
    message = SimpleNamespace(
        reasoning="private reasoning",
        reasoning_content="",
        content='\n\n{"action":"finish"}',
    )

    reasoning, action = response_policy_parts(message, thinking_prefilled=True)

    assert reasoning == "private reasoning"
    assert action == '\n\n{"action":"finish"}'
    assert response_content(message, enable_thinking=True) == '{"action":"finish"}'
