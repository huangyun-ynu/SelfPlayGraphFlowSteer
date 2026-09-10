from __future__ import annotations

from selfplay_graph_flowsteer.rollouts import DirectorTurn, tokenize_director_turns_with_spans


class ByteTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return list(text.encode())


def test_provider_attested_policy_call_does_not_reconstruct_thinking_history():
    from selfplay_graph_flowsteer.rollouts import tokenize_director_policy_calls

    class TransformingTemplate(ByteTokenizer):
        def encode_chat_trajectory(self, messages):
            raise AssertionError("must not re-render provider-attested policy calls")

    turn = DirectorTurn(
        model_response='reasoning</think>{"action":"finish"}',
        prompt_token_ids=(10, 11),
        completion_token_ids=(12, 13, 14),
        behavior_log_probs=(-0.1, -0.2, -0.3),
    )
    (call,) = tokenize_director_policy_calls([turn], TransformingTemplate(), max_tokens=5)
    assert call.token_ids == (10, 11, 12, 13, 14)
    assert call.action_mask == (0, 0, 1, 1, 1)
    assert call.behavior_log_probs == (0.0, -0.1, -0.2, -0.3)


class ChatTokenizer(ByteTokenizer):
    def encode_chat_trajectory(self, messages):
        token_ids = []
        mask = []
        spans = []
        for message in messages:
            encoded = list(message["content"].encode())
            start = len(token_ids)
            token_ids.extend(encoded)
            trainable = message["role"] == "assistant"
            mask.extend([int(trainable)] * len(encoded))
            if trainable:
                spans.append((start, len(token_ids)))
        return token_ids, mask, spans


def test_flowsteer_action_mask_excludes_canvas_feedback() -> None:
    ids, mask, _ = tokenize_director_turns_with_spans(
        [DirectorTurn("act", "feedback"), DirectorTurn("go", "ok")], ByteTokenizer()
    )
    assert len(ids) == len(mask)
    assert mask == (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0)


def test_trajectory_truncation_keeps_mask_aligned() -> None:
    ids, mask, _ = tokenize_director_turns_with_spans(
        [DirectorTurn("abcdef", "uvwxyz")], ByteTokenizer(), max_tokens=6
    )
    assert ids == tuple(b"abcxyz")
    assert mask == (1, 1, 1, 0, 0, 0)


def test_responsibility_violation_tokens_are_not_trainable() -> None:
    ids, mask, spans = tokenize_director_turns_with_spans(
        [
            DirectorTurn("accepted", "ok"),
            DirectorTurn("leaked-answer", "rejected", trainable=False),
        ],
        ByteTokenizer(),
    )
    assert len(ids) == len(mask)
    assert spans[0] is not None and spans[1] is not None
    assert all(mask[index] == 0 for index in range(*spans[1]))
    assert any(mask[index] == 1 for index in range(*spans[0]))


def test_chat_mask_rebuilds_turns_after_constrained_context_reset() -> None:
    system = {"role": "system", "content": "rules"}
    turns = [
        DirectorTurn(
            "add-agent",
            prompt_messages=(system, {"role": "user", "content": "task"}),
        ),
        DirectorTurn(
            "leaked-answer",
            prompt_messages=(system, {"role": "user", "content": "retry only"}),
            trainable=False,
        ),
    ]
    _, mask, spans = tokenize_director_turns_with_spans(turns, ChatTokenizer())
    assert len(spans) == 2
    assert all(mask[index] == 1 for index in range(*spans[0]))
    assert all(mask[index] == 0 for index in range(*spans[1]))
