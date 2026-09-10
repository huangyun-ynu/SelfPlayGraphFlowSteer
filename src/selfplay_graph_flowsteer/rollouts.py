from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


class Tokenizer(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...


@dataclass(frozen=True)
class DirectorTurn:
    """One FlowSteer-style model action followed by immutable environment feedback."""

    model_response: str
    feedback: str = ""
    accepted: bool = False
    graph_prefix: dict[str, Any] = field(default_factory=dict)
    prompt_messages: tuple[dict[str, str], ...] = ()
    trainable: bool = True
    call_id: str = ""
    completion_token_ids: tuple[int, ...] = ()
    behavior_log_probs: tuple[float, ...] = ()
    raw_reasoning_text: str = ""
    raw_action_text: str = ""
    prompt_token_ids: tuple[int, ...] = ()
    action_character_span: tuple[int, int] | None = None
    turn_kind: str = "graph_action"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TokenizedPolicyCall:
    call_id: str
    token_ids: tuple[int, ...]
    action_mask: tuple[int, ...]
    behavior_log_probs: tuple[float, ...] = ()
    action_token_span: tuple[int, int] | None = None
    relation_token_span: tuple[int, int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.token_ids) != len(self.action_mask):
            raise ValueError("policy-call token IDs and mask must have equal length")
        if self.behavior_log_probs and len(self.behavior_log_probs) != len(self.token_ids) - 1:
            raise ValueError("behavior log probabilities must align with next-token targets")


@dataclass(frozen=True)
class TokenizedDirectorTrajectory:
    rollout_id: str
    task_id: str
    token_ids: tuple[int, ...]
    action_mask: tuple[int, ...]
    reward: float
    graph: dict[str, Any]
    seed: int = 0
    executor_version: str = "v1"
    metadata: dict[str, Any] = field(default_factory=dict)
    policy_calls: tuple[TokenizedPolicyCall, ...] = ()

    def __post_init__(self) -> None:
        if len(self.token_ids) != len(self.action_mask):
            raise ValueError("token_ids and action_mask must have equal length")
        if any(value not in (0, 1) for value in self.action_mask):
            raise ValueError("action_mask values must be binary")
        if not any(self.action_mask) and self.metadata.get("training_eligible") is not False:
            raise ValueError("trajectory has no trainable Director action tokens")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["token_ids"] = list(self.token_ids)
        payload["action_mask"] = list(self.action_mask)
        payload["policy_calls"] = [asdict(call) for call in self.policy_calls]
        return payload


def tokenize_director_policy_calls(
    turns: list[DirectorTurn],
    tokenizer: Tokenizer,
    *,
    max_tokens: int = 4096,
) -> tuple[TokenizedPolicyCall, ...]:
    """Tokenize each actual policy call independently; never splice long contexts."""

    calls: list[TokenizedPolicyCall] = []
    chat_encoder = getattr(tokenizer, "encode_chat_trajectory", None)
    for index, turn in enumerate(turns):
        raw_completion = turn.raw_reasoning_text + turn.raw_action_text
        if not raw_completion:
            raw_completion = turn.model_response
        messages = [
            *[dict(message) for message in turn.prompt_messages],
            {"role": "assistant", "content": raw_completion},
        ]
        if turn.prompt_token_ids and turn.completion_token_ids:
            # Both sides of this actual request are already provider-attested.
            # Re-rendering a multi-turn chat can intentionally remove historical
            # reasoning and is not a valid prerequisite for using these exact IDs.
            token_ids = [*turn.prompt_token_ids, *turn.completion_token_ids]
            completion_span = (len(turn.prompt_token_ids), len(token_ids))
            action_mask = [0] * len(turn.prompt_token_ids) + [int(turn.trainable)] * len(
                turn.completion_token_ids
            )
        elif callable(chat_encoder):
            token_ids, _mask, assistant_spans = chat_encoder(messages)
            if not assistant_spans:
                raise ValueError("chat tokenizer did not expose the policy completion span")
            reconstructed_span = assistant_spans[-1]
            if turn.completion_token_ids:
                # The provider's sampled IDs are authoritative. Decoding then
                # re-tokenizing generated text is not lossless for every tokenizer
                # and can also synthesize a chat-template terminal token that the
                # policy did not sample.
                token_ids = [
                    *token_ids[: reconstructed_span[0]],
                    *(int(value) for value in turn.completion_token_ids),
                ]
                completion_span = (reconstructed_span[0], len(token_ids))
            else:
                completion_span = reconstructed_span
            action_mask = [0] * len(token_ids)
            if turn.trainable:
                action_mask[completion_span[0] : completion_span[1]] = [1] * (
                    completion_span[1] - completion_span[0]
                )
        else:
            prompt_ids: list[int] = []
            for message in turn.prompt_messages:
                prompt_ids.extend(tokenizer.encode(message["content"], add_special_tokens=False))
            completion = list(
                turn.completion_token_ids
                or tokenizer.encode(raw_completion, add_special_tokens=False)
            )
            token_ids = prompt_ids + completion
            completion_span = (len(prompt_ids), len(token_ids))
            action_mask = [0] * len(prompt_ids) + [int(turn.trainable)] * len(completion)
        effective_limit = (
            max_tokens * 64 if getattr(tokenizer, "approximate_token_count", False) else max_tokens
        )
        if len(token_ids) > effective_limit:
            raise ValueError(
                f"Director policy call {turn.call_id or index} has {len(token_ids)} tokens; "
                f"limit is {effective_limit}; discontinuous truncation is forbidden"
            )
        completion_length = completion_span[1] - completion_span[0]
        if turn.completion_token_ids and tuple(token_ids[slice(*completion_span)]) != tuple(
            turn.completion_token_ids
        ):
            raise ValueError(
                f"policy call {turn.call_id or index} completion tokens do not match the "
                "rollout-time tokenizer record"
            )
        if turn.prompt_token_ids and tuple(token_ids[: completion_span[0]]) != tuple(
            turn.prompt_token_ids
        ):
            raise ValueError(
                f"policy call {turn.call_id or index} prompt tokens do not match the "
                "rollout-time provider record"
            )
        behavior = [0.0] * max(0, len(token_ids) - 1)
        if turn.behavior_log_probs:
            if len(turn.behavior_log_probs) != completion_length:
                raise ValueError(
                    f"policy call {turn.call_id or index} rollout behavior log probabilities "
                    f"({len(turn.behavior_log_probs)}) do not match completion span "
                    f"({completion_length})"
                )
            for offset, value in enumerate(turn.behavior_log_probs):
                target_position = completion_span[0] + offset - 1
                if target_position < 0:
                    raise ValueError("policy completion cannot begin before a next-token context")
                behavior[target_position] = float(value)
        action_span = completion_span
        # All tokens actually generated by an ordinary Director call receive the
        # graph-level advantage.  Mapping parsed JSON character offsets back onto
        # provider token IDs is neither required nor generally reversible.  The
        # only local-credit span is the separately sampled one-token relation call.
        relation_span = completion_span if turn.turn_kind == "relation_choice" else None
        calls.append(
            TokenizedPolicyCall(
                call_id=turn.call_id or f"call-{index}",
                token_ids=tuple(int(value) for value in token_ids),
                action_mask=tuple(action_mask),
                behavior_log_probs=tuple(behavior) if turn.behavior_log_probs else (),
                action_token_span=action_span,
                relation_token_span=relation_span,
                metadata=dict(turn.metadata),
            )
        )
    return tuple(calls)


def tokenize_director_turns_with_spans(
    turns: list[DirectorTurn],
    tokenizer: Tokenizer,
    *,
    max_tokens: int = 1024,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[tuple[int, int] | None, ...]]:
    """Legacy audit tokenizer; new trainable policy calls use the no-splice path above."""

    chat_encoder = getattr(tokenizer, "encode_chat_trajectory", None)
    if callable(chat_encoder) and turns and turns[-1].prompt_messages:
        messages = _director_chat_messages(turns)
        token_ids, action_mask, original_spans = chat_encoder(messages)
        if len(original_spans) != len(turns):
            raise ValueError("chat tokenizer did not return one Director action span per turn")
        for turn, span in zip(turns, original_spans, strict=True):
            if not turn.trainable:
                start, end = span
                action_mask[start:end] = [0] * (end - start)
        return _truncate_with_spans(token_ids, action_mask, original_spans, max_tokens=max_tokens)

    token_ids: list[int] = []
    action_mask: list[int] = []
    original_spans: list[tuple[int, int]] = []
    for turn in turns:
        model_ids = tokenizer.encode(turn.model_response, add_special_tokens=False)
        feedback_ids = tokenizer.encode(turn.feedback, add_special_tokens=False)
        start = len(token_ids)
        token_ids.extend(model_ids)
        original_spans.append((start, len(token_ids)))
        action_mask.extend([int(turn.trainable)] * len(model_ids))
        token_ids.extend(feedback_ids)
        action_mask.extend([0] * len(feedback_ids))
    return _truncate_with_spans(token_ids, action_mask, original_spans, max_tokens=max_tokens)


def _director_chat_messages(turns: list[DirectorTurn]) -> list[dict[str, str]]:
    """Rebuild all turns even when a constrained retry intentionally resets context."""

    first_prompt = turns[0].prompt_messages
    system = next(
        (dict(message) for message in first_prompt if message.get("role") == "system"),
        None,
    )
    messages: list[dict[str, str]] = [system] if system is not None else []
    for turn in turns:
        user = next(
            (
                dict(message)
                for message in reversed(turn.prompt_messages)
                if message.get("role") == "user"
            ),
            None,
        )
        if user is None:
            raise ValueError("Director turn has no user request in prompt_messages")
        messages.append(user)
        messages.append({"role": "assistant", "content": turn.model_response})
    return messages


def _truncate_with_spans(
    token_ids: list[int],
    action_mask: list[int],
    original_spans: list[tuple[int, int]],
    *,
    max_tokens: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[tuple[int, int] | None, ...]]:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if len(token_ids) > max_tokens:
        head = max_tokens // 2
        tail = max_tokens - head
        retained = list(range(len(token_ids)))[:head] + list(range(len(token_ids)))[-tail:]
        token_ids = token_ids[:head] + token_ids[-tail:]
        action_mask = action_mask[:head] + action_mask[-tail:]
        remap = {old: new for new, old in enumerate(retained)}
        spans: list[tuple[int, int] | None] = []
        for start, end in original_spans:
            positions = [remap[position] for position in range(start, end) if position in remap]
            spans.append((min(positions), max(positions) + 1) if positions else None)
        return tuple(token_ids), tuple(action_mask), tuple(spans)
    return tuple(token_ids), tuple(action_mask), tuple(original_spans)


@dataclass(frozen=True)
class TrainingSample:
    rollout_id: str
    task_id: str
    token_ids: tuple[int, ...]
    action_mask: tuple[int, ...]
    reward: float
    advantage: float
    density: float = 1.0
    canonical_graph_key: str = ""
    graph_features: tuple[float, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    policy_calls: tuple[TokenizedPolicyCall, ...] = ()


@dataclass(frozen=True)
class TrainingBatch:
    """Trainer-ready data boundary. Optimizer/model references are intentionally absent."""

    role: str
    samples: tuple[TrainingSample, ...]
    objective: str = "masked_graph_level_grpo"
    optimizer_steps: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.optimizer_steps != 0:
            raise ValueError("rollout batches must stop before optimizer execution")

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "objective": self.objective,
            "optimizer_steps": self.optimizer_steps,
            "samples": [asdict(sample) for sample in self.samples],
            "metadata": self.metadata,
        }
