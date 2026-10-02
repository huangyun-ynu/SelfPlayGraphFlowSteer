import copy
import json

import pytest

from selfplay_graph_flowsteer.answer_submission import AnswerFinalizer
from selfplay_graph_flowsteer.contracts import AgentArtifact
from selfplay_graph_flowsteer.healthbench_artifact import (
    ARRAY_FIELDS,
    HealthBenchRepair,
    normalize,
    validate_preserved_artifact,
)
from selfplay_graph_flowsteer.observability import TaskSpec
from selfplay_graph_flowsteer.runtime import _finalization_recovery_messages


def state():
    return HealthBenchRepair({"agent_id": "solver", "execution_id": "1:solver",
                              "question_attempt_id": "q1", "input_sha256": "input-1"})


def consume(repair, payload, *, kind="initial", attempt=0, metadata=None):
    repair.consume(json.dumps(payload, ensure_ascii=False), metadata or {"finish_reason": "stop"},
                   kind=kind, model_attempt=attempt)


def artifact(repair):
    result = AgentArtifact.from_model_text(
        text=repair.raw_response, validated_payload=repair.payload, preserve_answer=True,
        artifact_id="a1", agent_id="solver",
    )
    result.normalized_payload = copy.deepcopy(repair.payload)
    result.healthbench_repair = repair.audit()
    return result


@pytest.mark.parametrize("key", ARRAY_FIELDS)
@pytest.mark.parametrize("value", ["", "None", "N/A", "  exact\nUnicode 中文  ", [], ["original"]])
def test_local_normalization_is_lossless_and_idempotent(key, value):
    answer = '  Unicode 中文 🧪 \\path\n<think>literal data</think>\n' * 200 + " END  \n"
    original = {"answer": answer, key: value, "confidence": 0.7}
    repair = state()
    consume(repair, original)
    assert repair.accepted
    assert repair.payload["answer"] == answer
    expected = value if isinstance(value, list) else [value] if value else []
    assert repair.payload[key] == expected
    assert normalize(repair.payload) == (repair.payload, [])
    assert validate_preserved_artifact(artifact(repair), question_attempt_id="q1")
    assert original[key] == value


@pytest.mark.parametrize("text", [
    '{"answer":"first", "answer":"second"}', '{"answer":"one"}{"answer":"two"}',
    '{"answer":"unfinished', '{"answer":null}', '{"answer":true}', '{"answer":123}',
    '{"answer":"  "}', '{}', '{"answer":"sure"}',
    '{"answer":"valid", "tool_calls":[]}', '{"answer":"valid", "confidence":NaN}',
    'prefix {"answer":"valid"} suffix',
])
def test_ambiguous_incomplete_or_unqualified_answers_are_not_locked(text):
    repair = state()
    repair.consume(text, {"finish_reason": "stop"}, kind="initial", model_attempt=0)
    assert not repair.accepted and repair.protected_answer is None
    assert repair.mode == "full_artifact_recovery"


@pytest.mark.parametrize("metadata", [{"finish_reason": "length"}, {"finish_reason": "MAX_TOKENS"},
                                      {"status": "incomplete"}, {}, {"finish_reason": "content_filter"}])
def test_completion_is_independent_of_json_validity(metadata):
    repair = state()
    repair.consume('{"answer":"a valid but possibly partial answer"}', metadata,
                   kind="initial", model_attempt=0)
    assert not repair.accepted and repair.protected_answer is None


def test_only_outer_reasoning_and_single_fence_are_unwrapped():
    repair = state()
    repair.consume('<think>outside</think>\n```json\n{"answer":"keep <think>inside</think>"}\n```',
                   {"status": "completed"}, kind="initial", model_attempt=0)
    assert repair.accepted and repair.payload["answer"] == "keep <think>inside</think>"


@pytest.mark.parametrize("bad", [None, 12, {}, True])
def test_unsafe_metadata_requires_patch_without_coercion(bad):
    repair = state()
    consume(repair, {"answer": "  complete body  ", "evidence": bad, "confidence": 8, "summary": []})
    assert not repair.accepted and repair.mode == "metadata_patch"
    assert repair.payload["confidence"] == 8 and repair.payload["evidence"] == bad
    assert set(repair.errors) == {"evidence", "confidence", "summary"}
    consume(repair, {"answer": "short", "evidence": [], "confidence": 0.5, "summary": "fixed"},
            kind="metadata_patch", attempt=1)
    assert repair.issue == "unauthorized_metadata_patch_fields"
    assert repair.protected_answer == "  complete body  "
    consume(repair, {"evidence": [], "confidence": 0.5, "summary": "fixed"},
            kind="metadata_patch", attempt=2)
    assert repair.accepted and validate_preserved_artifact(artifact(repair))
    assert repair.audit()["model_repair_count"] == 2


@pytest.mark.parametrize("mutation", ["answer", "payload", "raw", "patch", "agent", "version", "source"])
def test_persisted_artifact_replays_provenance_and_rejects_tampering(mutation):
    repair = state()
    consume(repair, {"answer": "  complete\nanswer  ", "summary": 42})
    consume(repair, {"summary": "fixed"}, kind="metadata_patch", attempt=1)
    packet = AgentArtifact(**json.loads(json.dumps(artifact(repair).to_dict())))
    assert validate_preserved_artifact(packet, question_attempt_id="q1")
    assert not validate_preserved_artifact(packet, question_attempt_id="q2")
    if mutation == "answer":
        packet.answer = packet.answer.strip()
    elif mutation == "payload":
        packet.normalized_payload["answer"] = "short"
    elif mutation == "raw":
        packet.raw_response += "changed"
    elif mutation == "patch":
        packet.healthbench_repair["records"][1]["text"] = '{"summary":"fixed", "answer":"short"}'
    elif mutation == "agent":
        packet.agent_id = "another"
    elif mutation == "version":
        packet.healthbench_repair["version"] = "untrusted"
    else:
        packet.healthbench_repair["records"][0]["raw_response_sha256"] = "bad"
    assert not validate_preserved_artifact(packet)


def test_healthbench_finalizer_preserves_whitespace_only_under_new_policy():
    raw = "  long medical answer\n\n"
    task = TaskSpec("q1", "prompt", metadata={"dataset": "healthbench_professional",
                                             "worker_usage": {"policy": "reported_usage_threshold_v1"}})
    assert AnswerFinalizer().finalize(task, raw).submitted_answer == raw
    task.metadata.pop("worker_usage")
    assert AnswerFinalizer().finalize(task, raw).submitted_answer == raw.strip()


def test_healthbench_full_recovery_does_not_excerpt_long_previous_response():
    raw = "x" * 9000 + "keep-tail"
    messages = _finalization_recovery_messages(
        instruction="answer", react_trace=[], previous_attempt_issue="invalid_json",
        visible_context={}, previous_response=raw, dataset="healthbench_professional",
        preserve_healthbench_response=True,
    )
    context = json.loads(messages[-1]["content"])
    assert context["previous_response"] == raw
    assert context["previous_response_excerpted"] is False


def test_full_recovery_and_patch_share_one_two_attempt_limit():
    repair = state()
    repair.consume("{broken", {"finish_reason": "stop"}, kind="initial", model_attempt=0)
    consume(repair, {"answer": "new complete answer", "summary": 5}, kind="full_artifact_recovery", attempt=1)
    consume(repair, {"summary": "fixed"}, kind="metadata_patch", attempt=2)
    assert validate_preserved_artifact(artifact(repair))
    assert repair.audit()["model_repair_count"] == 2
