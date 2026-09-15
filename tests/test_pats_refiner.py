from __future__ import annotations

import json
from dataclasses import replace

import pytest
from jsonschema import Draft202012Validator, ValidationError, validate

from selfplay_graph_flowsteer.llm import LLMResponse, MockBackend
from selfplay_graph_flowsteer.observability import TaskSpec
from selfplay_graph_flowsteer.pats import PatsConfig, PatsController
from selfplay_graph_flowsteer.pats_refiner import (
    build_review_messages,
    normalize_evidence_references,
    parse_review_response,
    public_response_audit,
    response_generation_audit,
    review_json_schema,
    review_system_prompt,
)
from selfplay_graph_flowsteer.skill_evolution_v2 import SkillStore


def evidence(count=32):
    return [
        {
            "id": f"e{i:02d}",
            "task_id": f"task{i}",
            "step": 3 if i < 4 else 2,
            "reward_mean": float(i % 2),
            "examples": [{"trace": {"feedback": "bounded public process " * 100}}],
        }
        for i in range(count)
    ]


def build(config, records):
    return build_review_messages(
        scope='["nq_open","qa","unspecified"]',
        mode="REVISE",
        policy_snapshot="p3",
        record={"ema": 0.5, "cards": [], "evidence": records},
        config=config,
        token_counter=len,
    )


def canonical_supplied(messages, audit):
    return [
        dict(item, id=audit["evidence_aliases"][item["id"]])
        for item in json.loads(messages[-1]["content"])["evidence"]
    ]


def test_budget_preserves_whole_distinct_recent_and_contrasting_evidence(monkeypatch):
    monkeypatch.setattr(
        "selfplay_graph_flowsteer.pats_refiner.review_system_prompt", lambda *_: "Review"
    )
    original = evidence()
    config = PatsConfig(enabled=True, max_review_input_tokens=12288)
    messages, admitted, audit = build(config, original)
    supplied = canonical_supplied(messages, audit)
    assert 2 <= len(admitted) < len(original)
    assert supplied == admitted
    assert all(e in original for e in supplied)
    assert len({e["task_id"] for e in supplied}) >= 2
    assert {e["reward_mean"] for e in supplied[:2]} == {0.0, 1.0}
    assert all(e["step"] == 3 for e in supplied[:2])
    assert audit["input_tokens_director_tokenizer"] <= 12288
    assert audit["evidence_groups_omitted"] == len(original) - len(supplied)
    assert build(config, list(reversed(original))) == (messages, admitted, audit)
    assert len(original) == 32


def test_too_small_budget_fails_instead_of_sending_unusable_evidence():
    with pytest.raises(ValueError, match="budget cannot fit"):
        build(replace(PatsConfig(enabled=True), max_review_input_tokens=100), evidence())


def test_large_preferred_group_does_not_block_two_complete_smaller_tasks(monkeypatch):
    # Isolate the packing regression from wording changes to the real prompt.
    monkeypatch.setattr(
        "selfplay_graph_flowsteer.pats_refiner.review_system_prompt",
        lambda config, mode: "Review the supplied public evidence.",
    )
    original = [
        {
            "id": key,
            "task_id": key,
            "step": 0,
            "reward_mean": reward,
            "examples": [{"detail": "x" * size}],
        }
        for key, reward, size in (("a", 0, 9000), ("b", 1, 4500), ("c", 0.5, 4500))
    ]
    config = PatsConfig(enabled=True, max_review_input_tokens=12288)
    messages, admitted, audit = build(config, original)
    assert {e["id"] for e in admitted} == {"b", "c"}
    assert canonical_supplied(messages, audit) == admitted
    assert all(item in original for item in admitted)
    assert audit["input_tokens_director_tokenizer"] <= 12288
    assert audit["evidence_groups_omitted"] == 1
    assert build(config, list(reversed(original))) == (messages, admitted, audit)


def test_small_group_fallback_does_not_fill_admission_with_duplicate_source_tasks(monkeypatch):
    monkeypatch.setattr(
        "selfplay_graph_flowsteer.pats_refiner.review_system_prompt", lambda *_: "Review"
    )
    original = [
        {
            "id": key,
            "task_id": task,
            "step": 0,
            "reward_mean": reward,
            "examples": [{"detail": "x" * size}],
        }
        for key, task, reward, size in (
            ("a", "big", 0, 9000),
            ("b1", "source-b", 1, 4000),
            ("b2", "source-b", 1, 4000),
            ("c", "source-c", 0.5, 4500),
        )
    ]
    _, admitted, _ = build(PatsConfig(enabled=True, max_review_input_tokens=12288), original)
    assert {e["task_id"] for e in admitted} == {"source-b", "source-c"}


def test_unfit_review_has_input_error_audit_and_zero_backend_calls(tmp_path):
    calls = []

    def respond(messages, role):
        calls.append(messages)
        return '{"operations": []}'

    tasks = {key: TaskSpec(key, "Find public evidence", task_type="qa") for key in ("a", "b")}
    rows = [
        {"rollout_id": key, "task_id": key, "reward": 0, "metadata": {"reward_known": True}}
        for key in tasks
    ]
    controller = PatsController(
        SkillStore(tmp_path / "bank.db"),
        replace(PatsConfig(enabled=True), max_review_input_tokens=100),
        len,
    )
    receipt = controller.maintain(
        tasks, rows, run="r", step=0, policy_snapshot="p", backend=MockBackend(handler=respond)
    )
    assert receipt["refiner_calls"] == 0
    assert calls == []
    assert receipt["reviews"][0]["status"] == "rejected"
    assert receipt["reviews"][0]["error_stage"] == "input_validation"
    assert "budget cannot fit" in receipt["reviews"][0]["validation_error"]


def test_only_complete_json_or_complete_outer_fence_is_accepted():
    payload = {"operations": []}
    assert parse_review_response(json.dumps(payload)) == payload
    assert parse_review_response('```json\n{"operations": []}\n```') == payload
    with pytest.raises(ValueError):
        parse_review_response('explanation\n```json\n{"operations": []}\n```')
    with pytest.raises(ValueError):
        parse_review_response('[{"operations": []}]')
    with pytest.raises(ValueError):
        parse_review_response('{"operations": []}\n{"operations": []}')


def test_modes_describe_actual_maintenance_strategy():
    config = PatsConfig()
    assert "recurring process failures" in review_system_prompt(config, "EXPAND")
    assert "more broadly applicable" in review_system_prompt(config, "REVISE")
    assert "Withdraw redundant" in review_system_prompt(config, "COMPRESS")
    assert "reached capacity" in review_system_prompt(config, "FORCED_PRUNE")


def proposal():
    return {
        "operations": [
            {
                "op": "ADD",
                "evidence_ids": ["E1", "E2"],
                "card": {
                    "name": "Check disagreements",
                    "description": "Resolve disagreement using public sources",
                    "trigger": "Visible sources disagree",
                    "plan": "Compare source dates and ask a checker to resolve contradictions",
                    "pitfall": "Do not treat repeated claims as independent support",
                    "constraint": "Use only available public evidence",
                    "kind": "verification",
                },
            }
        ],
    }


def run_review(tmp_path, backend):
    tasks = {key: TaskSpec(key, "Find public evidence", task_type="qa") for key in ("a", "b")}
    rows = [
        {"rollout_id": key, "task_id": key, "reward": 0, "metadata": {"reward_known": True}}
        for key in tasks
    ]
    controller = PatsController(
        SkillStore(tmp_path / "bank.db"), PatsConfig(enabled=True, max_tokens=4000), len
    )
    receipt = controller.maintain(
        tasks, rows, run="r", step=0, policy_snapshot="p", backend=backend
    )
    return receipt, controller.snapshot()


def test_short_ids_change_only_request_identifiers_and_resolve_exactly():
    original = evidence(2)
    messages, admitted, audit = build(PatsConfig(enabled=True), original)
    wire = json.loads(messages[-1]["content"])
    assert wire["allowed_evidence_ids"] == ["E1", "E2"]
    assert [item["id"] for item in wire["evidence"]] == ["E1", "E2"]
    assert canonical_supplied(messages, audit) == admitted
    payload = proposal()
    normalized = normalize_evidence_references(payload, audit["evidence_aliases"])
    assert normalized["operations"][0]["evidence_ids"] == [
        audit["evidence_aliases"]["E1"],
        audit["evidence_aliases"]["E2"],
    ]
    assert payload["operations"][0]["evidence_ids"] == ["E1", "E2"]
    assert normalize_evidence_references(normalized, audit["evidence_aliases"]) == normalized


@pytest.mark.parametrize(
    "bad", ["E0", "E01", "e1", " E1", "E1 ", "source-task", "a" * 12, "c" * 64]
)
def test_reference_normalization_rejects_unknown_or_approximate_identifiers(bad):
    payload = proposal()
    payload["operations"][0]["evidence_ids"] = ["E1", bad]
    with pytest.raises(ValueError, match="unknown evidence"):
        normalize_evidence_references(payload, {"E1": "a" * 64, "E2": "b" * 64})


def test_strict_schema_constrains_operations_card_kinds_and_supplied_identifiers():
    aliases = {"E1": "a" * 64, "E2": "b" * 64}
    records = [{"card": {"skill_id": "existing"}}]
    schema = review_json_schema(
        config=PatsConfig(), mode="EXPAND", records=records, evidence_aliases=aliases
    )
    Draft202012Validator.check_schema(schema)
    validate(proposal(), schema)
    validate(
        {"operations": [{"op": "DELETE", "skill_id": "existing", "evidence_ids": ["E1", "E2"]}]},
        schema,
    )
    for field, value in (("evidence_ids", ["E1", "E3"]), ("skill_id", "invented")):
        payload = proposal()
        payload["operations"][0][field] = value
        with pytest.raises(ValidationError):
            validate(payload, schema)
    for bad in ("unknown_kind", 17):
        payload = proposal()
        payload["operations"][0]["card"]["kind"] = bad
        with pytest.raises(ValidationError):
            validate(payload, schema)
    payload = proposal()
    del payload["operations"][0]["card"]["plan"]
    with pytest.raises(ValidationError):
        validate(payload, schema)
    for mode in ("COMPRESS", "FORCED_PRUNE"):
        restrictive = review_json_schema(
            config=PatsConfig(), mode=mode, records=records, evidence_aliases=aliases
        )
        with pytest.raises(ValidationError):
            validate(proposal(), restrictive)
    empty = review_json_schema(
        config=PatsConfig(), mode="COMPRESS", records=[], evidence_aliases=aliases
    )
    Draft202012Validator.check_schema(empty)
    validate({"operations": []}, empty)
    with pytest.raises(ValidationError):
        validate(proposal(), empty)


def test_controller_uses_schema_normalizes_refs_and_audits_public_response(tmp_path):
    calls = []

    class Backend:
        def generate_json(self, messages, *, role, schema):
            assert role == "skill-distiller"
            calls.append(messages)
            if "allowed_evidence_ids" not in json.loads(messages[-1]["content"]):
                response = {
                    "cards": {
                        alias: {
                            "approved": True,
                            "reason": "The card preserves public-evidence limits",
                        }
                        for alias in json.loads(messages[-1]["content"])["cards"]
                    }
                }
                validate(response, schema)
                return LLMResponse(text=json.dumps(response), model="independent-test-checker")
            assert json.loads(messages[-1]["content"])["allowed_evidence_ids"] == ["E1", "E2"]
            validate(proposal(), schema)
            return LLMResponse(
                text=json.dumps(proposal()),
                model="test-refiner",
                token_out=607,
                metadata={
                    "finish_reason": "stop",
                    "generation_attempts": [
                        {
                            "finish_reason": "length",
                            "token_in": 10,
                            "token_out": 20,
                            "max_output_tokens": 20,
                        },
                        {
                            "finish_reason": "stop",
                            "token_in": 20,
                            "token_out": 587,
                            "max_output_tokens": 1000,
                        },
                    ],
                },
            )

        def generate(self, *args, **kwargs):
            raise AssertionError("structured backend must not use unconstrained generation")

    receipt, snapshot = run_review(tmp_path, Backend())
    review = receipt["reviews"][0]
    assert receipt["refiner_calls"] == receipt["semantic_checker_calls"] == 1
    assert len(calls) == 2
    assert review["status"] == "updated" and review["structured_output"] == "json_schema"
    assert review["generation_attempts_count"] == 2 and review["finish_reason"] == "stop"
    assert review["response_text"] == json.dumps(proposal())
    assert review["returned_evidence_ids"] == [["E1", "E2"]]
    canonical = [review["evidence_aliases"][alias] for alias in ("E1", "E2")]
    assert review["operations"][0]["evidence_ids"] == canonical
    assert next(iter(snapshot["scopes"].values()))["cards"][0]["source_cases"] == sorted(canonical)


def test_explicit_unsupported_schema_falls_back_without_inflating_review_count(tmp_path):
    calls = []

    class Backend:
        def generate_json(self, *args, **kwargs):
            raise NotImplementedError("surface does not support JSON schema before any API request")

        def generate(self, messages, *, role):
            calls.append(messages)
            return LLMResponse(text='{"operations": []}', model="text-only")

    receipt, _ = run_review(tmp_path, Backend())
    assert receipt["refiner_calls"] == len(calls) == 1
    assert receipt["reviews"][0]["status"] == "unchanged"
    assert receipt["reviews"][0]["structured_output"] == "unsupported_fallback"


@pytest.mark.parametrize(
    "error", [ValueError("provider API failed"), RuntimeError("provider API failed")]
)
def test_schema_api_failure_is_not_silently_retried_as_plain_text(tmp_path, error):
    calls = []

    class Backend:
        def generate_json(self, *args, **kwargs):
            calls.append("schema")
            raise error

        def generate(self, *args, **kwargs):
            calls.append("text")
            return LLMResponse(text='{"operations": []}', model="test")

    receipt, _ = run_review(tmp_path, Backend())
    assert receipt["refiner_calls"] == 1 and calls == ["schema"]
    review = receipt["reviews"][0]
    assert review["status"] == "rejected" and review["error_stage"] == "refiner_request"
    assert "provider API failed" not in json.dumps(review)


@pytest.mark.parametrize(
    "raw, error",
    [
        ('{"operations":[{"op":"ADD","evidence_ids":["E1","unknown"]}]}', "unknown evidence"),
        ('{"operations":[{"op":"ADD" "evidence_ids":["E1","E2"]}]}', "Expecting ',' delimiter"),
    ],
)
def test_real_run_rejection_classes_retain_diagnostics_without_committing(tmp_path, raw, error):
    receipt, snapshot = run_review(tmp_path, MockBackend(handler=lambda *_: raw))
    review = receipt["reviews"][0]
    assert review["status"] == "rejected" and review["error_stage"] == "output_validation"
    assert error in review["validation_error"] and review["response_text"] == raw
    assert next(iter(snapshot["scopes"].values()))["cards"] == []
    if "unknown" in raw:
        assert review["returned_evidence_ids"] == [["E1", "unknown"]]


def test_response_audit_removes_reasoning_and_bounds_generation_details():
    raw = '<think>PRIVATE_THOUGHT</think>{"operations":[],"reasoning":"PRIVATE_REASONING" "diagnostic":"malformed"}'
    audit = public_response_audit(raw)
    assert "PRIVATE_" not in audit["response_text"] and "malformed" in audit["response_text"]
    assert public_response_audit("x" * 20000)["response_text_truncated"] is True
    assert len(public_response_audit("x" * 20000)["response_text"]) == 12000
    response = LLMResponse(
        text="",
        model="test",
        metadata={
            "generation_attempts": [{"finish_reason": "stop", "token_out": 1, "private": "SECRET"}]
            * 12
        },
    )
    audit = response_generation_audit(response)
    assert audit["generation_attempts_count"] == 12 and audit["generation_attempts_omitted"] == 4
    assert len(audit["generation_attempts"]) == 8 and "SECRET" not in json.dumps(audit)
