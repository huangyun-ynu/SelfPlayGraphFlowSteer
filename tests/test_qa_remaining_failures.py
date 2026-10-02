"""Real repair replays and actual Canvas paths, without prescribing a QA workflow."""

import importlib
import copy
import json
from pathlib import Path

import pytest

from selfplay_graph_flowsteer.qa_result_contract import QA_RESULT_CONTRACT_VERSION
from selfplay_graph_flowsteer.qa_schema_repair import complete_payload

from .test_director_relation_audit_context import TokenBackend

PAIRS = json.loads((Path(__file__).parent / "fixtures/qa_remaining_failures/historical_schema_repairs.json").read_text())["pairs"]
QUESTION = "[Public source]\nAn artist lived in Example County.\n\nQuestion: Where did the artist live?"


def artifact(answer="Example County", **fields):
    return json.dumps(dict(answer=answer, summary="The evidence resolves the requested location.",
                          confidence=0.8, evidence=["A supplied quotation"], unresolved_issues=[],
                          tool_summary=[], **fields))


@pytest.fixture(params=["selfplay_graph_flowsteer"])
def variant(request):
    package = request.param
    return lambda suffix: importlib.import_module(package + "." + suffix)


def dataset(variant):
    return "musique" if variant("runtime").__package__ == "musique_ood" else "hotpotqa"


def execute(variant, responses, *, output=True):
    backend = variant("llm").MockBackend(responses)
    node = variant("contracts").AgentNode(
        agent_id="reader", prompt="Identify the person mentioned in the passages.",
        operation_policy_configured=True,
        metadata={"system_managed_contract": {"dataset": dataset(variant)},
                  "submission_protocol": "unified_task_result_v1",
                  "result_scope": "task_result" if output else "subtask"},
    )
    result = variant("runtime").ModelAgentExecutor(backend).execute(
        task=QUESTION, node=node, upstream=[], peers=[], revision=False, seed=0,
    )
    return result, backend


@pytest.mark.parametrize("pair", PAIRS, ids=[p["task_id"].split("/")[-1] for p in PAIRS])
def test_archived_schema_repair_preserves_first_answer_and_valid_fields(variant, pair):
    result, backend = execute(variant, [pair["original"], pair["repair"]])
    frozen = variant("qa_schema_repair").preserved_fields(pair["original"])
    assert result.answer == frozen["answer"].strip()
    assert len(backend.calls) == 2
    final = complete_payload(result.raw_response)
    assert all(final[key] == value and type(final[key]) is type(value) for key, value in frozen.items())
    assert json.loads(backend.calls[-1]["messages"][-1]["content"])["preserved_artifact_fields"] == frozen


def test_consecutive_repairs_keep_first_fields_and_actual_provider_usage(variant):
    valid = json.loads(artifact("Bill Pullman"))
    first = {**valid, "confidence": "high", "tool_summary": "No tools used"}
    second = {**valid, "answer": "unknown", "confidence": True, "summary": "Reconsidered"}
    third = {**valid, "answer": "A different actor", "evidence": ["Invented"]}
    result, backend = execute(variant, list(map(json.dumps, [first, second, third])))
    assert len(backend.calls) == 3
    assert result.answer == "Bill Pullman" and result.summary == valid["summary"]
    assert result.evidence == valid["evidence"]
    assert sum(event.get("stage", "").endswith("schema_preservation") for event in result.protocol_diagnostics) == 2


@pytest.mark.parametrize("bad", [
    "", "{broken JSON", '{"answer":"Par', '{"answer":null}',
    '{"answer":"Paris","answer":"London"}', '{"answer":"Paris"} {"answer":"London"}',
    '<think>{"answer":"Paris"}</think>', '{"answer":"Paris","confidence":NaN}',
    '{"answer":true}', '{"answer":["Paris","London"]}', '{"answer":"Paris","confidence":1e400}',
])
def test_unrecoverable_answer_never_triggers_hidden_resolving_or_guessing(variant, bad):
    result, backend = execute(variant, [bad, artifact("A guessed replacement")])
    assert len(backend.calls) == 1
    assert result.answer == "WORKER_PROTOCOL_FAILURE"
    assert any(d.get("stage") == "qa_schema_repair_unavailable" and d["no_request_dispatched"]
               for d in result.protocol_diagnostics)


@pytest.mark.parametrize("answer", ["48.8 percent", "two", "2", "Trinidad and Tobago", "unknown", "Paris or London"])
@pytest.mark.parametrize("output", [True, False])
def test_valid_values_are_untouched_and_original_target_survives_local_assignment(variant, answer, output):
    result, backend = execute(variant, [artifact(answer)], output=output)
    assert result.answer == answer and len(backend.calls) == 1
    context = json.loads(backend.calls[0]["messages"][-1]["content"])
    assert context["original_question"] == "Where did the artist live?"
    assert context["public_task_context"] == QUESTION
    assert context["result_scope"] == ("task_result" if output else "subtask")
    assert "person" in context["assigned_task"]
    assert context["qa_result_contract_version"] == QA_RESULT_CONTRACT_VERSION


@pytest.mark.parametrize("other_dataset", ["nq_open", "aime", "healthbench_professional"])
def test_other_datasets_keep_their_existing_context_and_recovery(variant, other_dataset):
    backend = variant("llm").MockBackend(["{broken JSON", artifact()])
    node = variant("contracts").AgentNode(
        agent_id="reader", prompt="Resolve the public task", operation_policy_configured=True,
        metadata={"system_managed_contract": {"dataset": other_dataset},
                  "submission_protocol": "unified_task_result_v1", "result_scope": "task_result"},
    )
    result = variant("runtime").ModelAgentExecutor(backend).execute(
        task=QUESTION, node=node, upstream=[], peers=[], revision=False, seed=0,
    )
    assert result.answer == "Example County" and len(backend.calls) == 2
    for call in backend.calls:
        context = json.loads(call["messages"][-1]["content"])
        assert "qa_result_contract_version" not in context
        assert "For QA result_scope=" not in call["messages"][0]["content"]
        assert "preserved_artifact_fields" not in context


def make_canvas(variant, tmp_path, responses):
    backend = variant("llm").MockBackend(responses)
    canvas = variant("canvas").GraphCanvas(
        task=QUESTION, dataset=dataset(variant), binary_relation_policy=True,
        runtime=variant("runtime").MultiAgentRuntime(variant("runtime").ModelAgentExecutor(backend)),
        config=variant("config").CanvasConfig(
            submission_protocol="unified_task_result_v1", submission_journal_dir=str(tmp_path),
            max_rounds=50,
        ),
    )
    canvas.run_id = "qa-regression"
    return canvas, backend


def prompt(target, scope="task_result"):
    return dict(action="set_prompt", target=target, role="Analyst",
                objective="Resolve the requested responsibility using public evidence",
                scope="The supplied passages", expected_output="Supported findings", result_scope=scope)


def add(canvas, target, scope="task_result"):
    for action in [dict(action="add_agent", agent_id=target), prompt(target, scope)]:
        result = canvas.step(json.dumps(action))
        assert result.accepted, result.feedback


def finish(variant, canvas, target):
    call_id = "finish-" + str(len(canvas.history))
    canvas.observe_submission_candidates(call_id)
    return canvas.step(json.dumps(dict(action="finish", target=target)), director_context=
                       variant("submission_contract")._director_call_context(canvas.run_id, call_id))


def test_missing_answer_has_explicit_director_recovery_path(variant, tmp_path):
    canvas, worker = make_canvas(variant, tmp_path, ["{broken JSON", artifact()])
    add(canvas, "a")
    assert len(worker.calls) == 1
    state = canvas.control_snapshot()
    assert not state["result_assessments"]["a"]["submit_ready"]
    assert "a" in state["legal_action_parameters"]["run_agent"]["targets"]
    result = canvas.step(json.dumps(dict(action="run_agent", target="a")))
    assert result.accepted, result.feedback
    assert len(worker.calls) == 2
    assert finish(variant, canvas, "a").accepted
    assert len(worker.calls) == 2


def test_scope_promotion_reexecutes_and_off_remains_a_valid_director_choice(variant, tmp_path):
    canvas, worker = make_canvas(variant, tmp_path, [artifact("An intermediate person"), artifact("A second finding"), artifact()])
    add(canvas, "a", "subtask")
    add(canvas, "b", "subtask")
    assert canvas.step(json.dumps(dict(action="consider_relation", source="a", target="b"))).accepted
    result = canvas.resolve_relation_choice("off")
    assert result.accepted, result.feedback
    state = canvas.control_snapshot()
    assert not canvas.graph.directed_edges and not canvas.graph.bidirectional_edges
    assert "consider_relation" not in state["allowed_actions"]
    assert "finish" not in state["allowed_actions"]
    assert not state["legal_action_parameters"]["set_prompt"]["revision_requires_evidence"]
    assert "revision_evidence_by_target" not in state["legal_action_parameters"]["set_prompt"]
    # The Director chooses a scope edit and cleanup. Canvas neither forces on nor
    # relabels the old local result as a complete answer.
    old_id = canvas.runtime.artifacts["a"].artifact_id
    promoted = canvas.step(json.dumps(prompt("a")))
    assert promoted.accepted, promoted.feedback
    assert len(worker.calls) == 3 and canvas.runtime.artifacts["a"].artifact_id != old_id
    assert json.loads(worker.calls[-1]["messages"][-1]["content"])["result_scope"] == "task_result"
    assert not canvas.graph.bidirectional_edges
    assert canvas.step(json.dumps(dict(action="delete_agent", target="b"))).accepted
    assert canvas.control_snapshot()["progress"]["no_progress_count"] == 0
    assert finish(variant, canvas, "a").accepted
    assert len(worker.calls) == 3


def test_critical_issue_visible_in_compact_but_does_not_force_extra_work(variant, tmp_path, monkeypatch):
    monkeypatch.setenv("SPGFS_DIRECTOR_CONTEXT_MODE", "append_only")
    draft = json.loads(artifact("A candidate"))
    draft["unresolved_issues"] = ["The identity connecting the two passages is not established."]
    canvas, worker = make_canvas(variant, tmp_path, [json.dumps(draft)])
    add(canvas, "a")
    director = TokenBackend([json.dumps(dict(action="finish", target="a"))])
    run = variant("director").GraphDirector(
        backend=director, canvas=canvas, prompt_variant="v3",
        observation_schema="compact_factual_v1", tokenizer=director.tokenizer,
    ).run()
    assert run.finished and len(worker.calls) == 1
    sent = json.dumps(director.calls[0]["messages"])
    assert draft["answer"] in sent and draft["unresolved_issues"][0] in sent
    assert "not_assessed_by_runtime" in sent
    assert "submit_ready certifies only submission eligibility" in sent


def test_unchanged_reports_reference_exact_prior_content_without_editing_history(variant, tmp_path):
    canvas, _ = make_canvas(variant, tmp_path, [artifact("Answer with a necessary qualifier")])
    add(canvas, "a")
    builder = variant("director_observation").DirectorObservationBuilder(dataset=dataset(variant))
    snapshot = canvas.control_snapshot()
    first = builder.render(snapshot, "first", prior_messages=[])
    history = [{"role": "user", "content": first}]
    untouched = copy.deepcopy(history)
    second = builder.render(snapshot, "second", prior_messages=history)
    payload = json.loads(second.split("\n", 1)[1])
    ref = payload["worker_results"]["a"]["worker_report_ref"]
    assert ref == {"observation_index": 0, "agent_id": "a"}
    assert history == untouched
    prior = json.loads(first.split("\n", 1)[1])["worker_results"]["a"]
    assert prior["answer"] == canvas.runtime.artifacts["a"].answer
    history.append({"role": "user", "content": second})
    # Refs always point to the original full report, never chains of refs.
    third = builder.render(snapshot, "third", prior_messages=history)
    assert json.loads(third.split("\n", 1)[1])["worker_results"]["a"]["worker_report_ref"] == ref
    # The same artifact ID does not hide a changed answer, issue, or freshness bit.
    for key, value in [("answer", "Changed"), ("pending_reexecution", True), ("unresolved_count", 99)]:
        changed = copy.deepcopy(snapshot)
        changed["worker_results"]["a"][key] = value
        text = builder.render(changed, "changed", prior_messages=history)
        report = json.loads(text.split("\n", 1)[1])["worker_results"]["a"]
        assert "worker_report_ref" not in report and report[key] == value


def test_report_reference_online_history_keeps_exact_sampled_prefix(variant, tmp_path, monkeypatch):
    monkeypatch.setenv("SPGFS_DIRECTOR_CONTEXT_MODE", "append_only")
    canvas, worker = make_canvas(variant, tmp_path, [artifact()])
    add(canvas, "a")
    # An illegal action exposes the same report again, then the Director submits.
    backend = TokenBackend([json.dumps(dict(action="run_agent", target="missing")),
                            json.dumps(dict(action="finish", target="a"))])
    run = variant("director").GraphDirector(
        backend=backend, canvas=canvas, prompt_variant="v3", tokenizer=backend.tokenizer,
        observation_schema="compact_factual_v1",
    ).run()
    assert run.finished and len(worker.calls) == 1
    first, second = run.turns
    prefix = first.prompt_token_ids + first.completion_token_ids
    assert second.prompt_token_ids[:len(prefix)] == prefix
    assert "worker_report_ref" in second.prompt_messages[-1]["content"]
    for turn in run.turns:
        audit = run.observation_audits[turn.action_diagnostics["observation_audit_index"]]
        replay = variant("director_observation").DirectorObservationBuilder(dataset=dataset(variant)).render(
            audit["control_snapshot"], audit["raw_feedback"], prior_messages=turn.prompt_messages[:-1])
        assert replay == audit["observation"]


@pytest.mark.parametrize("contract_dataset", ["hotpotqa", "musique", "nq_open"])
@pytest.mark.parametrize("truncated_text", ['{"answer":"Par', artifact("Original answer")])
def test_physical_gateway_cannot_hide_a_qa_length_retry(variant, monkeypatch, contract_dataset, truncated_text):
    """Exercise the actual gateway + executor, not MockBackend.generate."""
    from openai.types.chat import ChatCompletion

    boundary = importlib.import_module(
        "tests.test_musique_qwen_boundary" if dataset(variant) == "musique"
        else "tests.test_qwen_director_boundary"
    )

    def completion(text, reason, index):
        return ChatCompletion.model_validate({
            "id": f"physical-{index}", "created": 1, "object": "chat.completion",
            "model": "Qwen3.5-9B", "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            "choices": [{"index": 0, "finish_reason": reason,
                         "message": {"role": "assistant", "content": text}}],
        })

    backend, requests = boundary.gateway(monkeypatch, [
        completion(truncated_text, "length", 1),
        completion(artifact("Unauthorized replacement"), "stop", 2),
    ], thinking=False)
    backend.config.roles["worker"] = variant("config").ModelRoleConfig(enable_thinking=False)
    node = variant("contracts").AgentNode(
        agent_id="reader", prompt="Resolve the public question", operation_policy_configured=True,
        metadata={"system_managed_contract": {"dataset": contract_dataset},
                  "submission_protocol": "unified_task_result_v1", "result_scope": "task_result"},
    )
    result = variant("runtime").ModelAgentExecutor(backend).execute(
        task=QUESTION, node=node, upstream=[], peers=[], revision=False, seed=0,
    )
    if contract_dataset == "nq_open":
        assert len(requests) == 2
        assert result.answer == "Unauthorized replacement"
    else:
        if complete_payload(truncated_text):
            assert result.answer == "Original answer"
            # A complete answer may pass directly or receive a runtime schema
            # repair due to finish_reason=length; that repair freezes its value.
            assert len(requests) in {1, 2}
            if len(requests) == 2:
                assert json.loads(requests[1]["messages"][-1]["content"])["preserved_artifact_fields"]["answer"] == "Original answer"
        else:
            assert len(requests) == 1
            assert result.answer == "WORKER_PROTOCOL_FAILURE"
        assert not variant("llm")._FINALIZATION_REQUEST.get(), "scope must reset after execution"
