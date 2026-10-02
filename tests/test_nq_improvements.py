import json

import pytest

from selfplay_graph_flowsteer.actions import CanvasAction, ActionType
from selfplay_graph_flowsteer.canvas import GraphCanvas
from selfplay_graph_flowsteer.config import CanvasConfig
from selfplay_graph_flowsteer.contracts import AgentArtifact
from selfplay_graph_flowsteer.llm import MockBackend
from selfplay_graph_flowsteer.nq_answer_selection import apply_selection, selection_messages
from selfplay_graph_flowsteer.nq_evidence import NQSearchBudgetExhausted
from selfplay_graph_flowsteer.runtime import ModelAgentExecutor, MultiAgentRuntime, _enforce_artifact_integrity

from .test_nq_corpus_evidence import SEARCH, SearchTool, add_evidence, context, node, run, supported


def choice(answer="Polonium", quote="polonium after Poland", **checks):
    return {"answer": answer, "evidence_refs": [{"evidence_id": "ev_0001", "quote": quote}],
            "checks": {"relation": "supported", "time": "not_applicable", "type": "supported", **checks},
            "reason": "The quote establishes the requested relationship."}


def selection(candidate=None):
    return json.dumps({"candidates": [candidate or choice()], "selected": 0})


def test_cached_query_at_exhausted_budget_is_bounded_and_grants_no_new_permissions():
    ledger = context(max_search_calls_per_task=1)
    add_evidence(ledger)
    for _ in range(2):
        call = ledger.reserve_search("a", "curie")
        assert ledger.skip_duplicate_search(call)["result"][0]
    assert ledger.audit()["search_calls_used"] == 1
    with pytest.raises(NQSearchBudgetExhausted, match="no_progress"):
        ledger.reserve_search("a", "curie")
    ledger.begin_agent("unconnected", [])
    with pytest.raises(NQSearchBudgetExhausted):
        ledger.reserve_search("unconnected", "curie")
    assert not ledger.validate("unconnected", supported())["valid"]


def test_cache_cannot_satisfy_required_independent_nonempty_searches():
    ledger = context(min_nonempty_searches_before_answer=2)
    add_evidence(ledger)
    call = ledger.reserve_search("a", "curie")
    ledger.skip_duplicate_search(call)
    assert ledger.validate("a", supported())["reason"] == "nonempty_corpus_search_required"


def test_executor_finalizes_from_evidence_when_last_search_credit_is_used():
    ledger, tool = context(max_search_calls_per_task=1), SearchTool()
    backend = MockBackend([SEARCH, supported()])
    artifact = run(ModelAgentExecutor(backend, tools={"search": tool}, nq_evidence_context=ledger))
    _enforce_artifact_integrity(artifact)
    assert artifact.answer == "Polonium"
    assert artifact.runtime_tool_evidence["nq_corpus"]["valid"]
    assert not artifact.runtime_tool_evidence["terminal_failure"]
    assert len(tool.calls) == 1 and len(backend.calls) == 2
    assert not backend.calls[-1].get("actions")


@pytest.mark.parametrize("grounded", [True, False])
def test_budget_stop_waiver_requires_runtime_validated_evidence(grounded):
    artifact = AgentArtifact("id", "a", "Polonium", react_trace=[{
        "action": {"name": "search"},
        "observation": {"status": "error", "error": {"code": "nq_task_search_budget_exhausted"}},
    }])
    artifact.runtime_tool_evidence["nq_corpus"] = {"valid": grounded, "status": "supported"}
    _enforce_artifact_integrity(artifact)
    assert artifact.runtime_tool_evidence["terminal_failure"] is not grounded
    assert ("all_tool_actions_failed" in artifact.integrity_risks) is not grounded
    assert artifact.runtime_tool_evidence["failed_count"] == 1


def test_nq_canvas_disallows_new_unconnected_agent_after_search_exhaustion(tmp_path):
    runtime = MultiAgentRuntime(ModelAgentExecutor(MockBackend([])))
    runtime.configure_nq_corpus(task_id="q", policy={"max_search_calls_per_task": 1})
    add_evidence(runtime.nq_evidence_context)
    canvas = GraphCanvas(task="What element?", dataset="nq_open", runtime=runtime,
                         config=CanvasConfig(submission_protocol="unified_task_result_v1",
                                             submission_journal_dir=str(tmp_path)))
    step = canvas.step(CanvasAction(ActionType.ADD_AGENT, agent_id="blind"))
    assert not step.accepted and step.rejection_code == "nq_task_search_budget_exhausted"
    assert not canvas.graph.nodes
    state = canvas.control_snapshot()
    assert "add_agent" not in state["allowed_actions"]
    assert state["nq_search_budget"]["agents_with_evidence"] == ["a"]


def test_candidate_ledger_preserves_valid_answer_only_for_owner_or_connected_inputs():
    ledger = context(answer_selection_enabled=True)
    add_evidence(ledger)
    ledger.bind_artifact("candidate-1", ledger.validate("a", supported()),
                         agent_id="a", raw_response=supported())
    assert ledger.begin_agent("a", [])["answer_candidates"][0]["answer"] == "Polonium"
    assert ledger.begin_agent("b", [])["answer_candidates"] == []
    assert ledger.begin_agent("b", ["candidate-1"])["answer_candidates"][0]["answer"] == "Polonium"


def test_selector_cannot_promote_a_model_supported_fabricated_quote():
    ledger = context(); add_evidence(ledger)
    output, audit = apply_selection("What element?", selection(choice(quote="invented quote")),
                                    supported(), ledger=ledger, agent_id="a")
    assert output == supported() and not audit["accepted"]
    assert audit["candidates"][0]["provenance_issue"] == "quote_not_in_returned_document"


def test_selector_preserves_draft_when_semantic_check_is_uncertain():
    ledger = context(); add_evidence(ledger)
    output, audit = apply_selection("What element?", selection(choice(relation="uncertain")),
                                    supported(), ledger=ledger, agent_id="a")
    assert output == supported() and audit["draft_preserved"]


def test_missing_advisory_selection_does_not_discard_verified_candidates():
    ledger = context(); add_evidence(ledger)
    draft = json.loads(supported()); draft["answer"] = "Radium"
    candidate = choice(); candidate["selected"] = 0
    output, audit = apply_selection("What element?", json.dumps({"candidates": [candidate]}),
                                    json.dumps(draft), ledger=ledger, agent_id="a")
    assert json.loads(output)["answer"] == "Polonium" and audit["accepted"]


def test_missing_selection_cannot_license_invalid_citation():
    ledger = context(); add_evidence(ledger)
    output, audit = apply_selection("What element?", json.dumps({"candidates": [choice(quote="made up")]}),
                                    supported(), ledger=ledger, agent_id="a")
    assert output == supported() and not audit["accepted"]


def test_explicit_year_constraint_cannot_be_skipped_by_time_not_applicable():
    ledger = context(); add_evidence(ledger)
    output, audit = apply_selection("Who won in 2000?", selection(), supported(), ledger=ledger, agent_id="a")
    assert not audit["candidates"][0]["eligible"]
    assert output == supported()


def test_numeric_person_answer_is_rejected_even_when_model_marks_type_supported():
    ledger = context(); add_evidence(ledger)
    output, audit = apply_selection("Who discovered it?", selection(choice(answer="1900")),
                                    supported(), ledger=ledger, agent_id="a")
    assert audit["candidates"][0]["type_issue"] == "numeric_answer_to_who_question"
    assert output == supported()


def test_clear_relation_rejection_can_abstain_but_does_not_invent_a_replacement():
    ledger = context(); add_evidence(ledger)
    output, audit = apply_selection("What element?", selection(choice(relation="unsupported")),
                                    supported(), ledger=ledger, agent_id="a")
    assert json.loads(output)["answer"] == "insufficient_evidence"
    assert ledger.validate("a", output)["valid"]
    assert audit["reason"] == "draft_semantically_rejected"


def test_executor_records_selection_usage_and_preserves_audit_through_integrity():
    ledger = context(answer_selection_enabled=True)
    draft = json.loads(supported()); draft["answer"] = "Radium"
    backend = MockBackend([SEARCH, json.dumps(draft), selection()])
    artifact = run(ModelAgentExecutor(backend, tools={"search": SearchTool()}, nq_evidence_context=ledger))
    _enforce_artifact_integrity(artifact)
    assert artifact.answer == "Polonium" and len(backend.calls) == 3
    assert artifact.runtime_tool_evidence["nq_answer_selection"]["changed_answer"]
    assert artifact.runtime_tool_evidence["nq_corpus"]["valid"]


def test_selection_interface_contains_public_evidence_only():
    payload = json.loads(selection_messages("Question?", {"documents": [{"text": "public"}],
                          "reference": "SECRET", "private": "SECRET"}, "draft")[1]["content"])
    assert set(payload) == {"question", "constraints", "documents", "stored_candidates", "draft"}
    assert "SECRET" not in json.dumps(payload)
