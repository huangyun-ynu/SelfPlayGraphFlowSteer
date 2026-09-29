from __future__ import annotations

import json
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor

import pytest

from selfplay_graph_flowsteer.agent_tools import SearchServiceTool
from selfplay_graph_flowsteer.contracts import AgentNode
from selfplay_graph_flowsteer.graph import MultiAgentGraph
from selfplay_graph_flowsteer.llm import MockBackend
from selfplay_graph_flowsteer.nq_evidence import NQEvidenceContext
from selfplay_graph_flowsteer.public_evidence import public_document
from selfplay_graph_flowsteer.runtime import ModelAgentExecutor, MultiAgentRuntime, RoutedModelAgentExecutor


def context(**overrides):
    return NQEvidenceContext(task_id="same-task", policy={
        "max_search_calls_per_task": 4, "max_submission_repairs": 1, **overrides,
    })


def search_result(text="Marie Curie named polonium after Poland."):
    return {"queries": ["Curie"], "result": [[{
        "document": {"id": "p-7", "title": "Marie Curie", "text": text,
                     "corpus_id": "wiki18", "faiss_row": 7, "has_answer": True},
        "score": 0.9,
    }]]}


def supported(evidence_id="ev_0001", quote="polonium after Poland"):
    return json.dumps({"answer": "Polonium", "answerability": "supported",
                       "evidence_refs": [{"evidence_id": evidence_id, "quote": quote}]})


def insufficient():
    return json.dumps({"answer": "insufficient_evidence", "answerability": "insufficient_evidence",
                       "evidence_refs": []})


def add_evidence(ledger, agent="a", text="Marie Curie named polonium after Poland."):
    ledger.begin_agent(agent, [])
    call_id = ledger.reserve_search(agent, "Curie")
    return ledger.record_search(call_id, search_result(text))


def test_atomic_search_budget_shared_across_workers_and_revisions():
    ledger = context()

    def attempt(number):
        try:
            return ledger.reserve_search(f"worker-{number}", "repeat query")
        except RuntimeError:
            return None

    with ThreadPoolExecutor(max_workers=12) as pool:
        admitted = [value for value in pool.map(attempt, range(50)) if value]
    assert len(admitted) == len(set(admitted)) == 4
    assert ledger.audit()["search_calls_used"] == 4
    ledger.begin_agent("worker-0", [])
    with pytest.raises(RuntimeError, match="budget_exhausted"):
        ledger.reserve_search("worker-0", "revision")


def test_disconnected_agent_can_repeat_query_to_obtain_own_evidence():
    ledger = context()
    add_evidence(ledger, agent="a")
    ledger.begin_agent("b", [])
    call = ledger.reserve_search("b", "  ＣＵＲＩＥ  ")
    assert ledger.skip_duplicate_search(call) is None
    assert not ledger.validate("b", supported())["valid"]
    ledger.record_search(call, search_result())
    assert ledger.validate("b", supported())["valid"]
    assert ledger.audit()["search_calls_used"] == 2
    assert [item["status"] for item in ledger.audit()["calls"]] == ["ok", "ok"]


def test_duplicate_query_spends_budget_when_prior_evidence_is_visible():
    ledger = context()
    add_evidence(ledger, agent="a")
    validated = ledger.validate("a", supported(), require_submission=False)
    ledger.bind_artifact("from-a", validated)
    assert [item["evidence_id"] for item in ledger.begin_agent("b", ["from-a"])["documents"]] == ["ev_0001"]
    call = ledger.reserve_search("b", "  ＣＵＲＩＥ  ")
    feedback = ledger.skip_duplicate_search(call)
    assert feedback is not None
    assert feedback["status"] == "duplicate_query"
    assert feedback["duplicate_of_call_id"] == "nq_search_0001"
    assert feedback["remaining_search_calls"] == 2
    assert "reformulate" in feedback["guidance"]
    assert ledger.audit()["search_calls_used"] == 2
    assert ledger.audit()["calls"][1]["status"] == "duplicate_query"
    assert ledger.validate("b", supported())["valid"]


def test_partial_upstream_visibility_does_not_suppress_repeat_search():
    ledger = context()
    ledger.begin_agent("a", [])
    first = ledger.reserve_search("a", "Curie")
    result = search_result()
    result["result"][0].append({
        "document": {"id": "p-8", "title": "Marie Curie", "text": "She also discovered radium.",
                     "corpus_id": "wiki18"},
        "score": 0.8,
    })
    ledger.record_search(first, result)
    assert ledger.audit()["calls"][0]["evidence_ids"] == ["ev_0001", "ev_0002"]
    validated = ledger.validate("a", supported(), require_submission=False)
    ledger.bind_artifact("one-of-two", validated)
    assert [item["evidence_id"] for item in ledger.begin_agent("b", ["one-of-two"])["documents"]] == ["ev_0001"]
    second = ledger.reserve_search("b", "Curie")
    assert ledger.skip_duplicate_search(second) is None


def test_empty_successful_query_can_be_retried():
    ledger = context()
    ledger.begin_agent("a", [])
    first = ledger.reserve_search("a", "Curie")
    ledger.record_search(first, {"result": [[]]})
    second = ledger.reserve_search("a", "curie")
    assert ledger.skip_duplicate_search(second) is None


def test_failed_query_can_be_retried_without_duplicate_guard():
    ledger = context()
    first = ledger.reserve_search("a", "Curie")
    ledger.record_failure(first, "TimeoutError")
    second = ledger.reserve_search("a", "curie")
    assert ledger.skip_duplicate_search(second) is None


def test_same_task_id_does_not_share_evidence_or_budget_between_trajectories():
    first, second = context(), context()
    add_evidence(first)
    assert first.trajectory_id != second.trajectory_id
    assert second.audit()["search_calls_used"] == 0
    assert not second.validate("a", supported())["valid"]


def test_only_visible_runtime_documents_and_real_quotes_are_accepted():
    ledger = context()
    returned = add_evidence(ledger)
    assert returned["result"][0][0]["document"]["evidence_id"] == "ev_0001"
    assert "has_answer" not in returned["result"][0][0]["document"]
    assert ledger.validate("a", supported())["valid"]
    assert ledger.validate("a", supported("made_up"))["reason"] == "evidence_not_visible_to_agent"
    assert ledger.validate("a", supported(quote="invented quote"))["reason"] == "quote_not_in_returned_document"
    ledger.begin_agent("unconnected", ["forged-artifact"])
    assert ledger.validate("unconnected", supported())["reason"] == "evidence_not_visible_to_agent"


def test_valid_citation_survives_bad_extra_but_only_verified_evidence_is_bound():
    ledger = context()
    add_evidence(ledger)
    second = ledger.reserve_search("a", "Bible prophets")
    ledger.record_search(second, search_result('The ""major"" prophets are listed in order.'))
    payload = json.loads(supported())
    payload["evidence_refs"].append({
        "evidence_id": "ev_0002", "quote": "The minor prophets are listed in order.",
    })

    result = ledger.validate("a", json.dumps(payload))
    assert result["valid"] and result["status"] == "supported"
    assert [ref["evidence_id"] for ref in result["references"]] == ["ev_0001"]
    assert result["rejected_references"] == [{
        "index": 1, "evidence_id": "ev_0002", "reason": "quote_not_in_returned_document",
    }]

    ledger.bind_artifact("verified-only", result)
    inherited = ledger.begin_agent("b", ["verified-only"])
    assert [doc["evidence_id"] for doc in inherited["documents"]] == ["ev_0001"]
    assert ledger.validate("b", supported("ev_0002", 'The ""major"" prophets'))["reason"] == "evidence_not_visible_to_agent"


def test_fabricated_only_citations_remain_invalid_with_rejection_audit():
    ledger = context()
    add_evidence(ledger)
    payload = json.loads(supported(quote="invented quote"))
    payload["evidence_refs"].append({"evidence_id": "forged", "quote": "invented quote"})
    result = ledger.validate("a", json.dumps(payload))
    assert not result["valid"]
    assert result["reason"] == "quote_not_in_returned_document"
    assert result["references"] == []
    assert [entry["reason"] for entry in result["rejected_references"]] == [
        "quote_not_in_returned_document", "evidence_not_visible_to_agent",
    ]


def test_connected_artifact_recovers_verbatim_evidence_without_relay_prose():
    ledger = context()
    add_evidence(ledger)
    validated = ledger.validate("a", supported(), require_submission=False)
    ledger.bind_artifact("artifact-1", validated)
    visible = ledger.begin_agent("b", ["artifact-1"])
    assert visible["documents"][0]["text"] == "Marie Curie named polonium after Poland."
    assert ledger.validate("b", supported())["valid"]
    # Rewiring removes upstream permission; previously inherited evidence alone
    # is not the same as the Worker's own tool observations.
    ledger.begin_agent("b", [])
    assert not ledger.validate("b", supported())["valid"]


def test_quote_whitespace_offsets_are_original_and_case_is_not_fuzzed():
    ledger = context()
    text = "Marie Curie named polonium\n  after Poland."
    add_evidence(ledger, text=text)
    result = ledger.validate("a", supported())
    ref = result["references"][0]
    assert text[ref["quote_start"]:ref["quote_end"]] == "polonium\n  after Poland"
    assert not ledger.validate("a", supported(quote="Polonium after Poland"))["valid"]


def test_mediawiki_doubled_quotes_match_with_original_span_and_hash():
    import hashlib

    ledger = context()
    text = 'The Latter Prophets include the ""major"" prophets, Isaiah, Jeremiah, Ezekiel, Daniel.'
    add_evidence(ledger, text=text)
    result = ledger.validate("a", supported(quote='the "major" prophets, Isaiah, Jeremiah, Ezekiel, Daniel'))
    assert result["valid"]
    ref = result["references"][0]
    assert ref["quote_match_mode"] == "mediawiki_doubled_quotes"
    assert text[ref["quote_start"]:ref["quote_end"]] == (
        'the ""major"" prophets, Isaiah, Jeremiah, Ezekiel, Daniel'
    )
    assert ref["content_sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_failed_response_does_not_grant_partial_evidence_and_failure_is_not_abstention():
    ledger = context()
    ledger.begin_agent("a", [])
    call = ledger.reserve_search("a", "Curie")
    result = search_result()
    result["result"][0].append({"document": {"text": "missing ID"}})
    with pytest.raises(RuntimeError, match="requires_id"):
        ledger.record_search(call, result)
    ledger.record_failure(call, "RuntimeError")
    assert ledger.audit()["documents"] == {}
    assert ledger.validate("a", insufficient())["reason"] == "retrieval_service_failure"


def test_insufficient_evidence_is_explicit_and_does_not_license_substantive_answer():
    ledger = context()
    assert ledger.validate("a", insufficient())["reason"] == "corpus_search_required_before_abstention"
    ledger.begin_agent("a", [])
    call = ledger.reserve_search("a", "Curie")
    ledger.record_search(call, {"result": [[]]})
    assert ledger.validate("a", insufficient())["valid"]
    assert ledger.validate("a", insufficient())["status"] == "insufficient_evidence"
    payload = json.loads(insufficient())
    payload["answer"] = "Polonium"
    assert not ledger.validate("a", json.dumps(payload))["valid"]
    assert not ledger.validate("a", supported())["valid"]


class SearchTool:
    name = "search"
    description = "Search the test corpus"
    parameters = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}

    def __init__(self):
        self.calls = []

    def execute(self, arguments):
        self.calls.append(arguments)
        return json.dumps(search_result())


def node(agent_id="a", selected=True):
    return AgentNode(agent_id, "Answer the public question", allowed_tools=("search",),
                     metadata={"_runtime_is_output_agent": selected,
                               "system_managed_contract": {"dataset": "nq_open"}})


def run(executor, worker=None, upstream=None):
    return executor.execute(task="What element did Curie name after Poland?", node=worker or node(),
                            upstream=upstream or [], peers=[], revision=False, seed=0)


SEARCH = json.dumps({"action_call": {"name": "search", "arguments": {"query": "Curie"}}})


def test_executor_attaches_evidence_ids_and_accepts_supported_answer():
    backend, tool, ledger = MockBackend([SEARCH, supported()]), SearchTool(), context()
    artifact = run(ModelAgentExecutor(backend, tools={"search": tool}, nq_evidence_context=ledger))
    assert artifact.runtime_tool_evidence["nq_corpus"]["status"] == "supported"
    assert artifact.runtime_tool_evidence["nq_corpus"]["valid"]
    assert ledger.audit()["search_calls_used"] == 1
    assert "ev_0001" in backend.calls[1]["messages"][-1]["content"]


def test_executor_returns_reformulation_guidance_without_second_search_call():
    repeat = json.dumps({"action_call": {"name": "search", "arguments": {"query": " CURIE "}}})
    backend, tool, ledger = MockBackend([SEARCH, repeat, supported()]), SearchTool(), context()
    artifact = run(ModelAgentExecutor(backend, tools={"search": tool}, nq_evidence_context=ledger))
    assert artifact.runtime_tool_evidence["nq_corpus"]["valid"]
    assert len(tool.calls) == 1
    assert ledger.audit()["search_calls_used"] == 2
    observation = backend.calls[2]["messages"][-1]["content"]
    assert "duplicate_query" in observation
    assert "reformulate" in observation


def test_executor_repeats_identical_search_for_disconnected_agent():
    backend = MockBackend([SEARCH, supported(), SEARCH, supported()])
    tool, ledger = SearchTool(), context()
    executor = ModelAgentExecutor(backend, tools={"search": tool}, nq_evidence_context=ledger)
    assert run(executor, worker=node("a")).runtime_tool_evidence["nq_corpus"]["valid"]
    assert run(executor, worker=node("b")).runtime_tool_evidence["nq_corpus"]["valid"]
    assert len(tool.calls) == 2
    assert ledger.audit()["search_calls_used"] == 2
    assert [item["status"] for item in ledger.audit()["calls"]] == ["ok", "ok"]


def test_executor_repairs_forged_citation_once_with_real_context():
    ledger = context()
    backend = MockBackend([SEARCH, supported("forged"), supported()])
    artifact = run(ModelAgentExecutor(backend, tools={"search": SearchTool()}, nq_evidence_context=ledger))
    assert artifact.runtime_tool_evidence["nq_corpus"]["valid"]
    assert ledger.audit()["submission_repairs_used"] == 1
    assert len(backend.calls) == 3
    assert "Marie Curie named polonium after Poland." in backend.calls[2]["messages"][-1]["content"]
    assert any(item["stage"] == "nq_evidence_repair" for item in artifact.protocol_diagnostics)


def test_executor_keeps_valid_reference_when_extra_quote_is_bad():
    ledger = context()
    payload = json.loads(supported())
    payload["evidence_refs"].append({"evidence_id": "ev_0001", "quote": "invented quote"})
    backend = MockBackend([SEARCH, json.dumps(payload)])
    artifact = run(ModelAgentExecutor(backend, tools={"search": SearchTool()}, nq_evidence_context=ledger))
    validation = artifact.runtime_tool_evidence["nq_corpus"]
    assert validation["valid"]
    assert len(validation["references"]) == 1
    assert validation["rejected_references"][0]["reason"] == "quote_not_in_returned_document"
    assert ledger.audit()["submission_repairs_used"] == 0


def test_premature_abstention_gets_one_search_protocol_repair():
    ledger = context()
    backend = MockBackend([insufficient(), SEARCH, supported()])
    artifact = run(ModelAgentExecutor(backend, tools={"search": SearchTool()}, nq_evidence_context=ledger))
    assert artifact.runtime_tool_evidence["nq_corpus"]["valid"]
    assert ledger.audit()["submission_repairs_used"] == 1
    assert ledger.audit()["search_calls_used"] == 1
    assert len(backend.calls) == 3


def test_routed_executor_keeps_one_context_across_fresh_node_executors():
    ledger = context(max_search_calls_per_task=1, max_submission_repairs=0)
    tool = SearchTool()
    backend = MockBackend([SEARCH, supported(), SEARCH, insufficient()])
    executor = RoutedModelAgentExecutor({"deepseek": backend}, routes=("deepseek",), tools={"search": tool})
    executor.nq_evidence_context = ledger
    first = run(executor)
    second = run(executor, node("b"))
    assert first.runtime_tool_evidence["nq_corpus"]["valid"]
    assert second.runtime_tool_evidence["nq_corpus"]["status"] == "insufficient_evidence"
    assert len(tool.calls) == 1
    assert ledger.audit()["search_calls_used"] == 1


def test_runtime_reset_drops_context_and_cannot_reconfigure_mid_trajectory():
    executor = ModelAgentExecutor(MockBackend([]))
    runtime = MultiAgentRuntime(executor)
    runtime.configure_nq_corpus(task_id="q", policy={})
    original = runtime.nq_evidence_context
    with pytest.raises(RuntimeError, match="already configured"):
        runtime.configure_nq_corpus(task_id="q", policy={})
    runtime.reset()
    assert runtime.nq_evidence_context is None and executor.nq_evidence_context is None
    runtime.configure_nq_corpus(task_id="q", policy={})
    assert runtime.nq_evidence_context is not original


def test_runtime_graph_transfers_validated_refs_even_when_relay_text_is_truncated():
    backend = MockBackend([SEARCH, supported(), supported()])
    runtime = MultiAgentRuntime(ModelAgentExecutor(backend, tools={"search": SearchTool()}), relay_max_chars=1)
    runtime.configure_nq_corpus(task_id="q", policy={})
    graph = MultiAgentGraph()
    for name in ("researcher", "answerer"):
        graph.add_agent(name)
        graph.set_prompt(name, "Resolve the question")
        graph.nodes[name].allowed_tools = ("search",)
        graph.nodes[name].metadata["system_managed_contract"] = {"dataset": "nq_open"}
    graph.set_layer("answerer", 1)
    graph.set_relation("researcher", "answerer", "directed")
    graph.set_output("answerer")
    runtime.execute(task="What did Curie name after Poland?", graph=graph)
    assert runtime.validate_nq_submission("answerer")["valid"]
    assert runtime.nq_evidence_audit()["search_calls_used"] == 1
    assert runtime.nq_evidence_status("researcher")["status"] == "intermediate"
    prompt = json.loads(backend.calls[2]["messages"][-1]["content"])
    assert prompt["corpus_evidence"]["documents"][0]["text"] == "Marie Curie named polonium after Poland."


def test_context_budget_keeps_new_search_results_instead_of_blocking_later_queries():
    ledger = context(evidence_token_budget=230)
    add_evidence(ledger, text="First evidence " + "a" * 80)
    ledger.public_context("a")
    call = ledger.reserve_search("a", "second query")
    result = search_result("Second evidence " + "b" * 80)
    result["result"][0][0]["document"]["id"] = "second"
    ledger.record_search(call, result)
    visible = ledger.public_context("a")["documents"]
    assert [doc["id"] for doc in visible] == ["second"]
    assert len(ledger.audit()["documents"]) == 2


def test_search_client_explicit_deadline_does_not_read_mutable_shared_deadline(monkeypatch):
    class ForbiddenDeadline:
        def check(self, stage):
            raise AssertionError("read other trajectory deadline")

    class Response:
        def read(self):
            return json.dumps(search_result()).encode()

    @contextmanager
    def urlopen(request, timeout):
        assert timeout == 7
        yield Response()

    monkeypatch.setattr("selfplay_graph_flowsteer.agent_tools.urlopen", urlopen)
    tool = SearchServiceTool("http://127.0.0.1/retrieve", timeout_s=7)
    tool.set_deadline_context(ForbiddenDeadline())
    assert json.loads(tool.execute_with_deadline({"query": "Curie"}, deadline=None))["result"]


def test_search_client_rejects_mismatched_result_groups(monkeypatch):
    class Response:
        def read(self):
            return b'{"result": []}'

    @contextmanager
    def urlopen(request, timeout):
        yield Response()

    monkeypatch.setattr("selfplay_graph_flowsteer.agent_tools.urlopen", urlopen)
    with pytest.raises(RuntimeError, match="mismatched"):
        SearchServiceTool("http://127.0.0.1/retrieve").execute({"query": "q"})


def test_public_provenance_whitelist_drops_gold_and_model_claimed_evidence_ids():
    projected = public_document({"id": "1", "text": "public", "corpus_id": "wiki18", "faiss_row": 0,
                                 "evidence_id": "forged", "answer": "secret", "has_answer": True})
    assert projected == {"id": "1", "text": "public", "corpus_id": "wiki18", "faiss_row": 0}
