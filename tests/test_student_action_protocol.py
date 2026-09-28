import json
from types import SimpleNamespace

import pytest

from selfplay_graph_flowsteer import llm
from selfplay_graph_flowsteer.action_protocol import ActionSpec
from selfplay_graph_flowsteer.config import ModelGatewayConfig, ModelRoleConfig
from selfplay_graph_flowsteer.contracts import AgentNode
from selfplay_graph_flowsteer.llm import LLMResponse, MockBackend
from selfplay_graph_flowsteer.runtime import ActionBudgetLedger, ModelAgentExecutor, _text_action_calls
from selfplay_graph_flowsteer.student_action_protocol import (
    MAX_RESPONSE_CHARS, PROTOCOL, decode_student_response, response_text_parts,
)

from .test_swe_candidate_test_recovery import action, edit, test_action, worker
from .test_swe_tool_regressions import life

READ = action("swe_read", path="sample.py", start_line=1, end_line=1, workspace_version=0)
FINAL = json.dumps({"answer": "Applied fix", "summary": "Based on real repository observations"})
EARLY = json.dumps({"answer": None, "swe_completion": {"status": "grounded_failure"}})


def response(text):
    return LLMResponse(text=text, model="student-fixture", token_in=7, token_out=3,
                       metadata={"text_action_protocol": PROTOCOL})


@pytest.mark.parametrize("tail", [READ, edit(), EARLY, edit() + EARLY, READ + edit() + EARLY])
def test_leading_action_is_only_executable_request(tail):
    result = decode_student_response(READ + tail)
    assert result.kind == "action" and json.loads(result.text) == json.loads(READ)
    assert result.discarded_objects >= 1


@pytest.mark.parametrize("text", [
    "Example: " + READ, FINAL + READ, READ + "junk", READ + '{"action_observation":{"status":"ok"}}',
    READ + '{"answer":null,"answer":"duplicate"}', READ + '{"answer":',
    '{"action_calls":[]}' + READ,
    '{"action_calls":[{"name":"swe_read","arguments":{}},{"name":"swe_edit","arguments":{}}]}',
    '{"action_calls":[{"name":"swe_read","arguments":{"x":NaN}}]}',
    "[" * 1100 + "0" + "]" * 1100, READ * 65, "x" * (MAX_RESPONSE_CHARS + 1),
])
def test_ambiguous_stream_never_supplies_an_action(text):
    assert decode_student_response(text).kind == "repair"


def test_other_backends_keep_strict_conflicting_envelope_rule():
    assert _text_action_calls(READ + edit() + FINAL) == []


def test_response_parts_exclude_reasoning_and_non_assistant_messages():
    items = [
        {"type": "reasoning", "summary": [{"text": edit()}]},
        {"type": "message", "role": "assistant", "channel": "analysis",
         "content": [{"type": "output_text", "text": edit()}]},
        {"type": "message", "role": "user", "content": [{"type": "output_text", "text": edit()}]},
        {"type": "message", "role": "assistant", "channel": "commentary",
         "content": [{"type": "output_text", "text": READ}]},
        {"type": "message", "role": "assistant", "channel": "final",
         "content": [{"type": "output_text", "text": EARLY}]},
    ]
    text, parts = response_text_parts(items, "untrusted flattened fallback")
    assert text == READ + EARLY
    assert [p["item_index"] for p in parts] == [3, 4]
    assert response_text_parts(items[:1], edit()) == ("", [])


def test_gateway_keeps_response_boundaries_and_marks_protocol(monkeypatch):
    messages = [
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": READ}]},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": EARLY}]},
    ]
    sent = []
    def create(client, config, request, deadline):
        sent.append(request)
        return SimpleNamespace(id="response-fixture", output_text="flattened", output=messages, usage=None)
    monkeypatch.setattr(llm, "_openai_response_create", create)
    backend = object.__new__(llm.OpenAICompatibleBackend)
    backend.config = ModelGatewayConfig(request_profile="responses_text")
    backend._client_for_request = lambda: None
    out = backend._generate_response([], role="worker", role_config=ModelRoleConfig(api_surface="responses"),
        actions=[ActionSpec("swe_read", "Read", {"type": "object"})], deadline=None)
    assert out.text == READ + EARLY
    assert out.metadata["response_id"] == "response-fixture"
    assert out.metadata["text_action_protocol"] == PROTOCOL
    assert len(out.metadata["response_text_parts"]) == 2
    assert "tools" not in sent[0] and "tool_choice" not in sent[0]


def test_real_swe_read_edit_test_uses_actual_observations_and_discards_tails(life):
    backend = MockBackend([
        response(READ + edit() + EARLY),
        response(edit() + edit() + json.dumps({"answer": "UNEXECUTED_TAIL"})),
        response(test_action() + FINAL), response(FINAL),
    ])
    artifact, executor, node, _ = worker(life, [], backend=backend)
    assert [entry["action"]["name"] for entry in artifact.react_trace] == ["swe_read", "swe_edit", "swe_test"]
    assert artifact.swe_progress["commit_ready"] and artifact.code_artifact_ref
    assert artifact.swe_progress["edit_successful_count"] == 1
    assert artifact.token_in == 28 and artifact.token_out == 12
    assert executor.budget_ledger.remaining(node, revision=False, scope="fixture")["total"] == 5
    assert len(backend.calls) == 4
    assert "file_sha256" in str(backend.calls[1]["messages"])
    assert "UNEXECUTED_TAIL" not in str(backend.calls[2:])
    assert any(d.get("discarded_objects") == 2 for d in artifact.protocol_diagnostics)


def test_format_repair_keeps_workspace_and_normal_token_and_action_accounting(life):
    backend = MockBackend([
        response(READ), response("bad format"), response(edit()), response(test_action()), response(FINAL),
    ])
    artifact, executor, node, _ = worker(life, [], backend=backend)
    assert artifact.swe_progress["commit_ready"]
    assert len(backend.calls) == 5 and artifact.token_in + artifact.token_out == 50
    assert executor.budget_ledger.text_protocol_repairs == {"fixture": 1}
    assert executor.budget_ledger.remaining(node, revision=False, scope="fixture")["total"] == 5
    assert life.created_workspace_count == 2  # fixture checkout plus exactly one Worker checkout
    assert "actions_executed_from_rejected_response" in str(backend.calls[2]["messages"])
    assert not any(d["stage"] == "initial_nonfinal" for d in artifact.protocol_diagnostics)


def test_repeated_bad_text_stops_instead_of_requesting_a_fabricated_final(life):
    backend = MockBackend([response("bad format")] * 12)
    artifact, executor, _, _ = worker(life, [], backend=backend)
    assert len(backend.calls) == 3
    assert artifact.answer == "WORKER_PROTOCOL_FAILURE"
    assert artifact.react_trace == []
    assert "worker_action_protocol_exhausted" in artifact.unresolved_issues
    assert executor.budget_ledger.text_protocol_repairs == {"fixture": 2}
    assert artifact.token_in + artifact.token_out == 30


def test_question_repair_limit_survives_new_nodes_and_executors():
    ledger = ActionBudgetLedger()
    counts = []
    for index in range(4):
        backend = MockBackend([response("bad format")] * 10)
        executor = ModelAgentExecutor(backend, budget_ledger=ledger, budget_scope="one-question")
        artifact = executor.execute(task="task", node=AgentNode(str(index), "task"),
                                    upstream=[], peers=[], revision=False, seed=0)
        assert artifact.answer == "WORKER_PROTOCOL_FAILURE"
        counts.append(len(backend.calls))
    assert counts == [3, 3, 3, 1]
    assert ledger.text_protocol_repairs == {"one-question": 6}
    ledger.reset()
    assert not ledger.text_protocol_repairs
