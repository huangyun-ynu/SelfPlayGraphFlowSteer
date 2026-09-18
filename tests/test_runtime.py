from __future__ import annotations

import json

import pytest

from selfplay_graph_flowsteer.action_protocol import ActionCall
from selfplay_graph_flowsteer.contracts import AgentArtifact, AgentNode, RelayPacket
from selfplay_graph_flowsteer.dataset_actions import (
    DatasetActionAdapter,
    DatasetActionRegistry,
    default_dataset_action_registry,
)
from selfplay_graph_flowsteer.graph import MultiAgentGraph
from selfplay_graph_flowsteer.llm import LLMResponse, MockBackend
from selfplay_graph_flowsteer.observability import TaskSpec
from selfplay_graph_flowsteer.runtime import (
    WORKER_PROTOCOL_FAILURE_SENTINEL,
    ModelAgentExecutor,
    MultiAgentRuntime,
    RoutedModelAgentExecutor,
    WorkerWallClockLimitExceeded,
    _enforce_artifact_integrity,
    _finalize_swe_progress,
)

from .helpers import RecordingExecutor


class FakeSearchTool:
    name = "search"
    description = "search test evidence"
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }

    def __init__(self) -> None:
        self.calls = []

    def execute(self, arguments):
        self.calls.append(arguments)
        return json.dumps({"result": [{"document": {"text": "evidence"}}]})


class FakeStatefulSearchTool(FakeSearchTool):
    stateful = True


class FakePythonTool:
    name = "python_exec"
    description = "execute test Python"

    def __init__(self) -> None:
        self.calls = []

    def execute(self, arguments):
        self.calls.append(arguments)
        return json.dumps({"status": "ok", "stdout": str(len(self.calls))})


class SequencedPythonTool(FakePythonTool):
    def __init__(self, statuses: list[str]) -> None:
        super().__init__()
        self.statuses = list(statuses)

    def execute(self, arguments):
        self.calls.append(arguments)
        status = self.statuses[len(self.calls) - 1]
        return json.dumps(
            {
                "status": status,
                "stdout": "verified" if status == "ok" else "",
                "stderr": "ImportError: module sympy is not allowed" if status == "error" else "",
            }
        )


class FakeNamedTool:
    description = "test-only dataset Action"
    parameters = {"type": "object", "additionalProperties": True}

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = []

    def execute(self, arguments):
        self.calls.append(arguments)
        return json.dumps({"status": "ok", "done": self.name == "alfworld_step"})


def test_agent_artifact_extracts_final_json_after_reasoning_with_example_json() -> None:
    text = (
        '<think>Do not emit {"action_call":{"name":"search"}} now.</think>\n'
        '{"answer":"2013","summary":"final result","confidence":0.9}'
    )

    artifact = AgentArtifact.from_model_text(
        text=text,
        artifact_id="artifact-1",
        agent_id="researcher",
    )

    assert artifact.answer == "2013"
    assert artifact.summary == "final result"
    assert artifact.raw_response == text


def test_model_agent_can_search_then_return_final_artifact() -> None:
    backend = MockBackend(
        [
            json.dumps({"tool_calls": [{"name": "search", "arguments": {"query": "q"}}]}),
            json.dumps({"answer": "grounded", "evidence": ["evidence"]}),
        ]
    )
    search = FakeSearchTool()
    executor = ModelAgentExecutor(backend, tools={"search": search})
    node = AgentNode("researcher", "solve", allowed_tools=("search",))

    artifact = executor.execute(
        task="question", node=node, upstream=[], peers=[], revision=False, seed=0
    )

    assert artifact.answer == "grounded"
    assert search.calls == [{"query": "q"}]
    assert artifact.tool_summary == ['search: {"query": "q"}']
    assert len(backend.calls) == 2


def test_worker_receives_action_schema_and_action_observation_protocol() -> None:
    backend = MockBackend(
        [
            json.dumps({"action_call": {"name": "search", "arguments": {"query": "q"}}}),
            json.dumps({"answer": "grounded"}),
        ]
    )
    search = FakeSearchTool()
    executor = ModelAgentExecutor(backend, tools={"search": search})
    node = AgentNode("researcher", "find evidence", allowed_tools=("search",))

    executor.execute(task="question", node=node, upstream=[], peers=[], revision=False, seed=0)

    first_context = json.loads(backend.calls[0]["messages"][-1]["content"])
    first_instruction = backend.calls[0]["messages"][0]["content"]
    observation = json.loads(backend.calls[1]["messages"][-1]["content"])
    assert first_context["available_actions"] == [
        {
            "name": "search",
            "description": "search test evidence",
            "parameters": FakeSearchTool.parameters,
        }
    ]
    assert first_context["assigned_task"] == "find evidence"
    assert "task" not in first_context
    assert "Never call an Action named finalize" in first_instruction
    assert "not an Action call" in first_instruction
    assert "shortest answer span" in first_instruction
    assert "assigned_task field is the only task visible" in first_instruction
    assert "do not infer or attempt to reconstruct" in first_instruction
    assert observation["action_observation"]["status"] == "ok"


@pytest.mark.parametrize(
    ("adapter_id", "dataset"),
    [
        ("aime", "aime"),
        ("retrieval_qa", "nq_open"),
        ("retrieval_qa", "hotpotqa"),
        ("webshop", "webshop"),
        ("alfworld", "alfworld"),
    ],
)
def test_all_existing_dataset_adapters_receive_public_task_separately_from_delegation(
    adapter_id: str,
    dataset: str,
) -> None:
    action_names = (
        "symbolic_compute",
        "finite_search",
        "python_exec",
        "search",
        "webshop_search",
        "webshop_click",
        "alfworld_step",
    )
    tools = {name: FakeNamedTool(name) for name in action_names}
    registry = default_dataset_action_registry(tools)
    adapter = registry.resolve(TaskSpec("task", "public", metadata={"dataset": dataset}))
    assert adapter is not None and adapter.adapter_id == adapter_id
    responses: list[str | LLMResponse] = [json.dumps({"answer": "public-answer"})]
    if adapter_id == "alfworld":
        responses.insert(
            0,
            LLMResponse(
                text="",
                model="mock",
                action_calls=[ActionCall("step", "alfworld_step", {"action_id": "public-action"})],
            ),
        )
    backend = MockBackend(responses)
    node = AgentNode(
        "worker",
        "BOUNDED DIRECTOR RESPONSIBILITY",
        allowed_tools=adapter.action_names,
        operation_policy_configured=True,
        initial_tool_budget=adapter.initial_action_budget,
        revision_tool_budget=adapter.revision_action_budget,
        total_tool_budget=adapter.total_action_budget,
        metadata={"action_adapter": adapter.adapter_id},
    )

    ModelAgentExecutor(backend, tools=tools, action_registry=registry).execute(
        task="PUBLIC TASK Q WITHOUT PRIVATE GOLD",
        node=node,
        upstream=[],
        peers=[],
        revision=False,
        seed=0,
    )

    context = json.loads(backend.calls[0]["messages"][-1]["content"])
    instruction = backend.calls[0]["messages"][0]["content"]
    assert context["assigned_task"] == "BOUNDED DIRECTOR RESPONSIBILITY"
    assert context["public_task_context"] == "PUBLIC TASK Q WITHOUT PRIVATE GOLD"
    assert context["action_environment"]["adapter"] == adapter.adapter_id
    assert "complete trusted public task" in instruction
    assert "assigned_task field is the only task visible" not in instruction
    for call in backend.calls:
        serialized = json.dumps(call["messages"], ensure_ascii=False)
        assert "PUBLIC TASK Q WITHOUT PRIVATE GOLD" in serialized


@pytest.mark.parametrize("dataset", ["nq_open", "hotpotqa"])
def test_short_qa_without_retrieval_adapter_still_receives_public_task(dataset: str) -> None:
    backend = MockBackend([json.dumps({"answer": "public-answer"})])
    node = AgentNode(
        "worker",
        "BOUNDED DIRECTOR RESPONSIBILITY",
        metadata={
            "system_managed_contract": {
                "version": "dataset-output-contract-v1",
                "dataset": dataset,
                "rule_ids": ["concise_answer_span"],
            }
        },
    )

    ModelAgentExecutor(backend).execute(
        task="PUBLIC TASK Q WITHOUT PRIVATE GOLD",
        node=node,
        upstream=[],
        peers=[],
        revision=False,
        seed=0,
    )

    context = json.loads(backend.calls[0]["messages"][-1]["content"])
    instruction = backend.calls[0]["messages"][0]["content"]
    assert context["assigned_task"] == "BOUNDED DIRECTOR RESPONSIBILITY"
    assert context["public_task_context"] == "PUBLIC TASK Q WITHOUT PRIVATE GOLD"
    assert "complete trusted public question" in instruction
    assert "assigned_task field is the only task visible" not in instruction


def test_worker_prefers_native_action_call_and_returns_native_tool_result() -> None:
    backend = MockBackend(
        [
            LLMResponse(
                text="",
                model="native-mock",
                action_calls=[ActionCall("call-1", "search", {"query": "q"})],
            ),
            json.dumps({"answer": "grounded"}),
        ]
    )
    search = FakeSearchTool()
    executor = ModelAgentExecutor(backend, tools={"search": search})
    node = AgentNode("researcher", "find evidence", allowed_tools=("search",))

    artifact = executor.execute(
        task="question", node=node, upstream=[], peers=[], revision=False, seed=0
    )

    assert artifact.answer == "grounded"
    assert search.calls == [{"query": "q"}]
    assert backend.calls[0]["actions"][0]["name"] == "search"
    tool_message = backend.calls[1]["messages"][-1]
    assert tool_message["role"] == "tool"
    assert tool_message["call_id"] == "call-1"
    assert json.loads(tool_message["content"])["status"] == "ok"


def test_worker_batches_multiple_stateless_action_calls() -> None:
    backend = MockBackend(
        [
            LLMResponse(
                text="",
                model="native-mock",
                action_calls=[
                    ActionCall("call-1", "search", {"query": "q1"}),
                    ActionCall("call-2", "search", {"query": "q2"}),
                ],
            ),
            json.dumps({"answer": "grounded"}),
        ]
    )
    search = FakeSearchTool()
    executor = ModelAgentExecutor(backend, tools={"search": search})
    node = AgentNode("researcher", "find evidence", allowed_tools=("search",))

    artifact = executor.execute(
        task="question", node=node, upstream=[], peers=[], revision=False, seed=0
    )

    assert artifact.answer == "grounded"
    assert search.calls == [{"query": "q1"}, {"query": "q2"}]
    assert [entry["observation"]["status"] for entry in artifact.react_trace] == [
        "ok",
        "ok",
    ]
    assert [entry["batch_index"] for entry in artifact.react_trace] == [0, 1]
    tool_messages = [
        message for message in backend.calls[1]["messages"] if message["role"] == "tool"
    ]
    assert [message["call_id"] for message in tool_messages] == ["call-1", "call-2"]


def test_worker_executes_only_first_legal_stateful_call_then_reobserves() -> None:
    backend = MockBackend(
        [
            LLMResponse(
                text="",
                model="native-mock",
                action_calls=[
                    ActionCall("call-1", "search", {"query": "q1"}),
                    ActionCall("call-2", "search", {"query": "stale-q2"}),
                ],
            ),
            LLMResponse(
                text="",
                model="native-mock",
                action_calls=[ActionCall("call-3", "search", {"query": "fresh-q2"})],
            ),
            json.dumps({"answer": "grounded"}),
        ]
    )
    search = FakeStatefulSearchTool()
    executor = ModelAgentExecutor(backend, tools={"search": search})
    node = AgentNode("shopper", "buy", allowed_tools=("search",))

    artifact = executor.execute(
        task="question", node=node, upstream=[], peers=[], revision=False, seed=0
    )

    assert artifact.answer == "grounded"
    assert search.calls == [{"query": "q1"}, {"query": "fresh-q2"}]
    assert artifact.react_trace[0]["observation"]["status"] == "ok"
    assert artifact.react_trace[1]["observation"]["error"]["code"] == ("stateful_action_deferred")
    assert artifact.react_trace[2]["observation"]["status"] == "ok"
    first_context = json.loads(backend.calls[0]["messages"][-1]["content"])
    assert first_context["available_actions"][0]["stateful"] is True
    assert "only the first legal stateful call runs" in backend.calls[0]["messages"][0]["content"]


def test_worker_validates_action_arguments_before_execution() -> None:
    backend = MockBackend(
        [
            LLMResponse(
                text="",
                model="native-mock",
                action_calls=[ActionCall("call-1", "search", {"query": 7})],
            ),
            json.dumps({"answer": "recovered"}),
        ]
    )
    search = FakeSearchTool()
    executor = ModelAgentExecutor(backend, tools={"search": search})
    node = AgentNode("researcher", "find evidence", allowed_tools=("search",))

    artifact = executor.execute(
        task="question", node=node, upstream=[], peers=[], revision=False, seed=0
    )

    assert artifact.answer == "recovered"
    assert search.calls == []
    error = artifact.react_trace[0]["observation"]["error"]
    assert error["code"] == "invalid_action_arguments"
    assert error["details"]["path"] == ["query"]
    assert error["details"]["budget_consumed"] is False
    assert error["details"]["retry_allowed"] is True


def test_invalid_action_call_does_not_consume_configured_execution_budget() -> None:
    backend = MockBackend(
        [
            LLMResponse(
                text="",
                model="native-mock",
                action_calls=[ActionCall("bad", "search", {"query": 7})],
            ),
            LLMResponse(
                text="",
                model="native-mock",
                action_calls=[ActionCall("good", "search", {"query": "q"})],
            ),
            json.dumps({"answer": "grounded"}),
        ]
    )
    search = FakeSearchTool()
    executor = ModelAgentExecutor(backend, tools={"search": search})
    node = AgentNode(
        "researcher",
        "find evidence",
        allowed_tools=("search",),
        operation_policy_configured=True,
        initial_tool_budget=1,
        total_tool_budget=1,
    )

    artifact = executor.execute(
        task="question", node=node, upstream=[], peers=[], revision=False, seed=0
    )

    assert artifact.answer == "grounded"
    assert search.calls == [{"query": "q"}]
    assert artifact.react_trace[0]["remaining_budget"] == {"phase": 1, "total": 1}
    assert artifact.react_trace[1]["remaining_budget"] == {"phase": 0, "total": 0}


def test_force_finalize_starts_immediately_after_action_budget_is_consumed() -> None:
    call = LLMResponse(
        text="",
        model="native-mock",
        action_calls=[
            ActionCall(
                "call-1",
                "python_exec",
                {"purpose": "compute", "code": "print(1)"},
            )
        ],
    )
    backend = MockBackend(
        [
            call,
            json.dumps(
                {
                    "answer": "1",
                    "summary": "done",
                    "confidence": 1,
                    "evidence": [],
                    "unresolved_issues": [],
                    "tool_summary": [],
                }
            ),
        ]
    )
    tool = FakePythonTool()
    executor = ModelAgentExecutor(backend, tools={"python_exec": tool})
    node = AgentNode(
        "solver",
        "solve",
        allowed_tools=("python_exec",),
        operation_policy_configured=True,
        initial_tool_budget=1,
        total_tool_budget=1,
    )

    artifact = executor.execute(
        task="problem", node=node, upstream=[], peers=[], revision=False, seed=0
    )

    assert artifact.answer == "1"
    assert len(tool.calls) == 1
    assert len(artifact.react_trace) == 1
    assert len(backend.calls) == 2
    assert backend.calls[-1]["actions"] == []
    assert "Action phase is over" in backend.calls[-1]["messages"][0]["content"]
    assert "shortest answer span" in backend.calls[-1]["messages"][0]["content"]


def test_runtime_caps_unsupported_high_confidence_after_failed_tool() -> None:
    backend = MockBackend(
        [
            json.dumps(
                {
                    "action_call": {
                        "name": "python_exec",
                        "arguments": {"code": "import sympy", "purpose": "verify"},
                    }
                }
            ),
            json.dumps(
                {
                    "answer": "25",
                    "summary": "The Python computation successfully verified the answer.",
                    "confidence": 1.0,
                    "evidence": ["Numerical calculation confirmed 25."],
                    "unresolved_issues": [],
                    "tool_summary": ["python_exec succeeded"],
                }
            ),
        ]
    )
    executor = ModelAgentExecutor(
        backend,
        tools={"python_exec": SequencedPythonTool(["error"])},
    )
    graph = MultiAgentGraph()
    graph.add_agent("solver")
    graph.set_prompt("solver", "solve and verify")
    node = graph.nodes["solver"]
    node.allowed_tools = ("python_exec",)
    node.operation_policy_configured = True
    node.initial_tool_budget = 1
    node.total_tool_budget = 1

    report = MultiAgentRuntime(executor).execute(task="problem", graph=graph)
    artifact = report.artifacts["solver"]

    assert artifact.answer == "25"
    assert artifact.claimed_confidence == 1.0
    assert artifact.confidence == 0.25
    assert artifact.runtime_tool_evidence == {
        "trusted": True,
        "attempted_count": 1,
        "successful_count": 0,
        "failed_count": 1,
        "successful_call_ids": [],
        "failed_call_ids": ["text-action-0"],
        "failure_codes": ["ImportError"],
        "terminal_failure": True,
        "recovered_failure": False,
        "all_actions_failed": True,
        "unsupported_tool_verification_claim": True,
        "claimed_confidence": 1.0,
        "effective_confidence": 0.25,
        "confidence_caps": {
            "all_tool_actions_failed": 0.35,
            "terminal_tool_failure": 0.5,
            "unsupported_tool_verification_claim": 0.25,
        },
    }
    assert artifact.integrity_risks == [
        "all_tool_actions_failed",
        "terminal_tool_failure",
        "unsupported_tool_verification_claim",
    ]
    assert artifact.unresolved_issues[:3] == [
        "runtime_integrity:all_tool_actions_failed",
        "runtime_integrity:terminal_tool_failure",
        "runtime_integrity:unsupported_tool_verification_claim",
    ]


def test_runtime_recognizes_tool_failure_recovered_by_later_success() -> None:
    backend = MockBackend(
        [
            json.dumps(
                {
                    "action_call": {
                        "name": "python_exec",
                        "arguments": {"code": "import sympy", "purpose": "compute"},
                    }
                }
            ),
            json.dumps(
                {
                    "action_call": {
                        "name": "python_exec",
                        "arguments": {"code": "print(45)", "purpose": "verify"},
                    }
                }
            ),
            json.dumps(
                {
                    "answer": "45",
                    "summary": "The later Python computation verified 45.",
                    "confidence": 0.95,
                    "evidence": ["The successful observation returned 45."],
                    "unresolved_issues": [],
                }
            ),
        ]
    )
    executor = ModelAgentExecutor(
        backend,
        tools={"python_exec": SequencedPythonTool(["error", "ok"])},
    )
    graph = MultiAgentGraph()
    graph.add_agent("solver")
    graph.set_prompt("solver", "solve and verify")
    node = graph.nodes["solver"]
    node.allowed_tools = ("python_exec",)
    node.operation_policy_configured = True
    node.initial_tool_budget = 2
    node.total_tool_budget = 2

    artifact = MultiAgentRuntime(executor).execute(task="problem", graph=graph).artifacts["solver"]

    assert artifact.claimed_confidence == 0.95
    assert artifact.confidence == 0.95
    assert artifact.integrity_risks == []
    assert artifact.runtime_tool_evidence["failed_count"] == 1
    assert artifact.runtime_tool_evidence["successful_count"] == 1
    assert artifact.runtime_tool_evidence["terminal_failure"] is False
    assert artifact.runtime_tool_evidence["recovered_failure"] is True


def test_post_success_action_budget_rejection_is_not_severe_integrity_failure() -> None:
    artifact = AgentArtifact(
        "artifact-budget-stop",
        "retriever",
        "193",
        confidence=0.9,
        evidence=["trusted search result"],
        react_trace=[
            {
                "action": {"call_id": "search-1", "name": "search"},
                "observation": {"status": "ok", "output": {"status": "ok"}},
            },
            {
                "action": {"call_id": "search-2", "name": "search"},
                "observation": {
                    "status": "error",
                    "error": {"code": "initial_action_budget_exhausted"},
                },
            },
        ],
    )

    _enforce_artifact_integrity(artifact)

    assert artifact.integrity_risks == []
    assert artifact.runtime_tool_evidence["terminal_failure"] is False
    assert artifact.runtime_tool_evidence["recovered_failure"] is True
    assert artifact.runtime_tool_evidence["terminal_failure_observed"] is True
    assert artifact.runtime_tool_evidence["post_success_budget_rejection_waived"] is True
    assert artifact.runtime_tool_evidence["budget_rejection_code"] == (
        "initial_action_budget_exhausted"
    )


def test_swe_post_commit_repeated_read_failure_does_not_invalidate_patch() -> None:
    artifact = AgentArtifact(
        "artifact-1",
        "solver",
        "Implemented and syntax-tested the patch.",
        confidence=0.8,
        react_trace=[
            {
                "action": {"call_id": "deferred", "name": "swe_edit"},
                "observation": {
                    "status": "error",
                    "error": {"code": "stateful_action_deferred"},
                },
            },
            {
                "action": {"call_id": "edit", "name": "swe_edit"},
                "observation": {"status": "ok", "output": {"status": "ok"}},
            },
            {
                "action": {"call_id": "test", "name": "swe_test"},
                "observation": {"status": "ok", "output": {"status": "ok"}},
            },
            {
                "action": {"call_id": "repeat", "name": "swe_status"},
                "observation": {
                    "status": "error",
                    "error": {"code": "repeated_no_progress_action"},
                },
            },
        ],
        swe_progress={
            "trusted": True,
            "selected_as_output": True,
            "commit_ready": True,
            "workspace_changed": True,
            "test_after_latest_edit": True,
            # The selected Agent may already be patch+test ready before
            # SET_OUTPUT, so no separate final-fix execution is required.
            "final_fix_pass_count": 0,
        },
    )

    _enforce_artifact_integrity(artifact)

    assert artifact.integrity_risks == []
    assert artifact.runtime_tool_evidence["terminal_failure"] is False
    assert artifact.runtime_tool_evidence["recovered_failure"] is True
    assert artifact.runtime_tool_evidence["failure_codes"] == [
        "stateful_action_deferred",
        "repeated_no_progress_action",
    ]
    assert artifact.runtime_tool_evidence["terminal_failure_observed"] is True
    assert artifact.runtime_tool_evidence["post_commit_no_progress_failure_waived"] is True


def test_swe_read_only_policy_stall_becomes_trainable_typed_failure() -> None:
    policy_error = {
        "status": "error",
        "error": {"code": "swe_edit_test_reserve_required"},
    }
    artifact = AgentArtifact(
        "artifact-policy-stall",
        "committer",
        WORKER_PROTOCOL_FAILURE_SENTINEL,
        confidence=0.8,
        unresolved_issues=["nonfinal_response_after_action_phase"],
        react_trace=[
            {
                "action": {"call_id": "read", "name": "swe_read"},
                "observation": {
                    "status": "ok",
                    "output": {
                        "status": "ok",
                        "path": "src/value.py",
                        "start_line": 1,
                        "end_line": 10,
                        "file_sha256": "a" * 64,
                    },
                },
            },
            *[
                {
                    "action": {"call_id": f"stall-{index}", "name": "swe_status"},
                    "observation": dict(policy_error),
                }
                for index in range(4)
            ],
        ],
        protocol_diagnostics=[
            {
                "stage": "finalization_2",
                "accepted": False,
                "rejection_reason": "swe_commit_or_grounded_failure_required",
            }
        ],
    )
    node = AgentNode(
        "committer",
        "produce the final repository fix",
        metadata={"exclusive_capabilities": ["code_commit"]},
    )

    _finalize_swe_progress(artifact, node=node)
    _enforce_artifact_integrity(artifact)

    assert artifact.answer == "typed_policy_failure"
    assert artifact.confidence == 0.0
    assert artifact.swe_progress["state"] == "typed_policy_failure"
    assert artifact.swe_progress["commit_ready"] is True
    assert artifact.swe_progress["workspace_changed"] is False
    assert artifact.swe_progress["failure_code"] == "read_only_policy_stall"
    assert artifact.swe_progress["policy_failure"] == {
        "status": "typed_policy_failure",
        "code": "read_only_policy_stall",
        "attribution": "model_policy",
        "repository_observation_count": 1,
        "consecutive_runtime_rejection_count": 4,
        "runtime_rejection_codes": ["swe_edit_test_reserve_required"],
        "normalized_diff_empty": True,
        "official_reward": 0.0,
    }
    assert artifact.integrity_risks == []
    assert artifact.runtime_tool_evidence["terminal_failure"] is False
    assert artifact.runtime_tool_evidence["terminal_protocol_failure_observed"] is True
    assert artifact.runtime_tool_evidence["classified_as_typed_policy_failure"] is True


def test_swe_empty_patch_with_action_execution_failure_is_not_policy_stall() -> None:
    artifact = AgentArtifact(
        "artifact-mixed-failure",
        "committer",
        WORKER_PROTOCOL_FAILURE_SENTINEL,
        react_trace=[
            {
                "action": {"call_id": "read", "name": "swe_read"},
                "observation": {
                    "status": "ok",
                    "output": {"status": "ok", "path": "src/value.py"},
                },
            },
            {
                "action": {"call_id": "edit", "name": "swe_edit"},
                "observation": {
                    "status": "error",
                    "error": {"code": "action_execution_failed"},
                },
            },
            *[
                {
                    "action": {"call_id": f"stall-{index}", "name": "swe_status"},
                    "observation": {
                        "status": "error",
                        "error": {"code": "swe_edit_test_reserve_required"},
                    },
                }
                for index in range(4)
            ],
        ],
    )
    node = AgentNode(
        "committer",
        "produce the final repository fix",
        metadata={"exclusive_capabilities": ["code_commit"]},
    )

    _finalize_swe_progress(artifact, node=node)

    assert artifact.answer == WORKER_PROTOCOL_FAILURE_SENTINEL
    assert artifact.swe_progress["state"] == "edit_required"
    assert artifact.swe_progress["commit_ready"] is False
    assert artifact.swe_progress["policy_failure"] == {}


def test_relay_packet_propagates_effective_confidence_and_runtime_integrity() -> None:
    artifact = AgentArtifact(
        "artifact-1",
        "source",
        "25",
        confidence=0.25,
        claimed_confidence=1.0,
        runtime_tool_evidence={"trusted": True, "terminal_failure": True},
        integrity_risks=["terminal_tool_failure"],
        swe_progress={"trusted": True, "state": "tested", "commit_ready": True},
    )
    packet = RelayPacket.from_artifact(
        artifact,
        recipients=["sink"],
        message_id="message-1",
    )

    assert packet.confidence == 0.25
    assert packet.claimed_confidence == 1.0
    assert packet.runtime_tool_evidence["terminal_failure"] is True
    assert packet.integrity_risks == ("terminal_tool_failure",)
    assert packet.swe_progress["commit_ready"] is True


def test_nonfinal_response_without_an_action_uses_one_compact_finalization() -> None:
    backend = MockBackend(
        [
            LLMResponse(text="", model="reasoning-only", token_in=100, token_out=500),
            json.dumps(
                {
                    "answer": "7",
                    "summary": "done",
                    "confidence": 1,
                    "evidence": [],
                    "unresolved_issues": [],
                    "tool_summary": [],
                }
            ),
        ]
    )
    executor = ModelAgentExecutor(backend, tools={"python_exec": FakePythonTool()})
    node = AgentNode(
        "solver",
        "solve",
        allowed_tools=("python_exec",),
        operation_policy_configured=True,
        initial_tool_budget=3,
        total_tool_budget=4,
    )

    artifact = executor.execute(
        task="problem", node=node, upstream=[], peers=[], revision=False, seed=0
    )

    assert artifact.answer == "7"
    assert len(backend.calls) == 2
    assert backend.calls[1]["actions"] == []
    assert backend.calls[1]["max_tokens"] == 4096
    assert backend.calls[1]["enable_thinking"] is False
    assert "Action phase is over" in backend.calls[1]["messages"][0]["content"]
    recovery_context = json.loads(backend.calls[1]["messages"][1]["content"])
    assert recovery_context["assigned_task"] == "solve"
    assert "task" not in recovery_context
    assert "rejected_nonfinal_response" not in recovery_context


def test_worker_never_receives_global_task_in_initial_or_recovery_requests() -> None:
    global_task = "GLOBAL_TASK_SECRET_7f31 must never reach the Worker"
    backend = MockBackend(
        [
            LLMResponse(text="", model="reasoning-only"),
            json.dumps({"answer": "assigned result", "summary": "done"}),
        ]
    )
    executor = ModelAgentExecutor(backend, tools={"python_exec": FakePythonTool()})
    upstream = AgentArtifact("upstream-1", "source", "visible upstream evidence")
    node = AgentNode(
        "solver",
        "Synthesize the visible upstream evidence",
        allowed_tools=("python_exec",),
        operation_policy_configured=True,
        initial_tool_budget=1,
        total_tool_budget=1,
    )

    artifact = executor.execute(
        task=global_task,
        node=node,
        upstream=[
            RelayPacket.from_artifact(
                upstream,
                recipients=["solver"],
                message_id="upstream-message",
                phase="upstream",
            )
        ],
        peers=[],
        revision=False,
        seed=0,
    )

    assert artifact.answer == "assigned result"
    encoded_calls = json.dumps(backend.calls, ensure_ascii=False)
    assert global_task not in encoded_calls
    initial_context = json.loads(backend.calls[0]["messages"][-1]["content"])
    recovery_context = json.loads(backend.calls[1]["messages"][-1]["content"])
    assert initial_context["assigned_task"] == node.prompt
    assert initial_context["upstream_packets"][0]["answer"] == "visible upstream evidence"
    assert recovery_context["assigned_task"] == node.prompt
    assert "task" not in initial_context
    assert "task" not in recovery_context


def test_empty_compact_finalization_becomes_an_explicit_protocol_failure() -> None:
    backend = MockBackend(
        [
            LLMResponse(text="", model="reasoning-only"),
            LLMResponse(text="", model="reasoning-only"),
            LLMResponse(text="", model="reasoning-only"),
        ]
    )
    executor = ModelAgentExecutor(backend, tools={"python_exec": FakePythonTool()})
    node = AgentNode(
        "solver",
        "solve",
        allowed_tools=("python_exec",),
        operation_policy_configured=True,
        initial_tool_budget=3,
        total_tool_budget=4,
    )

    artifact = executor.execute(
        task="problem", node=node, upstream=[], peers=[], revision=False, seed=0
    )

    assert artifact.answer == "WORKER_PROTOCOL_FAILURE"
    assert artifact.unresolved_issues == ["nonfinal_response_after_action_phase"]
    assert len(backend.calls) == 3
    assert [call["max_tokens"] for call in backend.calls[1:]] == [4096, 2048]
    assert len(artifact.protocol_diagnostics) == 3
    assert artifact.protocol_diagnostics[0]["stage"] == "initial_nonfinal"
    assert artifact.protocol_diagnostics[-1]["rejection_reason"] == "missing_final_json"


def test_generic_acknowledgement_gets_one_clean_finalization_retry() -> None:
    backend = MockBackend(
        [
            json.dumps({"answer": "Understood"}),
            json.dumps({"answer": "Okay"}),
            json.dumps({"answer": "63", "summary": "computed"}),
        ]
    )
    executor = ModelAgentExecutor(backend, tools={"python_exec": FakePythonTool()})
    node = AgentNode(
        "solver",
        "solve the geometry task",
        allowed_tools=("python_exec",),
        operation_policy_configured=True,
        initial_tool_budget=3,
        total_tool_budget=4,
    )

    artifact = executor.execute(
        task="Find the requested integer.",
        node=node,
        upstream=[],
        peers=[],
        revision=False,
        seed=0,
    )

    assert artifact.answer == "63"
    assert len(backend.calls) == 3
    assert backend.calls[1]["max_tokens"] == 4096
    assert backend.calls[2]["max_tokens"] == 2048
    assert "Understood" not in backend.calls[1]["messages"][1]["content"]
    assert (
        json.loads(backend.calls[2]["messages"][1]["content"])["previous_attempt_issue"]
        == "generic_acknowledgement"
    )
    assert [item["accepted"] for item in artifact.protocol_diagnostics] == [
        False,
        False,
        True,
    ]


def test_json_encoded_native_action_arguments_are_decoded_once() -> None:
    backend = MockBackend(
        [
            LLMResponse(
                text="",
                model="native-mock",
                action_calls=[ActionCall("call-1", "search", json.dumps({"query": "q"}))],
            ),
            json.dumps({"answer": "grounded"}),
        ]
    )
    search = FakeSearchTool()
    executor = ModelAgentExecutor(backend, tools={"search": search})
    node = AgentNode("researcher", "find evidence", allowed_tools=("search",))

    artifact = executor.execute(
        task="question", node=node, upstream=[], peers=[], revision=False, seed=0
    )

    assert artifact.answer == "grounded"
    assert search.calls == [{"query": "q"}]
    assert artifact.react_trace[0]["action"]["arguments"] == {"query": "q"}


def test_non_object_native_action_arguments_finalize_without_repeating_payload() -> None:
    oversized = "x" * 20_000
    backend = MockBackend(
        [
            LLMResponse(
                text="",
                model="malformed-native",
                action_calls=[ActionCall("bad", "python_exec", oversized)],
            ),
            json.dumps({"answer": "bounded recovery"}),
        ]
    )
    tool = FakePythonTool()
    executor = ModelAgentExecutor(backend, tools={"python_exec": tool})
    node = AgentNode(
        "solver",
        "solve",
        allowed_tools=("python_exec",),
        operation_policy_configured=True,
        initial_tool_budget=3,
        total_tool_budget=4,
    )

    artifact = executor.execute(
        task="problem", node=node, upstream=[], peers=[], revision=False, seed=0
    )

    assert artifact.answer == "bounded recovery"
    assert tool.calls == []
    assert len(backend.calls) == 2
    recovery_context = backend.calls[1]["messages"][1]["content"]
    assert len(recovery_context) < 3000
    assert "arguments_type" in recovery_context
    assert oversized not in recovery_context


def test_runtime_rejects_node_visibility_that_differs_from_dataset_adapter() -> None:
    adapter = DatasetActionAdapter(
        adapter_id="retrieval_qa",
        datasets=("nq_open",),
        action_names=("search",),
        initial_action_budget=1,
        revision_action_budget=0,
        total_action_budget=1,
    )
    registry = DatasetActionRegistry([adapter], available_actions=("search", "python_exec"))
    node = AgentNode(
        "researcher",
        "solve",
        allowed_tools=("python_exec",),
        operation_policy_configured=True,
        initial_tool_budget=1,
        total_tool_budget=1,
        metadata={"action_adapter": "retrieval_qa"},
    )
    executor = ModelAgentExecutor(
        MockBackend(),
        tools={"search": FakeSearchTool(), "python_exec": FakePythonTool()},
        action_registry=registry,
    )

    with pytest.raises(ValueError, match="visibility differs"):
        executor.execute(task="question", node=node, upstream=[], peers=[], revision=False, seed=0)


def test_runtime_rejects_action_injection_for_healthbench_zero_action_adapter() -> None:
    adapter = DatasetActionAdapter(
        adapter_id="healthbench_professional",
        datasets=("healthbench_professional",),
        action_names=(),
        initial_action_budget=0,
        revision_action_budget=0,
        total_action_budget=0,
    )
    registry = DatasetActionRegistry([adapter], available_actions=("python_exec",))
    node = AgentNode(
        "clinician",
        "answer the clinical request",
        allowed_tools=("python_exec",),
        operation_policy_configured=True,
        initial_tool_budget=1,
        total_tool_budget=1,
        metadata={"action_adapter": "healthbench_professional"},
    )
    executor = ModelAgentExecutor(
        MockBackend(),
        tools={"python_exec": FakePythonTool()},
        action_registry=registry,
    )

    with pytest.raises(ValueError, match="visibility differs"):
        executor.execute(
            task="clinical request", node=node, upstream=[], peers=[], revision=False, seed=0
        )


def test_bidirectional_component_uses_one_proposal_and_one_revision() -> None:
    graph = MultiAgentGraph()
    for key in ("a", "b", "out"):
        graph.add_agent(key)
        graph.set_prompt(key, key)
    graph.set_layer("out", 1)
    graph.set_relation("a", "b", "bidirectional")
    graph.set_relation("a", "out", "directed")
    graph.set_relation("b", "out", "directed")
    graph.set_output("out")

    executor = RecordingExecutor()
    report = MultiAgentRuntime(executor).execute(task="task", graph=graph)

    a_calls = [call for call in executor.calls if call["agent_id"] == "a"]
    b_calls = [call for call in executor.calls if call["agent_id"] == "b"]
    out_call = next(call for call in executor.calls if call["agent_id"] == "out")
    assert [call["revision"] for call in a_calls] == [False, True]
    assert [call["revision"] for call in b_calls] == [False, True]
    assert a_calls[1]["peers"] == ["b"]
    assert b_calls[1]["peers"] == ["a"]
    assert a_calls[1]["prior"] == "a"
    assert b_calls[1]["prior"] == "b"
    assert "revision=False" in str(a_calls[1]["prior_answer"])
    assert "revision=False" in str(b_calls[1]["prior_answer"])
    assert out_call["upstream"] == ["a", "b"]
    assert report.output.startswith("out:")
    assert report.token_in == 10
    assert report.token_out == 15
    assert report.scheduled_agents == ["a", "b", "out"]
    assert report.initial_model_calls == 3
    assert report.revision_model_calls == 2
    assert report.worker_model_calls_total == 5
    assert report.cache_hits == 0
    assert report.component_execution_count == 2
    assert len(report.execution_events) == 5
    assert {event["phase"] for event in report.execution_events} == {"initial", "revision"}
    assert all(
        "bidirectional_revision" in event["reason_codes"]
        for event in report.execution_events
        if event["phase"] == "revision"
    )
    assert report.revision_wave_count == 1
    assert report.revision_skipped_agents == []
    assert report.revision_decisions == [
        {
            "component_agents": ["a", "b"],
            "healthy_agents": ["a", "b"],
            "policy": "always",
            "revision_required": True,
            "reason_codes": ["answer_disagreement", "missing_evidence", "policy_always"],
            "answer_agreement": False,
            "minimum_confidence": 1.0,
            "confidence_threshold": 0.8,
            "revision_wave_budget": 1,
            "revision_waves_used": 1,
            "revised_agents": ["a", "b"],
        }
    ]


def test_bidirectional_component_revises_even_for_supported_agreement() -> None:
    class AgreementExecutor(RecordingExecutor):
        def execute(self, **kwargs):
            artifact = super().execute(**kwargs)
            if kwargs["node"].agent_id in {"a", "b"}:
                artifact.answer = "42."
                artifact.summary = "independent derivation gives 42"
                artifact.confidence = 0.95
                artifact.evidence = ["checked derivation"]
            return artifact

    graph = MultiAgentGraph()
    for key in ("a", "b", "out"):
        graph.add_agent(key)
        graph.set_prompt(key, key)
    graph.set_layer("out", 1)
    graph.set_relation("a", "b", "bidirectional")
    graph.set_relation("a", "out", "directed")
    graph.set_relation("b", "out", "directed")
    graph.set_output("out")
    executor = AgreementExecutor()

    report = MultiAgentRuntime(executor).execute(task="task", graph=graph)

    assert [(call["agent_id"], call["revision"]) for call in executor.calls] == [
        ("a", False),
        ("b", False),
        ("a", True),
        ("b", True),
        ("out", False),
    ]
    assert report.initial_model_calls == 3
    assert report.revision_model_calls == 2
    assert report.worker_model_calls_total == 5
    assert report.revision_skipped_agents == []
    assert report.revision_wave_count == 1
    decision = report.revision_decisions[0]
    assert decision["revision_required"] is True
    assert decision["reason_codes"] == ["policy_always"]
    assert decision["answer_agreement"] is True
    assert decision["revision_waves_used"] == 1


@pytest.mark.parametrize(
    ("signal", "expected_reason"),
    [
        ("answer_disagreement", "answer_disagreement"),
        ("low_confidence", "low_confidence"),
        ("missing_evidence", "missing_evidence"),
        ("unresolved_issue", "unresolved_issue"),
        ("terminal_tool_failure", "terminal_tool_failure"),
        ("terminal_protocol_failure", "terminal_protocol_failure"),
    ],
)
def test_bidirectional_revision_gate_runs_for_structured_risk_signals(
    signal: str,
    expected_reason: str,
) -> None:
    class SignaledExecutor(RecordingExecutor):
        def execute(self, **kwargs):
            artifact = super().execute(**kwargs)
            artifact.answer = "42"
            artifact.summary = "answer 42"
            artifact.confidence = 0.95
            artifact.evidence = ["checked derivation"]
            if kwargs["node"].agent_id == "b" and not kwargs["revision"]:
                if signal == "answer_disagreement":
                    artifact.answer = "43"
                elif signal == "low_confidence":
                    artifact.confidence = 0.5
                elif signal == "missing_evidence":
                    artifact.evidence = []
                elif signal == "unresolved_issue":
                    artifact.unresolved_issues = ["independent check required"]
                elif signal == "terminal_tool_failure":
                    artifact.react_trace = [
                        {
                            "observation": {
                                "status": "ok",
                                "output": {"status": "error"},
                            }
                        }
                    ]
                elif signal == "terminal_protocol_failure":
                    artifact.protocol_diagnostics = [{"accepted": False}]
            return artifact

    graph = MultiAgentGraph()
    for key in ("a", "b"):
        graph.add_agent(key)
        graph.set_prompt(key, key)
    graph.set_relation("a", "b", "bidirectional")
    executor = SignaledExecutor()

    report = MultiAgentRuntime(executor).execute(task="task", graph=graph)

    assert report.revision_model_calls == 2
    assert report.revision_wave_count == 1
    assert expected_reason in report.revision_decisions[0]["reason_codes"]


def test_bidirectional_revision_gate_ignores_recovered_and_empty_signals() -> None:
    class RecoveredExecutor(RecordingExecutor):
        def execute(self, **kwargs):
            artifact = super().execute(**kwargs)
            artifact.answer = "42"
            artifact.confidence = 0.95
            artifact.evidence = ["checked derivation"]
            artifact.unresolved_issues = ["None."]
            artifact.react_trace = [
                {
                    "observation": {
                        "status": "ok",
                        "output": {"status": "error"},
                    }
                },
                {
                    "observation": {
                        "status": "ok",
                        "output": {"status": "ok"},
                    }
                },
            ]
            artifact.protocol_diagnostics = [
                {"accepted": False},
                {"accepted": True},
            ]
            return artifact

    graph = MultiAgentGraph()
    for key in ("a", "b"):
        graph.add_agent(key)
        graph.set_prompt(key, key)
    graph.set_relation("a", "b", "bidirectional")

    report = MultiAgentRuntime(RecoveredExecutor()).execute(task="task", graph=graph)

    assert report.revision_model_calls == 2
    assert report.revision_decisions[0]["reason_codes"] == ["policy_always"]


def test_bidirectional_always_policy_preserves_flowsteer_baseline() -> None:
    class AgreementExecutor(RecordingExecutor):
        def execute(self, **kwargs):
            artifact = super().execute(**kwargs)
            artifact.answer = "42"
            artifact.confidence = 1.0
            artifact.evidence = ["checked derivation"]
            return artifact

    graph = MultiAgentGraph()
    for key in ("a", "b"):
        graph.add_agent(key)
        graph.set_prompt(key, key)
    graph.set_relation("a", "b", "bidirectional")

    report = MultiAgentRuntime(AgreementExecutor(), bidirectional_revision_policy="always").execute(
        task="task", graph=graph
    )

    assert report.revision_model_calls == 2
    assert report.revision_decisions[0]["reason_codes"] == ["policy_always"]


def test_bidirectional_dirty_run_counts_cached_proposals_and_real_revisions() -> None:
    graph = MultiAgentGraph()
    for key in ("a", "b"):
        graph.add_agent(key)
        graph.set_prompt(key, key)
    executor = RecordingExecutor()
    runtime = MultiAgentRuntime(executor)
    runtime.execute(task="task", graph=graph)
    executor.calls.clear()
    mutation = graph.set_relation("a", "b", "bidirectional")

    report = runtime.execute(
        task="task",
        graph=graph,
        dirty_agents=mutation.dirty_agents,
        invalidation_reasons={
            "a": {"peer_changed"},
            "b": {"peer_changed"},
        },
    )

    assert [(call["agent_id"], call["revision"]) for call in executor.calls] == [
        ("a", True),
        ("b", True),
    ]
    assert report.executed_agents == ["a", "b"]
    assert report.reused_agents == []
    assert report.cache_reused_agents == []
    assert report.initial_model_calls == 0
    assert report.revision_model_calls == 2
    assert report.worker_model_calls_total == 2
    assert report.cache_hits == 2
    assert report.component_execution_count == 1
    assert len(report.execution_events) == 4
    assert [event["cache_hit"] for event in report.execution_events] == [
        True,
        True,
        False,
        False,
    ]


def test_bidirectional_dirty_relation_reuses_initial_then_runs_peer_revision() -> None:
    class AgreementExecutor(RecordingExecutor):
        def execute(self, **kwargs):
            artifact = super().execute(**kwargs)
            artifact.answer = "42"
            artifact.confidence = 0.95
            artifact.evidence = ["checked derivation"]
            return artifact

    graph = MultiAgentGraph()
    for key in ("a", "b"):
        graph.add_agent(key)
        graph.set_prompt(key, key)
    executor = AgreementExecutor()
    runtime = MultiAgentRuntime(executor)
    runtime.execute(task="task", graph=graph)
    executor.calls.clear()
    mutation = graph.set_relation("a", "b", "bidirectional")

    report = runtime.execute(
        task="task",
        graph=graph,
        dirty_agents=mutation.dirty_agents,
        invalidation_reasons={
            "a": {"peer_changed"},
            "b": {"peer_changed"},
        },
    )

    assert [(call["agent_id"], call["revision"]) for call in executor.calls] == [
        ("a", True),
        ("b", True),
    ]
    assert report.scheduled_agents == ["a", "b"]
    assert report.executed_agents == ["a", "b"]
    assert report.reused_agents == []
    assert report.cache_hits == 2
    assert report.initial_cache_hits == 2
    assert report.worker_model_calls_total == 2
    assert report.revision_model_calls == 2
    assert report.revision_skipped_agents == []
    assert report.revision_decisions[0]["revision_required"] is True


def test_bidirectional_component_does_not_revise_failed_first_passes() -> None:
    class ProtocolFailingExecutor(RecordingExecutor):
        def execute(self, **kwargs):
            artifact = super().execute(**kwargs)
            if kwargs["node"].agent_id == "a" and not kwargs["revision"]:
                artifact.answer = WORKER_PROTOCOL_FAILURE_SENTINEL
            return artifact

    graph = MultiAgentGraph()
    for key in ("a", "b"):
        graph.add_agent(key)
        graph.set_prompt(key, key)
    graph.set_relation("a", "b", "bidirectional")
    graph.set_output("b")
    executor = ProtocolFailingExecutor()

    report = MultiAgentRuntime(executor).execute(task="task", graph=graph)

    assert [(call["agent_id"], call["revision"]) for call in executor.calls] == [
        ("a", False),
        ("b", False),
    ]
    assert report.artifacts["a"].answer == WORKER_PROTOCOL_FAILURE_SENTINEL
    assert report.output.startswith("b:")
    assert report.token_in == 4
    assert report.token_out == 6


def test_revision_prompt_separates_prior_artifact_from_peer_packets() -> None:
    backend = MockBackend([json.dumps({"answer": "revised"})])
    executor = ModelAgentExecutor(backend)
    own = AgentArtifact("own-1", "a", "my first answer", evidence=["own evidence"])
    peer = AgentArtifact("peer-1", "b", "peer answer", evidence=["peer evidence"])

    artifact = executor.execute(
        task="question",
        node=AgentNode("a", "solve"),
        upstream=[],
        peers=[
            RelayPacket.from_artifact(
                peer, recipients=["a"], message_id="peer-message", phase="peer_proposal"
            )
        ],
        revision=True,
        seed=0,
        prior=RelayPacket.from_artifact(
            own, recipients=["a"], message_id="own-message", phase="self_proposal"
        ),
    )

    context = json.loads(backend.calls[0]["messages"][1]["content"])
    assert context["prior_artifact"]["answer"] == "my first answer"
    assert context["prior_artifact"]["evidence"] == ["own evidence"]
    assert context["peer_packets"][0]["answer"] == "peer answer"
    assert context["peer_packets"][0]["evidence"] == ["peer evidence"]
    assert artifact.source_artifact_ids == ["own-1", "peer-1"]


def test_worker_deadline_stops_before_another_backend_request() -> None:
    backend = MockBackend([json.dumps({"answer": "too late"})])
    executor = ModelAgentExecutor(backend, deadline_monotonic=0.0)

    with pytest.raises(WorkerWallClockLimitExceeded, match="wall-clock budget"):
        executor.execute(
            task="question",
            node=AgentNode("a", "solve"),
            upstream=[],
            peers=[],
            revision=False,
            seed=0,
        )

    assert not backend.calls


def test_dirty_execution_reuses_unaffected_branch() -> None:
    graph = MultiAgentGraph()
    for key in ("a", "c", "out"):
        graph.add_agent(key)
        graph.set_prompt(key, key)
    graph.set_layer("out", 1)
    graph.set_relation("a", "out", "directed")
    graph.set_relation("c", "out", "directed")
    graph.set_output("out")

    executor = RecordingExecutor()
    runtime = MultiAgentRuntime(executor)
    runtime.execute(task="task", graph=graph)
    executor.calls.clear()
    mutation = graph.set_prompt("a", "a changed")
    report = runtime.execute(task="task", graph=graph, dirty_agents=mutation.dirty_agents)

    assert [call["agent_id"] for call in executor.calls] == ["a", "out"]
    assert report.executed_agents == ["a", "out"]
    assert report.reused_agents == ["c"]
    assert report.scheduled_agents == ["a", "out"]
    assert report.skipped_clean_agents == ["c"]
    assert report.initial_model_calls == 2
    assert report.revision_model_calls == 0
    assert report.worker_model_calls_total == 2
    reasons = {event["agent_id"]: event["reason_codes"] for event in report.execution_events}
    assert "prompt_changed" in reasons["a"]
    assert "upstream_changed" in reasons["out"]


def test_routed_executor_obeys_solver_model_choice_for_each_agent() -> None:
    response = json.dumps({"answer": "ok", "summary": "ok"})
    minimax = MockBackend([response, response])
    grok = MockBackend([response])
    executor = RoutedModelAgentExecutor({"minimax": minimax, "grok": grok}, ("minimax", "grok"))
    assert executor.version == ("solver-routed-model-agent-v21-qa-request-credit:minimax,grok")

    first = executor.execute(
        task="task",
        node=AgentNode("agent_1", "analyze", metadata={"runtime_route": "minimax"}),
        upstream=[],
        peers=[],
        revision=False,
        seed=0,
    )
    second = executor.execute(
        task="task",
        node=AgentNode("agent_2", "verify", metadata={"runtime_route": "grok"}),
        upstream=[],
        peers=[],
        revision=False,
        seed=0,
    )
    revision = executor.execute(
        task="task",
        node=AgentNode("agent_1", "analyze", metadata={"runtime_route": "minimax"}),
        upstream=[],
        peers=[],
        revision=True,
        seed=0,
    )

    assert (first.model_route, second.model_route, revision.model_route) == (
        "minimax",
        "grok",
        "minimax",
    )
    assert len(minimax.calls) == 2
    assert len(grok.calls) == 1


def test_routed_executor_accepts_explicit_agent_route() -> None:
    response = json.dumps({"answer": "ok"})
    executor = RoutedModelAgentExecutor(
        {"minimax": MockBackend([response]), "grok": MockBackend([response])},
        ("minimax", "grok"),
    )
    node = AgentNode("specialist", "verify", metadata={"runtime_route": "grok"})

    artifact = executor.execute(
        task="task", node=node, upstream=[], peers=[], revision=False, seed=0
    )

    assert artifact.model_route == "grok"


def test_routed_executor_turns_transient_timeout_into_failure_artifact() -> None:
    class TimeoutBackend(MockBackend):
        def generate(self, *args, **kwargs):
            raise TimeoutError("gateway stalled")

    executor = RoutedModelAgentExecutor(
        {"minimax": TimeoutBackend()},
        ("minimax",),
    )

    artifact = executor.execute(
        task="task",
        node=AgentNode("solver", "solve", metadata={"runtime_route": "minimax"}),
        upstream=[],
        peers=[],
        revision=False,
        seed=0,
    )

    assert artifact.answer == "WORKER_BACKEND_FAILURE"
    assert artifact.confidence == 0.0
    assert artifact.model_route == "minimax"
    assert artifact.unresolved_issues == ["transient_backend_error:TimeoutError"]


def test_routed_executor_classifies_nexus_ssrf_blocked_as_route_failure() -> None:
    class PermissionDeniedError(RuntimeError):
        status_code = 403
        body = {"error": {"message": "目标地址不可达", "type": "ssrf_blocked"}}

    class BlockedBackend(MockBackend):
        def generate(self, *args, **kwargs):
            raise PermissionDeniedError("Error code: 403 - target unavailable (ssrf_blocked)")

    executor = RoutedModelAgentExecutor(
        {"gpt": BlockedBackend()},
        ("gpt",),
    )

    artifact = executor.execute(
        task="task",
        node=AgentNode("solver", "solve", metadata={"runtime_route": "gpt"}),
        upstream=[],
        peers=[],
        revision=False,
        seed=0,
    )

    assert artifact.answer == "WORKER_BACKEND_FAILURE"
    assert artifact.model_route == "gpt"
    assert artifact.unresolved_issues == ["transient_backend_error:PermissionDeniedError"]


def test_routed_executor_preserves_structured_424_failure_without_relaying_it() -> None:
    class FailedDependencyError(RuntimeError):
        status_code = 424

    class FailedBackend(MockBackend):
        def generate(self, *args, **kwargs):
            raise FailedDependencyError("upstream dependency unavailable")

    executor = RoutedModelAgentExecutor(
        {"gemini": FailedBackend()},
        ("gemini",),
    )
    artifact = executor.execute(
        task="task",
        node=AgentNode("solver", "solve", metadata={"runtime_route": "gemini"}),
        upstream=[],
        peers=[],
        revision=False,
        seed=0,
    )

    failure = artifact.backend_request_events[-1]
    assert failure["kind"] == "upstream_424"
    assert failure["origin"] == "relay"
    assert failure["retryable"] is True
    assert failure["counts_toward_route_circuit"] is True
    packet = RelayPacket.from_artifact(
        artifact,
        recipients=["reviewer"],
        message_id="relay-1",
    )
    assert "backend_request_events" not in packet.to_dict()


def test_routed_executor_marks_422_terminal_without_route_circuit() -> None:
    class InvalidRequestError(RuntimeError):
        status_code = 422

    class InvalidBackend(MockBackend):
        def generate(self, *args, **kwargs):
            raise InvalidRequestError("unprocessable request")

    executor = RoutedModelAgentExecutor(
        {"grok": InvalidBackend()},
        ("grok",),
    )
    artifact = executor.execute(
        task="task",
        node=AgentNode("solver", "solve", metadata={"runtime_route": "grok"}),
        upstream=[],
        peers=[],
        revision=False,
        seed=0,
    )

    assert artifact.answer == "WORKER_BACKEND_FAILURE"
    assert artifact.unresolved_issues == ["terminal_backend_error:InvalidRequestError"]
    failure = artifact.backend_request_events[-1]
    assert failure["kind"] == "invalid_request"
    assert failure["counts_toward_route_circuit"] is False


def test_routed_executor_rejects_missing_solver_choice_with_multiple_models() -> None:
    executor = RoutedModelAgentExecutor(
        {"minimax": MockBackend(), "grok": MockBackend()}, ("minimax", "grok")
    )

    try:
        executor.execute(
            task="task",
            node=AgentNode("unassigned", "solve"),
            upstream=[],
            peers=[],
            revision=False,
            seed=0,
        )
    except ValueError as exc:
        assert "Director did not assign runtime_route" in str(exc)
    else:
        raise AssertionError("missing Solver route must not silently select a model")


def test_aime_react_budget_is_shared_between_initial_and_revision() -> None:
    def tool_call(purpose: str) -> str:
        return json.dumps(
            {
                "tool_calls": [
                    {
                        "name": "python_exec",
                        "arguments": {"purpose": purpose, "code": "print(1)"},
                    }
                ]
            }
        )

    backend = MockBackend(
        [
            tool_call("compute"),
            tool_call("verify"),
            json.dumps({"answer": "1"}),
            tool_call("verify"),
            json.dumps({"answer": "1"}),
        ]
    )
    python_tool = FakePythonTool()
    executor = ModelAgentExecutor(backend, tools={"python_exec": python_tool})
    node = AgentNode(
        "calculator",
        "solve",
        allowed_tools=("python_exec",),
        operation_policy_configured=True,
        initial_tool_budget=2,
        revision_tool_budget=1,
        total_tool_budget=3,
    )

    initial = executor.execute(
        task="problem", node=node, upstream=[], peers=[], revision=False, seed=0
    )
    revision = executor.execute(
        task="problem", node=node, upstream=[], peers=[], revision=True, seed=0
    )

    assert initial.answer == revision.answer == "1"
    assert len(python_tool.calls) == 3
    assert len(initial.react_trace) == 2
    assert len(revision.react_trace) == 1
    assert revision.react_trace[-1]["remaining_budget"] == {"phase": 0, "total": 0}
    assert backend.calls[2]["actions"] == []
    assert backend.calls[4]["actions"] == []


def test_tool_failure_becomes_observation_instead_of_aborting_worker() -> None:
    class BrokenTool(FakePythonTool):
        def execute(self, arguments):
            raise RuntimeError("sandbox failed")

    backend = MockBackend(
        [
            json.dumps(
                {
                    "tool_calls": [
                        {
                            "name": "python_exec",
                            "arguments": {"purpose": "compute", "code": "bad"},
                        }
                    ]
                }
            ),
            json.dumps({"answer": "7"}),
        ]
    )
    executor = ModelAgentExecutor(backend, tools={"python_exec": BrokenTool()})
    node = AgentNode("solver", "solve", allowed_tools=("python_exec",))

    artifact = executor.execute(
        task="problem", node=node, upstream=[], peers=[], revision=False, seed=0
    )

    assert artifact.answer == "7"
    observation = artifact.react_trace[0]["observation"]
    assert observation == {
        "name": "python_exec",
        "status": "error",
        "error": {
            "code": "action_execution_failed",
            "message": "RuntimeError: sandbox failed",
        },
    }
    second_context = backend.calls[1]["messages"][-1]["content"]
    assert "sandbox failed" in second_context


def test_typed_policy_failure_keeps_claimed_confidence_but_effective_is_zero() -> None:
    artifact = AgentArtifact(
        "artifact-alfworld-policy-stall",
        "worker",
        "typed_policy_failure",
        confidence=0.0,
        claimed_confidence=0.95,
        unresolved_issues=["typed_policy_failure:alfworld_semantic_no_progress"],
        raw_response=json.dumps(
            {
                "answer": "continue",
                "confidence": 0.95,
                "unresolved_issues": ["task incomplete"],
            }
        ),
        alfworld_progress={
            "trusted": True,
            "state": "typed_policy_failure",
            "policy_failure": {
                "status": "typed_policy_failure",
                "code": "alfworld_semantic_no_progress",
                "attribution": "model_policy",
            },
        },
    )

    _enforce_artifact_integrity(artifact)

    assert artifact.claimed_confidence == 0.95
    assert artifact.confidence == 0.0
    assert "high_confidence_with_unresolved_issues" in artifact.integrity_risks
    assert artifact.runtime_tool_evidence["claimed_confidence"] == 0.95
    assert artifact.runtime_tool_evidence["effective_confidence"] == 0.0
    assert artifact.runtime_tool_evidence["confidence_caps"]["typed_policy_failure"] == 0.0


def test_webshop_trusted_staged_purchase_survives_truncated_final_json() -> None:
    artifact = AgentArtifact(
        "artifact-webshop-staged",
        "shopper",
        "purchase_staged_with_caution",
        confidence=0.0,
        unresolved_issues=["nonfinal_response_after_action_phase"],
        react_trace=[
            {
                "action": {"call_id": "buy", "name": "webshop_click"},
                "observation": {
                    "status": "ok",
                    "output": {"commit_pending": True, "commit_ready": True},
                },
            }
        ],
        protocol_diagnostics=[
            {
                "stage": "finalization_2",
                "accepted": False,
                "rejection_reason": "missing_final_json",
                "finish_reason": "length",
            }
        ],
        webshop_progress={
            "trusted": True,
            "state": "purchase_staged",
            "commit_ready": True,
            "staged_purchase": {
                "product": {"title": "public fixture"},
                "purchase_evidence_status": {"accepted": True},
            },
        },
    )

    _enforce_artifact_integrity(artifact)

    assert artifact.integrity_risks == []
    assert artifact.runtime_tool_evidence["terminal_protocol_failure_observed"] is True
    assert artifact.runtime_tool_evidence["staged_purchase_protocol_completion_waived"] is True


def test_webshop_untrusted_staged_claim_does_not_waive_protocol_failure() -> None:
    artifact = AgentArtifact(
        "artifact-webshop-untrusted",
        "shopper",
        "purchase_staged_with_caution",
        protocol_diagnostics=[
            {
                "stage": "finalization_2",
                "accepted": False,
                "rejection_reason": "missing_final_json",
            }
        ],
        webshop_progress={
            "trusted": True,
            "state": "purchase_staged",
            "commit_ready": True,
            "staged_purchase": {
                "purchase_evidence_status": {"accepted": False},
            },
        },
    )

    _enforce_artifact_integrity(artifact)

    assert artifact.integrity_risks == ["terminal_protocol_failure"]
