"""SWE resource admission must precede every new workspace/model execution."""

import json

import pytest

from selfplay_graph_flowsteer.canvas import GraphCanvas
from selfplay_graph_flowsteer.config import CanvasConfig
from selfplay_graph_flowsteer.contracts import CodeArtifactRef
from selfplay_graph_flowsteer.dataset_actions import default_dataset_action_registry
from selfplay_graph_flowsteer.graph import MultiAgentGraph
from selfplay_graph_flowsteer.llm import MockBackend
from selfplay_graph_flowsteer.runtime import (
    ActionBudgetLedger,
    AgentActionUsage,
    ModelAgentExecutor,
    MultiAgentRuntime,
)

from .helpers import RecordingExecutor
from .test_runtime import FakeNamedTool
from .test_unified_submission import add, prompt, step

SCOPE = "tool-rollout:whole-graph"
TOOLS = ("swe_list", "swe_search", "swe_read", "swe_edit", "swe_apply_artifact", "swe_test", "swe_status")


class BudgetedExecutor(RecordingExecutor):
    def __init__(self, *, used=0, charge=True):
        super().__init__()
        self.budget_ledger = ActionBudgetLedger({SCOPE: AgentActionUsage(total_used=used)})
        self.budget_scope = None
        self.charge = charge

    def set_budget_scope(self, scope):
        self.budget_scope = scope

    def execute(self, **kwargs):
        if self.charge:
            assert self.budget_ledger.consume(
                kwargs["node"], revision=kwargs["revision"], scope=self.budget_scope
            )[0]
        return super().execute(**kwargs)


def make_canvas(tmp_path, executor):
    tools = {name: FakeNamedTool(name) for name in TOOLS}
    registry = default_dataset_action_registry(
        tools, swe_budgets=(24, 8, 32), action_budget_policy="shared_total_v1"
    )
    return GraphCanvas(
        task="Fix the repository", dataset="swe_bench", runtime=MultiAgentRuntime(executor),
        action_adapter=registry.get("swe_bench"),
        config=CanvasConfig(
            submission_protocol="unified_task_result_v1", director_budget_policy="edits_v1",
            max_director_edits=24, submission_journal_dir=str(tmp_path),
        ),
    )


def graph_with_agents(*agents, limit=32):
    graph = MultiAgentGraph(submission_protocol="unified_task_result_v1")
    for agent_id in agents:
        graph.add_agent(agent_id)
        graph.set_prompt(agent_id, "Inspect assigned module " + agent_id)
        node = graph.nodes[agent_id]
        node.allowed_tools = ("swe_read",)
        node.operation_policy_configured = True
        node.initial_tool_budget = node.total_tool_budget = limit
        node.revision_tool_budget = 0
        node.metadata.update(
            submission_protocol="unified_task_result_v1", result_scope="subtask",
            action_adapter="swe_bench",
            dataset_capability_policy={"action_budget_policy": "shared_total_v1"},
        )
    return graph


@pytest.mark.parametrize("scope", ["subtask", "task_result"])
def test_zero_tools_blocks_new_configurations_edits_and_explicit_recovery(tmp_path, scope):
    executor = BudgetedExecutor(used=32)
    canvas = make_canvas(tmp_path, executor)
    add(canvas, "a", scope)
    blocked = canvas.history[-1]
    assert blocked.accepted  # The configuration itself is still a real graph edit.
    assert blocked.execution.blocked_agents == {"a": "swe_tool_budget_exhausted"}
    assert blocked.execution.to_dict()["blocked_agents"] == blocked.execution.blocked_agents
    assert blocked.remaining_dirty == ["a"]
    assert "swe_tool_budget_exhausted" in blocked.feedback
    assert not canvas._unified_can_run("a")
    assert "run_agent" not in canvas.control_snapshot()["allowed_actions"]
    assert not step(canvas, {"action": "run_agent", "target": "a"}).accepted
    revised = prompt("a", scope)
    revised["objective"] = "Check another requirement"
    assert step(canvas, revised).accepted
    assert step(canvas, {"action": "delete_agent", "target": "a"}).accepted
    add(canvas, "a", scope)
    assert not executor.calls and not canvas.runtime.artifacts
    assert canvas.total_tokens == 0
    assert canvas.control_snapshot()["action_budget"]["used"] == 32
    assert canvas.dirty_agents == {"a"}


@pytest.mark.parametrize("used,expected_calls", [(31, 1), (30, 2), (29, 3)])
def test_budget_can_end_during_initial_or_synchronous_revision_wave(used, expected_calls):
    graph = graph_with_agents("a", "b", "downstream")
    graph.set_relation("a", "b", "bidirectional")
    graph.set_layer("downstream", 1)
    graph.set_relation("a", "downstream", "directed")
    executor = BudgetedExecutor(used=used)
    runtime = MultiAgentRuntime(executor, bidirectional_revision_policy="always")
    report = runtime.execute(task="Fix the repository", graph=graph)
    assert len(executor.calls) == expected_calls
    assert report.worker_model_calls_total == expected_calls
    assert set(report.blocked_agents) == {"a", "b", "downstream"}
    assert "downstream" not in report.artifacts
    assert set(runtime.artifacts) <= {"a", "b"}
    assert not any(runtime.artifact_matches_current_input_signature(
        key, task="Fix the repository", graph=graph
    ) for key in graph.nodes)
    assert report.incomplete_bidirectional_components[-1]["reason"] == "execution_admission_blocked"
    assert runtime.shared_tool_budget_status(32)["remaining"] == 0


@pytest.mark.parametrize("dirty", [False, True])
def test_zero_tools_also_blocks_existing_failure_recovery(tmp_path, dirty):
    executor = BudgetedExecutor(used=31)
    canvas = make_canvas(tmp_path, executor)
    add(canvas, "a", "task_result")
    artifact = canvas.runtime.artifacts["a"]
    artifact.integrity_risks = ["terminal_protocol_failure"]
    if dirty:
        canvas.dirty_agents.add("a")
    assert not canvas._unified_can_run("a")
    assert not step(canvas, {"action": "run_agent", "target": "a"}).accepted
    assert len(executor.calls) == 1
    assert canvas._unified_recovery_used == 0
    assert canvas.runtime.artifacts["a"] is artifact


def test_unchanged_cache_can_be_reused_after_tools_run_out():
    graph = graph_with_agents("a")
    executor = BudgetedExecutor(used=31)
    runtime = MultiAgentRuntime(executor)
    first = runtime.execute(task="Fix the repository", graph=graph)
    artifact = first.artifacts["a"]
    # Restore from the same cached input rather than opening another workspace.
    runtime.artifacts.clear()
    cached = runtime.execute(task="Fix the repository", graph=graph)
    assert len(executor.calls) == 1
    assert cached.cache_hits == 1 and not cached.blocked_agents
    assert cached.artifacts["a"] is artifact
    assert runtime.artifact_matches_current_input_signature("a", task="Fix the repository", graph=graph)


def test_discovered_stale_node_can_use_last_tool_before_explicit_target(tmp_path):
    executor = BudgetedExecutor(charge=False)
    canvas = make_canvas(tmp_path, executor)
    add(canvas, "a", "subtask")
    add(canvas, "b", "task_result")
    executor.budget_ledger.usage[SCOPE].total_used = 31
    executor.charge = True
    canvas.runtime._stale_artifacts.add("a")
    canvas.runtime.cache.clear()
    before = len(executor.calls)
    result = step(canvas, {"action": "run_agent", "target": "b"})
    assert len(executor.calls) == before + 1
    assert result.execution.executed_agents == ["a"]
    assert result.execution.blocked_agents == {"b": "swe_tool_budget_exhausted"}
    assert result.remaining_dirty == ["b"]
    assert "Continuation for b was blocked" in result.feedback
    assert not canvas._unified_can_run("b")


def test_new_dependency_keeps_old_candidate_but_does_not_mark_it_current(tmp_path):
    executor = BudgetedExecutor(used=30)
    canvas = make_canvas(tmp_path, executor)
    add(canvas, "a", "subtask")
    add(canvas, "b", "task_result")
    candidate = canvas.runtime.artifacts["b"]
    candidate.code_artifact_ref = CodeArtifactRef("a" * 64, "fixture-1", "fixture/repo", "b" * 40, 40, ("sample.py",))
    candidate.swe_progress = {"trusted": True, "commit_ready": True, "test_after_latest_edit": True}
    assert canvas.submission_assessment("b")["blockers"] == ["graph:agents cannot influence output: a"]
    changed = step(canvas, {"action": "set_relation", "source": "a", "target": "b", "relation": "bidirectional"})
    assert changed.accepted
    assert len(executor.calls) == 2
    assert canvas.runtime.artifacts["b"] is candidate
    assert canvas.dirty_agents == {"a", "b"}
    assert "stale_artifact" in canvas.submission_assessment("b")["blockers"]
    assert not step(canvas, {"action": "finish", "target": "b"}, True).accepted


def test_in_flight_worker_can_finish_text_after_using_last_tool():
    class Lifecycle:
        def __init__(self):
            self.starts = self.ends = 0

        def begin_execution(self, **kwargs):
            self.starts += 1
            return {"workspace_version": 0}

        def end_execution(self):
            self.ends += 1

        def result_for(self, agent_id):
            return {}

    tool = FakeNamedTool("swe_read")
    tool.lifecycle = Lifecycle()
    tools = {name: FakeNamedTool(name) for name in TOOLS}
    tools[tool.name] = tool
    registry = default_dataset_action_registry(tools, swe_budgets=(1, 0, 1), action_budget_policy="shared_total_v1")
    backend = MockBackend([
        json.dumps({"action_call": {"name": "swe_read", "arguments": {"path": "sample.py"}}}),
        json.dumps({"answer": "Located the function; modification remains necessary"}),
    ])
    runtime = MultiAgentRuntime(ModelAgentExecutor(backend, tools=tools, action_registry=registry))
    graph = graph_with_agents("a", "b", limit=1)
    for node in graph.nodes.values():
        node.allowed_tools = TOOLS
    report = runtime.execute(task="Fix the repository", graph=graph)
    assert len(tool.calls) == 1
    assert tool.lifecycle.starts == tool.lifecycle.ends == 1
    assert report.artifacts["a"].answer == "Located the function; modification remains necessary"
    assert len(backend.calls) == 2
    assert report.blocked_agents == {"b": "swe_tool_budget_exhausted"}
    assert report.executed_agents == ["a"]
