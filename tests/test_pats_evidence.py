from __future__ import annotations

import json

from selfplay_graph_flowsteer.contracts import AgentArtifact, AgentNode, Relation, RelationType
from selfplay_graph_flowsteer.observability import (
    ExecutionTrace,
    TaskSpec,
    TraceEvent,
    VerificationResult,
)
from selfplay_graph_flowsteer.pats import collect_evidence
from selfplay_graph_flowsteer.pats_evidence import public_process_trace


def trace(task, *, steps=8, rejected=(), worker=False):
    events = []
    for index in range(steps):
        execution = None
        if worker and index == steps - 1:
            artifact = AgentArtifact(
                artifact_id="artifact-1",
                agent_id="checker",
                answer="PRIVATE_ANSWER_NOT_PROCESS",
                summary="<think>HIDDEN_REASONING</think>Public evidence conflicts; check source dates.",
                react_trace=[
                    {
                        "round_index": i,
                        "action": {
                            "name": "search",
                            "arguments": {"query": f"source-{i}", "reference": "PRIVATE_TOOL_REF"},
                        },
                        "observation": {
                            "status": "ok",
                            "output": f"Public observation {i}",
                            "rubric": "PRIVATE_TOOL_RUBRIC",
                        },
                    }
                    for i in range(5)
                ],
            )
            execution = {
                "artifacts": {"checker": artifact.to_dict()},
                "executed_agents": ["checker"],
            }
        events.append(
            TraceEvent(
                sequence=index,
                kind="canvas_step",
                payload={
                    "raw_action": json.dumps(
                        {
                            "action": "set_prompt",
                            "target": "checker",
                            "objective": "Compare source dates",
                            "reference_answer": "PRIVATE_ACTION_REF",
                        }
                    ),
                    "accepted": index not in rejected,
                    "feedback": "Repeated action changed nothing"
                    if index in rejected
                    else f"Public execution feedback at step {index}",
                    "rejection_code": "duplicate_action" if index in rejected else "",
                    "execution": execution,
                    "private_verifier_payload": {"answer": "PRIVATE_PAYLOAD"},
                },
            )
        )
    events.append(TraceEvent(steps, "verification", {"rubric": "PRIVATE_EVENT"}))
    return ExecutionTrace(
        "trace-run",
        task,
        events,
        {
            "nodes": [
                AgentNode("checker", prompt="Check source dates").to_dict(),
                AgentNode("formatter", prompt="Summarize evidence").to_dict(),
            ],
            "relations": [Relation("checker", "formatter", RelationType.DIRECTED).to_dict()],
            "output_agent": "formatter",
        },
        output="PRIVATE_FINAL_OUTPUT",
        verification=VerificationResult(0, False, "qa", "PRIVATE_VERIFICATION_DETAIL"),
    ).to_dict()


def test_actual_trace_payload_preserves_late_actions_feedback_workers_and_relations():
    task = TaskSpec("task", "Compare public evidence", reference="PRIVATE_TASK_REF")
    result = public_process_trace(trace(task, rejected=(6,), worker=True))
    assert result["summary"]["steps"] == 8
    assert result["omitted_events"] == 0
    assert result["events"][6]["action"]["objective"] == "Compare source dates"
    assert result["events"][6]["accepted"] is False
    assert result["events"][6]["rejection_code"] == "duplicate_action"
    assert result["events"][-1]["feedback"] == "Public execution feedback at step 7"
    worker = result["events"][-1]["workers"][0]
    assert worker["summary"] == "Public evidence conflicts; check source dates."
    assert [step["index"] for step in worker["tool_steps"]] == [0, 2, 4]
    assert worker["tool_steps_total"] == 5
    assert worker["tool_steps_omitted"] == 2
    assert result["final_graph"]["relations"] == [
        {"source": "checker", "target": "formatter", "relation": "directed"}
    ]
    assert "PRIVATE_" not in json.dumps(result)
    assert "HIDDEN_REASONING" not in json.dumps(result)


def test_long_process_covers_start_middle_end_and_summarizes_every_step():
    task = TaskSpec("task", "Compare evidence")
    result = public_process_trace(trace(task, steps=80, rejected=(37, 38, 39)), max_events=12)
    sequences = [event["sequence"] for event in result["events"]]
    assert sequences[0] == 0 and sequences[-1] == 79
    assert 37 in sequences and any(40 < index < 70 for index in sequences)
    assert result["summary"]["steps"] == 80
    assert result["summary"]["rejected_actions"] == 3
    assert result["summary"]["action_counts"] == {"set_prompt": 80}
    assert result["summary"]["repeated_actions"][0]["count"] == 80
    assert result["omitted_events"] == 80 - len(sequences)
    assert len(json.dumps(result, ensure_ascii=False)) <= 10000


def test_collector_compares_all_siblings_and_keeps_actual_frozen_skill_versions():
    task = TaskSpec("task", "Compare source dates", reference="PRIVATE_REFERENCE", task_type="qa")
    manifest = {
        "snapshot_id": "frozen-view",
        "pats_snapshot_id": "policy-aware-view",
        "selected": [{"id": "check-dates", "version": 3}],
        "context": "Private unrelated context is not copied wholesale",
    }
    rows = []
    for i, (reward, steps, rejected) in enumerate(
        ((0, 8, (4, 5, 6)), (1, 5, ()), (0, 12, (6, 7)), (0.5, 6, (2,)))
    ):
        rows.append(
            {
                "rollout_id": f"attempt-{i}",
                "task_id": task.task_id,
                "reward": reward,
                "metadata": {
                    "reward_known": True,
                    "solver_snapshot": "p0",
                    "skill_context": manifest,
                    "solver_trace": trace(task, steps=steps, rejected=rejected),
                },
            }
        )
    evidence = collect_evidence(
        {task.task_id: task}, rows, run="run", step=0, policy_snapshot="p0"
    )[0]
    assert evidence["skill_context"]["selected"] == [{"id": "check-dates", "version": 3}]
    assert evidence["skill_context"]["pats_snapshot_id"] == "policy-aware-view"
    assert evidence["task_goal"] == "Compare source dates"
    contrast = evidence["process_contrast"]
    assert contrast["failure"]["attempts"] == 2
    assert contrast["failure"]["action_counts"] == {"set_prompt": 20}
    assert contrast["failure"]["rejection_counts"] == {"duplicate_action": 5}
    assert contrast["partial"]["attempts"] == 1
    assert contrast["success"]["attempts"] == 1
    assert contrast["success_failure_action_rates"] == [
        {"action": "set_prompt", "success_per_attempt": 5, "failure_per_attempt": 10}
    ]
    assert [e["rollout_id"] for e in evidence["examples"]] == ["attempt-2", "attempt-1"]
    assert "PRIVATE_" not in json.dumps(evidence)
    assert "unrelated context" not in json.dumps(evidence)


def test_nested_json_labels_and_unclosed_reasoning_never_enter_process_evidence():
    task = TaskSpec("task", "Public task")
    raw = trace(task, steps=1)
    raw["events"][0]["payload"]["feedback"] = json.dumps(
        {"status": "public", "rubric": "SECRET_RUBRIC", "answer": "SECRET_ANSWER"}
    )
    raw["events"][0]["payload"]["raw_action"] = (
        '<think>SECRET_REASONING</think>{"action":"finish","private_answer":"SECRET_ACTION"}'
    )
    result = public_process_trace(raw)
    assert result["events"][0]["action"] == {"action": "finish"}
    assert json.loads(result["events"][0]["feedback"]) == {"status": "public"}
    assert "SECRET_" not in json.dumps(result)
    raw["events"][0]["payload"]["feedback"] = "Visible feedback<think>SECRET_UNCLOSED"
    assert public_process_trace(raw)["events"][0]["feedback"] == "Visible feedback"


def test_binary_relation_gate_retains_public_choice_and_endpoints_without_policy_internals():
    raw = trace(TaskSpec("task", "Public task"), steps=1)
    payload = raw["events"][0]["payload"]
    payload["raw_action"] = "on"
    payload["relation_decision"] = {
        "source": "solver",
        "target": "checker",
        "relation_type": "bidirectional",
        "choice": "on",
        "policy": {"token_ids": {"on": 99}, "metadata": "PRIVATE_POLICY"},
    }
    result = public_process_trace(raw)
    assert result["events"][0]["action"] == {
        "action": "relation_gate",
        "choice": "on",
        "source": "solver",
        "target": "checker",
        "relation_type": "bidirectional",
    }
    assert result["summary"]["action_counts"] == {"relation_gate": 1}
    assert "PRIVATE_POLICY" not in json.dumps(result)


def test_oversized_public_process_has_a_hard_bound_with_honest_omission_counts():
    raw = trace(TaskSpec("task", "Public task"), steps=80, rejected=range(80))
    for index, event in enumerate(raw["events"][:-1]):
        event["payload"]["raw_action"] = json.dumps(
            {
                "action": "set_prompt",
                **{
                    key: "long " * 400
                    for key in (
                        "agent_id",
                        "target",
                        "source",
                        "role",
                        "objective",
                        "scope",
                        "expected_output",
                        "prompt",
                        "tools",
                    )
                },
            }
        )
        event["payload"]["rejection_code"] = str(index) + "x" * 100
    result = public_process_trace(raw, max_chars=3000)
    assert len(json.dumps(result, ensure_ascii=False)) <= 3000
    assert result["events"][0]["sequence"] == 0
    assert result["events"][-1]["sequence"] == 79
    assert result["summary"]["rejected_actions"] == 80
    assert (
        sum(result["summary"]["rejection_counts"].values()) + result["summary"]["other_rejections"]
        == 80
    )
    assert result["omitted_events"] == 80 - len(result["events"])
