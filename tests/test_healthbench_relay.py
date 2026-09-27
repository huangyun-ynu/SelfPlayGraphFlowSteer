from __future__ import annotations

import json

import pytest

from selfplay_graph_flowsteer.graph import MultiAgentGraph
from selfplay_graph_flowsteer.dataset_actions import default_dataset_action_registry
from selfplay_graph_flowsteer.llm import MockBackend
from selfplay_graph_flowsteer.runtime import ModelAgentExecutor, MultiAgentRuntime


@pytest.mark.parametrize("healthbench", [True, False])
def test_formal_relay_matches_selected_4696_baseline_in_worker_requests(healthbench):
    # The complete-relay experiment remains frozen under its experiment directory.
    # Formal training explicitly selects the earlier unified-contract behavior.
    summary = "Brief internal summary."
    graph = MultiAgentGraph()
    for key in ("a", "b", "out"):
        graph.add_agent(key)
        graph.set_prompt(key, key)
        if healthbench:
            graph.nodes[key].metadata["action_adapter"] = "healthbench_professional"
            graph.nodes[key].operation_policy_configured = True
    graph.set_layer("out", 1)
    graph.set_relation("a", "b", "bidirectional")
    graph.set_relation("a", "out", "directed")
    graph.set_relation("b", "out", "directed")
    graph.set_output("out")
    answers = [f"draft-{i}: " + "完整解释与限定条件。" * 150 + f" END-{i}" for i in range(5)]
    backend = MockBackend([json.dumps({
        "answer": answer, "summary": summary, "confidence": 0.9,
        "evidence": [], "unresolved_issues": [], "tool_summary": [],
    }, ensure_ascii=False) for answer in answers])
    runtime = MultiAgentRuntime(
        ModelAgentExecutor(backend, action_registry=default_dataset_action_registry(())),
        relay_max_chars=1000,
    )
    report = runtime.execute(task="Synthetic public conversation.", graph=graph)
    assert report.worker_model_calls_total == len(backend.calls) == 5
    contexts = [json.loads(call["messages"][1]["content"]) for call in backend.calls]
    packets = [
        (contexts[2]["prior_artifact"], 0),
        (contexts[2]["peer_packets"][0], 1),
        (contexts[3]["prior_artifact"], 1),
        (contexts[3]["peer_packets"][0], 0),
        (contexts[4]["upstream_packets"][0], 2),
        (contexts[4]["upstream_packets"][1], 3),
    ]
    for packet, _ in packets:
        assert packet["answer"] == ""
    assert report.output == answers[4]
    # The original artifacts must remain intact regardless of relay policy.
    assert runtime.artifacts["a"].answer == answers[2]
    assert runtime.artifacts["b"].answer == answers[3]
    for packet in report.packets:
        assert packet.answer == ""
