from __future__ import annotations

import json

import pytest

from selfplay_graph_flowsteer.curriculum import ADSBoundaryScheduler, FixedTaskPool, TSDSRetriever
from selfplay_graph_flowsteer.llm import MockBackend
from selfplay_graph_flowsteer.selfplay import (
    FixedPoolQwenProposer, SelfPlaySeed, _selection_prompt_preview,
)


def flow_prompt(question):
    return (
        "Based on the following passages, answer the question.\n\n"
        "[Article]\n" + "Public evidence sentence. " * 100
        + "\n\nQuestion: " + question
    )


@pytest.mark.parametrize("anchor_source", ["pool", "external"])
def test_hotpot_selector_sees_question_and_solver_keeps_full_task(tmp_path, anchor_source):
    questions = ["Who designed the Silver Observatory?", "When did the Amber Observatory open?"]
    rows = [dict(id=f"hp:{i}", dataset="hotpotqa", split="train",
                 prompt=flow_prompt(q), reference="PRIVATE_ANSWER_SENTINEL",
                 verifier="flowsteer_qa", embedding=[1.0, i / 10])
            for i, q in enumerate(questions)]
    path = tmp_path / "pool.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    pool = FixedTaskPool.from_jsonl([path])
    scheduler = ADSBoundaryScheduler(pool, active_clusters=1, mini_cluster_size=2, cooldown=0)
    backend = MockBackend(['{"candidate_id":"hp:1"}'])
    proposer = FixedPoolQwenProposer(
        backend, pool, scheduler, TSDSRetriever(pool, max_k=2, kde_k=2, sigma=0),
        candidate_count=4,
    )
    proposal = proposer.propose(
        SelfPlaySeed(rows[0]["prompt"], seed_id="hp:0" if anchor_source == "pool" else "external",
                     metadata={"dataset": "hotpotqa"}), task_id="selected",
    )
    payload = json.loads(backend.calls[0]["messages"][1]["content"])
    assert payload["anchor"].startswith("Question: " + questions[0])
    for c in payload["candidates"]:
        q = questions[int(c["candidate_id"].split(":")[1])]
        assert c["prompt_preview"].startswith("Question: " + q)
        assert "[Article]" in c["prompt_preview"]
        assert len(c["prompt_preview"]) <= 500
    assert "PRIVATE_ANSWER_SENTINEL" not in json.dumps(backend.calls[0]["messages"])
    assert proposal.task.prompt == rows[1]["prompt"]
    assert proposal.task.reference == "PRIVATE_ANSWER_SENTINEL"
    assert pool.tasks["hp:1"].task.prompt == rows[1]["prompt"]


@pytest.mark.parametrize("prompt,dataset", [
    ("Who built it?\n\nEvidence:\nQuestion: a quoted heading in evidence", "hotpotqa"),
    (flow_prompt(""), "hotpotqa"),
    (flow_prompt("Who built it?"), "aime"),
])
def test_non_flowsteer_or_missing_question_keeps_original_preview(prompt, dataset):
    assert _selection_prompt_preview(prompt, {}, dataset=dataset, max_chars=500) == prompt[:500]


def test_question_longer_than_budget_still_obeys_preview_limit():
    question = "Long question " * 80
    assert _selection_prompt_preview(flow_prompt(question), {}, dataset="hotpotqa", max_chars=500) == (
        "Question: " + question
    )[:500]
