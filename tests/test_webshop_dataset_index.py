import importlib.util
import json
from pathlib import Path
import random

import pytest

from .test_webshop_public_evidence import begin, assessment

SCRIPT = Path(__file__).parents[1] / "scripts/formal/webshop_dataset_index.py"
spec = importlib.util.spec_from_file_location("webshop_dataset_index", SCRIPT)
index = importlib.util.module_from_spec(spec)
spec.loader.exec_module(index)


def test_raw_to_session_permutation_preserves_duplicates_and_quarantines_splits():
    instructions = [f"public requirement {i}" for i in range(1600)]
    instructions[160] = instructions[1500] = "same public text for distinct release rows"
    order = list(range(len(instructions)))
    random.Random(233).shuffle(order)
    rows = [{"id": f"webshop/goal-{i:05d}", "prompt": instructions[i],
             "cluster_id": "cluster", "metadata": {"goal_id": f"goal-{i:05d}"}}
            for i in range(1600)]
    repaired, quarantine = index.repair_raw_records(rows, instructions, order)
    assert len(repaired) == 100 and len(quarantine) == 1500
    assert not index.validate_records(repaired + quarantine, instructions, order)
    duplicate = [r for r in repaired + quarantine if r["prompt"].startswith("same public")]
    assert len(duplicate) == 2 and len({r["id"] for r in duplicate}) == 2
    assert len({r["metadata"]["webshop_index_repair"]["raw_release_index"] for r in duplicate}) == 2
    assert all(r["cluster_size"] == 100 for r in repaired)
    assert [r["rank_in_cluster"] for r in repaired] == list(range(100))
    with pytest.raises(ValueError, match="provenance"):
        index.repair_raw_records([{**rows[0], "prompt": "wrong"}], instructions, order)


def test_task_instruction_is_not_product_evidence():
    shop, life, state = begin()
    raw = shop.state()
    raw["page_text"] = "Instruction: [SEP] waterproof shirt [SEP] Description: ordinary shirt"
    life._public_evidence.observe(raw)
    source = next(s for s in life._public_evidence.current_sources if s["field"] == "page_text")
    decision = assessment(state)
    decision["candidate"]["checks"][0]["references"] = [{"source_id": source["source_id"], "quote": "waterproof"}]
    with pytest.raises(ValueError, match="observed source"):
        life._public_evidence.updated(decision, raw)


def test_wrong_environment_task_is_closed_before_worker_can_act():
    from selfplay_graph_flowsteer.observability import TaskSpec
    shop, life, _ = begin()
    life.close_all()
    closed = []
    shop.close_session = closed.append
    shop.create_session = lambda *args, **kwargs: {"session_id": "wrong-session", "public_task_statement": "Buy tea"}
    life.bind_task(TaskSpec("task", "Buy a shirt", metadata={"goal_id": "goal-1"}))
    with pytest.raises(ValueError, match="goal index"):
        life.begin_execution(agent_id="owner", seed=0, revision=False)
    assert closed == ["wrong-session"]
    assert life.owner_agent is None and not shop.calls


def test_dispatch_snapshot_keeps_public_projection_and_omits_private_message_text():
    from selfplay_graph_flowsteer.runtime import _webshop_request_public_snapshot
    messages = [
        {"role": "system", "content": "system instructions"},
        {"role": "user", "content": json.dumps({"public_task_context": "Buy a shirt",
         "action_environment": {"state": {"page_type": "product", "product": {"asin": "A"},
          "selected_options": {"color": "blue"}, "arbitrary_private_value": "fixture secret"},
          "remaining": {"total": 3}}})},
        {"role": "assistant", "content": "private reasoning fixture"},
    ]
    result = _webshop_request_public_snapshot(messages, interaction_round=1, request_events=[{"event_id": "event"}])
    serialized = json.dumps(result)
    assert result["projection_found"] and result["state"]["selected_options"] == {"color": "blue"}
    assert result["backend_event_ids"] == ["event"] and len(result["messages_sha256"]) == 64
    assert "fixture secret" not in serialized and "private reasoning fixture" not in serialized
