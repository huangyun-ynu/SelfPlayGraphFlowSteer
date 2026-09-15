from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, replace

import pytest

from selfplay_graph_flowsteer.application import AdaptiveApplicationConfig
from selfplay_graph_flowsteer.pats import PatsConfig, PatsController, resolve_scope
from selfplay_graph_flowsteer.selfplay_runtime import ByteTokenizer
from selfplay_graph_flowsteer.skill_evolution_v2 import (
    SCHEMA,
    DirectorSkillBankV2,
    freeze_collection,
    load_bank,
    store_for_config,
)
from selfplay_graph_flowsteer.skills import SkillCard

TASK_METADATA = {"dataset": "nq_open"}
SCOPE = resolve_scope("qa", TASK_METADATA)


class RankedEmbedder:
    """Use explicit scores to separate ranking from eligibility and rendering."""

    def __init__(self):
        self.model_path = f"ranked-selection-test-{id(self)}"

    def encode(self, texts, *, query):
        if query:
            return [(1.0,) for _ in texts]
        vectors = []
        for text in texts:
            try:
                score = float(text.split(" || ")[0])
            except ValueError:
                score = 0.05  # Unselected real global seeds in the freeze integration test.
            vectors.append((score,))
        return vectors


def record(skill_id, score, *, learned=False, version=1, plan="Check public evidence"):
    return {
        "card": SkillCard(
            skill_id=skill_id,
            name=skill_id,
            description=str(score),
            trigger="An unresolved question",
            plan=plan,
            task_types=["qa"],
        ).to_dict(),
        "version": version,
        "status": "active" if learned else "seed",
        "provenance": "pats_scoped_unvalidated" if learned else "human_seed",
        "required_tools": [],
        "excluded_task_types": [],
    }


def snapshot(records, revision=None):
    pats = {"snapshot_id": "pats-frozen", "scopes": {SCOPE: {"cards": records}}}
    if revision is not None:
        pats["selection_revision"] = revision
    return {
        "schema": SCHEMA,
        "snapshot_id": "frozen",
        "cards": [record("global_seed", 1.0)],
        "pats": pats,
    }


def select(bank, **kwargs):
    return bank.select_context(
        "Question",
        task_type="qa",
        tokenizer=ByteTokenizer(),
        task_metadata=TASK_METADATA,
        **kwargs,
    )


def test_legacy_snapshot_preserves_e5_order_and_exact_manifest():
    records = [record("seed_a", 0.9), record("seed_b", 0.8), record("learned", 0.2, learned=True)]
    selected, context, manifest = select(
        DirectorSkillBankV2(snapshot(records), embedder=RankedEmbedder(), top_k=2)
    )
    assert [card.skill_id for card in selected] == ["seed_a", "seed_b"]
    assert manifest == {
        "schema": SCHEMA,
        "snapshot_id": "frozen",
        "selected": [{"id": "seed_a", "version": 1}, {"id": "seed_b", "version": 1}],
        "prompt_tokens": len(ByteTokenizer().encode(context, add_special_tokens=False)),
        "context": context,
        "pats_enabled": True,
        "pats_scope": SCOPE,
        "pats_snapshot_id": "pats-frozen",
        "context_sha256": hashlib.sha256(context.encode()).hexdigest(),
    }
    explicit = select(
        DirectorSkillBankV2(snapshot(records, "e5_only_v1"), embedder=RankedEmbedder(), top_k=2)
    )
    assert explicit[1:] == (context, manifest)


def test_new_revision_prioritizes_learned_and_updated_seed_with_e5_inside_each_group():
    records = [
        record("seed_high", 0.99),
        record("seed_low", 0.8),
        record("pats_new", 0.1, learned=True),
        record("seed_updated", 0.2, learned=True, version=3),
    ]
    selected, context, manifest = select(
        DirectorSkillBankV2(snapshot(records, "learned_first_v1"), embedder=RankedEmbedder())
    )
    assert [card.skill_id for card in selected] == ["seed_updated", "pats_new", "seed_high"]
    assert "[seed_updated@3]" in context
    assert manifest["selected"][0] == {"id": "seed_updated", "version": 3}
    assert manifest["selection_revision"] == "learned_first_v1"
    assert manifest["prompt_tokens"] <= 1024


def test_learned_priority_retains_all_eligibility_filters_and_score_cutoff():
    records = [record("seed", 0.8), record("valid", 0.3, learned=True)]
    for skill_id, score in [("zero", 0), ("negative", -1), ("below_cutoff", 0.19)]:
        records.append(record(skill_id, score, learned=True))
    wrong_type = record("wrong_type", 1, learned=True)
    wrong_type["card"]["task_types"] = ["coding"]
    excluded = record("excluded", 1, learned=True)
    excluded["excluded_task_types"] = ["qa"]
    tool = record("needs_tool", 1, learned=True)
    tool["required_tools"] = ["search"]
    records.extend([wrong_type, excluded, tool])
    bank = DirectorSkillBankV2(
        snapshot(records, "learned_first_v1"),
        embedder=RankedEmbedder(),
        retrieval_min_score=0.2,
    )
    assert [card.skill_id for card in select(bank)[0]] == ["valid", "seed"]
    assert [card.skill_id for card in select(bank, tools=["search"])[0]] == [
        "needs_tool",
        "valid",
        "seed",
    ]


def test_new_revision_skips_oversized_card_and_backfills_within_actual_token_and_count_budget():
    records = [record("too_long", 1, learned=True, plan="x" * 4000)] + [
        record(f"seed_{i}", 0.9 - i * 0.1) for i in range(4)
    ]
    bank = DirectorSkillBankV2(
        snapshot(records, "learned_first_v1"),
        embedder=RankedEmbedder(),
        top_k=99,
        prompt_token_budget=1024,
    )
    selected, context, manifest = select(bank)
    assert [card.skill_id for card in selected] == ["seed_0", "seed_1", "seed_2"]
    assert (
        len(ByteTokenizer().encode(context, add_special_tokens=False)) == manifest["prompt_tokens"]
    )
    assert manifest["prompt_tokens"] <= 1024
    tiny = DirectorSkillBankV2(
        snapshot(records, "learned_first_v1"),
        embedder=RankedEmbedder(),
        prompt_token_budget=1,
    )
    assert select(tiny)[0:2] == ([], "")
    # The old selector intentionally keeps its historical pre-budget top-k truncation.
    legacy = DirectorSkillBankV2(snapshot(records), embedder=RankedEmbedder())
    assert [card.skill_id for card in select(legacy)[0]] == ["seed_0", "seed_1"]


@pytest.mark.parametrize("revision", [None, "e5_only_v1", "learned_first_v1"])
def test_empty_scope_is_authoritative_for_each_revision(revision):
    selected, context, manifest = select(
        DirectorSkillBankV2(snapshot([], revision), embedder=RankedEmbedder())
    )
    assert selected == [] and context == ""
    assert manifest["selected"] == [] and manifest["prompt_tokens"] == 0
    assert ("selection_revision" in manifest) is (revision == "learned_first_v1")


def test_unknown_revision_is_rejected_before_embedding():
    class UnexpectedEmbedder:
        def encode(self, *args, **kwargs):
            pytest.fail("unknown revision must fail before embedding")

    with pytest.raises(ValueError, match="unknown PATS selection revision"):
        DirectorSkillBankV2(snapshot([], "future_v2"), embedder=UnexpectedEmbedder())


def test_new_cycles_freeze_revision_and_scoped_versions_without_changing_old_resume(
    tmp_path, monkeypatch
):
    from selfplay_graph_flowsteer import pats_semantics

    # This test isolates selector-version compatibility. The semantic admission
    # and real SQLite overlay are exercised separately in test_pats_semantic_freeze.
    monkeypatch.setattr(
        pats_semantics,
        "semantic_approvals",
        lambda store, scope, records: {
            pats_semantics.card_identity(scope, record): True for record in records
        },
    )
    config = replace(
        AdaptiveApplicationConfig(),
        pats=PatsConfig(enabled=True),
        skillbank_enabled=True,
        skillbank_mode=SCHEMA,
        skillbank_training=True,
        skillbank_path=tmp_path / "bank.json",
        skill_cases_path=tmp_path / "cases.json",
        skillbank_embedding_model_path=None,
    )
    store = store_for_config(config)
    controller = PatsController(store, config.pats, len)
    state = {
        "config": asdict(config.pats),
        "run": str(tmp_path.resolve()),
        "step": 0,
        "scopes": {
            SCOPE: {
                "cards": [record("seed", 0.9), record("learned", 0.2, learned=True)],
                "ema": 0.1,
                "policy_snapshot": "p0",
                "mode": "EXPAND",
            }
        },
    }
    with store.connect() as db:
        db.execute("INSERT INTO pats_state(id,payload) VALUES(1,?)", (json.dumps(state),))
    initial = controller.snapshot()
    assert initial["selection_revision"] == "learned_first_v1"
    old = freeze_collection(config, tmp_path / "cycle0", step=1)
    old_payload = json.loads(old.skillbank_path.read_text())
    old_payload["pats"].pop("selection_revision")
    old.skillbank_path.write_text(json.dumps(old_payload))
    old_bytes = old.skillbank_path.read_bytes()
    old_result = select(load_bank(old, embedder=RankedEmbedder()))
    assert [card.skill_id for card in old_result[0]] == ["seed", "learned"]

    updated = copy.deepcopy(state)
    updated["step"] = 1
    updated["scopes"][SCOPE]["cards"][1]["version"] = 2
    with store.connect() as db:
        db.execute("UPDATE pats_state SET payload=? WHERE id=1", (json.dumps(updated),))
    new = freeze_collection(config, tmp_path / "cycle1", step=2)
    new_result = select(load_bank(new, embedder=RankedEmbedder()))
    assert [card.skill_id for card in new_result[0]] == ["learned", "seed"]
    assert new_result[2]["selected"][0] == {"id": "learned", "version": 2}
    assert new_result[2]["selection_revision"] == "learned_first_v1"
    assert initial["snapshot_id"] != controller.snapshot()["snapshot_id"]
    assert (
        freeze_collection(config, tmp_path / "cycle0", step=1).skillbank_path.read_bytes()
        == old_bytes
    )
    assert select(load_bank(old, embedder=RankedEmbedder()))[1:] == old_result[1:]
    # Sibling groups retain the same frozen card versions even after live state changes.
    newer = copy.deepcopy(updated)
    newer["scopes"][SCOPE]["cards"] = []
    with store.connect() as db:
        db.execute("UPDATE pats_state SET payload=? WHERE id=1", (json.dumps(newer),))
    assert select(load_bank(new, embedder=RankedEmbedder()))[1:] == new_result[1:]
    assert select(load_bank(config, embedder=RankedEmbedder()))[0] == []

    broken = json.loads(new.skillbank_path.read_text())
    broken["pats"]["selection_revision"] = "unknown_v0"
    new.skillbank_path.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="unknown PATS selection revision"):
        freeze_collection(config, tmp_path / "cycle1", step=2)
    with pytest.raises(ValueError, match="unknown PATS selection revision"):
        load_bank(new)
