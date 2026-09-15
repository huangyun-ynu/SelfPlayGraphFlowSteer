from __future__ import annotations

import copy
import json
from dataclasses import asdict, replace

import pytest

from selfplay_graph_flowsteer.application import AdaptiveApplicationConfig
from selfplay_graph_flowsteer.llm import LLMResponse
from selfplay_graph_flowsteer.pats import PatsConfig, PatsController, resolve_scope
from selfplay_graph_flowsteer.pats_semantics import (
    SEMANTIC_REVISION,
    contract_hash,
    semantic_approvals,
)
from selfplay_graph_flowsteer.selfplay_runtime import ByteTokenizer
from selfplay_graph_flowsteer.skill_evolution_v2 import (
    SCHEMA,
    freeze_collection,
    load_bank,
    store_for_config,
)
from selfplay_graph_flowsteer.skills import SkillCard


class Checker:
    def __init__(self, *, reject=(), error=False):
        self.calls = 0
        self.reject = set(reject)
        self.error = error

    def generate_json(self, messages, *, role, schema):
        self.calls += 1
        assert role == "skill-distiller"
        assert schema["additionalProperties"] is False
        if self.error:
            raise RuntimeError("checker request failed")
        cards = json.loads(messages[-1]["content"])["cards"]
        payload = {
            "cards": {
                alias: {
                    "approved": item["card"]["name"] not in self.reject,
                    "reason": "Deterministic fixture verdict for this public card.",
                }
                for alias, item in cards.items()
            }
        }
        return LLMResponse(text=json.dumps(payload), model="fixture-checker")


def card(name, *, human=False, version=1):
    return {
        "card": SkillCard(
            skill_id=name,
            name=name,
            description="Check public evidence",
            trigger="Check evidence before acting",
            plan="Check public evidence within the exposed runtime contract",
            pitfall="Avoid invented state",
            constraint="Respect public controls",
            task_types=["qa"],
        ).to_dict(),
        "version": version,
        "status": "seed" if human else "active",
        "provenance": "human_seed" if human else "pats_scoped_unvalidated",
        "required_tools": [],
        "excluded_task_types": [],
    }


def setup_state(tmp_path, *, scopes=None, max_reviews=2):
    config = replace(
        AdaptiveApplicationConfig(),
        pats=PatsConfig(enabled=True, max_reviews_per_cycle=max_reviews),
        skillbank_enabled=True,
        skillbank_mode=SCHEMA,
        skillbank_training=True,
        skillbank_path=tmp_path / "bank.json",
        skill_cases_path=tmp_path / "cases.json",
        skillbank_embedding_model_path=None,
    )
    scopes = scopes or {"nq_open": [card("seed", human=True), card("good"), card("bad")]}
    state = {
        "config": asdict(config.pats),
        "run": str(tmp_path.resolve()),
        "step": 1,
        "scopes": {
            resolve_scope("qa", {"dataset": dataset}): {
                "cards": records,
                "ema": 0.12,
                "mode": "EXPAND",
                "policy_snapshot": "behavior-1",
            }
            for dataset, records in scopes.items()
        },
    }
    store = store_for_config(config)
    controller = PatsController(store, config.pats, len)
    with store.connect() as db:
        db.execute("INSERT INTO pats_state VALUES(1,?)", (json.dumps(state),))
        db.execute("INSERT INTO pats_cycles VALUES('original',?)", ('{"historical":true}',))
    return config, store, controller, state


def stored_history(store):
    with store.connect() as db:
        state = db.execute("SELECT payload FROM pats_state WHERE id=1").fetchone()[0]
        cycles = list(db.execute("SELECT id,payload FROM pats_cycles ORDER BY id"))
    return state, cycles


def select(bank, dataset="nq_open"):
    return bank.select_context(
        "Check public evidence",
        task_type="qa",
        tokenizer=ByteTokenizer(),
        task_metadata={"dataset": dataset},
    )


def test_new_freeze_semantic_review_is_bounded_audited_and_does_not_rewrite_history(tmp_path):
    config, store, controller, state = setup_state(tmp_path)
    before = stored_history(store)
    checker = Checker(reject={"bad"})
    directory = tmp_path / "cycle2"
    frozen = freeze_collection(
        config, directory, step=2, semantic_backend=checker, tokenizer=ByteTokenizer()
    )
    assert checker.calls == 1 and stored_history(store) == before
    payload = json.loads(frozen.skillbank_path.read_text())
    scope = resolve_scope("qa", {"dataset": "nq_open"})
    assert [record["card"]["skill_id"] for record in payload["pats"]["scopes"][scope]["cards"]] == [
        "seed",
        "good",
    ]
    assert payload["pats"]["scopes"][scope]["semantic_withheld_count"] == 1
    assert payload["pats"]["semantic_gate_revision"] == SEMANTIC_REVISION
    assert payload["pats"]["semantic_contract_sha256"] == contract_hash()
    result = select(load_bank(frozen))
    assert {item.skill_id for item in result[0]} == {"seed", "good"}
    assert result[2]["semantic_gate_revision"] == SEMANTIC_REVISION
    assert result[2]["semantic_contract_sha256"] == contract_hash()
    receipt = json.loads((directory / "pats_semantic_preflight.json").read_text())["attempts"][0]
    assert receipt["source_pats_step"] == 1 and receipt["next_collection_step"] == 2
    assert receipt["semantic_checker_calls"] == 1
    assert receipt["source_state_unchanged"] is True
    assert receipt["pats_state_mutated_by_preflight"] is False
    assert receipt["policy_maintenance_replayed"] is False
    assert receipt["reviews"][0]["approved_count"] == receipt["reviews"][0]["rejected_count"] == 1
    original = frozen.skillbank_path.read_bytes()
    freeze_collection(
        config, directory, step=2, semantic_backend=checker, tokenizer=ByteTokenizer()
    )
    assert checker.calls == 1 and frozen.skillbank_path.read_bytes() == original
    # A second boundary reuses both exact approval and exact rejection without new calls.
    freeze_collection(
        config, tmp_path / "cycle3", step=3, semantic_backend=checker, tokenizer=ByteTokenizer()
    )
    assert checker.calls == 1
    assert not (tmp_path / "cycle3/pats_semantic_preflight.json").exists()


def test_historical_snapshot_replay_keeps_original_cards_and_manifest(tmp_path):
    config, store, controller, state = setup_state(tmp_path)
    directory = tmp_path / "old_cycle"
    directory.mkdir()
    path = directory / "director_skill_snapshot.v2.json"
    legacy = {
        "schema": SCHEMA,
        "snapshot_id": "legacy-bank",
        "cards": [],
        "pats": {
            "snapshot_id": "legacy-pats",
            "config": asdict(config.pats),
            "scopes": state["scopes"],
        },
    }
    path.write_text(json.dumps(legacy))
    original = path.read_bytes()
    checker = Checker(error=True)
    frozen = freeze_collection(
        config, directory, step=2, semantic_backend=checker, tokenizer=ByteTokenizer()
    )
    assert checker.calls == 0 and path.read_bytes() == original
    selected, context, manifest = select(load_bank(frozen))
    assert {item.skill_id for item in selected} == {"seed", "good", "bad"}
    assert "semantic_gate_revision" not in manifest and "semantic_contract_sha256" not in manifest
    assert not (directory / "pats_semantic_preflight.json").exists()
    broken = copy.deepcopy(legacy)
    broken["pats"].update(semantic_gate_revision="unknown", semantic_contract_sha256="a" * 64)
    path.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="unknown PATS semantic gate"):
        load_bank(frozen)
    with pytest.raises(ValueError, match="unknown PATS semantic gate"):
        freeze_collection(config, directory, step=2)


@pytest.mark.parametrize("change", ["version", "content", "scope", "contract"])
def test_live_view_requires_exact_approval_identity_but_frozen_context_is_immutable(
    tmp_path, monkeypatch, change
):
    from selfplay_graph_flowsteer import pats_semantics

    config, store, controller, state = setup_state(tmp_path, scopes={"nq_open": [card("good")]})
    checker = Checker()
    frozen = freeze_collection(
        config, tmp_path / "cycle2", step=2, semantic_backend=checker, tokenizer=ByteTokenizer()
    )
    original = select(load_bank(frozen))
    assert len(original[0]) == 1
    scope = next(iter(state["scopes"]))
    if change == "version":
        state["scopes"][scope]["cards"][0]["version"] += 1
    elif change == "content":
        state["scopes"][scope]["cards"][0]["card"]["plan"] += " changed"
    elif change == "scope":
        state["scopes"] = {resolve_scope("qa", {"dataset": "hotpotqa"}): state["scopes"][scope]}
    else:
        monkeypatch.setattr(pats_semantics, "contract_hash", lambda: "b" * 64)
    with store.connect() as db:
        db.execute("UPDATE pats_state SET payload=? WHERE id=1", (json.dumps(state),))
    live = load_bank(config)
    assert select(live, "hotpotqa" if change == "scope" else "nq_open")[0:2] == ([], "")
    assert select(load_bank(frozen))[1:] == original[1:]


def test_review_budget_withholds_unreviewed_scope_without_seed_fallback(tmp_path):
    config, store, controller, state = setup_state(
        tmp_path,
        scopes={"nq_open": [card("good")], "hotpotqa": [card("other")]},
        max_reviews=1,
    )
    checker = Checker()
    frozen = freeze_collection(
        config, tmp_path / "cycle2", step=2, semantic_backend=checker, tokenizer=ByteTokenizer()
    )
    assert checker.calls == 1
    views = load_bank(frozen)._pats["scopes"]
    assert sorted(len(scope["cards"]) for scope in views.values()) == [0, 1]
    withheld_dataset = next(
        json.loads(scope)[0] for scope, view in views.items() if not view["cards"]
    )
    assert select(load_bank(frozen), withheld_dataset)[0:2] == ([], "")


def test_checker_error_is_audited_without_approval_or_history_mutation(tmp_path):
    config, store, controller, state = setup_state(tmp_path)
    before = stored_history(store)
    checker = Checker(error=True)
    directory = tmp_path / "cycle2"
    frozen = freeze_collection(
        config, directory, step=2, semantic_backend=checker, tokenizer=ByteTokenizer()
    )
    assert checker.calls == 1 and stored_history(store) == before
    scope = next(iter(state["scopes"]))
    assert semantic_approvals(store, scope, state["scopes"][scope]["cards"]) == {}
    assert [item.skill_id for item in select(load_bank(frozen))[0]] == ["seed"]
    receipt = json.loads((directory / "pats_semantic_preflight.json").read_text())["attempts"][0]
    assert receipt["reviews"][0]["status"] == "error"


def test_context_off_and_absent_backend_do_not_call_semantic_checker(tmp_path):
    config, store, controller, state = setup_state(tmp_path)
    checker = Checker(error=True)
    disabled = replace(config, skillbank_usage="off")
    freeze_collection(
        disabled, tmp_path / "disabled", step=2, semantic_backend=checker, tokenizer=ByteTokenizer()
    )
    assert checker.calls == 0
    assert not (tmp_path / "disabled/pats_semantic_preflight.json").exists()
    frozen = freeze_collection(config, tmp_path / "mock_cycle", step=2)
    assert not (tmp_path / "mock_cycle/pats_semantic_preflight.json").exists()
    assert [item.skill_id for item in select(load_bank(frozen))[0]] == ["seed"]
