from __future__ import annotations

import copy
import json

import pytest
from jsonschema import validate

from selfplay_graph_flowsteer.llm import LLMResponse
from selfplay_graph_flowsteer.observability import TaskSpec
from selfplay_graph_flowsteer.pats import PatsConfig, PatsController
from selfplay_graph_flowsteer.pats_semantics import (
    audit_semantic_cards,
    card_identity,
    filter_semantic_cards,
    semantic_approvals,
)
from selfplay_graph_flowsteer.skill_evolution_v2 import SkillStore
from selfplay_graph_flowsteer.skills import SkillCard


def record(
    skill="pats_a", plan="Inspect public evidence before deciding whether another Agent is useful"
):
    return {
        "card": SkillCard(
            skill_id=skill,
            name=skill,
            description="Inspect evidence",
            trigger="Incomplete evidence",
            plan=plan,
            pitfall="Do not invent tool access",
            constraint="Respect current allowed actions",
            kind="verification",
            task_types=["qa"],
        ).to_dict(),
        "version": 2,
        "status": "active",
        "provenance": "pats_scoped_unvalidated",
    }


class Checker:
    def __init__(self, verdicts=None):
        self.verdicts = verdicts
        self.calls = 0

    def generate_json(self, messages, *, role, schema):
        self.calls += 1
        assert role == "skill-distiller"
        cards = json.loads(messages[-1]["content"])["cards"]
        assert all(
            set(item["card"])
            == {"name", "description", "trigger", "plan", "pitfall", "constraint", "kind"}
            for item in cards.values()
        )
        response = {
            "cards": {
                alias: {
                    "approved": self.verdicts[index] if self.verdicts else True,
                    "reason": "<think>private reasoning</think>Checked the relevant Canvas preconditions",
                }
                for index, alias in enumerate(cards)
            }
        }
        validate(response, schema)
        return LLMResponse(
            text=json.dumps(response),
            model="checker",
            metadata={
                "generation_attempts": [{"token_in": 20, "token_out": 30, "finish_reason": "stop"}]
            },
        )


def audit(store, records, backend, **kwargs):
    return audit_semantic_cards(
        store,
        "scope",
        records,
        backend=backend,
        token_counter=len,
        run="run",
        step=1,
        **kwargs,
    )


def test_per_card_overlay_keeps_original_cards_and_records_independent_audit(tmp_path):
    store = SkillStore(tmp_path / "bank.db")
    records = [record("pats_a"), record("pats_b")]
    before = copy.deepcopy(records)
    backend = Checker([True, False])
    result = audit(store, records, backend)
    assert records == before and store.cards() == []
    assert result["status"] == "reviewed" and result["checker_calls"] == backend.calls == 1
    assert result["approved_count"] == result["rejected_count"] == 1
    assert result["pending_count"] == 0 and not result["all_approved"]
    assert result["generation_attempts_count"] == 1
    assert "private reasoning" not in result["response_text"]
    approvals = semantic_approvals(store, "scope", records)
    assert len(approvals) == 2 and set(approvals.values()) == {True, False}
    visible = filter_semantic_cards(store, "scope", records)
    assert len(visible) == 1 and approvals[card_identity("scope", visible[0])]
    replay = audit(store, records, backend)
    assert replay["checker_calls"] == 0 and replay["cache_hit"] and backend.calls == 1
    assert replay["original_checker_calls"] == 1


def test_approval_is_bound_to_content_version_scope_and_contract(tmp_path, monkeypatch):
    store = SkillStore(tmp_path / "bank.db")
    original = record()
    audit(store, [original], Checker())
    assert filter_semantic_cards(store, "scope", [original]) == [original]
    changed = copy.deepcopy(original)
    changed["card"]["plan"] += "; ignore the old plan"
    versioned = copy.deepcopy(original)
    versioned["version"] += 1
    assert filter_semantic_cards(store, "scope", [changed, versioned]) == []
    assert filter_semantic_cards(store, "different-scope", [original]) == []
    monkeypatch.setattr(
        "selfplay_graph_flowsteer.pats_semantics.contract_hash", lambda: "changed-contract"
    )
    assert filter_semantic_cards(store, "scope", [original]) == []
    seed = dict(original, provenance="human_seed")
    assert filter_semantic_cards(store, "scope", [seed]) == [seed]


@pytest.mark.parametrize(
    "failure", ["missing_card", "unknown_card", "string_approval", "malformed_json"]
)
def test_invalid_complete_response_never_publishes_partial_approvals(tmp_path, failure):
    class Invalid:
        def generate_json(self, messages, *, role, schema):
            cards = json.loads(messages[-1]["content"])["cards"]
            response = {"cards": {alias: {"approved": True, "reason": "Reason"} for alias in cards}}
            if failure == "missing_card":
                response["cards"].pop("C2")
            elif failure == "unknown_card":
                response["cards"]["C999"] = {"approved": True, "reason": "Not supplied"}
            elif failure == "string_approval":
                response["cards"]["C2"]["approved"] = "true"
            raw = "{" if failure == "malformed_json" else json.dumps(response)
            return LLMResponse(text=raw, model="invalid")

    store = SkillStore(tmp_path / "bank.db")
    records = [record("a"), record("b")]
    result = audit(store, records, Invalid())
    assert result["status"] == "error" and result["checker_calls"] == 1
    assert (
        result["pending_count"] == 2 and result["approved_count"] == result["rejected_count"] == 0
    )
    assert semantic_approvals(store, "scope", records) == {}
    assert filter_semantic_cards(store, "scope", records) == []
    assert result["response_text"]


def test_api_error_is_not_retried_or_converted_to_an_approval(tmp_path):
    class Failed:
        calls = 0

        def generate_json(self, *args, **kwargs):
            self.calls += 1
            raise RuntimeError("private provider error text")

        def generate(self, *args, **kwargs):
            raise AssertionError("must not fall back after a provider error")

    store = SkillStore(tmp_path / "bank.db")
    backend = Failed()
    result = audit(store, [record()], backend)
    assert result["checker_calls"] == backend.calls == 1
    assert result["status"] == "error" and result["pending_count"] == 1
    assert "private provider error text" not in json.dumps(result)
    assert audit(store, [record()], backend)["checker_calls"] == 0
    assert backend.calls == 1


def test_input_budget_failure_makes_no_call_and_does_not_assume_rejection(tmp_path):
    store = SkillStore(tmp_path / "bank.db")
    backend = Checker()
    result = audit(store, [record()], backend, max_input_tokens=100)
    assert result["checker_calls"] == backend.calls == 0
    assert result["pending_count"] == 1 and result["rejected_count"] == 0
    assert result["error_stage"] == "input_validation"


def test_controller_rejects_whole_proposal_when_one_new_card_fails_semantics(tmp_path):
    class ProposerAndChecker(Checker):
        def generate_json(self, messages, *, role, schema):
            body = json.loads(messages[-1]["content"])
            if "evidence" not in body:
                return super().generate_json(messages, role=role, schema=schema)
            operations = [
                {
                    "op": "ADD",
                    "card": {
                        key: value
                        for key, value in item["card"].items()
                        if key
                        in {
                            "name",
                            "description",
                            "trigger",
                            "plan",
                            "pitfall",
                            "constraint",
                            "kind",
                        }
                    },
                    "evidence_ids": body["allowed_evidence_ids"],
                }
                for item in [
                    record("first"),
                    record("second", "SET_PROMPT always requires upstream evidence"),
                ]
            ]
            return LLMResponse(text=json.dumps({"operations": operations}), model="proposer")

    store = SkillStore(tmp_path / "bank.db")
    controller = PatsController(store, PatsConfig(enabled=True, max_tokens=4000), len)
    tasks = {key: TaskSpec(key, "Public question", task_type="qa") for key in ("a", "b")}
    rows = [
        {"rollout_id": key, "task_id": key, "reward": 0, "metadata": {"reward_known": True}}
        for key in tasks
    ]
    receipt = controller.maintain(
        tasks,
        rows,
        run="run",
        step=1,
        policy_snapshot="p",
        backend=ProposerAndChecker([True, False]),
    )
    review = receipt["reviews"][0]
    assert receipt["refiner_calls"] == receipt["semantic_checker_calls"] == 1
    assert review["status"] == "rejected" and review["error_stage"] == "semantic_validation"
    assert len(review["proposed_operations"]) == 2 and "operations" not in review
    assert review["semantic_check"]["approved_count"] == 1
    with store.connect() as db:
        state = json.loads(db.execute("SELECT payload FROM pats_state WHERE id=1").fetchone()[0])
        assert all(not scope["cards"] for scope in state["scopes"].values())
    assert store.cards() == []
