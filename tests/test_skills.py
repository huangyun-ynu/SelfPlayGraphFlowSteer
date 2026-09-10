from __future__ import annotations

import json

from selfplay_graph_flowsteer.llm import MockBackend
from selfplay_graph_flowsteer.skills import (
    SESASolverSkillDistiller,
    SkillCard,
    SolverFailureCase,
    SolverSkillBank,
    SolverSkillLifecycle,
)


def card(skill_id: str, plan: str) -> SkillCard:
    return SkillCard(
        skill_id=skill_id,
        name="Verify unresolved evidence",
        description="Repair workflows that synthesize before checking unresolved evidence.",
        trigger="Use when agent artifacts contain unresolved evidence gaps.",
        plan=plan,
        pitfall="Do not invent evidence.",
        kind="bug-repair",
        task_types=["qa"],
    )


class _SemanticEmbedder:
    def encode(self, texts: list[str], *, query: bool) -> list[tuple[float, ...]]:
        del query
        return [(1.0, 0.0) if "alpha" in text else (0.0, 1.0) for text in texts]


def test_solver_skillbank_deduplicates_tracks_outcomes_and_persists(tmp_path) -> None:
    path = tmp_path / "solver_skills.json"
    bank = SolverSkillBank(path, dedup_threshold=0.5)
    skill_id, status = bank.add_or_deduplicate(card("solver_000", "Inspect evidence then verify."))
    assert status == "retained"
    duplicate_id, duplicate_status = bank.add_or_deduplicate(
        card("solver_001", "Inspect every evidence packet, resolve gaps, then verify the output.")
    )
    assert (duplicate_id, duplicate_status) == (skill_id, "deduplicated")
    bank.record_outcome(skill_id, step=1, helpful=True)
    assert bank.retrieve("unresolved evidence", task_type="qa")[0].skill_id == skill_id
    loaded = SolverSkillBank(path)
    assert loaded.skills[skill_id].stats.helpful_count == 1


def test_solver_skillbank_uses_e5_style_vectors_for_retrieval_and_deduplication() -> None:
    bank = SolverSkillBank(embedder=_SemanticEmbedder(), dedup_threshold=0.9)
    alpha = card("solver_000", "alpha workflow")
    beta = card("solver_001", "beta workflow")
    assert bank.add_or_deduplicate(alpha)[1] == "retained"
    assert bank.add_or_deduplicate(beta)[1] == "retained"
    assert bank.retrieve("alpha question")[0].skill_id == "solver_000"
    assert bank.add_or_deduplicate(card("solver_002", "alpha alternative")) == (
        "solver_000",
        "deduplicated",
    )


def test_sesa_lifecycle_distills_an_informative_frontier_failure() -> None:
    response = json.dumps(
        {
            "name": "Repair missing verification",
            "description": "Add a verification pass when the first answer is unsupported.",
            "trigger": "The Solver output has unresolved evidence.",
            "plan": "Inspect the unresolved issue, ask a verifier agent to test the claim, and synthesize only after the evidence is resolved.",
            "pitfall": "Do not expose this Solver skill to the Proposer.",
            "constraint": "Use only trace-visible evidence.",
            "kind": "verification",
        }
    )
    bank = SolverSkillBank()
    lifecycle = SolverSkillLifecycle(bank, SESASolverSkillDistiller(MockBackend([response])))
    lifecycle.collect_failure(
        SolverFailureCase(
            task="q",
            task_type="qa",
            failure_trace="synthesized too early",
            failure_mode="unresolved_evidence_gap",
            evidence_refs=("artifact_1",),
            frontier_score=0.5,
        )
    )
    changes = lifecycle.evolve(step=10, force=True)
    assert changes == [("solver_000", "retained")]
    assert bank.skills["solver_000"].kind == "verification"


def test_lifecycle_waits_for_sesa_periodic_consolidation_boundary() -> None:
    response = json.dumps(
        {
            "name": "Recover failed branch",
            "description": "Use the successful branch pattern after a matching failure.",
            "trigger": "A Solver branch collapses before producing evidence.",
            "plan": "Compare the failed branch with the successful trace, restore the missing evidence step, and verify the result before synthesis.",
            "kind": "bug-repair",
        }
    )
    bank = SolverSkillBank()
    lifecycle = SolverSkillLifecycle(bank, SESASolverSkillDistiller(MockBackend([response])))
    lifecycle.collect_failure(
        SolverFailureCase("failed task", "qa", "failure", "branch_collapse", frontier_score=0.5)
    )
    assert lifecycle.evolve(step=1) == []
    assert lifecycle.evolve(step=2) == []
    assert not bank.skills


def test_lifecycle_deduplicates_resumed_failure_by_rollout_uid() -> None:
    lifecycle = SolverSkillLifecycle(SolverSkillBank(), SESASolverSkillDistiller(MockBackend()))
    failure = SolverFailureCase(
        "failed task",
        "qa",
        "failure",
        "branch_collapse",
        frontier_score=0.5,
        uid="task-1-r0",
    )

    lifecycle.collect_failure(failure)
    lifecycle.collect_failure(failure)

    assert [case.uid for case in lifecycle.failures] == ["task-1-r0"]
