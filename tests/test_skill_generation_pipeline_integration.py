from __future__ import annotations

import json

from selfplay_graph_flowsteer.llm import MockBackend
from selfplay_graph_flowsteer.observability import TaskSpec
from selfplay_graph_flowsteer.skill_evolution_v2 import SkillStore, distill_pending, ingest_rollouts


class _ConstantEmbedder:
    """Deterministic stand-in for E5; it makes the semantic-dedup branch observable."""

    model_path = "offline-test-e5"

    def encode(self, texts, *, query):
        return [(1.0, 0.0) for _ in texts]


def _row(task_id: str, rollout_id: str, reward: float, *, failure: str = "missing_verification"):
    return {
        "task_id": task_id,
        "rollout_id": rollout_id,
        "reward": reward,
        "metadata": {
            "reward_known": True,
            "failure_mode": failure,
            "solver_answer": "public answer",
            "solver_trace": {
                "events": [{"kind": "canvas_step", "action": "check", "output": "public"}],
                "final_graph": {"status": "complete"},
            },
        },
    }


def _proposal(description: str) -> str:
    return json.dumps(
        {
            "name": "Targeted evidence check",
            "description": description,
            "trigger": "Evidence is incomplete",
            "plan": "Assign one narrow check and inspect its public result",
            "pitfall": "Do not restart valid work without a concrete gap",
            "constraint": "Use only public evidence and preserve valid state",
            "kind": "verification",
        }
    )


def test_offline_rollout_to_skill_generation_and_e5_dedup(tmp_path):
    """Exercise the complete production path without model services or formal state."""
    store = SkillStore(tmp_path / "skills.db", embedder=_ConstantEmbedder())
    store.initialize_seeds()
    tasks = {
        task_id: TaskSpec(
            task_id,
            f"public synthetic task {task_id}",
            task_type="qa",
            metadata={"dataset": "synthetic", "source_task_id": task_id},
        )
        for task_id in ("t1", "t2")
    }
    rows = [
        _row("t1", "t1-low", 0.2),
        _row("t1", "t1-high", 0.8),
        _row("t2", "t2-low", 0.1),
        _row("t2", "t2-high", 0.9),
        # A known reward is still excluded when the failure is infrastructure-related.
        _row("t1", "t1-infra", 1.0, failure="worker_backend_failure"),
    ]
    ingest_rollouts(store, tasks, rows, run="offline-run", step=1)

    with store.connect() as db:
        cases = list(db.execute("SELECT id, status, payload FROM cases ORDER BY id"))
    assert len(cases) == 2
    assert all(row[1] == "pending" for row in cases)
    assert all("t1-infra" not in row[2] for row in cases)

    backend = MockBackend(
        [
            _proposal("Inspect the missing evidence before synthesis"),
            _proposal("Inspect the missing evidence before synthesis carefully"),
        ]
    )
    changes = distill_pending(store, backend, step=10, min_pending=1, concurrency=1)
    assert len(changes) == 2
    assert len(backend.calls) == 2
    assert all(call["role"] == "skill-distiller" for call in backend.calls)
    assert all("SECRET" not in call["messages"][1]["content"] for call in backend.calls)

    cards = store.cards()
    candidates = [item for item in cards if item["status"] == "candidate"]
    assert len(candidates) == 1
    assert candidates[0]["provenance"] == "distilled"
    assert len(candidates[0]["source_cases"]) == 2
    with store.connect() as db:
        statuses = dict(db.execute("SELECT status, COUNT(*) FROM cases GROUP BY status"))
        audit_kinds = [row[0] for row in db.execute("SELECT kind FROM audit")]
    assert statuses == {"processed": 2}
    assert "skill_semantic_dedup" in audit_kinds
