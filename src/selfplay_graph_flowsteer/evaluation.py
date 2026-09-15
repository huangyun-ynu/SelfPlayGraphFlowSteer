from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .application import AdaptiveApplicationResult


@dataclass(frozen=True)
class EvaluationRecord:
    task_id: str
    system: str
    answer: str
    score: float
    passed: bool
    seed: int = 0
    token_cost: int = 0
    checkpoint: str = ""
    duration_s: float = 0.0
    trajectory: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def from_adaptive_result(
    result: AdaptiveApplicationResult, *, seed: int = 0, checkpoint: str = ""
) -> EvaluationRecord:
    verification = result.solver_result.verification
    payload = result.to_dict()
    solver_payload = result.solver_result.to_dict()
    trajectory = dict(solver_payload["trace"])
    trajectory.update(
        {
            "director_run": solver_payload["director_run"],
            "flowsteer_structure": solver_payload["flowsteer_structure"],
            "skills_used": solver_payload["skills_used"],
            "skill_context": solver_payload["skill_context"],
            "answer_submission": solver_payload["answer_submission"],
        }
    )
    return EvaluationRecord(
        task_id=result.task.task_id,
        system="selfplay_graph_flowsteer",
        answer=str(payload["answer"]),
        score=verification.score if verification else 0.0,
        passed=verification.passed if verification else False,
        seed=seed,
        token_cost=int(payload["token_in"]) + int(payload["token_out"]),
        checkpoint=checkpoint,
        duration_s=0.0,
        trajectory=trajectory,
    )


def from_flowsteer_trajectory(payload: dict[str, Any]) -> EvaluationRecord:
    """Normalize a FlowSteer evaluation/trajectory JSON object for paired comparison."""

    score = float(payload.get("score", payload.get("reward", 0.0)))
    return EvaluationRecord(
        task_id=str(payload.get("task_id", payload.get("id", ""))),
        system="flowsteer",
        answer=str(payload.get("answer", payload.get("output", ""))),
        score=score,
        passed=bool(payload.get("passed", score >= 0.5)),
        seed=int(payload.get("seed", 0)),
        token_cost=int(payload.get("token_cost", payload.get("tokens", 0))),
        checkpoint=str(payload.get("checkpoint", "")),
        duration_s=float(payload.get("duration_s", payload.get("latency", 0.0))),
        trajectory=dict(payload),
    )
