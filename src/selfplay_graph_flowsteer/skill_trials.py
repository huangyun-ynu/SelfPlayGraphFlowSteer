"""Explicit, bounded development-only on/off trials through the formal Solver.

This entry point uses already served, frozen models. It never starts services or
updates weights. Do not use a service whose adapter is being changed by training.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from .skill_evolution_v2 import (
    SCHEMA,
    DirectorSkillBankV2,
    SkillStore,
    atomic_json,
    store_for_config,
)
from .skills import SkillCard


class TrialContextBank(DirectorSkillBankV2):
    """Hold other selected cards constant; add only the one target in the on arm."""

    def __init__(self, snapshot, candidate, *, enabled, prompt_token_budget, embedder=None):
        super().__init__(
            snapshot, embedder=embedder, top_k=2, prompt_token_budget=prompt_token_budget
        )
        self.candidate = candidate
        self.enabled = enabled

    def select_context(self, query, *, task_type, tokenizer, tools=()):
        # Reserve the target's space in BOTH arms, preventing another skill from
        # being displaced only on the treatment side.
        chosen, _, _ = super().select_context(
            query, task_type=task_type, tokenizer=tokenizer, tools=tools
        )
        target = SkillCard.from_dict(self.candidate["card"])
        if target.task_types and task_type not in target.task_types:
            raise ValueError("trial task outside candidate applicability")
        if task_type in self.candidate.get("excluded_task_types", []) or not set(
            self.candidate.get("required_tools", [])
        ) <= set(tools):
            raise ValueError("trial tool/task applicability mismatch")
        self.records[target.skill_id] = self.candidate
        while (
            chosen
            and len(
                tokenizer.encode(
                    self.format_prompt_context([*chosen, target]), add_special_tokens=False
                )
            )
            > self.prompt_token_budget
        ):
            chosen.pop()
        if (
            len(
                tokenizer.encode(
                    self.format_prompt_context([*chosen, target]), add_special_tokens=False
                )
            )
            > self.prompt_token_budget
        ):
            raise ValueError("target skill cannot fit trial token budget")
        selected = [*chosen, target] if self.enabled else chosen
        context = self.format_prompt_context(selected)
        return (
            selected,
            context,
            {
                "schema": SCHEMA,
                "snapshot_id": self.snapshot_id,
                "selected": [
                    {"id": c.skill_id, "version": self.records[c.skill_id]["version"]}
                    for c in selected
                ],
                "context": context,
                "prompt_tokens": len(tokenizer.encode(context, add_special_tokens=False)),
            },
        )


def run_application_trial(
    config,
    tasks,
    *,
    store: SkillStore,
    skill_id: str,
    version: int,
    output: Path,
    budget: float = 900,
    min_tasks: int = 5,
    minimum_mean_gain: float = 0,
    mock: bool = False,
):
    from .application import create_adaptive_application
    from .deadline import RolloutDeadline
    from .selfplay_runtime import ByteTokenizer, HuggingFaceTokenizer, adaptive_result_to_rollout
    from .skills import E5SkillEmbedder

    if not 0 < budget <= 900:
        raise ValueError("trial arm budget must be in (0,900]")
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("trial task IDs must be unique")
    if any(
        task.metadata.get("skill_evaluation_split") != "development"
        or task.metadata.get("is_final_test") is True
        for task in tasks
    ):
        raise ValueError("every task must be explicitly marked skill_evaluation_split=development")
    # A first trial may be the first v2 use of this state directory. Initialize
    # seeds here so a seed skill can itself be evaluated without a prior cycle.
    store.initialize_seeds()
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "trial.json"
    # Persist public configuration rather than credentials. Serving must stay frozen;
    # matching configuration alone does not attest remote parameter immutability.
    conditions = {
        "solver_snapshot": config.solver_model.to_dict(),
        "worker_config": config.model_manifest(),
        "budget": budget,
        "canvas_config": asdict(config.canvas),
    }
    conditions = json.loads(json.dumps(conditions))
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if (
            saved["conditions"] != conditions
            or saved["skill_id"] != skill_id
            or saved["version"] != version
            or saved["task_ids"] != [t.task_id for t in tasks]
            or saved["mock"] != mock
        ):
            raise ValueError("trial resume differs from frozen protocol")
        trial_id = saved["trial_id"]
    else:
        candidates = [
            c
            for c in store.cards()
            if c["card"]["skill_id"] == skill_id and c["version"] == version
        ]
        if not candidates:
            raise ValueError("unknown candidate version")
        snapshot = store.snapshot(output / "source_bank.json")
        snapshot["cards"] = [c for c in snapshot["cards"] if c["card"]["skill_id"] != skill_id]
        atomic_json(output / "other_skills.json", snapshot)
        atomic_json(output / "candidate.json", candidates[0])
        # Local evidence includes private verifier inputs but never enters the bank/distiller.
        atomic_json(output / "tasks.json", [asdict(task) for task in tasks])
        (output / "tasks.json").chmod(0o600)
        protocol = dict(
            conditions,
            split="development",
            task_ids=[t.task_id for t in tasks],
            other_skills_snapshot=snapshot["snapshot_id"],
            min_tasks=min_tasks,
            minimum_mean_gain=minimum_mean_gain,
            synthetic=mock,
        )
        trial_id = store.begin_trial(skill_id, version, protocol=protocol)
        atomic_json(
            manifest,
            {
                "trial_id": trial_id,
                "skill_id": skill_id,
                "version": version,
                "conditions": conditions,
                "task_ids": protocol["task_ids"],
                "mock": mock,
            },
        )
    saved_tasks = json.loads((output / "tasks.json").read_text())
    if json.loads(json.dumps([asdict(t) for t in tasks])) != saved_tasks:
        raise ValueError("trial task contents changed during resume")
    tokenizer = (
        ByteTokenizer() if mock else HuggingFaceTokenizer(config.solver_model.base_model_path)
    )
    embedder = (
        E5SkillEmbedder(config.skillbank_embedding_model_path)
        if config.skillbank_embedding_model_path and not mock
        else None
    )
    snapshot = json.loads((output / "other_skills.json").read_text())
    candidate = json.loads((output / "candidate.json").read_text())
    by_id = {task.task_id: task for task in tasks}

    def evaluate(*, task_id, enabled, protocol):
        task = by_id[task_id]
        index = list(by_id).index(task_id)
        arm_path = output / f"task-{index:04d}-{'on' if enabled else 'off'}.json"
        if arm_path.exists():
            return json.loads(arm_path.read_text())["assessment"]
        application = None
        assessment = {
            key: protocol[key]
            for key in (
                "solver_snapshot",
                "worker_config",
                "budget",
                "other_skills_snapshot",
                "skill_id",
            )
        }
        assessment.update(task_id=task_id, skill_enabled=enabled, skill_version=version)
        try:
            arm_config = replace(
                config,
                skillbank_enabled=False,
                seed=config.seed + index,
                persist_runtime_updates=False,
                trace_path=output / f"trace-{index}-{enabled}.jsonl",
            )
            application = create_adaptive_application(
                arm_config, mock=mock, director_tokenizer=tokenizer
            )
            application.solver.skillbank = TrialContextBank(
                snapshot,
                candidate,
                enabled=enabled,
                embedder=embedder,
                prompt_token_budget=config.skillbank_prompt_token_budget,
            )
            application.set_rollout_deadline(
                RolloutDeadline(
                    total_timeout_s=budget,
                    no_progress_timeout_s=budget,
                    request_timeout_s=min(120, budget),
                    absolute_wall_timeout_s=budget,
                )
            )
            result = application.solve(
                task.prompt,
                task_id=task.task_id,
                task_type=task.task_type,
                reference=task.reference,
                metadata=dict(task.metadata),
                private_verifier_payload=dict(task.private_verifier_payload),
                run_id=f"skill-trial-{trial_id}-{index}-{enabled}",
            )
            rollout = adaptive_result_to_rollout(
                result, tokenizer, rollout_index=int(enabled), seed=arm_config.seed
            )
            metadata = rollout.trajectory.metadata
            assessment.update(
                reward=rollout.trajectory.reward,
                reward_known=metadata.get("reward_known") is True,
                infrastructure_failure=any(
                    bool(result.task.metadata.get(k))
                    for k in (
                        "worker_backend_failure",
                        "swe_infrastructure_failure",
                        "unresolved_tool_failure",
                    )
                ),
                skill_context=result.solver_result.skill_context,
                artifact_path=str(arm_path),
            )
            payload = {"assessment": assessment, "result": result.to_dict()}
        except Exception as exc:
            assessment.update(
                reward=0,
                reward_known=False,
                infrastructure_failure=True,
                failure_type=type(exc).__name__,
                # Keep enough transport detail to distinguish a local service
                # outage from a task/verifier failure; never serialize a full
                # traceback or request headers that could contain credentials.
                failure_detail=str(exc)[:1000],
            )
            payload = {"assessment": assessment}
        finally:
            if application is not None:
                from .alfworld import alfworld_lifecycles
                from .swebench import swe_lifecycles
                from .webshop import webshop_lifecycles

                tools = getattr(application.runtime.executor, "tools", {})
                for lifecycle in (
                    *alfworld_lifecycles(tools),
                    *swe_lifecycles(tools),
                    *webshop_lifecycles(tools),
                ):
                    lifecycle.close_all()
                application.close()
        atomic_json(arm_path, payload)
        return assessment

    summary = store.run_trial(trial_id, evaluate)
    atomic_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--tasks", type=Path, required=True, help="JSON list of development TaskSpec objects"
    )
    parser.add_argument("--skill", required=True)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=float, default=900)
    parser.add_argument("--min-tasks", type=int, default=5)
    parser.add_argument("--minimum-mean-gain", type=float, default=0)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    from .application import load_adaptive_config
    from .observability import TaskSpec

    config = load_adaptive_config(args.config)
    if config.skillbank_mode != SCHEMA:
        parser.error("requires director_skill_v2 mode")
    tasks = [TaskSpec(**row) for row in json.loads(args.tasks.read_text())]
    store = store_for_config(config)
    if args.mock:
        # An offline CLI check must never change the production lifecycle state.
        isolated = SkillStore(args.output / "mock-skill-store.sqlite3")
        with isolated.connect() as db:
            for item in store.cards():
                db.execute(
                    "INSERT OR IGNORE INTO cards VALUES(?,?,?,?)",
                    (item["card"]["skill_id"], item["version"], item["status"], json.dumps(item)),
                )
        store = isolated
    summary = run_application_trial(
        config,
        tasks,
        store=store,
        skill_id=args.skill,
        version=args.version,
        output=args.output,
        budget=args.budget,
        min_tasks=args.min_tasks,
        minimum_mean_gain=args.minimum_mean_gain,
        mock=args.mock,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
