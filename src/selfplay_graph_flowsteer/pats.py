"""Policy-aware Director scaffolding, adapted from PATS (Apache-2.0).

Adapted from shi-yipeng/PATS (Apache-2.0). The review bands and per-type,
zero-initialized EMA follow the official
``experience_bank.choose_review_mode`` / evidence-bank design at upstream
commit bad468b5c73081c2f5aa74c4e0011c6fb2872dbf. This integration uses existing
Director cards, audited task rewards (continuous where applicable), strict
atomic edit validation, actual tokenizer counts and immutable scoped views.
See NOTICE.md and third_party/notices for attribution. No PPO data is edited.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from .config import canonical_dataset_name
from .pats_evidence import (
    process_contrast,
    public_process_trace,
    public_text,
    selected_skill_context,
)
from .skills import SKILL_KINDS, SkillCard, SkillStats, SolverFailureCase, _public_failure_case

SCHEMA = "policy_aware_director_scaffold_v1"
CARD_FIELDS = ("name", "description", "trigger", "plan", "pitfall", "constraint", "kind")
MODES = {"EXPAND", "REVISE", "COMPRESS", "FORCED_PRUNE"}


@dataclass(frozen=True)
class PatsConfig:
    enabled: bool = False
    ema_alpha: float = 0.1
    revise_threshold: float = 0.3
    compress_threshold: float = 0.85
    max_skills: int = 30
    max_tokens: int = 2000
    min_groups: int = 2
    min_evidence_cards: int = 2
    review_interval: int = 1
    max_edits: int = 4
    max_reviews_per_cycle: int = 2
    max_evidence_groups: int = 32
    max_policy_lag: int = 1
    max_review_input_tokens: int = 20480

    def validate(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("PATS enabled must be boolean")
        if not math.isfinite(self.ema_alpha) or not 0 < self.ema_alpha <= 1:
            raise ValueError("PATS ema_alpha must be in (0, 1]")
        if not all(
            math.isfinite(x) for x in (self.revise_threshold, self.compress_threshold)
        ) or not (0 <= self.revise_threshold <= self.compress_threshold <= 1):
            raise ValueError("PATS thresholds require 0 <= revise <= compress <= 1")
        for name in (
            "max_skills",
            "max_tokens",
            "min_groups",
            "min_evidence_cards",
            "review_interval",
            "max_edits",
            "max_reviews_per_cycle",
            "max_evidence_groups",
            "max_review_input_tokens",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"PATS {name} must be a positive integer")
        if self.min_groups < 2 or self.min_evidence_cards < 2:
            raise ValueError("PATS requires at least two distinct evidence groups")
        if self.max_evidence_groups < max(self.min_groups, self.min_evidence_cards):
            raise ValueError("PATS evidence capacity is smaller than admission requirements")
        if type(self.max_policy_lag) is not int or self.max_policy_lag < 0:
            raise ValueError("PATS max_policy_lag must be a nonnegative integer")


def _json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def resolve_scope(task_type: str, task_metadata: dict[str, Any] | None = None) -> str:
    """Only public, fixed task metadata defines a scope; never use observed reward."""
    metadata = task_metadata or {}
    dataset = canonical_dataset_name(str(metadata.get("dataset") or task_type or "general"))
    difficulty = metadata.get("difficulty_bucket", metadata.get("difficulty", "unspecified"))
    if not isinstance(difficulty, (str, int, float)) or isinstance(difficulty, bool):
        difficulty = "unspecified"
    if isinstance(difficulty, float) and not math.isfinite(difficulty):
        difficulty = "unspecified"
    return _json([dataset, str(task_type or "general"), str(difficulty)])


def choose_review_mode(
    *,
    bank_pressure: float,
    group_sr_ema: float,
    revise_threshold: float = 0.3,
    compress_threshold: float = 0.85,
) -> str:
    """PATS review bands; capacity pressure takes priority over competence."""
    if not 0 <= revise_threshold <= compress_threshold <= 1:
        raise ValueError("invalid PATS thresholds")
    if bank_pressure >= 1:
        return "FORCED_PRUNE"
    if group_sr_ema >= compress_threshold:
        return "COMPRESS"
    return "REVISE" if group_sr_ema >= revise_threshold else "EXPAND"


def render_cards(records: list[dict[str, Any]]) -> str:
    """Match the existing DirectorSkillBankV2 rendered card contents exactly."""
    if not records:
        return ""
    return "## Director SkillBank — optional orchestration guidance\n" + "\n\n".join(
        f"[{r['card']['skill_id']}@{r['version']}] {r['card']['name']}\n"
        f"When: {r['card']['trigger']}\nPlan: {r['card']['plan']}\n"
        f"Pitfall: {r['card']['pitfall']}\nConstraint: {r['card']['constraint']}"
        for r in records
    )


def _credible(row: dict[str, Any]) -> bool:
    metadata = row.get("metadata", {})
    if metadata.get("reward_known") is not True:
        return False
    if any(
        metadata.get(key)
        for key in (
            "infrastructure_failure",
            "worker_backend_failure",
            "swe_infrastructure_failure",
            "worker_artifact_integrity_failure",
            "unresolved_tool_failure",
            "swe_non_train_split",
            "swe_synthetic_evaluation",
            "uncertain_attribution_zero",
        )
    ):
        return False
    if (
        "frozen_policy_probability_mismatch" in metadata.get("training_exclusion_reasons", [])
        or metadata.get("reward_admission_reason") == "uncertain_attribution_zero"
        or metadata.get("terminal_graph_status") == "unsafe_partial"
    ):
        return False
    reason = " ".join(
        str(metadata.get(k, ""))
        for k in (
            "failure_mode",
            "training_exclusion_reasons",
            "error_type",
            "error",
            "terminal_reason",
        )
    )
    if re.search(
        r"api[_ -]?error|backend|route|upstream|connection|judge.*fail|"
        r"environment.*timeout|attribution_unresolved",
        reason,
        re.I,
    ):
        return False
    try:
        reward = float(row["reward"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return (
        math.isfinite(reward) and 0 <= reward <= 1 and metadata.get("task_reward", reward) == reward
    )


def apply_metadata_updates(
    rows: list[dict[str, Any]], updates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Apply the collector's authoritative post-collection audits without rewriting raw logs."""
    by_id = {}
    for row in rows:
        key = str(row.get("rollout_id", ""))
        if key in by_id and by_id[key] != row:
            raise ValueError("conflicting duplicate PATS rollout")
        by_id[key] = copy.deepcopy(row)
    for update in updates:
        key = str(update.get("rollout_id", ""))
        if key in by_id and "metadata" in update:
            if not isinstance(update["metadata"], dict):
                raise ValueError("PATS rollout metadata audit must be an object")
            by_id[key]["metadata"] = copy.deepcopy(update["metadata"])
    return list(by_id.values())


def collect_evidence(
    tasks: dict[str, Any], rows: list[dict[str, Any]], *, run: str, step: int, policy_snapshot: str
) -> list[dict[str, Any]]:
    """Include all-failure/all-success groups, excluding unknown and infrastructure outcomes."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: dict[str, str] = {}
    for row in rows:
        task = tasks.get(row.get("task_id"))
        if (
            task is None
            or task.metadata.get("is_final_test") is True
            or (task.metadata.get("skill_evaluation_split") == "final_test")
        ):
            continue
        if canonical_dataset_name(str(task.metadata.get("dataset", task.task_type))) == "swe_bench":
            from .swebench import swe_task_is_training_split

            if not swe_task_is_training_split(task.metadata):
                continue
        identity = str(row.get("rollout_id", ""))
        if not identity:
            raise ValueError("PATS evidence requires rollout identity")
        signature = _digest(row)
        if identity in seen:
            if seen[identity] != signature:
                raise ValueError("conflicting duplicate PATS rollout")
            continue
        seen[identity] = signature
        if _credible(row):
            groups[str(row["task_id"])].append(row)
    evidence = []
    for task_id, group in sorted(groups.items()):
        task = tasks[task_id]
        contexts = {_digest(r.get("metadata", {}).get("skill_context", {})) for r in group}
        policies = {
            str(r.get("metadata", {}).get("solver_snapshot", policy_snapshot)) for r in group
        }
        if len(contexts) != 1 or policies != {policy_snapshot}:
            continue  # Mixed policy/context groups cannot describe one learner condition.
        rewards = [float(row["reward"]) for row in group]
        attempts = []
        skill_context = selected_skill_context(group[0].get("metadata", {}).get("skill_context"))
        for row in group:
            metadata = row.get("metadata", {})
            public = _public_failure_case(
                SolverFailureCase(
                    task=task.prompt,
                    task_type=task.task_type,
                    failure_trace=json.dumps(metadata.get("solver_trace", {})),
                    failure_mode=str(metadata.get("failure_mode", "trusted_outcome")),
                    solver_answer="",
                    used_skill_ids=tuple(item["id"] for item in skill_context["selected"]),
                )
            )
            attempts.append(
                {
                    "rollout_id": str(row["rollout_id"]),
                    "reward": float(row["reward"]),
                    "failure_mode": public["failure_mode"],
                    "trace": public_process_trace(public["failure_trace"]),
                }
            )
        # Use every sibling for contrast. Representatives favor the longest
        # low-reward attempt and shortest high-reward attempt, with stable ties.
        ordered = sorted(
            attempts,
            key=lambda item: (
                item["reward"],
                -item["trace"]["summary"]["steps"],
                item["rollout_id"],
            ),
        )
        examples = list({item["rollout_id"]: item for item in (ordered[0], ordered[-1])}.values())
        scope = resolve_scope(task.task_type, task.metadata)
        source_task = str(task.metadata.get("source_task_id", task_id))
        evidence.append(
            {
                "id": _digest([run, step, scope, task_id]),
                "task_id": source_task,
                "scope": scope,
                "step": step,
                "policy_snapshot": policy_snapshot,
                "context_hash": next(iter(contexts)),
                "skill_context": skill_context,
                "task_goal": public_text(task.prompt, 1200),
                "rollout_ids": sorted(r["rollout_id"] for r in group),
                "reward_mean": sum(rewards) / len(rewards),
                "valid_rollouts": len(rewards),
                "all_failed": all(r == 0 for r in rewards),
                "all_passed": all(r == 1 for r in rewards),
                "process_contrast": process_contrast(attempts),
                "examples": examples,
            }
        )
    return evidence


def apply_operations(
    records: list[dict[str, Any]],
    payload: dict[str, Any],
    *,
    mode: str,
    evidence: list[dict[str, Any]],
    config: PatsConfig,
    token_counter: Callable[[str], int],
    scope: str,
    step: int,
) -> list[dict[str, Any]]:
    """Validate the whole proposed transaction before publishing any change."""
    config.validate()
    if mode not in MODES or not isinstance(payload, dict) or set(payload) != {"operations"}:
        raise ValueError("PATS requires a JSON object containing only operations")
    operations = payload["operations"]
    if not isinstance(operations, list) or len(operations) > config.max_edits:
        raise ValueError("PATS edit budget exceeded")
    if not operations:
        return copy.deepcopy(records)
    by_id = {record["card"]["skill_id"]: copy.deepcopy(record) for record in records}
    evidence_by_id = {e["id"]: e for e in evidence if e.get("scope") == scope}
    touched: set[str] = set()
    additions = 0
    for index, operation in enumerate(operations):
        if not isinstance(operation, dict) or set(operation) - {
            "op",
            "skill_id",
            "card",
            "evidence_ids",
        }:
            raise ValueError("invalid PATS operation fields")
        op = operation.get("op")
        if op not in {"ADD", "UPDATE", "DELETE"}:
            raise ValueError("unknown PATS operation")
        refs = operation.get("evidence_ids")
        if not isinstance(refs, list) or any(
            not isinstance(r, str) or r not in evidence_by_id for r in refs
        ):
            raise ValueError("PATS edit references unknown evidence")
        if len({evidence_by_id[r]["task_id"] for r in refs}) < config.min_evidence_cards:
            raise ValueError("PATS edits require distinct-task supporting evidence")
        skill_id = operation.get("skill_id")
        if op == "ADD":
            if skill_id is not None or mode in {"COMPRESS", "FORCED_PRUNE"}:
                raise ValueError("PATS addition is forbidden in this review mode")
            additions += 1
            if additions > (2 if mode == "EXPAND" else 1):
                raise ValueError("PATS mode addition budget exceeded")
            skill_id = "pats_" + _digest([scope, step, index, operation])[:24]
            if skill_id in by_id:
                raise ValueError("PATS duplicate addition")
        elif not isinstance(skill_id, str) or skill_id not in by_id:
            raise ValueError("PATS edit references an unknown skill")
        if skill_id in touched:
            raise ValueError("PATS cannot edit a skill twice in one transaction")
        touched.add(skill_id)
        if op == "DELETE":
            if "card" in operation:
                raise ValueError("PATS DELETE cannot carry a card")
            del by_id[skill_id]
            continue
        card = operation.get("card")
        if not isinstance(card, dict) or set(card) != set(CARD_FIELDS):
            raise ValueError("PATS ADD/UPDATE requires the complete card schema")
        if any(not isinstance(value, str) or not value.strip() for value in card.values()):
            raise ValueError("PATS card fields must be nonempty text")
        text = " ".join(card.values())
        if (
            card["kind"] not in SKILL_KINDS
            or len(text) > 6000
            or re.search(
                r"gold_answer|reference_answer|rubric_items|api[_ -]?key|ignore .*system",
                text,
                re.I,
            )
        ):
            raise ValueError("PATS card violates static content boundary")
        previous = by_id.get(skill_id)
        version = int(previous["version"]) + 1 if previous else 1
        record = {
            "card": SkillCard(
                skill_id=skill_id,
                **card,
                task_types=[json.loads(scope)[1]],
                evidence=sorted(set(refs)),
                stats=SkillStats(creation_step=step),
            ).to_dict(),
            "version": version,
            "parent_version": previous["version"] if previous else None,
            "status": "active",
            "provenance": "pats_scoped_unvalidated",
            "required_tools": copy.deepcopy(previous.get("required_tools", [])) if previous else [],
            "excluded_task_types": [],
            "source_cases": sorted(set(refs)),
            "pats_scope": scope,
            "pats_mode": mode,
        }
        record["card"]["evidence"] = sorted(set(refs))
        SkillCard.from_dict(record["card"]).validate()
        by_id[skill_id] = record
    result = [by_id[key] for key in sorted(by_id)]
    before, after = token_counter(render_cards(records)), token_counter(render_cards(result))
    if len(result) > config.max_skills or after > config.max_tokens:
        raise ValueError("PATS resulting view exceeds capacity")
    if mode in {"COMPRESS", "FORCED_PRUNE"} and after >= before:
        raise ValueError("PATS compression/pruning must reduce actual rendered tokens")
    return result


class PatsController:
    """One bounded synchronous review per admitted scope; scoped state never edits global cards."""

    def __init__(self, store, config: PatsConfig, token_counter: Callable[[str], int]):
        config.validate()
        self.store, self.config, self.token_counter = store, config, token_counter
        with store.connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS pats_state(id INTEGER PRIMARY KEY, payload TEXT NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS pats_cycles(id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
            )

    def snapshot(self, *, run: str | None = None, next_step: int | None = None) -> dict[str, Any]:
        from .pats_semantics import (
            SEMANTIC_REVISION,
            contract_hash,
            filter_semantic_cards,
        )

        with self.store.connect() as db:
            row = db.execute("SELECT payload FROM pats_state WHERE id=1").fetchone()
        state = json.loads(row[0]) if row else {"scopes": {}, "step": -1}
        if row and state["config"] != asdict(self.config):
            raise ValueError("PATS configuration changed; use a fresh scoped state")
        if row and run is not None and state["run"] != run:
            raise ValueError(
                "PATS state belongs to another run; configure a separate skill cases path"
            )
        if row and next_step is not None and next_step <= state["step"]:
            raise ValueError("PATS cannot collect a new cycle at an already committed step")
        view = {
            "schema": SCHEMA,
            "selection_revision": "learned_first_v1",
            "semantic_gate_revision": SEMANTIC_REVISION,
            "semantic_contract_sha256": contract_hash(),
            "config": asdict(self.config),
            "step": state["step"],
            "run": state.get("run"),
            "scopes": {
                key: {
                    **{k: value[k] for k in ("ema", "policy_snapshot", "mode")},
                    "cards": filter_semantic_cards(self.store, key, value["cards"]),
                }
                for key, value in sorted(state["scopes"].items())
            },
        }
        for scope, value in view["scopes"].items():
            value["semantic_withheld_count"] = (
                len(state["scopes"][scope]["cards"]) - len(value["cards"])
            )
        view["snapshot_id"] = _digest(view)
        return view

    def maintain(
        self,
        tasks,
        rows,
        *,
        run: str,
        step: int,
        policy_snapshot: str | None = None,
        backend=None,
        mock: bool = False,
    ) -> dict[str, Any]:
        if not self.config.enabled:
            return {"enabled": False}
        if not isinstance(step, int) or step < 0:
            raise ValueError("PATS cycle step must be nonnegative")
        if not policy_snapshot:
            raise ValueError("PATS evidence requires the behavior Solver snapshot")
        identity = _json([run, step])
        input_hash = _digest(
            {
                "config": asdict(self.config),
                "policy": policy_snapshot,
                "rows": sorted(rows, key=lambda r: str(r.get("rollout_id", ""))),
                "tasks": {
                    k: {"prompt": t.prompt, "type": t.task_type, "metadata": t.metadata}
                    for k, t in sorted(tasks.items())
                },
                "mock": mock,
            }
        )
        # Do not hold a SQLite transaction across inference. This lock protects the
        # full read/review/commit cycle from another controller process.
        with self.store.path.with_suffix(".pats.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            with self.store.connect() as db:
                old = db.execute(
                    "SELECT payload FROM pats_cycles WHERE id=?", (identity,)
                ).fetchone()
                if old:
                    receipt = json.loads(old[0])
                    if receipt["input_hash"] != input_hash:
                        raise ValueError("PATS cycle replay changed input/configuration")
                    return receipt
                current = db.execute("SELECT payload FROM pats_state WHERE id=1").fetchone()
            state = (
                json.loads(current[0])
                if current
                else {
                    "config": asdict(self.config),
                    "run": run,
                    "step": -1,
                    "scopes": {},
                }
            )
            if state["config"] != asdict(self.config) or state["run"] != run:
                raise ValueError("PATS configuration/run changed; use a fresh scoped state")
            if step <= state["step"]:
                raise ValueError("PATS cycles must commit in increasing order")
            evidence = collect_evidence(
                tasks, rows, run=run, step=step, policy_snapshot=policy_snapshot
            )
            grouped = defaultdict(list)
            for item in evidence:
                grouped[item["scope"]].append(item)
            base = [r for r in self.store.cards() if r["status"] in {"seed", "active"}]
            reviews = []
            calls = 0
            semantic_calls = 0
            ordered_scopes = sorted(
                grouped,
                key=lambda key: (
                    state["scopes"].get(key, {}).get("last_review_step", -1),
                    key,
                ),
            )
            for scope in ordered_scopes:
                current_groups = grouped[scope]
                task_type = json.loads(scope)[1]
                record = state["scopes"].setdefault(
                    scope,
                    {
                        "ema": 0.0,
                        "groups_seen": 0,
                        "evidence": [],
                        "mode": "EXPAND",
                        "policy_snapshot": policy_snapshot,
                        "cards": copy.deepcopy(
                            [
                                r
                                for r in base
                                if (
                                    not r["card"].get("task_types")
                                    or task_type in r["card"]["task_types"]
                                )
                                and task_type not in r.get("excluded_task_types", [])
                            ]
                        ),
                    },
                )
                mean = sum(e["reward_mean"] for e in current_groups) / len(current_groups)
                record["ema"] = (1 - self.config.ema_alpha) * record[
                    "ema"
                ] + self.config.ema_alpha * mean
                record["groups_seen"] += len(current_groups)
                record["policy_snapshot"] = policy_snapshot
                recent = [
                    e for e in record["evidence"] if step - e["step"] <= self.config.max_policy_lag
                ]
                recent.extend(current_groups)
                recent.sort(key=lambda e: (-e["step"], e["id"]))
                record["evidence"] = recent[: self.config.max_evidence_groups]
                pressure = max(
                    len(record["cards"]) / self.config.max_skills,
                    self.token_counter(render_cards(record["cards"])) / self.config.max_tokens,
                )
                mode = choose_review_mode(
                    bank_pressure=pressure,
                    group_sr_ema=record["ema"],
                    revise_threshold=self.config.revise_threshold,
                    compress_threshold=self.config.compress_threshold,
                )
                record["mode"] = mode
                review = {
                    "scope": scope,
                    "mode": mode,
                    "ema": record["ema"],
                    "mean_reward": mean,
                    "credible_groups": len(current_groups),
                    "bank_pressure": pressure,
                    "policy_snapshot": policy_snapshot,
                    "status": "evidence_only",
                }
                reviews.append(review)
                distinct = {e["task_id"] for e in record["evidence"]}
                if len(distinct) < max(self.config.min_groups, self.config.min_evidence_cards):
                    review["status"] = "insufficient_evidence"
                    continue
                if step % self.config.review_interval:
                    review["status"] = "interval_not_due"
                    continue
                if calls >= self.config.max_reviews_per_cycle:
                    review["status"] = "cycle_review_budget"
                    continue
                if mock or backend is None:
                    review["status"] = "mock_no_refiner" if mock else "refiner_unavailable"
                    continue
                record["last_review_step"] = step
                review_stage = "input_validation"
                try:
                    from .pats_refiner import (
                        build_review_messages,
                        normalize_evidence_references,
                        parse_review_response,
                        public_response_audit,
                        response_generation_audit,
                        returned_evidence_references,
                        review_json_schema,
                    )

                    messages, supplied_evidence, request_audit = build_review_messages(
                        scope=scope,
                        mode=mode,
                        policy_snapshot=policy_snapshot,
                        record=record,
                        config=self.config,
                        token_counter=self.token_counter,
                    )
                    review.update(request_audit, request_sha256=_digest(messages))
                    schema = review_json_schema(
                        config=self.config,
                        mode=mode,
                        records=record["cards"],
                        evidence_aliases=request_audit["evidence_aliases"],
                    )
                    review["response_schema_sha256"] = _digest(schema)
                    review_stage = "refiner_request"
                    generate_json = getattr(backend, "generate_json", None)
                    response = None
                    if callable(generate_json):
                        review["structured_output"] = "json_schema"
                        calls += 1
                        try:
                            response = generate_json(
                                messages, role="skill-distiller", schema=schema
                            )
                        except NotImplementedError:
                            # The backend contract raises this only before a
                            # provider request. API errors must not take this path.
                            calls -= 1
                            review["structured_output"] = "unsupported_fallback"
                    else:
                        review["structured_output"] = "unavailable_fallback"
                    if review["structured_output"] != "json_schema":
                        calls += 1
                        response = backend.generate(messages, role="skill-distiller")
                    review_stage = "output_validation"
                    raw = response.raw_action_text or response.text
                    review.update(public_response_audit(raw))
                    review.update(response_generation_audit(response))
                    if isinstance(raw, str):
                        review["response_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
                    review["response_tokens"] = getattr(response, "token_out", None)
                    payload = parse_review_response(raw)
                    review["returned_evidence_ids"] = returned_evidence_references(payload)
                    payload = normalize_evidence_references(
                        payload, request_audit["evidence_aliases"]
                    )
                    revised = apply_operations(
                        record["cards"],
                        payload,
                        mode=mode,
                        evidence=supplied_evidence,
                        config=self.config,
                        token_counter=self.token_counter,
                        scope=scope,
                        step=step,
                    )
                    from .pats_semantics import audit_semantic_cards, card_identity

                    original_identities = {
                        card_identity(scope, card) for card in record["cards"]
                    }
                    changed_cards = [
                        card for card in revised
                        if card_identity(scope, card) not in original_identities
                    ]
                    if changed_cards:
                        review_stage = "semantic_validation"
                        semantic = audit_semantic_cards(
                            self.store, scope, changed_cards, backend=backend,
                            token_counter=self.token_counter,
                            max_input_tokens=self.config.max_review_input_tokens,
                            run=run, step=step,
                        )
                        semantic_calls += semantic["checker_calls"]
                        review["semantic_check"] = semantic
                        if not semantic["all_approved"]:
                            # Keep the proposed transaction visible in audit, but never
                            # publish even its approved subset or accompanying DELETEs.
                            review["proposed_operations"] = payload["operations"]
                            raise ValueError("PATS proposed cards did not pass independent interface check")
                    review["before_tokens"] = self.token_counter(render_cards(record["cards"]))
                    review["after_tokens"] = self.token_counter(render_cards(revised))
                    review["operations"] = payload["operations"]
                    review["status"] = "updated" if revised != record["cards"] else "unchanged"
                    record["cards"] = revised
                except Exception as exc:
                    review.update(
                        status="rejected", error_type=type(exc).__name__, error_stage=review_stage
                    )
                    # Local validation errors contain fixed diagnostics. Provider error text
                    # is deliberately excluded because it can include request credentials.
                    if isinstance(exc, ValueError) and review_stage != "refiner_request":
                        review["validation_error"] = str(exc)[:300]
            state["step"] = step
            from .pats_semantics import SEMANTIC_REVISION, contract_hash

            receipt = {
                "schema": SCHEMA,
                "input_hash": input_hash,
                "step": step,
                "run": run,
                "policy_snapshot": policy_snapshot,
                "credible_groups": len(evidence),
                "refiner_calls": calls,
                "semantic_checker_calls": semantic_calls,
                "semantic_validator_revision": SEMANTIC_REVISION,
                "semantic_contract_sha256": contract_hash(),
                "reviews": reviews,
            }
            with self.store.connect() as db:
                db.execute("INSERT OR REPLACE INTO pats_state VALUES(1,?)", (_json(state),))
                db.execute("INSERT INTO pats_cycles VALUES(?,?)", (identity, _json(receipt)))
            return receipt
