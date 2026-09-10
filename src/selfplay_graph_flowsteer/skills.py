from __future__ import annotations

import json
import re
import threading
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Protocol

from .llm import ChatBackend

SKILL_KINDS = frozenset(
    {
        "orchestration",
        "decomposition",
        "communication",
        "verification",
        "tool-use",
        "bug-repair",
    }
)


@dataclass
class SkillStats:
    usage_count: int = 0
    helpful_count: int = 0
    hurt_count: int = 0
    last_used_step: int = -1
    creation_step: int = 0
    is_seed: bool = False


@dataclass
class SkillCard:
    skill_id: str
    name: str
    description: str
    trigger: str
    plan: str
    pitfall: str = ""
    constraint: str = ""
    kind: str = "bug-repair"
    task_types: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    stats: SkillStats = field(default_factory=SkillStats)

    def validate(self) -> None:
        if self.kind not in SKILL_KINDS:
            raise ValueError(f"unknown Solver skill kind: {self.kind}")
        for name, value in (
            ("skill_id", self.skill_id),
            ("name", self.name),
            ("description", self.description),
            ("trigger", self.trigger),
            ("plan", self.plan),
        ):
            if not str(value).strip():
                raise ValueError(f"skill {name} cannot be empty")

    def format_for_prompt(self) -> str:
        return (
            f"[{self.skill_id}] {self.name}\n"
            f"When: {self.trigger}\n"
            f"Plan: {self.plan}\n"
            f"Pitfall: {self.pitfall or 'none recorded'}\n"
            f"Reliability: {self.stats.helpful_count}/{self.stats.usage_count} helpful"
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SkillCard:
        raw = dict(payload)
        stats = dict(raw.get("stats", {}))
        stats.pop("flow_score", None)
        raw["stats"] = SkillStats(**stats)
        return cls(**raw)


@dataclass(frozen=True)
class SolverFailureCase:
    task: str
    task_type: str
    failure_trace: str
    failure_mode: str
    evidence_refs: tuple[str, ...] = ()
    reference: str = ""
    solver_answer: str = ""
    used_skill_ids: tuple[str, ...] = ()
    frontier_score: float = 0.0
    uid: str = ""


class SkillEmbedder(Protocol):
    def encode(self, texts: list[str], *, query: bool) -> list[tuple[float, ...]]: ...


class E5SkillEmbedder:
    """CPU E5 encoder migrated from SESA's SkillBank retrieval path."""

    _cache: ClassVar[dict[str, tuple[Any, Any]]] = {}
    _cache_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, model_path: str | Path) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        source = str(model_path)
        self.model_path = source
        with self._cache_lock:
            cached = self._cache.get(source)
            if cached is None:
                cached = (
                    AutoTokenizer.from_pretrained(source),
                    AutoModel.from_pretrained(source).eval().cpu(),
                )
                self._cache[source] = cached
        self.tokenizer, self.model = cached

    def encode(self, texts: list[str], *, query: bool) -> list[tuple[float, ...]]:
        if not texts:
            return []
        prefix = "query: " if query else "passage: "
        vectors: list[tuple[float, ...]] = []
        with self.torch.no_grad():
            for offset in range(0, len(texts), 32):
                batch = [prefix + text for text in texts[offset : offset + 32]]
                inputs = self.tokenizer(
                    batch, padding=True, truncation=True, max_length=256, return_tensors="pt"
                )
                output = self.model(**inputs)
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                pooled = (output.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
                vectors.extend(tuple(float(value) for value in row) for row in pooled)
        return vectors


class SolverSkillBank:
    """SESA lifecycle scoped exclusively to the Workflow Solver Director."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        max_skills: int = 800,
        dedup_threshold: float = 0.93,
        retrieve_top_k: int = 3,
        min_retrieved_for_evict: int = 3,
        embedder: SkillEmbedder | None = None,
    ) -> None:
        self.path = Path(path) if path else None
        self.max_skills = int(max_skills)
        self.dedup_threshold = float(dedup_threshold)
        self.retrieve_top_k = int(retrieve_top_k)
        self.min_retrieved_for_evict = int(min_retrieved_for_evict)
        self.embedder = embedder
        self.skills: dict[str, SkillCard] = {}
        self._embeddings: dict[str, tuple[float, ...]] = {}
        if self.path and self.path.exists():
            self._load()
        self._rebuild_embeddings()

    def add_or_deduplicate(self, candidate: SkillCard) -> tuple[str, str]:
        candidate.validate()
        duplicate = self._nearest(candidate)
        if duplicate and duplicate[0] >= self.dedup_threshold:
            existing = duplicate[1]
            return existing.skill_id, "deduplicated"
        if len(self.skills) >= self.max_skills:
            removable = sorted(
                (skill for skill in self.skills.values() if not skill.stats.is_seed),
                key=lambda skill: (
                    skill.stats.helpful_count - skill.stats.hurt_count,
                    skill.stats.usage_count,
                ),
            )
            if not removable:
                return candidate.skill_id, "rejected_capacity"
            del self.skills[removable[0].skill_id]
        if not candidate.skill_id or candidate.skill_id in self.skills:
            candidate.skill_id = self.next_skill_id()
        self.skills[candidate.skill_id] = candidate
        self._rebuild_embeddings()
        self.save()
        return candidate.skill_id, "retained"

    def retrieve(
        self, query: str, *, task_type: str = "", top_k: int | None = None
    ) -> list[SkillCard]:
        limit = self.retrieve_top_k if top_k is None else max(0, int(top_k))
        scored: list[tuple[float, str, SkillCard]] = []
        query_tokens = _token_set(f"{query} {task_type}")
        query_vector = self.embedder.encode([query], query=True)[0] if self.embedder else None
        for skill in self.skills.values():
            skill_tokens = _token_set(
                f"{skill.name} {skill.description} {skill.trigger} {skill.plan} "
                + " ".join(skill.task_types)
            )
            relevance = (
                _dot(query_vector, self._embeddings[skill.skill_id])
                if query_vector is not None
                else _jaccard(query_tokens, skill_tokens)
            )
            scored.append((relevance, skill.skill_id, skill))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [skill for _, _, skill in scored[:limit]]

    def prompt_context(self, query: str, *, task_type: str = "", top_k: int | None = None) -> str:
        return self.format_prompt_context(self.retrieve(query, task_type=task_type, top_k=top_k))

    @staticmethod
    def format_prompt_context(selected: list[SkillCard]) -> str:
        if not selected:
            return ""
        return "## Solver SkillBank\n\n" + "\n\n---\n\n".join(
            skill.format_for_prompt() for skill in selected
        )

    def record_outcome(
        self, skill_id: str, *, step: int, helpful: bool, hurt: bool = False
    ) -> None:
        skill = self.skills[skill_id]
        skill.stats.usage_count += 1
        skill.stats.last_used_step = int(step)
        skill.stats.helpful_count += int(helpful)
        skill.stats.hurt_count += int(hurt)
        self.save()

    def prune(self, *, min_usage: int | None = None) -> list[str]:
        removed: list[str] = []
        for skill_id, skill in list(self.skills.items()):
            stats = skill.stats
            required = self.min_retrieved_for_evict if min_usage is None else int(min_usage)
            net_score = stats.helpful_count - stats.hurt_count
            if not stats.is_seed and stats.usage_count >= required and net_score < 0:
                del self.skills[skill_id]
                removed.append(skill_id)
        self._rebuild_embeddings()
        self.save()
        return removed

    def next_skill_id(self) -> str:
        index = 0
        while f"solver_{index:03d}" in self.skills:
            index += 1
        return f"solver_{index:03d}"

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {"scope": "workflow-solver", "skills": [s.to_dict() for s in self.skills.values()]},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _load(self) -> None:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("scope") != "workflow-solver":
            raise ValueError("not a Workflow Solver SkillBank")
        self.skills = {
            skill.skill_id: skill
            for skill in (SkillCard.from_dict(item) for item in payload.get("skills", []))
        }

    def _rebuild_embeddings(self) -> None:
        if not self.embedder or not self.skills:
            self._embeddings = {}
            return
        skills = list(self.skills.values())
        texts = [f"{skill.description} || {skill.trigger} || {skill.plan}" for skill in skills]
        vectors = self.embedder.encode(texts, query=False)
        self._embeddings = {
            skill.skill_id: vector for skill, vector in zip(skills, vectors, strict=True)
        }

    def _nearest(self, candidate: SkillCard) -> tuple[float, SkillCard] | None:
        if self.embedder:
            vector = self.embedder.encode(
                [f"{candidate.description} || {candidate.trigger} || {candidate.plan}"],
                query=False,
            )[0]
            matches = [
                (_dot(vector, self._embeddings[skill.skill_id]), skill)
                for skill in self.skills.values()
            ]
            return max(matches, key=lambda item: item[0]) if matches else None
        candidate_tokens = _token_set(
            f"{candidate.name} {candidate.description} {candidate.trigger} {candidate.plan}"
        )
        matches = [
            (
                _jaccard(
                    candidate_tokens,
                    _token_set(f"{skill.name} {skill.description} {skill.trigger} {skill.plan}"),
                ),
                skill,
            )
            for skill in self.skills.values()
        ]
        return max(matches, key=lambda item: item[0]) if matches else None


class SESASolverSkillDistiller:
    """Distill a SESA corrective skill for Solver orchestration failures."""

    def __init__(self, backend: ChatBackend) -> None:
        self.backend = backend

    def distill(self, case: SolverFailureCase, *, skill_id: str, step: int) -> SkillCard:
        prompt = (
            "Extract one concrete corrective skill from this failed Workflow Solver Director "
            "trajectory. Return one JSON object with "
            "name, description, trigger, plan, pitfall, constraint, and kind. The skill must "
            "repair graph construction, role prompting, communication, verification, or output "
            "selection. Do not teach the worker the task answer and do not mention the specific "
            "task. kind must be bug-repair, orchestration, "
            "decomposition, communication, verification, or tool-use."
        )
        response = self.backend.generate(
            [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": json.dumps(_public_failure_case(case), ensure_ascii=False),
                },
            ],
            role="skill-distiller",
        )
        payload = _json_object(response.text)
        card = SkillCard(
            skill_id=skill_id,
            name=str(payload["name"]),
            description=str(payload["description"]),
            trigger=str(payload["trigger"]),
            plan=str(payload["plan"]),
            pitfall=str(payload.get("pitfall", "")),
            constraint=str(payload.get("constraint", "")),
            kind=str(payload.get("kind", "bug-repair")),
            task_types=[case.task_type],
            evidence=list(case.evidence_refs),
            stats=SkillStats(creation_step=int(step)),
        )
        card.validate()
        return card


def _public_failure_case(case: SolverFailureCase) -> dict[str, Any]:
    """Only pre-verification process evidence may reach a future policy skill.

    Archived cases still retain labels for offline analysis. Do not forward them
    wholesale or trust a prompt instruction to hide labels from the distiller.
    """
    private_keys = {
        "reference",
        "references",
        "reference_answer",
        "reference_answers",
        "gold",
        "gold_answer",
        "gold_answers",
        "target_answer",
        "target_answers",
        "private_verifier_payload",
        "verification",
        "verifier_result",
        "verification_detail",
        "rubric_items",
        "rubrics",
        "physician_response",
        "correct_answer",
        "correct_answers",
        "ground_truth",
        "solution",
        "solutions",
        "test_patch",
        "reference_patch",
        "hidden_goal",
        "goal_contract",
        "goal_id",
        "target_asin",
        "target_object",
        "target_receptacle",
    }

    def scrub(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: scrub(item)
                for key, item in value.items()
                if str(key).casefold() not in private_keys
                and not any(word in str(key).casefold() for word in ("canary", "private_", "gold_"))
            }
        if isinstance(value, list):
            return [scrub(item) for item in value]
        if isinstance(value, str) and (
            value.lstrip().startswith("{")
            or re.match(r'\s*\[\s*(?:[\[\]"{]|true|false|null|-?\d)', value)
        ):
            try:
                return json.dumps(scrub(json.loads(value)), ensure_ascii=False)
            except ValueError:
                # Opaque serialized traces cannot establish a public-data boundary.
                return "[unparseable structured trace omitted]"
        return value

    try:
        trace = json.loads(case.failure_trace)
    except ValueError:
        trace = {}
    if not isinstance(trace, dict):
        trace = {}
    # Exclude post-verification summaries and arbitrary metadata at trace root.
    events = trace.get("events", [])
    events = events if isinstance(events, list) else []
    process = {
        "final_graph": trace.get("final_graph", {}),
        "events": [
            event
            for event in events
            if isinstance(event, dict) and event.get("kind") == "canvas_step"
        ],
    }
    return {
        "schema": "public_failure_case_v1",
        "task": case.task,
        "task_type": case.task_type,
        "failure_trace": scrub(process),
        "failure_mode": case.failure_mode
        if re.fullmatch(r"[a-z_]+", case.failure_mode)
        else "task_verification_failure",
        "solver_answer": case.solver_answer,
        "used_skill_ids": list(case.used_skill_ids),
    }


class SolverSkillLifecycle:
    def __init__(
        self,
        bank: SolverSkillBank,
        distiller: SESASolverSkillDistiller,
        *,
        update_freq: int = 10,
        pending_queue_max: int = 300,
        min_pending: int = 20,
        generate_per_update: int = 50,
    ) -> None:
        self.bank = bank
        self.distiller = distiller
        self.failures: list[SolverFailureCase] = []
        self.step = 0
        self.update_freq = int(update_freq)
        self.pending_queue_max = int(pending_queue_max)
        self.min_pending = int(min_pending)
        self.generate_per_update = int(generate_per_update)
        self.last_update_step = -1

    def collect_failure(self, case: SolverFailureCase) -> None:
        if not 0.0 < float(case.frontier_score) < 1.0:
            return
        if case.uid and any(existing.uid == case.uid for existing in self.failures):
            return
        self.failures.append(case)
        self.failures = self.failures[-self.pending_queue_max :]

    def evolve(self, *, step: int, force: bool = False) -> list[tuple[str, str]]:
        self.step = int(step)
        eligible = (
            self.step != self.last_update_step
            and self.step % self.update_freq == 0
            and len(self.failures) >= self.min_pending
        )
        if not force and not eligible:
            return []
        self.last_update_step = self.step
        changes: list[tuple[str, str]] = []
        uid_counts = Counter(case.uid for case in self.failures)
        selected = sorted(
            self.failures,
            key=lambda case: (
                10 * bool(case.used_skill_ids)
                + 5 * (uid_counts[case.uid] > 1)
                + 3 * (len(case.failure_trace) > 1500)
            ),
            reverse=True,
        )[: self.generate_per_update]
        self.failures = []
        changes.extend((skill_id, "negative_utility_evicted") for skill_id in self.bank.prune())
        for failure in selected:
            candidate = self.distiller.distill(
                failure, skill_id=self.bank.next_skill_id(), step=step
            )
            changes.append(self.bank.add_or_deduplicate(candidate))
        return changes

    def save_pending(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(
                {
                    "scope": "workflow-solver-skill-cases",
                    "step": self.step,
                    "last_update_step": self.last_update_step,
                    "failures": [asdict(case) for case in self.failures],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def load_pending(self, path: str | Path) -> None:
        source = Path(path)
        if not source.exists():
            return
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("scope") != "workflow-solver-skill-cases":
            raise ValueError("not a Workflow Solver skill-case store")
        self.step = int(payload.get("step", 0))
        self.last_update_step = int(payload.get("last_update_step", -1))
        self.failures = [
            SolverFailureCase(
                **{
                    **{
                        key: value
                        for key, value in item.items()
                        if key in SolverFailureCase.__dataclass_fields__
                    },
                    "evidence_refs": tuple(item.get("evidence_refs", [])),
                    "used_skill_ids": tuple(item.get("used_skill_ids", [])),
                }
            )
            for item in payload.get("failures", [])
        ]


def _json_object(text: str) -> dict[str, Any]:
    first, last = text.find("{"), text.rfind("}")
    payload = json.loads(text[first : last + 1] if first >= 0 and last > first else text)
    if not isinstance(payload, dict):
        raise TypeError("expected JSON object")
    return payload


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.casefold()))


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _dot(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding vectors must have equal length")
    return sum(a * b for a, b in zip(left, right, strict=True))
