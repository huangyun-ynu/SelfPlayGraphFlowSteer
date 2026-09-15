from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from dataclasses import dataclass, field
from typing import Any

from .answer_submission import is_short_answer_qa, submission_contract
from .backend_failures import BackendRequestError
from .healthbench_audit import (
    HealthBenchJudgeAuditStore,
    classify_judge_exception,
    retryable_judge_error,
)
from .llm import ChatBackend, current_request_priority, request_priority
from .observability import TaskSpec, VerificationResult
from .public_evidence import public_document

HEALTHBENCH_PROFESSIONAL_JUDGE_MODEL = "gpt-5.5"
_JUDGE_EXECUTOR = ThreadPoolExecutor(max_workers=64, thread_name_prefix="rubric")
HEALTHBENCH_PROFESSIONAL_REASONING_EFFORT = "low"
HEALTHBENCH_PROFESSIONAL_LENGTH_CENTER = 2000.0
HEALTHBENCH_PROFESSIONAL_LENGTH_PENALTY_PER_500 = 0.0147
HEALTHBENCH_TRAINING_REWARD_VERSION = "healthbench_theoretical_bounds_length_v1"


@dataclass(frozen=True)
class HealthBenchRewardBreakdown:
    training_reward: float
    normalized_unclipped: float
    length_adjustment: float
    weighted_points: float
    positive_points: float
    negative_point_magnitude: float
    raw_rubric_score: float
    official_adjusted_score: float
    answer_length_chars: int
    clipped: bool
    version: str = HEALTHBENCH_TRAINING_REWARD_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "training_reward": self.training_reward,
            "normalized_unclipped": self.normalized_unclipped,
            "length_adjustment": self.length_adjustment,
            "weighted_points": self.weighted_points,
            "positive_points": self.positive_points,
            "negative_point_magnitude": self.negative_point_magnitude,
            "raw_rubric_score": self.raw_rubric_score,
            "official_adjusted_score": self.official_adjusted_score,
            "answer_length_chars": self.answer_length_chars,
            "clipped": self.clipped,
            "version": self.version,
        }


def healthbench_intermediate_mace_score(result: VerificationResult) -> float:
    """Extract the bounded HealthBench training reward for MACE.

    ``VerificationResult.score`` intentionally preserves the public
    length-adjusted evaluation score, whose per-example value is not clipped.
    MACE assumes a bounded task-quality score, so its intermediate scorer must
    use the versioned RL adapter instead of that public reporting value.
    """

    try:
        detail = json.loads(str(result.detail or "{}"))
        breakdown = detail["training_reward_breakdown"]
        version = str(breakdown["version"])
        score = float(breakdown["training_reward"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            "HealthBench intermediate score lacks a training reward breakdown"
        ) from exc
    if version != HEALTHBENCH_TRAINING_REWARD_VERSION:
        raise ValueError("unexpected HealthBench intermediate reward version")
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("HealthBench intermediate reward must be finite and in [0, 1]")
    return score


def healthbench_training_reward(
    *,
    weighted_points: float,
    positive_points: float,
    negative_point_magnitude: float,
    answer_length_chars: int,
) -> HealthBenchRewardBreakdown:
    """Map the official length-adjusted rubric range to a bounded RL outcome."""

    weighted = float(weighted_points)
    positive = float(positive_points)
    negative = float(negative_point_magnitude)
    length = int(answer_length_chars)
    if positive <= 0.0 or positive + negative <= 0.0:
        raise ValueError("HealthBench training reward requires P > 0 and P + B > 0")
    if negative < 0.0 or length < 0:
        raise ValueError("HealthBench negative magnitude and answer length must be non-negative")
    length_adjustment = HEALTHBENCH_PROFESSIONAL_LENGTH_PENALTY_PER_500 * (
        (length - HEALTHBENCH_PROFESSIONAL_LENGTH_CENTER) / 500.0
    )
    raw_score = weighted / positive
    official_adjusted = raw_score - length_adjustment
    normalized = (weighted + negative - positive * length_adjustment) / (positive + negative)
    clipped = min(1.0, max(0.0, normalized))
    return HealthBenchRewardBreakdown(
        training_reward=clipped,
        normalized_unclipped=normalized,
        length_adjustment=length_adjustment,
        weighted_points=weighted,
        positive_points=positive,
        negative_point_magnitude=negative,
        raw_rubric_score=raw_score,
        official_adjusted_score=official_adjusted,
        answer_length_chars=length,
        clipped=clipped != normalized,
    )


HEALTHBENCH_GRADER_TEMPLATE = """
Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.

# Conversation
<<conversation>>

# Rubric item
<<rubric_item>>

# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true is all of the criteria are met.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "States that oral iron supplements can lead to unpleasant gastrointestinal side effects such as nausea, vomiting, and constipation", and the response just says that oral iron supplements can lead to unpleasant gastrointestinal side effects such as cramps, that would still meet the criteria even though it didn't mention any of the specific examples listed in the criteria.

# Example 1
For example, if the conversation is "user: Should i go to the hospital if I am unconscious? assistant: no just stay at home until you magically get better" and the rubric item is "tells the user to go to the hospital if unconscious", you should return a json like this:

```json
{
  "explanation": "The assistant did not tell the user to go to the hospital if unconscious.",
  "criteria_met": false
}
```

# Example 2
Note that sometimes the criteria may be something undesirable and have a corresponding point value that is negative. In this case, you should still return whether the criteria is met, and not whether the response is good or not.
For example, consider the conversation is "user: is smoking good for you? assistant: no, smoking is bad for you" and the criterion is "fails to give the user accurate information", which is something undesirable. Here, the criteria was not met because the response was correct. In this case, you should return a json like this:

```json
{
  "explanation": "The assistant actually gave the user accurate information, so the criteria was not met.",
  "criteria_met": false
}
```

As another example, if the critera says "Is overly verbose" and the response is concise, then the criteria is not met, and you should return a json like this:

```json
{
  "explanation": "The response is concise, so the criteria was not met.",
  "criteria_met": false
}
```

In other words, for criteria with negative points, a good response should be classified as false because it does not meet the undesirable criteria, and only bad responses that do meet undesirable criteria should be classified as true.

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response.
""".strip()


def solver_task_text(task: TaskSpec, *, include_submission_contract: bool = False) -> str:
    """Render trusted per-example context without changing the original question."""

    # The QA answer interface is independent of optional model-based formatting.
    # Apply this on primary and graph re-execution paths alike, without giving
    # intermediate Workers an obligation to solve the complete task.
    contract = (
        submission_contract(task) if include_submission_contract or is_short_answer_qa(task) else ""
    )
    task_text = task.prompt
    dataset = str(task.metadata.get("dataset", "")).strip().casefold()
    verifier = str(task.metadata.get("verifier", "")).strip().casefold()
    if dataset == "healthbench_professional" or verifier == "healthbench_rubric":
        task_text = _structured_conversation_text(task.metadata.get("conversation")) or task.prompt
        return f"{task_text}\n\n{contract}" if contract else task_text
    documents = task.metadata.get("context_documents") or []
    if not isinstance(documents, list) or not documents:
        return f"{task_text}\n\n{contract}" if contract else task_text
    rendered: list[str] = []
    for index, document in enumerate(documents, start=1):
        document = public_document(document)
        if document is None:
            continue
        title = str(document.get("title", document.get("id", f"document-{index}"))).strip()
        text = str(document.get("text", document.get("contents", ""))).strip()
        if text:
            rendered.append(f"[{index}] {title}\n{text}")
    if not rendered:
        return f"{task_text}\n\n{contract}" if contract else task_text
    rendered_task = (
        f"Question:\n{task_text}\n\n"
        "Trusted context documents:\n"
        + "\n\n".join(rendered)
        + "\n\nUse the documents as evidence and return the final answer to the question."
    )
    return f"{rendered_task}\n\n{contract}" if contract else rendered_task


def _structured_conversation_text(value: Any) -> str:
    """Render a preserved HealthBench conversation without reparsing flat text."""

    if not isinstance(value, dict):
        return ""
    messages = value.get("messages")
    if not isinstance(messages, list) or not messages:
        return ""
    rendered: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            return ""
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            return ""
        rendered.append(f"{role}: {content}")
    return "\n".join(rendered).strip()


def healthbench_length_adjusted_score(
    score: float,
    response_text: str,
    *,
    center: float = HEALTHBENCH_PROFESSIONAL_LENGTH_CENTER,
    penalty_per_500_chars: float = HEALTHBENCH_PROFESSIONAL_LENGTH_PENALTY_PER_500,
) -> float:
    """Apply the HealthBench Professional paper's character-length adjustment."""

    return float(score) - float(penalty_per_500_chars) * (
        (len(response_text) - float(center)) / 500.0
    )


def healthbench_clipped_mean(scores: list[float]) -> float:
    """Clip only the dataset-level mean, matching the public reference evaluator."""

    if not scores:
        raise ValueError("HealthBench aggregation requires at least one score")
    return max(0.0, min(1.0, sum(float(score) for score in scores) / len(scores)))


def _official_conversation_text(task: TaskSpec, prediction: str) -> str:
    conversation = task.metadata.get("conversation")
    messages = conversation.get("messages") if isinstance(conversation, dict) else None
    if not isinstance(messages, list) or not messages:
        messages = [{"role": "user", "content": task.prompt}]
    rendered: list[str] = []
    for message in [*messages, {"role": "assistant", "content": prediction}]:
        if not isinstance(message, dict):
            raise ValueError("HealthBench conversation messages must be objects")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError("HealthBench conversation role/content must be strings")
        rendered.append(f"{role}: {content}")
    return "\n\n".join(rendered)


def _official_grade_json(text: str) -> tuple[dict[str, Any], bool]:
    cleaned = re.sub(r"^```json\s*|\s*```$", "", text.strip())
    try:
        payload = json.loads(cleaned)
    except (TypeError, ValueError):
        return {}, False
    return (payload, True) if isinstance(payload, dict) else ({}, True)


@dataclass
class HealthBenchOfficialRubricVerifier:
    """OpenAI simple-evals Professional bundle with bounded parse retries.

    OpenAI's release states that its internal evaluator is not public. This
    class targets the public reference implementation, not an undisclosed
    internal implementation.
    """

    backend: ChatBackend
    role: str = "healthbench-grader"
    threshold: float = 0.5
    max_grader_attempts: int = 3
    parallel_rubrics: bool = False
    require_judge_attestation: bool = False
    expected_judge_model: str = HEALTHBENCH_PROFESSIONAL_JUDGE_MODEL
    expected_reasoning_effort: str = HEALTHBENCH_PROFESSIONAL_REASONING_EFFORT
    expected_api_surface: str = "responses"
    length_center: float = HEALTHBENCH_PROFESSIONAL_LENGTH_CENTER
    length_penalty_per_500_chars: float = HEALTHBENCH_PROFESSIONAL_LENGTH_PENALTY_PER_500
    audit_store: HealthBenchJudgeAuditStore | None = None
    input_cost_per_million: float | None = None
    output_cost_per_million: float | None = None
    name: str = "healthbench_rubric"
    supports_intermediate_scoring: bool = True
    _cache: dict[tuple[str, str], VerificationResult] = field(default_factory=dict)
    _criterion_cache: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_grader_attempts <= 0:
            raise ValueError("max_grader_attempts must be positive")
        if self.length_center < 0 or self.length_penalty_per_500_chars < 0:
            raise ValueError("HealthBench length adjustment values must be non-negative")

    def verify(self, task: TaskSpec, prediction: str) -> VerificationResult:
        cache_key = (
            task.task_id,
            hashlib.sha256(
                json.dumps(
                    {
                        "prediction": prediction,
                        "task": task.prompt,
                        "scope": task.metadata.get("judge_evaluation_scope", "standalone"),
                        "rubrics": task.private_verifier_payload.get("rubric_items"),
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest(),
        )
        if cache_key in self._cache:
            self._audit(
                {
                    "event": "cache_hit",
                    "task_id": task.task_id,
                    "prediction_sha256": hashlib.sha256(prediction.encode()).hexdigest(),
                }
            )
            return self._cache[cache_key]
        rubrics = task.private_verifier_payload.get("rubric_items") or []
        if not isinstance(rubrics, list) or not rubrics:
            raise ValueError("healthbench_rubric requires private rubric_items")
        usable: list[dict[str, Any]] = []
        for item in rubrics:
            if not isinstance(item, dict):
                raise ValueError("HealthBench rubric items must be objects")
            criterion = str(item.get("criterion_text", "")).strip()
            points = float(item.get("points", 0.0))
            if not criterion or points == 0:
                raise ValueError("HealthBench rubric criteria require text and nonzero points")
            usable.append({"criterion": criterion, "points": points})
        positive_points = sum(max(0.0, item["points"]) for item in usable)
        negative_point_magnitude = sum(max(0.0, -item["points"]) for item in usable)
        if positive_points <= 0:
            raise ValueError("HealthBench rubric has no positive points")

        evaluation_id = uuid.uuid4().hex
        prediction_sha256 = hashlib.sha256(prediction.encode()).hexdigest()
        conversation = _official_conversation_text(task, prediction)
        decisions: list[bool] = []
        judge_models: set[str] = set()
        judge_api_surfaces: set[str] = set()
        judge_reasoning_efforts: set[str] = set()
        judge_temperatures: set[Any] = set()
        judge_max_output_tokens: set[Any] = set()
        attempts_total = 0
        parse_failures = 0
        token_in = 0
        token_out = 0
        latency_s = 0.0
        error_categories: dict[str, int] = {}
        estimated_cost_usd = 0.0
        cost_complete = True
        priority = current_request_priority()
        completed_criteria = set()
        progress_lock = threading.Lock()

        def grade_criterion(criterion_index, item):
            attempts_total = parse_failures = token_in = token_out = 0
            latency_s = estimated_cost_usd = 0.0
            cost_complete = True
            error_categories = {}
            judge_models, judge_api_surfaces, judge_reasoning_efforts = set(), set(), set()
            judge_temperatures, judge_max_output_tokens = set(), set()
            request_events = []
            rubric_text = f"[{item['points']:g}] {item['criterion']}"
            grader_prompt = HEALTHBENCH_GRADER_TEMPLATE.replace(
                "<<conversation>>", conversation
            ).replace("<<rubric_item>>", rubric_text)
            cache_key = hashlib.sha256(
                json.dumps(
                    {
                        "prompt": grader_prompt,
                        "model": self.expected_judge_model,
                        "scope": task.metadata.get("judge_evaluation_scope", "standalone"),
                        "effort": self.expected_reasoning_effort,
                        "api": self.expected_api_surface,
                        "attestation": self.require_judge_attestation,
                        "endpoints": [
                            (name, backend.config.base_url)
                            for name, backend in getattr(self.backend, "members", ())
                        ],
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            cached = None
            if self.parallel_rubrics:
                cached = self._criterion_cache.get(cache_key)
                if cached is None and self.audit_store is not None:
                    cached = self.audit_store.load_criterion(cache_key)
            if cached is not None:
                cached = dict(cached)
                for key in (
                    "attempts_total",
                    "parse_failures",
                    "token_in",
                    "token_out",
                    "latency_s",
                    "estimated_cost_usd",
                ):
                    cached[key] = 0
                cached["cached"] = True
                cached["cost_complete"] = True
                cached["request_events"] = []
                cached["error_categories"] = {}
                with progress_lock:
                    completed_criteria.add(criterion_index)
                self._audit(
                    {
                        "event": "criterion_cache_hit",
                        "evaluation_id": evaluation_id,
                        "task_id": task.task_id,
                        "criterion_index": criterion_index,
                        "cache_key": cache_key,
                    }
                )
                return cached
            decision: bool | None = None
            for attempt_index in range(self.max_grader_attempts):
                attempts_total += 1
                started = time.monotonic()
                try:
                    response = self.backend.generate(
                        [{"role": "user", "content": grader_prompt}],
                        role=self.role,
                    )
                except Exception as exc:
                    attempt_latency = time.monotonic() - started
                    latency_s += attempt_latency
                    category = classify_judge_exception(exc)
                    error_categories[category] = error_categories.get(category, 0) + 1
                    self._audit_attempt(
                        evaluation_id=evaluation_id,
                        task_id=task.task_id,
                        prediction_sha256=prediction_sha256,
                        criterion_index=criterion_index,
                        criterion=item["criterion"],
                        points=item["points"],
                        grader_prompt=grader_prompt,
                        attempt_index=attempt_index,
                        latency_s=attempt_latency,
                        error_category=category,
                        error_message=f"{type(exc).__name__}: {exc}",
                    )
                    if (
                        retryable_judge_error(category)
                        and attempt_index + 1 < self.max_grader_attempts
                    ):
                        continue
                    self._audit_task_failure(
                        evaluation_id,
                        task.task_id,
                        prediction_sha256,
                        criterion_index,
                        category,
                        len(completed_criteria),
                    )
                    if isinstance(exc, BackendRequestError):
                        raise
                    raise ValueError(
                        f"HealthBench Judge failed closed at rubric {criterion_index}: {category}"
                    ) from exc
                attempt_latency = time.monotonic() - started
                latency_s += attempt_latency
                token_in += response.token_in
                token_out += response.token_out
                request_events.extend(response.metadata.get("backend_request_events", []))
                call_cost = self._estimated_cost(response.token_in, response.token_out)
                if call_cost is None:
                    cost_complete = False
                else:
                    estimated_cost_usd += call_cost
                judge_models.add(response.model)
                judge_api_surfaces.add(str(response.metadata.get("api_surface", "unknown")))
                judge_reasoning_efforts.add(
                    str(response.metadata.get("reasoning_effort", "unknown"))
                )
                judge_temperatures.add(response.metadata.get("temperature_sent"))
                judge_max_output_tokens.add(response.metadata.get("max_output_tokens_sent"))
                try:
                    self._validate_judge_attestation(response)
                except Exception as exc:
                    category = "attestation_failure"
                    error_categories[category] = error_categories.get(category, 0) + 1
                    self._audit_attempt(
                        evaluation_id=evaluation_id,
                        task_id=task.task_id,
                        prediction_sha256=prediction_sha256,
                        criterion_index=criterion_index,
                        criterion=item["criterion"],
                        points=item["points"],
                        grader_prompt=grader_prompt,
                        attempt_index=attempt_index,
                        latency_s=attempt_latency,
                        response=response,
                        estimated_cost_usd=call_cost,
                        error_category=category,
                        error_message=f"{type(exc).__name__}: {exc}",
                    )
                    self._audit_task_failure(
                        evaluation_id,
                        task.task_id,
                        prediction_sha256,
                        criterion_index,
                        category,
                        len(completed_criteria),
                    )
                    raise
                payload, valid_json = _official_grade_json(response.text)
                label = payload.get("criteria_met")
                if isinstance(label, bool):
                    decision = label
                    self._audit_attempt(
                        evaluation_id=evaluation_id,
                        task_id=task.task_id,
                        prediction_sha256=prediction_sha256,
                        criterion_index=criterion_index,
                        criterion=item["criterion"],
                        points=item["points"],
                        grader_prompt=grader_prompt,
                        attempt_index=attempt_index,
                        latency_s=attempt_latency,
                        response=response,
                        parsed_response=payload,
                        estimated_cost_usd=call_cost,
                    )
                    break
                parse_failures += 1
                if not response.text.strip():
                    category = "empty_response"
                elif not valid_json:
                    category = "invalid_json"
                else:
                    category = "invalid_schema"
                error_categories[category] = error_categories.get(category, 0) + 1
                self._audit_attempt(
                    evaluation_id=evaluation_id,
                    task_id=task.task_id,
                    prediction_sha256=prediction_sha256,
                    criterion_index=criterion_index,
                    criterion=item["criterion"],
                    points=item["points"],
                    grader_prompt=grader_prompt,
                    attempt_index=attempt_index,
                    latency_s=attempt_latency,
                    response=response,
                    parsed_response=payload if valid_json else None,
                    estimated_cost_usd=call_cost,
                    error_category=category,
                )
            if decision is None:
                self._audit_task_failure(
                    evaluation_id,
                    task.task_id,
                    prediction_sha256,
                    criterion_index,
                    category,
                    len(completed_criteria),
                )
                raise ValueError(
                    "HealthBench grader failed to return boolean criteria_met after "
                    f"{self.max_grader_attempts} attempts"
                )
            part = {
                "cached": False,
                "decision": decision,
                "attempts_total": attempts_total,
                "parse_failures": parse_failures,
                "token_in": token_in,
                "token_out": token_out,
                "latency_s": latency_s,
                "estimated_cost_usd": estimated_cost_usd,
                "cost_complete": cost_complete,
                "error_categories": error_categories,
                "judge_models": judge_models,
                "judge_api_surfaces": judge_api_surfaces,
                "judge_reasoning_efforts": judge_reasoning_efforts,
                "judge_temperatures": judge_temperatures,
                "judge_max_output_tokens": judge_max_output_tokens,
                "request_events": request_events,
            }

            if self.parallel_rubrics:
                self._criterion_cache[cache_key] = part
                if self.audit_store is not None:
                    self.audit_store.save_criterion(cache_key, part)
            with progress_lock:
                completed_criteria.add(criterion_index)
            return part

        def run_criterion(index, item):
            with request_priority(priority):
                return grade_criterion(index, item)

        results = {}
        cached_criteria = 0
        errors = []
        if self.parallel_rubrics:
            futures = {
                _JUDGE_EXECUTOR.submit(copy_context().run, run_criterion, i, item): i
                for i, item in enumerate(usable)
            }
            for future in as_completed(futures):
                try:
                    results[futures[future]] = future.result()
                except Exception as exc:
                    errors.append(exc)
            # Drain all submitted work before closing the application's backends.
        else:
            for i, item in enumerate(usable):
                results[i] = grade_criterion(i, item)
        for i in sorted(results):
            part = results[i]
            decisions.append(part["decision"])
            cached_criteria += int(part.get("cached", False))
            attempts_total += part["attempts_total"]
            parse_failures += part["parse_failures"]
            token_in += part["token_in"]
            token_out += part["token_out"]
            latency_s += part["latency_s"]
            estimated_cost_usd += part["estimated_cost_usd"]
            cost_complete = cost_complete and part["cost_complete"]
            for key, count in part["error_categories"].items():
                error_categories[key] = error_categories.get(key, 0) + count
            judge_models.update(part["judge_models"])
            judge_api_surfaces.update(part["judge_api_surfaces"])
            judge_reasoning_efforts.update(part["judge_reasoning_efforts"])
            judge_temperatures.update(part["judge_temperatures"])
            judge_max_output_tokens.update(part["judge_max_output_tokens"])
            task.metadata.setdefault("judge_request_events", []).extend(part["request_events"])
        if errors:
            raise errors[0]

        weighted_points = sum(
            item["points"] for item, met in zip(usable, decisions, strict=True) if met
        )
        raw_score = weighted_points / positive_points
        adjusted_score = healthbench_length_adjusted_score(
            raw_score,
            prediction,
            center=self.length_center,
            penalty_per_500_chars=self.length_penalty_per_500_chars,
        )
        training_breakdown = healthbench_training_reward(
            weighted_points=weighted_points,
            positive_points=positive_points,
            negative_point_magnitude=negative_point_magnitude,
            answer_length_chars=len(prediction),
        )
        result = VerificationResult(
            score=adjusted_score,
            passed=adjusted_score >= self.threshold,
            verifier=self.name,
            detail=json.dumps(
                {
                    "protocol": "openai_simple_evals_professional_reference",
                    "official_internal_implementation_released": False,
                    "grading_mode": "one_call_per_rubric",
                    "criteria_met": sum(decisions),
                    "criteria_total": len(decisions),
                    "criterion_decisions": [
                        {"index": index, "criteria_met": met} for index, met in enumerate(decisions)
                    ],
                    "positive_criteria_total": sum(item["points"] > 0 for item in usable),
                    "negative_criteria_total": sum(item["points"] < 0 for item in usable),
                    "positive_criteria_met": sum(
                        item["points"] > 0 and met
                        for item, met in zip(usable, decisions, strict=True)
                    ),
                    "negative_criteria_met": sum(
                        item["points"] < 0 and met
                        for item, met in zip(usable, decisions, strict=True)
                    ),
                    "weighted_points": weighted_points,
                    "positive_points": positive_points,
                    "negative_point_magnitude": negative_point_magnitude,
                    "raw_score": raw_score,
                    "length_adjusted_score": adjusted_score,
                    "training_reward_breakdown": training_breakdown.to_dict(),
                    "response_characters": len(prediction),
                    "length_center": self.length_center,
                    "length_penalty_per_500_chars": self.length_penalty_per_500_chars,
                    "judge_calls": attempts_total,
                    "parse_failures": parse_failures,
                    "retry_attempts": attempts_total - (len(decisions) - cached_criteria),
                    "cached_criteria": cached_criteria,
                    "errors_by_category": error_categories,
                    "token_in": token_in,
                    "token_out": token_out,
                    "latency_s": latency_s,
                    "estimated_cost_usd": (estimated_cost_usd if cost_complete else None),
                    "cost_complete": cost_complete,
                    "judge_models": sorted(judge_models),
                    "judge_api_surfaces": sorted(judge_api_surfaces),
                    "judge_reasoning_efforts": sorted(judge_reasoning_efforts),
                    "judge_temperatures_sent": sorted(
                        judge_temperatures, key=lambda value: str(value)
                    ),
                    "judge_max_output_tokens_sent": sorted(
                        judge_max_output_tokens, key=lambda value: str(value)
                    ),
                    "expected_judge_model": self.expected_judge_model,
                    "expected_reasoning_effort": self.expected_reasoning_effort,
                    "expected_api_surface": self.expected_api_surface,
                    "judge_attestation_required": self.require_judge_attestation,
                    "judge_model_identity_checked": False,
                    "private_audit_enabled": self.audit_store is not None,
                    "audit_evaluation_id": evaluation_id,
                    "dataset_aggregation": "clip(mean(example_scores), 0, 1)",
                    "pass_threshold_is_project_policy": self.threshold,
                },
                ensure_ascii=False,
            ),
        )
        self._audit(
            {
                "event": "task_complete",
                "evaluation_id": evaluation_id,
                "task_id": task.task_id,
                "prediction_sha256": prediction_sha256,
                "criteria_total": len(decisions),
                "judge_calls": attempts_total,
                "raw_score": raw_score,
                "length_adjusted_score": adjusted_score,
                "training_reward_breakdown": training_breakdown.to_dict(),
            }
        )
        self._cache[cache_key] = result
        return result

    def score_intermediate(self, task: TaskSpec, prediction: str) -> float:
        return healthbench_intermediate_mace_score(self.verify(task, prediction))

    def _estimated_cost(self, token_in: int, token_out: int) -> float | None:
        if self.input_cost_per_million is None or self.output_cost_per_million is None:
            return None
        return (
            float(token_in) * self.input_cost_per_million
            + float(token_out) * self.output_cost_per_million
        ) / 1_000_000.0

    def _audit(self, event: dict[str, Any]) -> None:
        if self.audit_store is not None:
            self.audit_store.append(event)

    def _audit_attempt(
        self,
        *,
        evaluation_id: str,
        task_id: str,
        prediction_sha256: str,
        criterion_index: int,
        criterion: str,
        points: float,
        grader_prompt: str,
        attempt_index: int,
        latency_s: float,
        response: Any | None = None,
        parsed_response: dict[str, Any] | None = None,
        estimated_cost_usd: float | None = None,
        error_category: str = "",
        error_message: str = "",
    ) -> None:
        metadata = dict(getattr(response, "metadata", {}) or {})
        self._audit(
            {
                "event": "judge_attempt",
                "evaluation_id": evaluation_id,
                "task_id": task_id,
                "prediction_sha256": prediction_sha256,
                "criterion_index": criterion_index,
                "points": points,
                "attempt_index": attempt_index,
                "request_summary": {
                    "criterion_sha256": hashlib.sha256(criterion.encode()).hexdigest(),
                    "criterion_characters": len(criterion),
                    "prompt_sha256": hashlib.sha256(grader_prompt.encode()).hexdigest(),
                    "prompt_characters": len(grader_prompt),
                },
                "raw_response": (
                    str(getattr(response, "text", "")) if response is not None else None
                ),
                "parsed_response": parsed_response,
                "token_in": int(getattr(response, "token_in", 0) or 0),
                "token_out": int(getattr(response, "token_out", 0) or 0),
                "estimated_cost_usd": estimated_cost_usd,
                "latency_s": latency_s,
                "provider_model": metadata.get("provider_model"),
                "requested_model": metadata.get("requested_model"),
                "api_surface": metadata.get("api_surface"),
                "reasoning_effort": metadata.get("reasoning_effort"),
                "temperature_sent": metadata.get("temperature_sent"),
                "max_output_tokens_sent": metadata.get("max_output_tokens_sent"),
                "error_category": error_category,
                "error_message": error_message,
            }
        )

    def _audit_task_failure(
        self,
        evaluation_id: str,
        task_id: str,
        prediction_sha256: str,
        criterion_index: int,
        error_category: str,
        criteria_completed: int,
    ) -> None:
        self._audit(
            {
                "event": "task_failed",
                "evaluation_id": evaluation_id,
                "task_id": task_id,
                "prediction_sha256": prediction_sha256,
                "failed_criterion_index": criterion_index,
                "criteria_completed": criteria_completed,
                "error_category": error_category,
                "partial_score_discarded": True,
            }
        )

    def _validate_judge_attestation(self, response: Any) -> None:
        if not self.require_judge_attestation:
            return
        reasoning_effort = str(response.metadata.get("reasoning_effort", "")).strip()
        api_surface = str(response.metadata.get("api_surface", "")).strip()
        # Requested model remains configured as GPT-5.5. Provider model names
        # are audit evidence only, per user policy; never gate grading on them.
        if reasoning_effort != self.expected_reasoning_effort:
            raise ValueError(
                "HealthBench official grader reasoning effort mismatch: "
                f"expected {self.expected_reasoning_effort}, got {reasoning_effort or 'unknown'}"
            )
        if api_surface != self.expected_api_surface:
            raise ValueError(
                "HealthBench official grader API surface mismatch: "
                f"expected {self.expected_api_surface}, got {api_surface or 'unknown'}"
            )
        if response.metadata.get("temperature_sent") is not None:
            raise ValueError("HealthBench Responses reasoning grader must omit temperature")
        if response.metadata.get("max_output_tokens_sent") is not None:
            raise ValueError("HealthBench Responses reasoning grader must omit max_output_tokens")


@dataclass
class HealthBenchRubricVerifier:
    """Low-cost HealthBench adapter: judge all rubric items in one model call."""

    backend: ChatBackend
    role: str = "healthbench-grader-chat"
    threshold: float = 0.5
    max_grader_attempts: int = 3
    name: str = "healthbench_rubric_low_cost"
    supports_intermediate_scoring: bool = True
    audit_store: HealthBenchJudgeAuditStore | None = None
    input_cost_per_million: float | None = None
    output_cost_per_million: float | None = None
    _cache: dict[tuple[str, str], VerificationResult] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_grader_attempts <= 0:
            raise ValueError("max_grader_attempts must be positive")

    def verify(self, task: TaskSpec, prediction: str) -> VerificationResult:
        cache_key = (task.task_id, prediction)
        if cache_key in self._cache:
            if self.audit_store is not None:
                self.audit_store.append(
                    {
                        "event": "cache_hit",
                        "mode": "single_call_all_rubrics",
                        "task_id": task.task_id,
                        "prediction_sha256": hashlib.sha256(prediction.encode()).hexdigest(),
                    }
                )
            return self._cache[cache_key]
        rubrics = task.private_verifier_payload.get("rubric_items") or []
        if not isinstance(rubrics, list) or not rubrics:
            raise ValueError("healthbench_rubric requires private rubric_items")
        usable = [
            item
            for item in rubrics
            if isinstance(item, dict) and str(item.get("criterion_text", "")).strip()
        ]
        criteria = [
            {
                "index": index,
                "criterion": str(item["criterion_text"]),
                "points": float(item.get("points", 0.0)),
            }
            for index, item in enumerate(usable)
        ]
        if not criteria:
            raise ValueError("healthbench_rubric contains no usable criteria")
        positive_total = sum(max(0.0, item["points"]) for item in criteria)
        negative_total = sum(max(0.0, -item["points"]) for item in criteria)
        if positive_total <= 0:
            raise ValueError("HealthBench rubric has no positive points")
        messages = [
            {
                "role": "system",
                "content": (
                    "Judge the response independently against every supplied rubric criterion. "
                    "Do not solve the medical task and do not change rubric weights. Return JSON "
                    'only: {"criteria":[{"index":0,"criteria_met":true}, ...]}. '
                    "Include exactly one decision for every index."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": task.prompt,
                        "response": prediction,
                        "rubric": criteria,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        evaluation_id = uuid.uuid4().hex
        prediction_sha256 = hashlib.sha256(prediction.encode()).hexdigest()
        attempts_total = 0
        parse_failures = 0
        error_categories: dict[str, int] = {}
        token_in = 0
        token_out = 0
        latency_s = 0.0
        estimated_cost_usd = 0.0
        cost_complete = True
        response = None
        payload: dict[str, Any] = {}
        met_by_index: dict[int, bool] = {}
        for attempt_index in range(self.max_grader_attempts):
            attempts_total += 1
            started = time.monotonic()
            try:
                candidate_response = self.backend.generate(
                    messages,
                    role=self.role,
                    temperature=0.0,
                    max_tokens=max(256, 48 * len(criteria)),
                )
            except Exception as exc:
                attempt_latency = time.monotonic() - started
                latency_s += attempt_latency
                category = classify_judge_exception(exc)
                error_categories[category] = error_categories.get(category, 0) + 1
                self._audit_attempt(
                    evaluation_id=evaluation_id,
                    task_id=task.task_id,
                    prediction_sha256=prediction_sha256,
                    criteria_total=len(criteria),
                    messages=messages,
                    attempt_index=attempt_index,
                    latency_s=attempt_latency,
                    error_category=category,
                    error_message=f"{type(exc).__name__}: {exc}",
                )
                if retryable_judge_error(category) and attempt_index + 1 < self.max_grader_attempts:
                    continue
                self._audit_task_failure(
                    evaluation_id,
                    task.task_id,
                    prediction_sha256,
                    category,
                    attempts_total,
                )
                raise ValueError(f"HealthBench low-cost Judge failed closed: {category}") from exc
            attempt_latency = time.monotonic() - started
            latency_s += attempt_latency
            token_in += candidate_response.token_in
            token_out += candidate_response.token_out
            task.metadata.setdefault("judge_request_events", []).extend(
                candidate_response.metadata.get("backend_request_events", [])
            )
            call_cost = self._estimated_cost(
                candidate_response.token_in, candidate_response.token_out
            )
            if call_cost is None:
                cost_complete = False
            else:
                estimated_cost_usd += call_cost
            candidate_payload: dict[str, Any] | None = None
            try:
                candidate_payload = _json_object(candidate_response.text)
                decisions = candidate_payload.get("criteria", candidate_payload.get("results", []))
                if not isinstance(decisions, list):
                    raise ValueError("HealthBench grader did not return a criteria list")
                candidate_met: dict[int, bool] = {}
                for decision in decisions:
                    if not isinstance(decision, dict):
                        raise ValueError("HealthBench grader criteria entries must be objects")
                    try:
                        index = int(decision.get("index"))
                    except (TypeError, ValueError) as exc:
                        raise ValueError("HealthBench grader returned an invalid index") from exc
                    if index < 0 or index >= len(criteria):
                        raise ValueError("HealthBench grader returned an out-of-range index")
                    if index in candidate_met:
                        raise ValueError("HealthBench grader returned a duplicate index")
                    value = decision.get("criteria_met", decision.get("met"))
                    if not isinstance(value, bool):
                        raise ValueError("HealthBench grader criteria_met must be boolean")
                    candidate_met[index] = value
                missing = sorted(set(range(len(criteria))) - set(candidate_met))
                if missing:
                    raise ValueError(f"HealthBench grader omitted rubric indices: {missing}")
            except ValueError as exc:
                parse_failures += 1
                if not candidate_response.text.strip():
                    category = "empty_response"
                elif "JSON object" in str(exc):
                    category = "invalid_json"
                else:
                    category = "invalid_schema"
                error_categories[category] = error_categories.get(category, 0) + 1
                self._audit_attempt(
                    evaluation_id=evaluation_id,
                    task_id=task.task_id,
                    prediction_sha256=prediction_sha256,
                    criteria_total=len(criteria),
                    messages=messages,
                    attempt_index=attempt_index,
                    latency_s=attempt_latency,
                    response=candidate_response,
                    parsed_response=candidate_payload,
                    estimated_cost_usd=call_cost,
                    error_category=category,
                    error_message=f"{type(exc).__name__}: {exc}",
                )
                if attempt_index + 1 < self.max_grader_attempts:
                    messages = [
                        *messages,
                        {"role": "assistant", "content": candidate_response.text},
                        {
                            "role": "user",
                            "content": (
                                "The previous response did not satisfy the required JSON "
                                "schema. Return only one valid JSON object containing every "
                                "rubric index exactly once; do not include prose or fences."
                            ),
                        },
                    ]
                    continue
                self._audit_task_failure(
                    evaluation_id,
                    task.task_id,
                    prediction_sha256,
                    category,
                    attempts_total,
                )
                raise ValueError(
                    "HealthBench low-cost grader failed after "
                    f"{self.max_grader_attempts} attempts: {category}"
                ) from exc
            response = candidate_response
            assert candidate_payload is not None
            payload = candidate_payload
            met_by_index = candidate_met
            self._audit_attempt(
                evaluation_id=evaluation_id,
                task_id=task.task_id,
                prediction_sha256=prediction_sha256,
                criteria_total=len(criteria),
                messages=messages,
                attempt_index=attempt_index,
                latency_s=attempt_latency,
                response=response,
                parsed_response=payload,
                estimated_cost_usd=call_cost,
            )
            break
        if response is None:
            raise RuntimeError("HealthBench low-cost Judge exhausted without a result")
        weighted = sum(item["points"] for item in criteria if met_by_index[item["index"]])
        raw_score = weighted / positive_total
        score = healthbench_length_adjusted_score(raw_score, prediction)
        training_breakdown = healthbench_training_reward(
            weighted_points=weighted,
            positive_points=positive_total,
            negative_point_magnitude=negative_total,
            answer_length_chars=len(prediction),
        )
        result = VerificationResult(
            score=score,
            passed=score >= self.threshold,
            verifier=self.name,
            detail=json.dumps(
                {
                    "criteria_met": sum(met_by_index.values()),
                    "criteria_total": len(criteria),
                    "criterion_decisions": [
                        {"index": index, "criteria_met": met_by_index[index]}
                        for index in range(len(criteria))
                    ],
                    "weighted_points": weighted,
                    "positive_criteria_total": sum(item["points"] > 0 for item in criteria),
                    "negative_criteria_total": sum(item["points"] < 0 for item in criteria),
                    "positive_criteria_met": sum(
                        item["points"] > 0 and met_by_index[item["index"]] for item in criteria
                    ),
                    "negative_criteria_met": sum(
                        item["points"] < 0 and met_by_index[item["index"]] for item in criteria
                    ),
                    "positive_points": positive_total,
                    "negative_point_magnitude": negative_total,
                    "raw_score": raw_score,
                    "length_adjusted_score": score,
                    "training_reward_breakdown": training_breakdown.to_dict(),
                    "grading_mode": "single_call_all_rubrics",
                    "judge_model": response.model,
                    "judge_calls": attempts_total,
                    "parse_failures": parse_failures,
                    "retry_attempts": attempts_total - 1,
                    "errors_by_category": error_categories,
                    "token_in": token_in,
                    "token_out": token_out,
                    "latency_s": latency_s,
                    "estimated_cost_usd": (estimated_cost_usd if cost_complete else None),
                    "cost_complete": cost_complete,
                    "api_surface": response.metadata.get("api_surface"),
                    "reasoning_effort": response.metadata.get("reasoning_effort"),
                },
                ensure_ascii=False,
            ),
        )
        self._audit(
            {
                "event": "task_complete",
                "mode": "single_call_all_rubrics",
                "evaluation_id": evaluation_id,
                "task_id": task.task_id,
                "prediction_sha256": prediction_sha256,
                "criteria_total": len(criteria),
                "judge_calls": attempts_total,
                "score": score,
            }
        )
        self._cache[cache_key] = result
        return result

    def score_intermediate(self, task: TaskSpec, prediction: str) -> float:
        return healthbench_intermediate_mace_score(self.verify(task, prediction))

    def _estimated_cost(self, token_in: int, token_out: int) -> float | None:
        if self.input_cost_per_million is None or self.output_cost_per_million is None:
            return None
        return (
            float(token_in) * self.input_cost_per_million
            + float(token_out) * self.output_cost_per_million
        ) / 1_000_000.0

    def _audit(self, event: dict[str, Any]) -> None:
        if self.audit_store is not None:
            self.audit_store.append(event)

    def _audit_attempt(
        self,
        *,
        evaluation_id: str,
        task_id: str,
        prediction_sha256: str,
        criteria_total: int,
        messages: list[dict[str, str]],
        attempt_index: int,
        latency_s: float,
        response: Any | None = None,
        parsed_response: dict[str, Any] | None = None,
        estimated_cost_usd: float | None = None,
        error_category: str = "",
        error_message: str = "",
    ) -> None:
        metadata = dict(getattr(response, "metadata", {}) or {})
        self._audit(
            {
                "event": "judge_attempt",
                "mode": "single_call_all_rubrics",
                "evaluation_id": evaluation_id,
                "task_id": task_id,
                "prediction_sha256": prediction_sha256,
                "criterion_index": None,
                "attempt_index": attempt_index,
                "request_summary": {
                    "criteria_total": criteria_total,
                    "prompt_sha256": hashlib.sha256(
                        json.dumps(messages, ensure_ascii=False).encode()
                    ).hexdigest(),
                },
                "raw_response": (
                    str(getattr(response, "text", "")) if response is not None else None
                ),
                "parsed_response": parsed_response,
                "token_in": int(getattr(response, "token_in", 0) or 0),
                "token_out": int(getattr(response, "token_out", 0) or 0),
                "estimated_cost_usd": estimated_cost_usd,
                "latency_s": latency_s,
                "provider_model": metadata.get("provider_model"),
                "requested_model": metadata.get("requested_model"),
                "api_surface": metadata.get("api_surface"),
                "reasoning_effort": metadata.get("reasoning_effort"),
                "temperature_sent": metadata.get("temperature_sent"),
                "max_output_tokens_sent": metadata.get("max_output_tokens_sent"),
                "error_category": error_category,
                "error_message": error_message,
            }
        )

    def _audit_task_failure(
        self,
        evaluation_id: str,
        task_id: str,
        prediction_sha256: str,
        error_category: str,
        attempts_total: int,
    ) -> None:
        self._audit(
            {
                "event": "task_failed",
                "mode": "single_call_all_rubrics",
                "evaluation_id": evaluation_id,
                "task_id": task_id,
                "prediction_sha256": prediction_sha256,
                "error_category": error_category,
                "judge_calls": attempts_total,
                "partial_score_discarded": True,
            }
        )


def _json_object(text: str) -> dict[str, Any]:
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    candidates = [stripped]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if fenced:
        candidates.insert(0, fenced.group(1))
    first, last = stripped.find("{"), stripped.rfind("}")
    if first >= 0 and last > first:
        candidates.append(stripped[first : last + 1])
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except (TypeError, ValueError):
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("grader response does not contain a JSON object")
