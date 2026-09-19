from __future__ import annotations

import json
import re
import string
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any

from .aime_submission import is_aime_dataset, parse_aime_answer
from .llm import ChatBackend
from .observability import TaskSpec
from .qa_submission import extract_qa_answer, is_short_qa_dataset


@dataclass(frozen=True)
class AnswerSubmissionConfig:
    """Dataset-aware final-answer submission without adding a Canvas Agent."""

    enabled: bool = False
    qa_model_enabled: bool = False
    runtime_route: str = ""
    max_tokens: int = 128
    require_source_span: bool = True

    def validate(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("answer_submission.max_tokens must be positive")
        if self.enabled and self.qa_model_enabled and not self.runtime_route.strip():
            raise ValueError(
                "answer_submission.runtime_route is required when QA model formatting is enabled"
            )


@dataclass(frozen=True)
class AnswerSubmission:
    raw_answer: str
    submitted_answer: str
    method: str
    changed: bool
    valid: bool
    formatter_route: str | None = None
    fallback_used: bool = False
    detail: str = ""
    formatter_token_in: int = 0
    formatter_token_out: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AnswerFinalizer:
    """Convert an output artifact into the exact value submitted to a verifier.

    The formatter never receives the reference answer. For open-domain QA, a
    model-authored extraction is accepted only when it is grounded as a literal
    span in the original answer. Summary is evidence, never another candidate
    pool from which the formatter can choose a different conclusion.
    """

    def __init__(
        self,
        config: AnswerSubmissionConfig | None = None,
        *,
        qa_backend: ChatBackend | None = None,
    ) -> None:
        self.config = config or AnswerSubmissionConfig()
        self.config.validate()
        self.qa_backend = qa_backend
        if self.config.enabled and self.config.qa_model_enabled and qa_backend is None:
            raise ValueError("QA model formatting is enabled but no formatter backend was provided")

    def finalize(
        self,
        task: TaskSpec,
        raw_answer: str,
        *,
        raw_summary: str = "",
        allow_model: bool = True,
    ) -> AnswerSubmission:
        raw = str(raw_answer or "").strip()
        # Dataset correctness boundaries cannot be bypassed by disabling optional
        # formatting. AIME never invokes the QA formatter or reads the summary.
        if is_aime_dataset(task.metadata.get("dataset", "")):
            parsed = parse_aime_answer(raw)
            return AnswerSubmission(
                raw_answer=raw,
                submitted_answer=parsed.answer,
                method="aime_strict_submission_v1",
                changed=parsed.answer != raw,
                valid=parsed.valid,
                detail=parsed.reason,
            )
        if not self.config.enabled:
            return _submission(raw, raw, method="disabled_passthrough")

        kind = submission_kind(task)
        fallback, fallback_method = _deterministic_submission(kind, raw)
        if kind == "qa" and not fallback:
            return _submission(raw, "", method="qa_ambiguous_submission")
        if (
            kind != "qa"
            or not allow_model
            or not self.config.qa_model_enabled
            or self.qa_backend is None
            or not raw
        ):
            return _submission(raw, fallback, method=fallback_method)

        # A committed, explicitly delimited answer was already extracted. A
        # second model must not re-interpret it (or consult another candidate).
        if fallback != raw:
            return _submission(raw, fallback, method=fallback_method)

        formatter_token_in = formatter_token_out = 0
        try:
            evidence_mode = str(task.metadata.get("evidence_mode", "")).strip().casefold()
            context_documents = task.metadata.get("context_documents", [])
            if not isinstance(context_documents, list):
                context_documents = []
            use_evidence_spans = is_short_answer_qa(task) and evidence_mode in {
                "provided_context",
                "provided_context_inline",
            }
            visible_context = (
                [
                    {
                        "id": str(doc.get("id", f"D{i + 1}")),
                        "text": str(doc.get("text", "")),
                    }
                    for i, doc in enumerate(context_documents[:20])
                    if isinstance(doc, dict) and str(doc.get("text", "")).strip()
                ]
                if use_evidence_spans
                else []
            )
            original_question = str(task.metadata.get("original_question", "")).strip()
            if not original_question:
                original_question = _question_from_inline_prompt(task.prompt)
            question_type = _nq_question_type(original_question)
            grounding_instruction = (
                "For provided-context QA, choose the answer from the supplied evidence passages. "
                "The answer must be copied as one shortest contiguous span from a passage; do "
                "not invent or paraphrase it. The raw answer is only a proposal and may be "
                "wrong. "
                if visible_context
                else "The answer must be copied from the supplied raw answer, not summary. "
            )
            response = self.qa_backend.generate(
                [
                    {
                        "role": "system",
                        "content": (
                            "Extract the shortest answer span that directly answers the question. "
                            "Do not solve the question again, add facts, or use outside knowledge. "
                            f"{grounding_instruction}"
                            f"Question type is {question_type}; select only an answer of this "
                            "type (person, date, location, number, yes/no, or explicitly "
                            "requested list). Never select a merely related entity. "
                            "Extract only an unambiguous answer committed to by the original author. "
                            "Preserve necessary units, date ranges, qualifications and precision. "
                            "Do not choose among conflicting candidates or regional dates, prefer "
                            "the first/last date, or convert a period into a specific year. "
                            "If extraction requires deciding the answer, return an empty answer. "
                            'Return exactly one JSON object: {"answer":"..."}.'
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "question": original_question,
                                "raw_answer": raw,
                                "raw_summary": str(raw_summary or ""),
                                "evidence_passages": visible_context,
                                "question_type": question_type,
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
                role="answer-formatter",
                actions=(),
                temperature=0.0,
                max_tokens=self.config.max_tokens,
                enable_thinking=False,
            )
            formatter_token_in = int(response.token_in)
            formatter_token_out = int(response.token_out)
            candidate = _json_answer(response.text)
            if not candidate:
                raise ValueError("formatter returned no answer")
            # This safety boundary is mandatory for short QA even in old
            # configurations that disabled the optional generic span check.
            source_text = raw
            if visible_context:
                source_text += "\n" + "\n".join(
                    str(doc.get("text", ""))
                    for doc in task.metadata.get("context_documents", [])
                    if isinstance(doc, dict)
                )
            if (
                is_short_answer_qa(task) or self.config.require_source_span
            ) and not _is_source_span(candidate, source_text):
                raise ValueError("formatter answer is not grounded in the source artifact")
            _validate_qa_format_only(raw, candidate)
            return AnswerSubmission(
                raw_answer=raw,
                submitted_answer=candidate,
                method="qa_model_source_span",
                changed=candidate != raw,
                valid=True,
                formatter_route=self.config.runtime_route,
                formatter_token_in=int(response.token_in),
                formatter_token_out=int(response.token_out),
            )
        except Exception as exc:  # formatting must not invalidate a completed primary
            return AnswerSubmission(
                raw_answer=raw,
                submitted_answer=fallback,
                method=fallback_method,
                changed=fallback != raw,
                valid=bool(fallback),
                formatter_route=self.config.runtime_route,
                fallback_used=True,
                detail=f"{type(exc).__name__}: {exc}",
                formatter_token_in=formatter_token_in,
                formatter_token_out=formatter_token_out,
            )


def _nq_question_type(question: str) -> str:
    """Coarse answer-type hint used only to constrain final span selection."""
    q = str(question or "").strip().casefold()
    if re.match(r"^(who|which person|whose)\b", q):
        return "person/entity"
    if re.match(r"^(when|what year|what date|in what year)\b", q):
        return "date/year"
    if re.match(r"^(where|what city|what country|what state)\b", q):
        return "location"
    if re.match(r"^(how many|how much|what number|what percentage)\b", q):
        return "number"
    if re.match(r"^(did|does|is|are|was|were|has|have|can|could|will)\b", q):
        return "yes/no or short fact"
    if any(word in q for word in ("list", "which countries", "what are the", "who are the")):
        return "explicit list"
    return "short entity/date/location/phrase"


def _question_from_inline_prompt(prompt: str) -> str:
    """Recover the question from the frozen-context prompt when metadata is old."""
    text = str(prompt or "")
    match = re.search(r"(?:^|\n)Question:\s*(.+?)(?:\nAnswer:\s*|$)", text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def submission_kind(task: TaskSpec) -> str:
    if is_short_answer_qa(task):
        return "qa"
    requested = str(task.metadata.get("verifier", "")).strip().casefold()
    if requested in {"numeric"}:
        return "numeric"
    if requested in {"multiple_choice"}:
        return "multiple_choice"
    if requested in {"exact_match", "multi_answer_exact_match", "flowsteer_qa"}:
        return "qa"
    task_type = task.task_type.casefold()
    if any(value in task_type for value in ("math", "numeric", "number")):
        return "numeric"
    if any(value in task_type for value in ("multiple_choice", "choice", "mcq", "gpqa")):
        return "multiple_choice"
    if any(value in task_type for value in ("qa", "question", "retrieval", "hotpot", "nq")):
        return "qa"
    dataset = str(task.metadata.get("dataset", "")).casefold()
    if any(value in dataset for value in ("hotpot", "naturalquestions", "nq_open", "trivia")):
        return "qa"
    return "passthrough"


def is_short_answer_qa(task: TaskSpec) -> bool:
    """Dataset identity only: never infer an answer type from private labels."""
    return is_short_qa_dataset(task.metadata.get("dataset", ""))


def submission_contract(task: TaskSpec) -> str:
    if is_aime_dataset(task.metadata.get("dataset", "")):
        return (
            "Submission contract: The selected output Agent must put one final integer "
            "from 0 to 999 in the answer field. Put derivations in summary or evidence. "
            "Intermediate Agents may retain local findings; lists or ambiguous candidates "
            "cannot be submitted as the final answer."
        )
    kind = submission_kind(task)
    if kind == "numeric":
        return (
            "Submission contract: Put only the final number in the answer field. Put every "
            "derivation, explanation, unit, and confidence statement in summary or evidence."
        )
    if kind == "multiple_choice":
        return (
            "Submission contract: Put only the final option label in the answer field. Put all "
            "reasoning in summary or evidence."
        )
    if kind == "qa":
        return (
            "Submission contract: The selected output Agent must put only the shortest entity, "
            "date, location, title, or phrase that directly answers the question in the answer "
            "field. Do not add explanations, "
            "appositives or related facts there; put them in summary or evidence. Preserve units "
            "and qualifications needed to express the answer without changing its meaning. "
            "Intermediate Agents may retain local findings for their assigned responsibilities; "
            "they are not required to answer the entire question."
        )
    return (
        "Submission contract: Put only the direct task result in the answer field and place all "
        "supporting explanation in summary or evidence."
    )


def qa_token_f1(task: TaskSpec, prediction: str) -> float | None:
    if submission_kind(task) != "qa" or task.reference is None:
        return None
    references = (
        list(task.reference) if isinstance(task.reference, (list, tuple, set)) else [task.reference]
    )
    predicted = _qa_tokens(prediction)
    if not predicted:
        return 0.0
    best = 0.0
    for reference in references:
        expected = _qa_tokens(str(reference))
        if not expected:
            continue
        remaining = list(expected)
        overlap = 0
        for token in predicted:
            if token in remaining:
                remaining.remove(token)
                overlap += 1
        if not overlap:
            continue
        precision = overlap / len(predicted)
        recall = overlap / len(expected)
        best = max(best, 2.0 * precision * recall / (precision + recall))
    return best


def _deterministic_submission(kind: str, raw: str) -> tuple[str, str]:
    if kind == "qa":
        return extract_qa_answer(raw), "qa_deterministic_extraction"
    explicit = _explicit_answer(raw)
    if kind == "numeric":
        numbers = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", explicit)
        return (numbers[-1] if numbers else explicit), "numeric_extraction"
    if kind == "multiple_choice":
        direct = re.fullmatch(r"\s*\(?([A-Z])\)?[.)]?\s*", explicit, flags=re.IGNORECASE)
        marked = re.findall(
            r"(?i)(?:option|choice|answer)\s*(?:is|=|:)?\s*\(?([A-Z])\)?",
            raw,
        )
        choice = direct.group(1) if direct else (marked[-1] if marked else "")
        return (choice.upper() if choice else explicit), "choice_extraction"
    return raw, "passthrough"


def _explicit_answer(raw: str) -> str:
    payload = _json_payload(raw)
    if isinstance(payload, dict):
        value = payload.get("answer", payload.get("final_answer"))
        if value not in (None, ""):
            return str(value).strip()
    matches = re.findall(
        r"(?i)(?:final\s+answer|short\s+answer|answer|答案)\s*(?:is|=|:|：)?\s*([^\n\r]+)",
        raw,
    )
    if matches:
        return matches[-1].strip()
    boxed = re.findall(r"\\boxed\s*\{([^{}]+)\}", raw)
    return boxed[-1].strip() if boxed else raw.strip()


def _json_answer(text: str) -> str:
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate formatter JSON key")
            result[key] = value
        return result

    stripped = str(text).strip()
    fenced = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", stripped, re.IGNORECASE)
    if fenced:
        stripped = fenced.group(1)
    payload = json.loads(stripped, object_pairs_hook=unique_pairs)
    if not isinstance(payload, dict):
        return ""
    if (
        "answer" in payload
        and "final_answer" in payload
        and payload["answer"] != payload["final_answer"]
    ):
        raise ValueError("conflicting formatter answer fields")
    value = payload.get("answer", payload.get("final_answer"))
    return value.strip() if isinstance(value, str) else ""


def _validate_qa_format_only(raw: str, candidate: str) -> None:
    """Conservative reference-blind guards, not a semantic correctness judge.

    Under-extraction is preferable to silently resolving uncertainty. These
    guards do not prove semantic equivalence; model formatting stays opt-in.
    """
    source, answer = _span_normalize(raw), _span_normalize(candidate)
    if source == answer:
        return
    if re.search(r"\b(?:or|either|versus|vs|alternatively|however)\b|[;；]|或者|或是", source):
        raise ValueError("formatter must not choose among alternative candidates")
    # Preserve all numerical precision, not just a substring of a date/range.
    if re.findall(r"\d+(?:[.,]\d+)*", source) != re.findall(r"\d+(?:[.,]\d+)*", answer):
        raise ValueError("formatter would change numeric/date precision or select a candidate")
    protected = (
        r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december|"
        r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec|"
        r"early|late|mid|about|approximately|around|between|before|after|until|"
        r"not|no|yes|never|neither|nor|possibly|perhaps|maybe|and|"
        r"years?|months?|days?|hours?|minutes?|seconds?|kg|km|cm|mm|"
        r"grams?|kilograms?|meters?|metres?|miles?|percent|dollars?)\b"
    )
    if re.findall(protected, source) != re.findall(protected, answer):
        raise ValueError("formatter would drop a qualification, polarity, or precision")


def _json_payload(text: str) -> Any:
    stripped = str(text).strip()
    try:
        return json.loads(stripped)
    except (TypeError, ValueError):
        pass
    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(stripped[index:])
        except ValueError:
            continue
        return payload
    return None


def _is_source_span(candidate: str, source: str) -> bool:
    normalized_candidate = _span_normalize(candidate)
    normalized_source = _span_normalize(source)
    if not normalized_candidate:
        return False
    pattern = re.escape(normalized_candidate)
    if normalized_candidate[0].isalnum():
        pattern = r"(?<!\w)" + pattern
    if normalized_candidate[-1].isalnum():
        pattern += r"(?!\w)"
    return re.search(pattern, normalized_source) is not None


def _span_normalize(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value).casefold()).strip()


def _qa_tokens(value: str) -> list[str]:
    lowered = unicodedata.normalize("NFKC", value).casefold()
    without_punctuation = "".join(
        " " if char in string.punctuation or unicodedata.category(char).startswith("P") else char
        for char in lowered
    )
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return re.sub(r"\s+", " ", without_articles).strip().split()


def _submission(raw: str, submitted: str, *, method: str) -> AnswerSubmission:
    return AnswerSubmission(
        raw_answer=raw,
        submitted_answer=submitted,
        method=method,
        changed=submitted != raw,
        valid=bool(submitted),
    )
