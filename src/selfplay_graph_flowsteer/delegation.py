from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .graph import MultiAgentGraph

DUPLICATE_RESPONSIBILITY_POLICIES = frozenset({"off", "record_only", "warn_once", "reject"})

_ANSWER_OR_SOLUTION_RE = re.compile(
    r"(?:final\s+answer|answer|答案)\s*(?:is|=|:|：)\s*\S+|"
    r"(?:candidate\s+answer|intermediate\s+calculation|solution\s+steps?)",
    flags=re.IGNORECASE,
)
_PROCEDURAL_SOLUTION_RE = re.compile(
    r"\b(?:use|apply)\b.{0,160}\b(?:to\s+)?"
    r"(?:derive|calculate|compute|prove|solve|obtain)\b|"
    r"\b(?:first|then|next)\b.{0,100}\b"
    r"(?:derive|calculate|compute|prove|solve|obtain)\b",
    flags=re.IGNORECASE,
)
_ROUTING_RE = re.compile(
    r"\b(?:mace|runtime|model)\s*(?:router|routing|selection)\b|"
    r"\b(?:select|choose|rank|evaluate)\b.{0,24}\b(?:candidate\s+)?models?\b",
    flags=re.IGNORECASE,
)
_GENERIC_TASK_RE = re.compile(
    r"\b(?:assigned|current|original|given)\s+(?:task|problem|question)\b",
    flags=re.IGNORECASE,
)
_DOWNSTREAM_CONTRIBUTION_RE = re.compile(
    r"\b(?:upstream|available|provided|received)\b.{0,40}"
    r"\b(?:evidence|findings|results?|answers?|artifacts?|data)\b|"
    r"\b(?:synthesi[sz]e|aggregate|combine|compare|critique|review|verify)\b",
    flags=re.IGNORECASE,
)
_WORDS_RE = re.compile(r"[a-z0-9]+", flags=re.IGNORECASE)

_RESPONSIBILITY_TYPE_PATTERNS: dict[str, re.Pattern[str]] = {
    "implementation": re.compile(
        r"\b(?:implement|implementation|modify|modification|patch|fix|repair|"
        r"add|change|update|developer|code)\b|\b_cdf\b",
        flags=re.IGNORECASE,
    ),
    "diagnosis": re.compile(
        r"\b(?:diagnos(?:e|is)|investigat(?:e|ion)|root cause|locali[sz](?:e|ation)|"
        r"identify the (?:cause|failure|bug))\b",
        flags=re.IGNORECASE,
    ),
    "testing": re.compile(
        r"\b(?:test|tests|testing|validate|validation|verify|verification|regression|"
        r"differentiat(?:e|ion))\b",
        flags=re.IGNORECASE,
    ),
    "review": re.compile(
        r"\b(?:review|audit|critique|check correctness|falsif(?:y|ication))\b",
        flags=re.IGNORECASE,
    ),
    "synthesis": re.compile(
        r"\b(?:synthesi[sz](?:e|er|is)|aggregate|combine|merge)\b|"
        r"\bintegrat(?:e|ion)\s+(?:findings|artifacts|results|patches)\b",
        flags=re.IGNORECASE,
    ),
    "environment_execution": re.compile(
        r"\b(?:execute|interact|navigate|purchase|configure|environment|stateful)\b",
        flags=re.IGNORECASE,
    ),
}

_DELIVERABLE_PATTERNS: dict[str, re.Pattern[str]] = {
    "code_change": re.compile(
        r"\b(?:code|patch|implementation|implementations|modification|modifications|"
        r"fixed|fix|method|methods|workspace change)\b|\b_cdf\b",
        flags=re.IGNORECASE,
    ),
    "tests": re.compile(
        r"\b(?:test|tests|testing|validation|validated|verified|regression|"
        r"differentiat(?:e|ion))\b",
        flags=re.IGNORECASE,
    ),
    "root_cause_report": re.compile(
        r"\b(?:root cause|diagnosis|diagnostic|findings|analysis|report)\b",
        flags=re.IGNORECASE,
    ),
    "final_answer": re.compile(
        r"\b(?:final answer|requested value|concise answer|answer span)\b",
        flags=re.IGNORECASE,
    ),
    "environment_completion": re.compile(
        r"\b(?:purchase completion|environment outcome|configured environment|"
        r"official outcome)\b",
        flags=re.IGNORECASE,
    ),
}

_RESPONSIBILITY_PRIMARY_ORDER = (
    "implementation",
    "environment_execution",
    "diagnosis",
    "testing",
    "review",
    "synthesis",
)


@dataclass(frozen=True)
class ResponsibilitySignature:
    """Deterministic semantic summary of one Director-authored responsibility."""

    types: tuple[str, ...]
    primary_type: str
    scope_entities: frozenset[str]
    objective_tokens: frozenset[str]
    expected_output_tokens: frozenset[str]
    deliverables: frozenset[str]


@dataclass(frozen=True)
class ResponsibilityComparison:
    """Pairwise responsibility overlap used for admission and offline audits."""

    same_primary_type: bool
    scope_overlap: float
    objective_overlap: float
    expected_output_overlap: float
    deliverable_overlap: float
    similarity: float
    shared_scope_entities: tuple[str, ...]
    high_confidence_duplicate: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "same_primary_type": self.same_primary_type,
            "scope_overlap": self.scope_overlap,
            "objective_overlap": self.objective_overlap,
            "expected_output_overlap": self.expected_output_overlap,
            "deliverable_overlap": self.deliverable_overlap,
            "similarity": self.similarity,
            "shared_scope_entities": list(self.shared_scope_entities),
            "high_confidence_duplicate": self.high_confidence_duplicate,
        }


DELEGATION_CONTRACT_VERSION = "dataset-output-contract-v1"
WEBSHOP_DELEGATION_CONTRACT_VERSION = "webshop-terminal-contract-v3-neutral"
DELEGATION_FIELD_LIMITS = {
    "role": 80,
    "objective": 320,
    "scope": 240,
    "expected_output": 200,
}
DELEGATION_TOTAL_LIMIT = 700

_WEBSHOP_EXPECTED_OUTPUT = (
    "Choose a product and requested options using public evidence; decide whether to call "
    "latest Buy Now. Record uncertainty honestly; never claim an unresolved constraint is verified."
)

_DATASET_OUTPUT_CONTRACTS: dict[str, tuple[tuple[str, str], ...]] = {
    "aime": (
        (
            "preserve_exact_forms",
            "Preserve exact integers, fractions, and radicals in the reported result.",
        ),
        (
            "identify_requested_value",
            "State the requested final value unambiguously with concise supporting evidence.",
        ),
    ),
    "nq_open": (
        (
            "concise_answer_span",
            "Return a concise answer span and evidence that directly supports it.",
        ),
        (
            "report_uncertainty",
            "Report unresolved uncertainty instead of inventing unsupported details.",
        ),
    ),
    "hotpotqa": (
        (
            "multi_fact_evidence",
            "Return a concise answer and the evidence connecting the relevant facts or entities.",
        ),
        (
            "report_uncertainty",
            "Report unresolved uncertainty instead of inventing unsupported details.",
        ),
    ),
    "webshop": (
        (
            "public_catalog_search",
            "Search accepts the Agent's query and returns public catalog results. The runtime does not supply a query, rank candidates by the hidden goal, or prescribe query terms.",
        ),
        (
            "interpret_search_previews",
            "Search-result option values are display defaults, not the session's selected options. Search history and product-page observations are public evidence; they do not reveal hidden candidate scores.",
        ),
        (
            "public_candidate_evidence",
            "Titles, prices, available options and observed sections are visible to the Agent. The Agent decides how to interpret them; no candidate or evidence-coverage order is selected by the runtime.",
        ),
        (
            "product_option_state",
            "One selected value is retained per option group; a later selection replaces it. The latest live selected_options is authoritative for the session. Observed Description and Features text is retained in webshop_progress. The Agent chooses whether to select an option or navigate.",
        ),
        (
            "agent_purchase_authority",
            "The Agent decides whether to inspect, compare, purchase or report a blocker. Purchase evidence records that decision and its unresolved constraints; it is not an oracle correctness check. The official environment determines the score.",
        ),
        (
            "complete_environment_purchase",
            "A recommendation or product page is not an environment purchase. A Buy Now Action stages a purchase in this session; it is committed only after Canvas selects the output Agent. No purchase is automatically chosen.",
        ),
        (
            "follow_live_subactions",
            "Use only target identifiers exposed by the latest environment observation; the runtime executes exactly one staged purchase after Canvas selects the output Agent.",
        ),
        (
            "report_environment_outcome",
            "Report staged status and unresolved constraints without claiming purchase success before the selected staged action is committed by the runtime.",
        ),
    ),
    "alfworld": (
        (
            "execute_environment_actions",
            "Use alfworld_step with an action_id from the latest observation; naming or recommending a command without calling the Action does not change the environment.",
        ),
        (
            "continue_until_environment_done",
            "Continue one stateful Action at a time, re-observing after every call, until the official environment reports success or termination.",
        ),
        (
            "follow_live_action_ids",
            "Use only state-scoped action identifiers exposed by the latest observation; never reuse stale identifiers.",
        ),
    ),
    "swe_bench": (
        (
            "produce_repository_patch",
            "The task outcome is the runtime-exported repository patch; analysis or a proposed change in text alone is not a completed fix.",
        ),
        (
            "ground_changes_in_repository",
            "Use the visible repository Actions to inspect the trusted checkout and edit it only when the observed code supports the change.",
        ),
        (
            "validate_workspace_change",
            "After changing code, use an available configured test profile when feasible and report unresolved failures without claiming success from model text.",
        ),
    ),
}


@dataclass(frozen=True)
class DelegationIssue:
    """Machine-readable reason why a Director responsibility is unsafe."""

    code: str
    message: str
    field: str | None = None
    details: dict[str, Any] | None = None


class DelegationValidationError(ValueError):
    """Raised with structured detail while preserving transactional rollback."""

    def __init__(self, issue: DelegationIssue) -> None:
        self.issue = issue
        suffix = f" (field={issue.field})" if issue.field else ""
        super().__init__(issue.message + suffix)


@dataclass(frozen=True)
class DelegationFieldRepair:
    """A deterministic normalization applied to one Director-authored field."""

    field: str
    reason: str
    original_length: int
    repaired_length: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "reason": self.reason,
            "original_length": self.original_length,
            "repaired_length": self.repaired_length,
        }


@dataclass(frozen=True)
class DelegationCompilation:
    """Compiled Worker prompt with provenance kept separate from system rules."""

    prompt: str
    director_fields: dict[str, str]
    dataset: str
    contract_version: str | None
    contract_rule_ids: tuple[str, ...]
    field_repairs: tuple[DelegationFieldRepair, ...] = ()

    def metadata(self) -> dict[str, Any]:
        return {
            "director_delegation": dict(self.director_fields),
            "system_managed_contract": {
                "version": self.contract_version,
                "dataset": self.dataset,
                "rule_ids": list(self.contract_rule_ids),
            },
            "delegation_field_repairs": [repair.to_dict() for repair in self.field_repairs],
        }


_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "answer",
        "agent",
        "analyze",
        "analysis",
        "as",
        "at",
        "be",
        "best",
        "by",
        "concise",
        "direct",
        "for",
        "from",
        "given",
        "identify",
        "in",
        "independent",
        "independently",
        "information",
        "is",
        "of",
        "on",
        "or",
        "objective",
        "output",
        "provide",
        "provided",
        "relevant",
        "report",
        "result",
        "return",
        "role",
        "scope",
        "specialist",
        "task",
        "the",
        "to",
        "using",
        "was",
        "when",
        "which",
        "with",
        "expected",
    }
)

_WEBSHOP_DELEGATION_PROCESS_WORDS = frozenset(
    {
        *_STOPWORDS,
        # Grammatical forms of "have" connect public attributes; they do not
        # introduce a product constraint (e.g. "flavored, has zero added sugar").
        # Keep this local: the shared stopwords also serve other delegation checks.
        "had",
        "has",
        "have",
        "having",
        "artifact",
        "artifacts",
        "visible",
        "available",
        "availability",
        "below",
        "budget",
        "buy",
        "before",
        "candidate",
        "candidates",
        "catalog",
        "category",
        "complete",
        "completion",
        "continue",
        "constraint",
        "constraints",
        "criteria",
        "detail",
        "details",
        "dimension",
        "dimensions",
        "dollar",
        "dollars",
        "environment",
        "evidence",
        "evaluate",
        "evaluation",
        "every",
        "exact",
        "exclude",
        "filter",
        "filtering",
        "find",
        "finding",
        "findings",
        "fulfill",
        "flavor",
        "inspect",
        "inspection",
        "installation",
        "interact",
        "interaction",
        "issue",
        "item",
        "items",
        "latest",
        "lightweight",
        "limit",
        "link",
        "links",
        "list",
        "listing",
        "locate",
        "lower",
        "match",
        "matches",
        "matching",
        "navigate",
        "navigation",
        "online",
        "only",
        "option",
        "options",
        "owner",
        "price",
        "priced",
        "prices",
        "product",
        "products",
        "public",
        "purchase",
        "purchasing",
        "qualifying",
        "recheck",
        "remaining",
        "retailer",
        "retailers",
        "resolve",
        "requested",
        "search",
        "select",
        "selected",
        "shop",
        "shopper",
        "shopping",
        "specified",
        "store",
        "stores",
        "under",
        "usd",
        "use",
        "value",
        "values",
        "verification",
        "verify",
        "webshop",
        "attempt",
        # Attribute class labels may describe the responsibility, but concrete
        # values still must occur in the trusted public task.
        "attribute",
        "attributes",
        "are",
        "color",
        "count",
        "each",
        "exactly",
        "fits",
        "flavors",
        "handling",
        "height",
        "high",
        "holds",
        "if",
        "inch",
        "inches",
        "length",
        "less",
        "material",
        "measuring",
        "measures",
        "meeting",
        "meets",
        "oz",
        "piece",
        "print",
        "printed",
        "size",
        "sized",
        "specification",
        "specifications",
        "style",
        "that",
        "then",
        "type",
        "unit",
        "volume",
        "wide",
        "width",
    }
)

_RESPONSIBILITY_SCOPE_STOPWORDS = frozenset(
    {
        *_STOPWORDS,
        "add",
        "all",
        "code",
        "correct",
        "correctly",
        "ensure",
        "full",
        "internal",
        "known",
        "method",
        "methods",
        "missing",
        "performance",
        "poor",
        "respective",
        "specific",
        "working",
    }
)


def responsibility_signature(fields: dict[str, object]) -> ResponsibilitySignature:
    """Extract a bounded, deterministic signature from Director-authored fields."""

    role = str(fields.get("role") or "")
    objective = str(fields.get("objective") or "")
    scope = str(fields.get("scope") or "")
    expected_output = str(fields.get("expected_output") or "")
    responsibility_text = " ".join((role, objective, expected_output))
    types = tuple(
        name
        for name, pattern in _RESPONSIBILITY_TYPE_PATTERNS.items()
        if pattern.search(responsibility_text)
    )
    primary_type = next(
        (name for name in _RESPONSIBILITY_PRIMARY_ORDER if name in types),
        "general",
    )
    deliverables = frozenset(
        name for name, pattern in _DELIVERABLE_PATTERNS.items() if pattern.search(expected_output)
    )
    return ResponsibilitySignature(
        types=types,
        primary_type=primary_type,
        scope_entities=_responsibility_tokens(
            scope,
            stopwords=_RESPONSIBILITY_SCOPE_STOPWORDS,
        ),
        objective_tokens=_responsibility_tokens(objective, stopwords=_STOPWORDS),
        expected_output_tokens=_responsibility_tokens(
            expected_output,
            stopwords=_STOPWORDS,
        ),
        deliverables=deliverables,
    )


def compare_responsibilities(
    left_fields: dict[str, object],
    right_fields: dict[str, object],
) -> ResponsibilityComparison:
    """Compare two responsibilities without using system-managed prompt text."""

    left = responsibility_signature(left_fields)
    right = responsibility_signature(right_fields)
    scope_overlap = _containment_overlap(left.scope_entities, right.scope_entities)
    objective_overlap = _jaccard(left.objective_tokens, right.objective_tokens)
    expected_output_overlap = _jaccard(
        left.expected_output_tokens,
        right.expected_output_tokens,
    )
    deliverable_overlap = _containment_overlap(left.deliverables, right.deliverables)
    similarity = (
        0.50 * scope_overlap
        + 0.20 * objective_overlap
        + 0.10 * expected_output_overlap
        + 0.20 * deliverable_overlap
    )
    same_primary_type = left.primary_type == right.primary_type
    high_confidence_duplicate = bool(
        same_primary_type
        and left.primary_type == "implementation"
        and scope_overlap >= 0.80
        and deliverable_overlap >= 0.75
        and similarity >= 0.78
    )
    return ResponsibilityComparison(
        same_primary_type=same_primary_type,
        scope_overlap=round(scope_overlap, 6),
        objective_overlap=round(objective_overlap, 6),
        expected_output_overlap=round(expected_output_overlap, 6),
        deliverable_overlap=round(deliverable_overlap, 6),
        similarity=round(similarity, 6),
        shared_scope_entities=tuple(sorted(left.scope_entities & right.scope_entities)[:24]),
        high_confidence_duplicate=high_confidence_duplicate,
    )


def _responsibility_tokens(
    text: str,
    *,
    stopwords: frozenset[str],
) -> frozenset[str]:
    return frozenset(
        token
        for token in (match.casefold() for match in _WORDS_RE.findall(str(text or "")))
        if len(token) >= 2 and token not in stopwords
    )


def _containment_overlap(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def delegation_safety_issue(
    text: str,
    *,
    action_names: Iterable[str] = (),
    field: str | None = None,
) -> DelegationIssue | None:
    """Return a deterministic responsibility issue, or ``None`` when safe.

    Task entities and goals are deliberately allowed.  The hard boundary covers
    only answer/solution leakage, explicit Action control, and runtime routing.
    """

    delegation = str(text or "").strip()
    if _ANSWER_OR_SOLUTION_RE.search(delegation):
        return DelegationIssue(
            "answer_or_solution_leak",
            "SET_PROMPT contains an answer clue or solution content",
            field,
        )
    if _PROCEDURAL_SOLUTION_RE.search(delegation):
        return DelegationIssue(
            "concrete_solution_procedure",
            "SET_PROMPT prescribes a concrete solution procedure",
            field,
        )
    if _ROUTING_RE.search(delegation):
        return DelegationIssue(
            "runtime_routing_control",
            "SET_PROMPT must not delegate runtime routing or model selection",
            field,
        )

    lowered = delegation.casefold()
    for raw_name in action_names:
        name = str(raw_name or "").strip().casefold()
        if not name:
            continue
        escaped = re.escape(name)
        code_like = "_" in name and re.search(rf"\b{escaped}\b", lowered)
        explicit_invocation = re.search(
            rf"\b(?:call|invoke|execute|run)\b.{{0,24}}\b{escaped}\b|"
            rf"\buse\b.{{0,16}}\b{escaped}\b"
            rf"(?!\s+(?:results?|output|evidence|findings|records?|documents?))|"
            rf"\b(?:use|select|choose)\b.{{0,16}}\b{escaped}\b"
            rf"\s+(?:action|tool|function|api)\b|"
            rf"\b{escaped}\b\s+(?:action|tool|function|api)\b",
            lowered,
        )
        if code_like or explicit_invocation:
            return DelegationIssue(
                "worker_action_control",
                "SET_PROMPT must not select or constrain Worker Actions",
                field,
            )
    return None


def delegation_safety_error(
    text: str,
    *,
    action_names: Iterable[str] = (),
) -> str | None:
    """Backward-compatible string form of :func:`delegation_safety_issue`."""

    issue = delegation_safety_issue(text, action_names=action_names)
    return issue.message if issue else None


def _webshop_alignment_text(text: str) -> str:
    """Normalize grammar for comparison only, never the Worker prompt."""

    return re.sub(r"\b([a-z0-9]+)['’]s\b", r"\1", text, flags=re.IGNORECASE)


def _webshop_public_word(token: str, public_tokens: set[str]) -> str:
    """Accept a regular plural only when its base word is public already.

    Do not use the shared stemmer: reducing arbitrary task/entity words would
    weaken the content gate and affect unrelated delegation consumers.
    """

    if token in public_tokens or token in {"news", "means", "series", "species", "clothes"}:
        return token
    candidates: list[str] = []
    if len(token) > 4 and token.endswith("ies"):
        candidates.append(token[:-3] + "y")
    if token.endswith("es") and token[:-2].endswith(("s", "x", "z", "ch", "sh")):
        candidates.append(token[:-2])
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is", "ics")):
        candidates.append(token[:-1])
    return next((base for base in candidates if base in public_tokens), token)


def delegation_task_alignment_issue(
    fields: dict[str, object],
    *,
    public_task: str,
    dataset: str,
) -> DelegationIssue | None:
    """Reject WebShop task nouns invented by a Director responsibility.

    The public task is the constraint authority.  Generic process vocabulary is
    allowed, but a new content token in objective/scope/expected_output can turn
    an ambiguous request into a different shopping task.  This gate uses only
    public text and never consults goal metadata or evaluator answers.
    """

    if str(dataset or "").strip().casefold() != "webshop":
        return None
    task_tokens = {
        token.casefold()
        for token in _WORDS_RE.findall(_webshop_alignment_text(str(public_task or "")))
    }
    novel_by_field: dict[str, set[str]] = {}
    tokens_by_field: dict[str, list[str]] = {}
    for field_name in ("objective", "scope"):
        value = _webshop_alignment_text(str(fields.get(field_name) or ""))
        # "Alternative search/queries" describes a process, not an alternative
        # product requirement. Do not globally allow "alternative" (e.g. colour).
        value = re.sub(
            r"\balternative\s+(?:search|queries|query)\b",
            "search",
            value,
            flags=re.IGNORECASE,
        )
        tokens_by_field[field_name] = [
            _webshop_public_word(token.casefold(), task_tokens)
            for token in _WORDS_RE.findall(value)
        ]
        novel_by_field[field_name] = {
            token
            for token in tokens_by_field[field_name]
            if not any(character.isdigit() for character in token)
            and token.casefold() not in task_tokens
            and token.casefold() not in _WEBSHOP_DELEGATION_PROCESS_WORDS
        }
    public_content_tokens = task_tokens - _WEBSHOP_DELEGATION_PROCESS_WORDS
    objective_tokens = tokens_by_field.get("objective", [])
    adjacent_novel: set[str] = set()
    for index, token in enumerate(objective_tokens):
        if token not in novel_by_field.get("objective", set()):
            continue
        if index == 0 or index + 1 >= len(objective_tokens):
            continue
        left = objective_tokens[index - 1]
        right = objective_tokens[index + 1]
        if left in public_content_tokens and right in public_content_tokens:
            adjacent_novel.add(token)
    repeated_novel = sorted(
        novel_by_field.get("objective", set()) & novel_by_field.get("scope", set())
    )
    scope_text = str(fields.get("scope") or "").casefold()
    explicitly_narrowed = bool(
        re.search(r"\b(?:only|exclude|excluding|not|category)\b", scope_text)
    )
    drift_terms = sorted(adjacent_novel | (set(repeated_novel) if explicitly_narrowed else set()))
    if drift_terms:
        return DelegationIssue(
            "task_constraint_drift",
            (
                "SET_PROMPT repeats WebShop content terms absent from the authoritative "
                "public task; preserve the original task wording and ambiguity"
            ),
            "objective",
            {"novel_terms": drift_terms[:16]},
        )
    return None


def compile_delegation(
    fields: dict[str, object],
    *,
    dataset: str = "",
    action_names: Iterable[str] = (),
) -> tuple[DelegationCompilation | None, DelegationIssue | None]:
    """Validate, locally compact, and deterministically compile a responsibility.

    Semantic safety is checked on the complete Director-authored text *before*
    any length repair.  Consequently compaction can never hide an answer clue,
    procedure, routing instruction, or Worker Action selection.  Only formatting
    and length are repaired automatically; semantic violations remain rejected.
    """

    normalized: dict[str, str] = {}
    repairs: list[DelegationFieldRepair] = []
    for name in DELEGATION_FIELD_LIMITS:
        original = str(fields.get(name) or "").strip()
        if not original:
            return None, DelegationIssue(
                "missing_field",
                f"SET_PROMPT field {name} is required",
                name,
            )
        one_line = " ".join(original.split())
        if one_line != original:
            repairs.append(
                DelegationFieldRepair(
                    name,
                    "whitespace_normalization",
                    len(original),
                    len(one_line),
                )
            )
        semantic_issue = delegation_safety_issue(
            one_line,
            action_names=action_names,
            field=name,
        )
        if semantic_issue is not None:
            return None, semantic_issue
        normalized[name] = one_line

    repaired = dict(normalized)
    dataset_key = str(dataset or "").strip().casefold()
    if dataset_key == "webshop" and repaired["expected_output"] != _WEBSHOP_EXPECTED_OUTPUT:
        original = repaired["expected_output"]
        repaired["expected_output"] = _WEBSHOP_EXPECTED_OUTPUT
        repairs.append(
            DelegationFieldRepair(
                "expected_output",
                "webshop_environment_completion",
                len(original),
                len(_WEBSHOP_EXPECTED_OUTPUT),
            )
        )
    for name, limit in DELEGATION_FIELD_LIMITS.items():
        value = repaired[name]
        if len(value) <= limit:
            continue
        shortened = _shorten_at_boundary(value, limit)
        repaired[name] = shortened
        repairs.append(
            DelegationFieldRepair(
                name,
                "field_length",
                len(value),
                len(shortened),
            )
        )

    _fit_total_delegation_limit(repaired, repairs)
    director_prompt = "\n".join(
        (
            f"Role: {repaired['role']}",
            f"Objective: {repaired['objective']}",
            f"Scope: {repaired['scope']}",
            f"Expected output: {repaired['expected_output']}",
        )
    )
    contract = _DATASET_OUTPUT_CONTRACTS.get(dataset_key, ())
    rule_ids = tuple(rule_id for rule_id, _text in contract)
    if contract:
        managed_contract_version = (
            WEBSHOP_DELEGATION_CONTRACT_VERSION
            if dataset_key == "webshop"
            else DELEGATION_CONTRACT_VERSION
        )
        managed = "\n".join(
            (
                f"System-managed output contract ({managed_contract_version}):",
                *(f"- {text}" for _rule_id, text in contract),
            )
        )
        prompt = director_prompt + "\n\n" + managed
        contract_version: str | None = managed_contract_version
    else:
        prompt = director_prompt
        contract_version = None
    return (
        DelegationCompilation(
            prompt=prompt,
            director_fields=repaired,
            dataset=dataset_key,
            contract_version=contract_version,
            contract_rule_ids=rule_ids,
            field_repairs=tuple(repairs),
        ),
        None,
    )


def director_delegation_text(prompt: str, metadata: dict[str, Any]) -> str:
    """Return only Director-authored responsibility text for final graph audit."""

    raw = metadata.get("director_delegation") if isinstance(metadata, dict) else None
    if not isinstance(raw, dict):
        return str(prompt or "")
    fields = [str(raw.get(name) or "").strip() for name in DELEGATION_FIELD_LIMITS]
    return " ".join(value for value in fields if value)


def _shorten_at_boundary(text: str, limit: int) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    if limit <= 1:
        return text[:limit]
    candidate = text[:limit].rstrip()
    boundary = max(candidate.rfind(" "), candidate.rfind(";"), candidate.rfind(","))
    if boundary >= max(16, limit // 2):
        candidate = candidate[:boundary].rstrip(" ,;")
    return candidate


def _fit_total_delegation_limit(
    fields: dict[str, str],
    repairs: list[DelegationFieldRepair],
) -> None:
    def total() -> int:
        return len(" ".join(fields.values()))

    minimums = {"scope": 48, "objective": 64, "expected_output": 40, "role": 24}
    for name in ("scope", "objective", "expected_output", "role"):
        excess = total() - DELEGATION_TOTAL_LIMIT
        if excess <= 0:
            return
        current = fields[name]
        target = max(minimums[name], len(current) - excess)
        if target >= len(current):
            continue
        shortened = _shorten_at_boundary(current, target)
        fields[name] = shortened
        repairs.append(
            DelegationFieldRepair(
                name,
                "total_length",
                len(current),
                len(shortened),
            )
        )


def graph_delegation_issues(task: str, graph: MultiAgentGraph) -> tuple[str, ...]:
    """Conservatively audit whether final graph assignments stay on task.

    This is intentionally graph-aware: downstream synthesis/checking nodes may be
    generic when they have incoming evidence, while source nodes must either name
    task anchors or explicitly operate on the assigned task.
    """

    task_tokens = _content_tokens(task)
    incoming = {agent_id: 0 for agent_id in graph.nodes}
    for _source, target in graph.directed_edges:
        incoming[target] = incoming.get(target, 0) + 1
    for source, target in graph.bidirectional_edges:
        incoming[target] = incoming.get(target, 0) + 1
        incoming[source] = incoming.get(source, 0) + 1

    issues: list[str] = []
    for agent_id, node in graph.nodes.items():
        assignment = director_delegation_text(node.prompt, node.metadata)
        safety_error = delegation_safety_error(
            assignment,
            action_names=node.allowed_tools,
        )
        if safety_error:
            issues.append(f"{agent_id}: {safety_error}")
            continue
        if not task_tokens or _GENERIC_TASK_RE.search(assignment):
            continue
        if incoming.get(agent_id, 0) > 0 and _DOWNSTREAM_CONTRIBUTION_RE.search(assignment):
            continue

        prompt_tokens = _content_tokens(assignment)
        shared = task_tokens & prompt_tokens
        minimum_shared = 1 if len(task_tokens) <= 4 else 2
        prompt_coverage = len(shared) / max(1, len(prompt_tokens))
        aligned = len(shared) >= 3 or (len(shared) >= minimum_shared and prompt_coverage >= 0.20)
        if not aligned:
            issues.append(
                f"{agent_id}: assignment lacks task anchors "
                f"(shared={len(shared)}, prompt_coverage={prompt_coverage:.3f})"
            )
    return tuple(issues)


def _content_tokens(text: str) -> set[str]:
    tokens = {_stem(token.casefold()) for token in _WORDS_RE.findall(str(text or ""))}
    stopwords = {_stem(token) for token in _STOPWORDS}
    return {token for token in tokens if len(token) > 1 and token not in stopwords}


def _stem(token: str) -> str:
    if token.startswith("erupt"):
        return "erupt"
    if token.startswith("form"):
        return "form"
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("s"):
        return token[:-1]
    return token
