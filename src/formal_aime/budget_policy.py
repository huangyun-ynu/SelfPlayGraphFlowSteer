"""Question-scoped actual usage and cleanup of retired allocation metadata."""

from .config import DEFAULT_DATASET_MAX_TOTAL_TOKENS

POLICY = "reported_usage_threshold_v1"
REPORTED_USAGE_DATASETS = frozenset(DEFAULT_DATASET_MAX_TOTAL_TOKENS) - {"webshop"}
LEGACY_ALLOCATION_FIELDS = (
    "_runtime_token_credit",
    "_runtime_reserved_closure_tokens",
    "_runtime_finalization_output_reserve",
    "_runtime_budget_phase",
    "_runtime_webshop_request_admission_enabled",
)


def default_usage_policy(threshold):
    return dict(
        policy=POLICY,
        start_threshold=threshold,
        accounting_scope="question_attempt",
        max_inflight_requests=1,
        unknown_usage_policy="continue_bounded",
        max_unsettled_attempts=2,
    )


def clear_legacy_allocations(node):
    for field in LEGACY_ALLOCATION_FIELDS:
        node.metadata.pop(field, None)
    if node.metadata.get("_runtime_budget_kind") != POLICY:
        node.metadata.pop("_runtime_budget_kind", None)


def use_reported_usage(node):
    clear_legacy_allocations(node)
    node.metadata["_runtime_budget_kind"] = POLICY
