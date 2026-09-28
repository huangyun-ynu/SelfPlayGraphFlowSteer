"""Dataset-aware contract for the result selected by a graph output Agent."""

from __future__ import annotations

from .config import canonical_dataset_name
from .qa_submission import is_short_qa_dataset

OUTPUT_CONTRACT_VERSION = "task_result_v1"
WORKER_OUTPUT_ROLE_VERSION = "worker_output_role_v1"

_OUTPUT_OWNED_BY_ENVIRONMENT = frozenset({"alfworld", "webshop", "swe_bench"})


def worker_output_role_changes_input(*, dataset: str, action_adapter: str = "") -> bool:
    """Whether selecting an output changes the prompt sent to this Worker."""

    return not (
        canonical_dataset_name(dataset) in _OUTPUT_OWNED_BY_ENVIRONMENT
        or canonical_dataset_name(action_adapter) in _OUTPUT_OWNED_BY_ENVIRONMENT
    )


def selected_output_instruction(
    *,
    dataset: str,
    action_adapter: str,
    short_answer_qa: bool,
    is_output_agent: bool,
) -> str:
    """Return a short role instruction only when output status is a Worker input."""

    if not is_output_agent or not worker_output_role_changes_input(
        dataset=dataset,
        action_adapter=action_adapter,
    ):
        return ""
    if canonical_dataset_name(dataset) == "hotpotqa":
        return (
            "You are the selected output Agent. Submit the answer to the original public "
            "question in public_task_context using the HotpotQA answer contract, including "
            "all requested hops, comparisons, and list members. "
        )
    if short_answer_qa or is_short_qa_dataset(dataset):
        return (
            "You are the selected output Agent. Your answer is submitted for the original "
            "public question in public_task_context. Complete that question using your "
            "delegated work and visible graph evidence; include every requested part, "
            "comparison, or yes/no conclusion. Put only the shortest sufficient answer in "
            "answer and keep explanation in summary or evidence."
        )
    adapter = canonical_dataset_name(action_adapter)
    if adapter == "aime" or canonical_dataset_name(dataset) == "aime":
        return (
            "You are the selected output Agent for the original public math problem. Solve "
            "the complete problem and put its final integer answer in answer; put reasoning "
            "in summary or evidence."
        )
    if (
        adapter == "healthbench_professional"
        or canonical_dataset_name(dataset) == "healthbench_professional"
    ):
        return (
            "You are the selected output Agent for the complete public healthcare conversation. "
            "Return the complete professional response required by its submission contract; "
            "do not submit only your local analysis."
        )
    if adapter:
        return ""
    return (
        "You are the selected output Agent. Your answer is submitted for the original public "
        "task in public_task_context. Complete that task using your delegated work and the "
        "visible graph evidence; do not submit only a local intermediate result."
    )


def recovery_answer_description(*, is_output_agent: bool, action_adapter: str) -> str:
    """Describe the answer field consistently when repairing a malformed Worker response."""

    if action_adapter in _OUTPUT_OWNED_BY_ENVIRONMENT:
        return "the Agent's result for its own environment or code session; never a claim that substitutes for trusted execution evidence"
    if is_output_agent:
        return "the final result for the original public task, using its dataset-specific submission contract"
    return "the assigned local result (string, number, array or object; not boolean/null)"
