"""Opt-in, reference-free AIME review guidance; never a correctness grader."""

POLICY = "bounded_review_v1"
RESULT_FIELDS = ("evidence", "summary", "confidence", "unresolved_issues", "tool_summary", "answer")


def output_instruction(*, is_output_agent: bool) -> str:
    scope = (
        "Solve the complete original problem in public_task_context. "
        if is_output_agent else "Complete your assigned mathematical responsibility. "
    )
    return (
        "Return one final JSON object in this key order: evidence, summary, confidence, "
        "unresolved_issues, tool_summary, answer. "
        + scope
        + "First resolve the mathematical reasoning using actual tool observations where useful. "
        "Check the decisive formula, domain restrictions, and requested quantity. A feasible "
        "example or heuristic search alone is not a proof of a maximum or minimum. State any "
        "missing proof, failed check, or contradictory evidence in unresolved_issues. "
        "Write answer LAST, consistent with the resolved conclusion. If you correct a candidate "
        "during reasoning, update answer too. Do not retain a candidate that your own evidence "
        "has refuted. Report remaining uncertainty honestly; do not invent verification. "
        + ("Put one final integer from 0 to 999 in answer. " if is_output_agent else "")
    )


REVIEW_INSTRUCTION = (
    "This is the task's one explicit AIME review. Treat the previous answer as an untrusted "
    "candidate, not a premise. Re-derive the decisive step from the public problem and check "
    "the candidate against every required constraint. Use the available computation allowance "
    "to test the decisive claim, not merely reformat the previous answer. For an extremum, "
    "distinguish a construction from a proof that no better value exists. Resolve contradictions "
    "between answer, summary, evidence and tool observations; report checks that remain "
    "inconclusive. Return a complete revised result with answer last."
)


def quality_warnings(artifact) -> list[str]:
    """Expose reported doubts and syntax only, without interpreting prose as a proof."""
    from .aime_submission import parse_aime_answer

    warnings = []
    if not parse_aime_answer(str(artifact.answer)).valid:
        warnings.append("answer_is_not_an_unambiguous_aime_integer")
    warnings.extend(str(issue)[:1000] for issue in artifact.unresolved_issues
                    if str(issue).strip().lower() not in {"", "none", "no", "n/a", "no unresolved issues"})
    return list(dict.fromkeys(warnings))[:8]
