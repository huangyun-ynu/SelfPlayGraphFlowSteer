"""Shared HealthBench deliverable contract for generation and submission."""


def healthbench_answer_instruction() -> str:
    """Keep the deliverable self-contained in the field submitted to the grader."""
    return (
        "For HealthBench, answer must contain the complete, self-contained result for your "
        "assigned responsibility, including the explanation, supporting evidence, "
        "qualifications, and uncertainty needed to understand and use that result. "
        "The reader must not need summary or evidence to understand answer. "
        "Use summary for a brief internal collaboration summary and evidence for internal "
        "evidence records; information essential to the reader must also appear in answer. "
        "An intermediate Worker should complete only its assigned sub-task. When your "
        "responsibility is the final reply, answer must be the complete user-facing response "
        "to the healthcare conversation. "
    )
