"""Public QA question provenance, separate from delegation and output instructions."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

QA_PUBLIC_TASK_VERSION = "qa_public_task_v1"
QA_DATASETS = frozenset({"hotpotqa", "musique"})
INLINE_HEADER = "Based on the following passages, answer the question.\n\n["


@dataclass(frozen=True)
class PublicQATask:
    question: str | None
    question_source: str
    task_id: str = ""
    version: str = QA_PUBLIC_TASK_VERSION

    def __post_init__(self) -> None:
        if self.version != QA_PUBLIC_TASK_VERSION:
            raise ValueError("unsupported public QA task version")
        if self.question is not None and (
            not isinstance(self.question, str) or not self.question.strip()
        ):
            raise ValueError("public QA question must be a nonempty string or None")
        if not isinstance(self.question_source, str) or not isinstance(self.task_id, str):
            raise ValueError("public QA provenance must be strings")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PublicQATask":
        # Never copy arbitrary TaskSpec metadata, references, or private labels.
        return cls(question=value.get("question"),
                   question_source=value["question_source"],
                   task_id=value.get("task_id", ""), version=value["version"])


def public_qa_task_from_prompt(prompt: str, *, task_id: str = "") -> PublicQATask:
    """Compatibility reader for the original, unwrapped inline passage template.

    Custom/free-form tasks retain their public text, without guessing a question.
    This reader is used at task ingestion, never on delegated instructions.
    """
    prefix, marker, question = prompt.rpartition("\n\nQuestion:")
    recognized = prefix.lstrip().startswith("[") or prefix.startswith(INLINE_HEADER)
    if (marker and recognized and question.strip()
            and "\n\nSubmission contract:" not in question):
        return PublicQATask(question.strip(), "rendered_qa_inline_v1.Question", task_id)
    return PublicQATask(None, "public_task_context", task_id)


def public_qa_task_from_record(record: Mapping[str, Any]) -> PublicQATask | None:
    metadata = record.get("metadata") or {}
    dataset = str(record.get("dataset", metadata.get("dataset", ""))).strip().casefold()
    if dataset not in QA_DATASETS:
        return None
    task_id = str(record.get("id", ""))
    question = record.get("question")
    if isinstance(question, str) and question.strip():
        return PublicQATask(question, "dataset.question", task_id)
    prompt = str(record.get("task", record.get("prompt", record.get("problem", "")))).strip()
    return public_qa_task_from_prompt(prompt, task_id=task_id)
