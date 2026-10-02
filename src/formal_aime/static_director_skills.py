from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import canonical_dataset_name


@dataclass(frozen=True)
class StaticDirectorSkill:
    skill_id: str


class StaticDirectorSkillBank:
    """One reviewed, answer-free orchestration Skill per benchmark dataset."""

    supports_task_metadata = True

    def __init__(self, root: str | Path, *, prompt_token_budget: int = 1024) -> None:
        self.root = Path(root)
        self.prompt_token_budget = int(prompt_token_budget)
        if self.prompt_token_budget <= 0:
            raise ValueError("static Director Skill token budget must be positive")

    def select_context(
        self,
        query: str,
        *,
        task_type: str,
        tokenizer: Any,
        tools=(),
        task_metadata=None,
    ):
        del query, task_type, tools
        dataset = canonical_dataset_name((task_metadata or {}).get("dataset", ""))
        path = self.root / f"{dataset}.md"
        if not path.is_file():
            return [], "", {
                "schema": "static_director_skill_v1",
                "dataset": dataset,
                "selected": [],
                "prompt_tokens": 0,
            }
        context = path.read_text(encoding="utf-8").strip()
        prompt_tokens = len(tokenizer.encode(context, add_special_tokens=False))
        if prompt_tokens > self.prompt_token_budget:
            raise ValueError(
                f"static Director Skill for {dataset} uses {prompt_tokens} tokens, "
                f"above budget {self.prompt_token_budget}"
            )
        skill = StaticDirectorSkill(f"static_director_{dataset}")
        return [skill], context, {
            "schema": "static_director_skill_v1",
            "dataset": dataset,
            "selected": [{"id": skill.skill_id, "version": 1}],
            "prompt_tokens": prompt_tokens,
            "context_sha256": hashlib.sha256(context.encode()).hexdigest(),
            "source_path": str(path),
        }
