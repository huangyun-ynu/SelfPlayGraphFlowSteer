"""Read-only QA diagnostics; never feed these values into reward or prompts.

Hotpot answer metrics follow the dataset authors' evaluator (not supporting
facts or joint metrics): github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py
NQ-open EM uses the DPR reader convention (not original NQ long/short-span
annotation evaluation): github.com/facebookresearch/DPR/blob/main/dpr/data/qa_validation.py
NQ F1 is an explicitly additional SQuAD-style token-overlap diagnostic.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any

from .config import canonical_dataset_name
from .qa_submission import extract_qa_answer, is_short_qa_dataset


def normalize_official_qa(value: str) -> str:
    lowered = str(value).lower()
    unpunctuated = "".join(char for char in lowered if char not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", unpunctuated).split())


def _pair_scores(prediction: str, reference: str, *, hotpot: bool) -> dict[str, float]:
    pred, gold = normalize_official_qa(prediction), normalize_official_qa(reference)
    result = {"em": float(pred == gold), "f1": 0.0, "precision": 0.0, "recall": 0.0}
    special = {"yes", "no", "noanswer"}
    if hotpot and pred != gold and (pred in special or gold in special):
        return result
    pred_tokens, gold_tokens = pred.split(), gold.split()
    common = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if common:
        precision, recall = common / len(pred_tokens), common / len(gold_tokens)
        result.update(
            f1=2 * precision * recall / (precision + recall), precision=precision, recall=recall
        )
    return result


def qa_official_metrics(
    dataset: object, prediction: str, references: object
) -> dict[str, Any] | None:
    """Score only the supplied submitted answer, never mine other artifacts.

    The direct `answer_*` fields run the evaluator on the submitted text with
    no extraction. `explicit_submission_*` additionally apply the project's
    existing reference-blind wrapper parser and are deliberately named apart.
    Neither variant modifies strict success, eligibility, or training reward.
    """
    if not is_short_qa_dataset(dataset) or references is None:
        return None
    refs = list(references) if isinstance(references, (list, tuple, set)) else [references]
    if not refs:
        return None
    hotpot = canonical_dataset_name(str(dataset).lower()) == "hotpotqa"
    if not hotpot:
        hotpot = re.sub(r"[^a-z]", "", str(dataset).lower()) in {"hotpot", "hotpotqa"}
    extracted = extract_qa_answer(prediction)

    def best(text: str) -> dict[str, float]:
        scores = [_pair_scores(text, str(ref), hotpot=hotpot) for ref in refs]
        return {name: max(score[name] for score in scores) for name in scores[0]}

    raw_scores = best(prediction)
    extracted_scores = best(extracted) if extracted else dict.fromkeys(raw_scores, 0.0)
    return {
        "schema": "hotpot_official_answer_v1" if hotpot else "nq_open_dpr_em_token_f1_v1",
        "diagnostic_only": True,
        "reference_count": len(refs),
        "explicit_submission_valid": bool(extracted),
        **{f"answer_{key}": value for key, value in raw_scores.items()},
        **{f"explicit_submission_{key}": value for key, value in extracted_scores.items()},
    }
