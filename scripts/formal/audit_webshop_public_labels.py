#!/usr/bin/env python3
"""Offline label audit and an isolated option-normalization proposal.

This is never imported by inference or the official scorer. Missing lexical
matches are review candidates, not proof of missing semantic requirements.
Goal options without group labels cannot support a group-aware replacement score.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re


def normalize_option(group, value, colors):
    value = " ".join(str(value).casefold().split())
    if str(group).casefold() not in {"color", "colour", "color name", "colour name"}:
        return value
    for color in colors:
        if re.search(r"(?<!\w)" + re.escape(color) + r"(?!\w)", value):
            return color
    return value


def audit(rows, colors):
    # First record public requests; hidden fields are used only in the following
    # offline label comparison and never used to revise an inference question.
    public = {row["task"]: " ".join(row["public_task"].casefold().split()) for row in rows}
    findings = []
    for row in rows:
        missing = [attribute for attribute in row.get("goal_attributes_offline_only", [])
                   if " ".join(str(attribute).casefold().split()) not in public[row["task"]]]
        options = []
        for group, value in (row.get("purchased_options") or {}).items():
            original = str(value).casefold()
            legacy = next((color for color in colors if color in original), original)
            proposed = normalize_option(group, value, colors)
            if legacy != proposed:
                options.append({"group": group, "observed_value": value,
                                "official_normalized": legacy, "proposed_normalized": proposed})
        if missing or options:
            findings.append({"task": row["task"], "official_score_unchanged": row["score"],
                             "attributes_without_literal_public_match": missing,
                             "semantic_review_required": bool(missing),
                             "option_normalization_differences": options})
    return findings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored", type=Path, required=True)
    parser.add_argument("--normalizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tree = ast.parse(args.normalizer.read_text())
    colors = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "COLOR_SET" for target in node.targets))
    rows = json.loads(args.scored.read_text())
    findings = audit(rows, colors)
    report = {"scope": "offline_only_not_an_alternative_official_score", "scorer_modified": False,
              "scored_input_sha256": hashlib.sha256(args.scored.read_bytes()).hexdigest(),
              "official_normalizer_sha256": hashlib.sha256(args.normalizer.read_bytes()).hexdigest(),
              "examples": len(rows), "official_successes_unchanged": sum(bool(row["passed"]) for row in rows),
              "lexical_review_candidates": sum(bool(f["attributes_without_literal_public_match"]) for f in findings),
              "tasks_with_normalization_differences": sum(bool(f["option_normalization_differences"]) for f in findings),
              "limitation": "Goal options lack reliable group labels; no replacement reward is reported. Synonyms require human semantic review.",
              "findings": findings}
    with args.output.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k != "findings"}))


if __name__ == "__main__":
    main()
