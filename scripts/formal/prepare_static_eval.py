from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("SPGFS_ROOT", Path(__file__).resolve().parents[2]))
P10 = Path(os.environ.get("SPGFS_PROTOCOL10_ROOT", ROOT / "assets/protocol-v10-v6"))
ALFWORLD_ROOT = Path(
    os.environ.get("SPGFS_ALFWORLD_DATA_ROOT", ROOT / "assets/alfworld-data/json_2.1.1")
)
NQ_DEV = Path(
    os.environ.get("SPGFS_NQ_DEV", ROOT / "state/datasets/nq-open/official/NQ-open.dev.jsonl")
)
OUT = Path(os.environ.get("SPGFS_STATIC_EVAL_ROOT", ROOT / "data/formal/eval"))
SEED = "spgfs-static-eval-20260916-v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select(rows: list[dict[str, Any]], dataset: str, count: int) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> bytes:
        return hashlib.sha256(
            f"{SEED}\0{dataset}\0{row['source_id']}".encode()
        ).digest()

    unique = {str(row["source_id"]): row for row in rows}
    if len(unique) < count:
        raise ValueError(f"{dataset} has only {len(unique)} unique official evaluation rows")
    return sorted(unique.values(), key=key)[:count]


def protocol_rows(population: str) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    public = read_jsonl(P10 / "populations" / f"{population}.jsonl")
    private = {
        str(row["source_id"]): row["private_payload"]
        for row in read_jsonl(P10 / "private-records" / f"{population}.jsonl")
    }
    return [(row, private[str(row["source_id"])]) for row in public]


def validate_alfworld_binding(
    task: dict[str, Any], private: dict[str, Any], trajectory: dict[str, Any]
) -> None:
    """Reject public/private ALFWorld pairings that do not describe one source trajectory."""

    relative = Path(str(private["trajectory_relative_path"]))
    if len(relative.parts) < 2:
        raise ValueError("ALFWorld trajectory path has no encoded task directory")
    encoded = relative.parts[-2].rsplit("-", 4)
    if len(encoded) != 5:
        raise ValueError("ALFWorld task directory does not encode task parameters")
    task_type, object_target, middle_target, destination_target, _ = encoded
    params = trajectory.get("pddl_params", {})
    expected_destination = (
        params.get("toggle_target", "")
        if task_type == "look_at_obj_in_light"
        else params.get("parent_target", "")
    )
    expected = (
        str(trajectory.get("task_type", "")),
        str(params.get("object_target", "")),
        str(params.get("mrecep_target", "") or "None"),
        str(expected_destination),
    )
    actual = (task_type, object_target, middle_target, destination_target)
    if actual != expected:
        raise ValueError(f"ALFWorld path/PDDL mismatch: encoded={actual!r}, pddl={expected!r}")
    query = " ".join(str(task.get("query", "")).casefold().split())
    annotations = trajectory.get("turk_annotations", {}).get("anns", ())
    descriptions = {
        " ".join(str(item.get("task_desc", "")).casefold().split())
        for item in annotations
        if isinstance(item, dict)
    }
    if not query or query not in descriptions:
        raise ValueError("ALFWorld public query is not an annotation of its bound trajectory")


def qa_row(
    public: dict[str, Any],
    private: dict[str, Any],
    *,
    dataset: str,
    task_type: str,
    verifier: str,
    source_split: str,
) -> dict[str, Any]:
    task = public["task"]
    answers = [str(value) for value in private["accepted_answers"]]
    return {
        "dataset": dataset,
        "id": str(task["task_id"]),
        "metadata": {
            "dataset": dataset,
            "scoring_rule": str(private["scoring_rule"]),
            "source_lineage": task["public_context"],
            "source_split": source_split,
            "split": "test",
            "task_type": task_type,
            "verifier": verifier,
        },
        "prompt": str(task["query"]),
        "reference": answers[0],
        "source_id": str(public["source_id"]),
        "split": "test",
        "target_answers": answers,
        "task_type": task_type,
        "verifier": verifier,
    }


def aime() -> list[dict[str, Any]]:
    return [
        qa_row(
            public,
            private,
            dataset="aime",
            task_type="math_reasoning",
            verifier="numeric",
            source_split="official_2026",
        )
        for public, private in protocol_rows("aime-2026-all-30")
    ]


def nq_open() -> list[dict[str, Any]]:
    rows = []
    for index, item in enumerate(read_jsonl(NQ_DEV)):
        source_id = f"nq_open/dev/{index:06d}"
        answers = [str(value) for value in item["answer"]]
        rows.append(
            {
                "dataset": "nq_open",
                "id": source_id,
                "metadata": {
                    "dataset": "nq_open",
                    "source_lineage": {
                        "dataset": "google-research-datasets/natural-questions",
                        "source_split": "dev",
                    },
                    "source_split": "dev",
                    "split": "test",
                    "task_type": "factual_qa",
                    "verifier": "flowsteer_qa",
                },
                "prompt": str(item["question"]),
                "reference": answers[0],
                "source_id": source_id,
                "split": "test",
                "target_answers": answers,
                "task_type": "factual_qa",
                "verifier": "flowsteer_qa",
            }
        )
    return select(rows, "nq_open", 128)


def hotpotqa() -> list[dict[str, Any]]:
    rows = [
        qa_row(
            public,
            private,
            dataset="hotpotqa",
            task_type="multi_hop_qa",
            verifier="flowsteer_qa",
            source_split="dev_distractor",
        )
        for public, private in protocol_rows("hotpotqa-v1.1-dev-distractor")
    ]
    return select(rows, "hotpotqa", 128)


def alfworld_population(population: str, source_split: str) -> list[dict[str, Any]]:
    rows = []
    for public, private in protocol_rows(population):
        task = public["task"]
        relative = str(private["trajectory_relative_path"])
        trajectory_path = ALFWORLD_ROOT / relative / "traj_data.json"
        game_path = ALFWORLD_ROOT / relative / "game.tw-pddl"
        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
        validate_alfworld_binding(task, private, trajectory)
        rows.append(
            {
                "dataset": "alfworld",
                "id": str(task["task_id"]),
                "metadata": {
                    "alfworld_task_type": trajectory["task_type"],
                    "dataset": "alfworld",
                    "game_path": str(game_path.resolve()),
                    "max_steps": int(private["max_steps"]),
                    "pddl_params": trajectory["pddl_params"],
                    "source_lineage": task["public_context"],
                    "source_split": source_split,
                    "split": "test",
                    "task_type": str(task["task_family"]),
                    "verifier": "alfworld_environment",
                },
                "prompt": str(task["query"]),
                "source_id": str(public["source_id"]),
                "split": "test",
                "task_type": str(task["task_family"]),
                "verifier": "alfworld_environment",
            }
        )
    return rows


def alfworld() -> list[dict[str, Any]]:
    seen = select(alfworld_population("alfworld-valid-seen", "valid_seen"), "alf_seen", 95)
    unseen = select(
        alfworld_population("alfworld-valid-unseen", "valid_unseen"), "alf_unseen", 33
    )
    return sorted(seen + unseen, key=lambda row: str(row["source_id"]))


def main() -> None:
    datasets = {
        "aime": aime(),
        "nq_open": nq_open(),
        "hotpotqa": hotpotqa(),
        "alfworld": alfworld(),
    }
    manifest: dict[str, Any] = {
        "schema": "spgfs-static-eval-splits-v1",
        "selection_seed": SEED,
        "datasets": {},
    }
    for name, rows in datasets.items():
        path = OUT / f"{name}_official_test.jsonl"
        write_jsonl(path, rows)
        manifest["datasets"][name] = {
            "path": str(path.relative_to(ROOT)),
            "rows": len(rows),
            "sha256": digest(path),
            "unique_source_rows": len({row["source_id"] for row in rows}),
        }
    path = OUT / "static_eval_split_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
