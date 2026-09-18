from __future__ import annotations

import hashlib
import json
import os
import random
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("SPGFS_ROOT", Path(__file__).resolve().parents[2]))
P10 = Path(os.environ.get("SPGFS_PROTOCOL10_ROOT", ROOT / "assets/protocol-v10-v6"))
ALFWORLD_ROOT = Path(
    os.environ.get("SPGFS_ALFWORLD_DATA_ROOT", ROOT / "assets/alfworld-data/json_2.1.1")
)
WEBSHOP_GOALS = Path(
    os.environ.get("SPGFS_WEBSHOP_GOALS", ROOT / "assets/webshop/prepared/goals.jsonl")
)
NQ_SOURCE = Path(
    os.environ.get("SPGFS_NQ_TRAIN", ROOT / "state/source-data/NQ-open.train.jsonl")
)
SWE_SOURCE = Path(
    os.environ.get("SPGFS_SWE_TRAIN", ROOT / "state/source-data/swe_train_raw.jsonl")
)
HEALTH_SOURCE = Path(
    os.environ.get(
        "SPGFS_HEALTHBENCH_TRAIN", ROOT / "state/source-data/healthbench_train_raw.jsonl"
    )
)
OUT = Path(os.environ.get("SPGFS_RAW_POOL_ROOT", ROOT / "state/formal-build/raw"))
WEBSHOP_SPLIT_ROOT = Path(
    os.environ.get("SPGFS_WEBSHOP_SPLIT_ROOT", ROOT / "state/formal-build/webshop-split")
)
SEED = "spgfs-pats-formal-3584-v1"
COUNT = 512
WEBSHOP_TEST_COUNT = 128
WEBSHOP_TEST_STOP = 500
WEBSHOP_EVAL_STOP = 1500
WEBSHOP_OFFICIAL_SHUFFLE_SEED = 233


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


def official_webshop_goals(goals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    shuffled = list(goals)
    random.Random(WEBSHOP_OFFICIAL_SHUFFLE_SEED).shuffle(shuffled)
    return shuffled


def selected(
    rows: list[dict[str, Any]],
    dataset: str,
    *,
    count: int = COUNT,
    seed: str = SEED,
) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> bytes:
        source_id = str(row.get("source_id", row.get("id", "")))
        return hashlib.sha256(f"{seed}\0{dataset}\0{source_id}".encode()).digest()

    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        source_id = str(row.get("source_id", row.get("id", ""))).strip()
        if not source_id:
            raise ValueError(f"{dataset} row has no source identity")
        unique.setdefault(source_id, row)
    ordered = sorted(unique.values(), key=key)
    if len(ordered) < count:
        raise ValueError(f"{dataset} has only {len(ordered)} unique rows")
    return ordered[:count]


def protocol_rows(population: str) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    public = read_jsonl(P10 / "populations" / f"{population}.jsonl")
    private = {
        str(row["source_id"]): row["private_payload"]
        for row in read_jsonl(P10 / "private-records" / f"{population}.jsonl")
    }
    result = []
    for row in public:
        source_id = str(row["source_id"])
        if source_id not in private:
            raise ValueError(f"{population} missing private row {source_id}")
        result.append((row, private[source_id]))
    return result


def base_row(
    *,
    task_id: str,
    source_id: str,
    dataset: str,
    task_type: str,
    verifier: str,
    prompt: str,
    metadata: dict[str, Any],
    split: str = "train",
) -> dict[str, Any]:
    return {
        "id": task_id,
        "source_id": source_id,
        "dataset": dataset,
        "split": split,
        "task_type": task_type,
        "verifier": verifier,
        "prompt": prompt.strip(),
        "metadata": {
            "dataset": dataset,
            "split": split,
            "task_type": task_type,
            "verifier": verifier,
            **metadata,
        },
    }


def qa_protocol_pool(
    population: str,
    dataset: str,
    task_type: str,
    verifier: str,
) -> list[dict[str, Any]]:
    rows = []
    for public, private in protocol_rows(population):
        task = public["task"]
        answers = list(private["accepted_answers"])
        row = base_row(
            task_id=str(task["task_id"]),
            source_id=str(public["source_id"]),
            dataset=dataset,
            task_type=task_type,
            verifier=verifier,
            prompt=str(task["query"]),
            metadata={
                "source_lineage": task["public_context"],
                "scoring_rule": private["scoring_rule"],
            },
        )
        row["target_answers"] = answers
        row["reference"] = answers[0]
        if (
            dataset != "hotpotqa"
            and len(str(answers[0]).strip()) >= 4
            and str(answers[0]).casefold() in row["prompt"].casefold()
        ):
            continue
        rows.append(row)
    return selected(rows, dataset)


def nq_pool() -> list[dict[str, Any]]:
    rows = []
    for index, item in enumerate(read_jsonl(NQ_SOURCE)):
        source_id = f"nq_open/train/{index:06d}"
        answers = list(item["answer"])
        row = base_row(
            task_id=source_id,
            source_id=source_id,
            dataset="nq_open",
            task_type="factual_qa",
            verifier="multi_answer_exact_match",
            prompt=str(item["question"]),
            metadata={
                "source_lineage": {
                    "dataset": "google-research-datasets/natural-questions",
                    "revision": "fb26a3073b1fe636c97302890a27b491d6530130",
                    "source_split": "train",
                }
            },
        )
        row["target_answers"] = answers
        row["reference"] = answers[0]
        if len(str(answers[0]).strip()) >= 4 and str(answers[0]).casefold() in row[
            "prompt"
        ].casefold():
            continue
        rows.append(row)
    return selected(rows, "nq_open")


def alfworld_pool() -> list[dict[str, Any]]:
    rows = []
    for public, private in protocol_rows("alfworld-train-main"):
        task = public["task"]
        relative = str(private["trajectory_relative_path"])
        trajectory_path = ALFWORLD_ROOT / relative / "traj_data.json"
        game_path = ALFWORLD_ROOT / relative / "game.tw-pddl"
        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
        annotations = trajectory["turk_annotations"]["anns"]
        high_descs = annotations[0]["high_descs"]
        row = base_row(
            task_id=str(task["task_id"]),
            source_id=str(public["source_id"]),
            dataset="alfworld",
            task_type=str(task["task_family"]),
            verifier="alfworld_environment",
            prompt=str(task["query"]),
            metadata={
                "task_id": str(task["task_id"]),
                "source_split": "train",
                "game_path": str(game_path),
                "alfworld_task_type": trajectory["task_type"],
                "pddl_params": trajectory["pddl_params"],
                "max_steps": int(private["max_steps"]),
                "source_lineage": task["public_context"],
            },
        )
        row["ads_target"] = high_descs
        rows.append(row)
    return selected(rows, "alfworld")


def webshop_rows(goals: list[dict[str, Any]], indices: range, *, split: str) -> list[dict[str, Any]]:
    rows = []
    official_goals = official_webshop_goals(goals)
    for goal_index in indices:
        goal = official_goals[goal_index]
        source_id = f"webshop/goal-{goal_index:05d}"
        row = base_row(
            task_id=source_id,
            source_id=source_id,
            dataset="webshop",
            task_type="shopping",
            verifier="webshop_environment",
            prompt=str(goal["instruction_text"]),
            split=split,
            metadata={
                "task_id": source_id,
                "goal_id": source_id,
                "goal_index": goal_index,
                "source_split": split,
                "source_lineage": {
                    "benchmark_id": "webshop",
                    "source_version": "official-12087-instruction-release",
                    "source_split": split,
                    "goal_order": "official_random_seed_233",
                    "official_goal_index_ranges": {
                        "test": [0, WEBSHOP_TEST_STOP - 1],
                        "eval": [WEBSHOP_TEST_STOP, WEBSHOP_EVAL_STOP - 1],
                        "train": [WEBSHOP_EVAL_STOP, len(goals) - 1],
                    },
                },
            },
        )
        row["ads_target"] = {
            "asin": goal["asin"],
            "goal_options": goal["goal_options"],
        }
        rows.append(row)
    return rows


def webshop_pool() -> list[dict[str, Any]]:
    goals = read_jsonl(WEBSHOP_GOALS)
    rows = webshop_rows(goals, range(WEBSHOP_EVAL_STOP, len(goals)), split="train")
    return selected(rows, "webshop")


def write_webshop_test_split() -> dict[str, Any]:
    goals = read_jsonl(WEBSHOP_GOALS)
    candidates = webshop_rows(goals, range(WEBSHOP_TEST_STOP), split="test")
    rows = selected(
        candidates,
        "webshop_test",
        count=WEBSHOP_TEST_COUNT,
        seed="spgfs-webshop-official-test-128-v1",
    )
    for row in rows:
        row.pop("ads_target", None)
    path = WEBSHOP_SPLIT_ROOT / "test_128.jsonl"
    write_jsonl(path, rows)
    manifest = {
        "schema": "spgfs-webshop-official-split-v1",
        "source": {
            "path": str(WEBSHOP_GOALS),
            "rows": len(goals),
            "official_goal_index_ranges": {
                "test": [0, WEBSHOP_TEST_STOP - 1],
                "eval": [WEBSHOP_TEST_STOP, WEBSHOP_EVAL_STOP - 1],
                "train": [WEBSHOP_EVAL_STOP, len(goals) - 1],
            },
        },
        "test": {
            "selection_seed": "spgfs-webshop-official-test-128-v1",
            "rows": len(rows),
            "unique_source_rows": len({row["source_id"] for row in rows}),
            "path": str(path),
            "sha256": digest(path),
        },
    }
    manifest_path = WEBSHOP_SPLIT_ROOT / "split_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def swe_pool() -> list[dict[str, Any]]:
    rows = read_jsonl(SWE_SOURCE)
    if len(rows) != COUNT or len({str(row["id"]) for row in rows}) != COUNT:
        raise ValueError("SWE Verified adapted training split must contain 512 unique rows")
    for row in rows:
        metadata = row.get("metadata", {})
        if (
            metadata.get("source_split") != "verified"
            or metadata.get("experiment_split") != "train"
            or not str(metadata.get("split_manifest_id", "")).strip()
            or not str(metadata.get("patch", "")).strip()
        ):
            raise ValueError("SWE Verified training row is missing its trusted split binding")
    return rows


def health_pool() -> list[dict[str, Any]]:
    rows = read_jsonl(HEALTH_SOURCE)
    if len(rows) != COUNT:
        raise ValueError("HealthBench adapted training split must contain 512 rows")
    return rows


def main() -> None:
    webshop_test_manifest = write_webshop_test_split()
    builders: dict[str, Callable[[], list[dict[str, Any]]]] = {
        "aime": lambda: qa_protocol_pool(
            "aime-1983-2024", "aime", "math_reasoning", "numeric"
        ),
        "nq_open": nq_pool,
        "hotpotqa": lambda: qa_protocol_pool(
            "hotpotqa-v1.1-train-main",
            "hotpotqa",
            "multi_hop_qa",
            "multi_answer_exact_match",
        ),
        "webshop": webshop_pool,
        "alfworld": alfworld_pool,
        "healthbench_professional": health_pool,
        "swe_bench": swe_pool,
    }
    manifest: dict[str, Any] = {
        "schema": "spgfs-formal-seven-dataset-raw-pools-v1",
        "selection_seed": SEED,
        "rows_per_dataset": COUNT,
        "datasets": {},
        "webshop_official_test": webshop_test_manifest["test"],
    }
    for dataset, builder in builders.items():
        rows = builder()
        if len(rows) != COUNT or len({str(row["id"]) for row in rows}) != COUNT:
            raise RuntimeError(f"{dataset} cardinality or ID uniqueness failed")
        output = OUT / f"{dataset}.jsonl"
        write_jsonl(output, rows)
        manifest["datasets"][dataset] = {
            "path": str(output),
            "rows": len(rows),
            "sha256": digest(output),
            "ads_target_sources": dict(
                Counter(
                    "explicit"
                    if "ads_target" in row
                    else "dataset_contract"
                    for row in rows
                )
            ),
        }
        print(dataset, len(rows), digest(output), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
