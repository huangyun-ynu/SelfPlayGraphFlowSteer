from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from selfplay_graph_flowsteer.ads_preprocessing import (
    ADSPreprocessingConfig,
    build_ads_records,
    compute_base_policy_nll,
    extract_base_policy_embeddings,
    load_ads_input_rows,
)
from selfplay_graph_flowsteer.curriculum import FixedTaskPool
from selfplay_graph_flowsteer.pool_audit import audit_fixed_task_pool
from selfplay_graph_flowsteer.swebench import assert_public_swe_payload

ROOT = Path(os.environ.get("SPGFS_ROOT", Path(__file__).resolve().parents[2]))
RAW_ROOT = Path(os.environ.get("SPGFS_RAW_POOL_ROOT", ROOT / "state/formal-build/raw"))
OUT = Path(os.environ.get("SPGFS_ADS_POOL_ROOT", ROOT / "state/formal-build/ads"))
MODEL = Path(os.environ.get("SPGFS_MODEL_PATH", ROOT / "models/Qwen3.5-9B"))
DATASETS = (
    "aime",
    "nq_open",
    "hotpotqa",
    "webshop",
    "alfworld",
    "healthbench_professional",
    "swe_bench",
)
START = time.time()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def event(stage: str, **values: object) -> None:
    payload = {
        "stage": stage,
        "elapsed_seconds": round(time.time() - START, 3),
        **values,
    }
    print(json.dumps(payload, ensure_ascii=False, allow_nan=False), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "run_events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")


def main() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("seven-dataset ADS requires exactly one visible CUDA GPU")
    free_bytes, _total_bytes = torch.cuda.mem_get_info(0)
    if free_bytes < 24 * 1024**3:
        raise RuntimeError(f"visible GPU has only {free_bytes / 1024**3:.1f} GiB free")

    rows_by_dataset = {
        dataset: load_ads_input_rows([RAW_ROOT / f"{dataset}.jsonl"])
        for dataset in DATASETS
    }
    if any(len(rows) != 512 for rows in rows_by_dataset.values()):
        raise RuntimeError("every formal dataset must contain exactly 512 raw rows")
    all_rows = [row for dataset in DATASETS for row in rows_by_dataset[dataset]]
    sample_ids = [str(row["_ads_sample_id"]) for row in all_rows]
    if len(all_rows) != 3584 or len(set(sample_ids)) != 3584:
        raise RuntimeError("formal raw pool cardinality or ID uniqueness failed")

    configs = {
        dataset: ADSPreprocessingConfig(
            model_path=MODEL,
            output_path=OUT / dataset / "task_pool_ads.jsonl",
            artifacts_dir=OUT / dataset / "artifacts",
            num_clusters=16,
            device="cuda:0",
            dtype="bfloat16",
            batch_size=1,
            max_length=8192,
            pca_dim=128,
            seed=20260915,
            overwrite=True,
        )
        for dataset in DATASETS
    }
    torch.manual_seed(20260915)
    np.random.seed(20260915)
    shared = OUT / "extraction"
    shared.mkdir(parents=True, exist_ok=True)
    embeddings_by_dataset: dict[str, np.ndarray] = {}
    embeddings_reused_by_dataset: dict[str, bool] = {}
    difficulties_by_dataset: dict[str, np.ndarray] = {}
    target_sources_by_dataset: dict[str, list[str]] = {}

    # Persist each completed dataset so shared-GPU contention never discards
    # successful work from the other formal datasets.
    for dataset in DATASETS:
        rows = rows_by_dataset[dataset]
        ids = [str(row["_ads_sample_id"]) for row in rows]
        extraction_dir = shared / dataset
        embeddings_path = extraction_dir / "base_policy_mean_embeddings.npy"
        ids_path = extraction_dir / "sample_ids.json"
        extraction_dir.mkdir(parents=True, exist_ok=True)
        reuse_embeddings = embeddings_path.is_file() and ids_path.is_file()
        if reuse_embeddings:
            embeddings = np.load(embeddings_path)
            stored_ids = json.loads(ids_path.read_text(encoding="utf-8"))
            reuse_embeddings = (
                stored_ids == ids
                and embeddings.shape[0] == len(rows)
                and np.isfinite(embeddings).all()
            )
        if not reuse_embeddings:
            event(
                "embeddings_start",
                dataset=dataset,
                samples=len(rows),
                gpu=torch.cuda.get_device_name(0),
            )
            embeddings = extract_base_policy_embeddings(rows, configs[dataset])
            if embeddings.shape[0] != len(rows) or not np.isfinite(embeddings).all():
                raise RuntimeError(f"{dataset}: base-policy embeddings are malformed")
            np.save(embeddings_path, embeddings)
            write_json(ids_path, ids)
            event("embeddings_complete", dataset=dataset, shape=list(embeddings.shape))
        else:
            event("embeddings_reused", dataset=dataset, shape=list(embeddings.shape))
        embeddings_by_dataset[dataset] = embeddings
        embeddings_reused_by_dataset[dataset] = reuse_embeddings

    for dataset in DATASETS:
        rows = rows_by_dataset[dataset]
        ids = [str(row["_ads_sample_id"]) for row in rows]
        extraction_dir = shared / dataset
        difficulties_path = extraction_dir / "difficulty_nll.npy"
        target_sources_path = extraction_dir / "target_sources.json"
        nll_ids_path = extraction_dir / "difficulty_sample_ids.json"
        reuse_nll = difficulties_path.is_file() and target_sources_path.is_file()
        if reuse_nll:
            difficulties = np.load(difficulties_path)
            target_sources = json.loads(target_sources_path.read_text(encoding="utf-8"))
            stored_nll_ids = (
                json.loads(nll_ids_path.read_text(encoding="utf-8"))
                if nll_ids_path.is_file()
                else ids if embeddings_reused_by_dataset[dataset] else []
            )
            reuse_nll = (
                difficulties.shape == (len(rows),)
                and np.isfinite(difficulties).all()
                and len(target_sources) == len(rows)
                and stored_nll_ids == ids
            )
        if not reuse_nll:
            event("nll_start", dataset=dataset, samples=len(rows))
            difficulties, target_sources = compute_base_policy_nll(rows, configs[dataset])
            if difficulties.shape != (len(rows),) or not np.isfinite(difficulties).all():
                bad = [ids[i] for i in np.flatnonzero(~np.isfinite(difficulties))]
                write_json(extraction_dir / "non_finite_nll_sample_ids.json", bad)
                raise RuntimeError(f"{dataset}: non-finite NLL for {len(bad)} rows")
            np.save(difficulties_path, difficulties)
            write_json(target_sources_path, target_sources)
            write_json(nll_ids_path, ids)
            event(
                "nll_complete",
                dataset=dataset,
                minimum=float(difficulties.min()),
                maximum=float(difficulties.max()),
            )
        else:
            if not nll_ids_path.is_file():
                write_json(nll_ids_path, ids)
            event("nll_reused", dataset=dataset, samples=len(rows))
        difficulties_by_dataset[dataset] = difficulties
        target_sources_by_dataset[dataset] = target_sources

    summaries: dict[str, object] = {}
    validated_paths: list[Path] = []
    for dataset in DATASETS:
        rows = rows_by_dataset[dataset]
        vectors = embeddings_by_dataset[dataset]
        scores = difficulties_by_dataset[dataset]
        sources = target_sources_by_dataset[dataset]
        records, manifest, features, labels = build_ads_records(
            rows, vectors, scores, sources, configs[dataset]
        )
        if len(set(labels.tolist())) != 16 or not np.isfinite(features).all():
            raise RuntimeError(f"{dataset}: cluster collapse or non-finite features")
        if any("ads_target" in row for row in records):
            raise RuntimeError(f"{dataset}: offline ADS target leaked into output")
        if dataset == "swe_bench":
            for row in records:
                assert_public_swe_payload(row)

        config = configs[dataset]
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        config.artifacts_dir.mkdir(parents=True, exist_ok=True)
        np.save(config.artifacts_dir / "base_policy_mean_embeddings.npy", vectors)
        np.save(config.artifacts_dir / "pca_cluster_features.npy", features)
        np.save(config.artifacts_dir / "difficulty_nll.npy", scores)
        np.save(config.artifacts_dir / "cluster_labels.npy", labels)
        write_json(config.artifacts_dir / "sample_ids.json", [row["_ads_sample_id"] for row in rows])
        write_json(config.artifacts_dir / "manifest.json", manifest)
        config.output_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n"
                for row in records
            ),
            encoding="utf-8",
        )
        audited = audit_fixed_task_pool(
            [config.output_path], output_dir=OUT / dataset / "validated"
        )
        if (audited.accepted, audited.rejected) != (512, 0):
            raise RuntimeError(f"{dataset}: post-ADS audit rejected rows")
        pool = FixedTaskPool.from_jsonl(
            [audited.validated_path],
            require_ads_metadata=True,
            require_validation_manifest=True,
        )
        if len(pool.ids) != 512:
            raise RuntimeError(f"{dataset}: validated cardinality changed")
        validated_paths.append(audited.validated_path)
        summaries[dataset] = {
            "rows": len(pool.ids),
            "clusters": len(set(labels.tolist())),
            "cluster_sizes": dict(sorted(Counter(labels.tolist()).items())),
            "nll_min": float(scores.min()),
            "nll_max": float(scores.max()),
            "validated_pool_sha256": digest(audited.validated_path),
        }
        event("dataset_complete", dataset=dataset, **summaries[dataset])

    combined = audit_fixed_task_pool(validated_paths, output_dir=OUT / "validated")
    if (combined.accepted, combined.rejected) != (3584, 0):
        raise RuntimeError("combined seven-dataset audit failed")
    pool = FixedTaskPool.from_jsonl(
        [combined.validated_path],
        require_ads_metadata=True,
        require_validation_manifest=True,
    )
    cluster_counts = {
        dataset: len(cluster_ids)
        for dataset, cluster_ids in __import__(
            "selfplay_graph_flowsteer.curriculum", fromlist=["ADSBoundaryScheduler"]
        ).ADSBoundaryScheduler(pool).dataset_cluster_ids.items()
    }
    if len(pool.ids) != 3584 or cluster_counts != {dataset: 16 for dataset in DATASETS}:
        raise RuntimeError(f"combined pool shape mismatch: {cluster_counts}")
    summary = {
        "schema": "spgfs-formal-seven-dataset-ads-summary-v1",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": torch.cuda.get_device_name(0),
        "rows": len(pool.ids),
        "dataset_cluster_counts": cluster_counts,
        "validated_pool": str(combined.validated_path),
        "validated_pool_sha256": digest(combined.validated_path),
        "validation_manifest": str(combined.manifest_path),
        "datasets": summaries,
        "elapsed_seconds": round(time.time() - START, 3),
    }
    write_json(OUT / "summary.json", summary)
    event("complete", rows=len(pool.ids), validated_pool=str(combined.validated_path))


if __name__ == "__main__":
    main()
