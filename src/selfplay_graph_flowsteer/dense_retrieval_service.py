"""Search-R1 Wiki-18 E5/FAISS retrieval, using the project's offline E5 encoder."""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import logging
import mmap
import threading
from pathlib import Path

import numpy as np

from .retrieval_service import RetrievalBatcher, RetrievalHandler, RetrievalServer
from .skills import E5SkillEmbedder


EXPECTED_SOURCES = {
    "part_aa": ("a4d31160a035f30764604f4827cd8f1d0315eb86", 42949672960,
                "a8a6a246951da4bbc8771a223283ef61963882a32864d9044ec00abb90fc3023"),
    "part_ab": ("a4d31160a035f30764604f4827cd8f1d0315eb86", 21609402413,
                "b6d9bc943626fe7cb44de4c849e9379e7f272ab216c0552acbcf2390cc033c11"),
    "wiki-18.jsonl.gz": ("69c1c00ffe7c5554c68d8548355cb22e46aabc51", 5123307260,
                         "7abd929223399cd63c52b499f289bf4f9039be1e9f8c43e1cb3938305b2317db"),
}
EXPECTED_ENCODER_REPO = "intfloat/e5-base-v2"
EXPECTED_ENCODER_REVISION = "f52bf8ec8c7124536f0efb74aca902b2995e5bcd"


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _e5_encoder_identity(model_path: Path) -> dict:
    model_manifest = json.loads((model_path / "download_manifest.json").read_text())
    if (model_manifest.get("repo_id"), model_manifest.get("revision")) != (
        EXPECTED_ENCODER_REPO, EXPECTED_ENCODER_REVISION
    ):
        raise ValueError("E5 encoder revision does not match the fixed retrieval profile")
    model_files = model_manifest.get("files")
    if not isinstance(model_files, list):
        raise ValueError("E5 encoder manifest is missing files")
    checked: dict[str, str] = {}
    for entry in model_files:
        name = entry.get("path") if isinstance(entry, dict) else None
        if not isinstance(name, str) or Path(name).name != name or name in checked:
            raise ValueError("Invalid E5 encoder manifest file")
        path = model_path / name
        if path.stat().st_size != entry.get("bytes") or _sha256(path) != entry.get("sha256"):
            raise ValueError(f"E5 encoder file digest mismatch: {name}")
        checked[name] = entry["sha256"]
    if not {"config.json", "model.safetensors", "tokenizer.json"}.issubset(checked):
        raise ValueError("E5 encoder manifest is incomplete")
    return {"repo": EXPECTED_ENCODER_REPO, "revision": EXPECTED_ENCODER_REVISION,
            "files_sha256": checked, "query_prefix": "query: ", "max_length": 256,
            "pooling": "attention_mask_mean", "l2_normalize": True}


def asset_identity(index_path: Path, corpus_path: Path, model_path: Path) -> dict:
    """Bind the service to the content-checked prepare manifests and E5 weights."""
    if index_path.parent.resolve() != corpus_path.parent.resolve():
        raise ValueError("FAISS index and corpus must share one prepared asset directory")
    manifest_path = corpus_path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") == "spgfs-r2d2-e5-faiss-v1":
        if (index_path.name != "e5_Flat.index" or corpus_path.name != "r2d2.jsonl"
            or index_path.stat().st_size != manifest.get("index_bytes")
            or corpus_path.stat().st_size != manifest.get("corpus_bytes")
            or _sha256(index_path) != manifest.get("index_sha256")
            or _sha256(corpus_path) != manifest.get("corpus_sha256")):
            raise ValueError("R2D2 index or corpus differs from its build manifest")
        if (manifest.get("encoder_repo"), manifest.get("encoder_revision"),
            manifest.get("encoder_passage_prefix"), manifest.get("encoder_max_length"),
            manifest.get("encoder_pooling"), manifest.get("encoder_l2_normalize")) != (
            EXPECTED_ENCODER_REPO, EXPECTED_ENCODER_REVISION, "passage: ", 256,
            "attention_mask_mean", True
        ):
            raise ValueError("R2D2 E5 build settings differ from the fixed retrieval profile")
        return {
            "corpus": {"repo": "local/R2D2-pruned-NQ", "sha256": manifest["corpus_sha256"],
                       "bytes": manifest["corpus_bytes"],
                       "source_archive_sha256": manifest["source_archive_sha256"]},
            "index": {"repo": "local/R2D2-E5-FlatIP", "sha256": manifest["index_sha256"],
                      "bytes": manifest["index_bytes"]},
            "encoder": _e5_encoder_identity(model_path),
        }
    sources = manifest.get("sources")
    if not isinstance(sources, list) or len(sources) != len(EXPECTED_SOURCES):
        raise ValueError("Search-R1 asset manifest has unexpected sources")
    by_name = {item.get("file"): item for item in sources if isinstance(item, dict)}
    if set(by_name) != set(EXPECTED_SOURCES):
        raise ValueError("Search-R1 asset manifest has unexpected files")
    for name, (revision, size, digest) in EXPECTED_SOURCES.items():
        source = by_name[name]
        repo = "PeterJinGo/wiki-18-corpus" if name.endswith(".gz") else "PeterJinGo/wiki-18-e5-index"
        if (source.get("repo"), source.get("revision"), source.get("bytes"),
            source.get("sha256")) != (repo, revision, size, digest):
            raise ValueError(f"Search-R1 asset source identity mismatch: {name}")
    if (index_path.stat().st_size != manifest.get("index_bytes")
        or corpus_path.stat().st_size != manifest.get("corpus_bytes")
        or not isinstance(manifest.get("corpus_sha256"), str)
        or len(manifest["corpus_sha256"]) != 64):
        raise ValueError("Search-R1 index/corpus size or digest manifest mismatch")
    # prepare_searchr1_retrieval.py hashes the sources and extracted corpus before
    # atomically writing its manifest. Changed data requires re-preparation.
    if max(index_path.stat().st_mtime_ns, corpus_path.stat().st_mtime_ns) > manifest_path.stat().st_mtime_ns:
        raise ValueError("Search-R1 assets changed after preparation; re-verify them")

    return {
        "corpus": {"repo": "PeterJinGo/wiki-18-corpus",
                   "revision": EXPECTED_SOURCES["wiki-18.jsonl.gz"][0],
                   "sha256": manifest["corpus_sha256"], "bytes": manifest["corpus_bytes"]},
        "index": {"repo": "PeterJinGo/wiki-18-e5-index",
                  "revision": EXPECTED_SOURCES["part_aa"][0],
                  "parts_sha256": [by_name[name]["sha256"] for name in ("part_aa", "part_ab")],
                  "bytes": manifest["index_bytes"]},
        "encoder": _e5_encoder_identity(model_path),
    }


class JsonlCorpus:
    """Preserve source row order; FAISS labels are row offsets, not document IDs."""

    def __init__(self, path: Path):
        self.path = path
        stat = path.stat()
        with path.open("rb") as source:
            header = source.read(512)
        if len(header) >= 262 and header[257:262] == b"ustar":
            raise ValueError(
                "Corpus path contains a TAR archive, not JSONL; rerun "
                "scripts/formal/prepare_searchr1_retrieval.py"
            )
        offsets_path = path.with_suffix(path.suffix + ".offsets")
        metadata_path = offsets_path.with_suffix(".metadata.json")
        identity = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        try:
            valid = json.loads(metadata_path.read_text()) == identity
        except (FileNotFoundError, ValueError):
            valid = False
        if not valid or not offsets_path.is_file():
            logging.info("Building corpus row offsets: %s", path)
            temporary = offsets_path.with_suffix(".partial")
            with path.open("rb") as source, temporary.open("wb") as target:
                offsets = array.array("Q")
                position = 0
                for row, line in enumerate(source):
                    if not line.strip():
                        raise ValueError(f"Empty corpus row {row}; row alignment is unsafe")
                    offsets.append(position)
                    position += len(line)
                    if len(offsets) == 100_000:
                        offsets.tofile(target)
                        offsets = array.array("Q")
                offsets.append(position)
                offsets.tofile(target)
            temporary.replace(offsets_path)
            metadata_path.write_text(json.dumps(identity))
        self.offsets = np.memmap(offsets_path, dtype=np.uint64, mode="r")
        if len(self.offsets) < 2 or int(self.offsets[-1]) != stat.st_size:
            raise ValueError("Invalid corpus offsets")
        with path.open("rb") as source:
            self.mapping = mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ)

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, row: int):
        if not 0 <= row < len(self):
            raise IndexError(row)
        start, end = int(self.offsets[row]), int(self.offsets[row + 1])
        document = json.loads(self.mapping[start:end])
        if not isinstance(document, dict) or not any(
            isinstance(document.get(key), str) for key in ("contents", "text")
        ):
            raise ValueError(f"Invalid public document at row {row}")
        return document


class DenseRetrievalIndex:
    schema = "spgfs-searchr1-e5-faiss-v1"

    def __init__(self, index_path: Path, corpus_path: Path, model_path: Path, threads: int = 4,
                 *, profile_id: str = "nq-dense8-v1", candidate_k: int = 128,
                 reranker_path: Path | None = None):
        import faiss
        import torch

        if not model_path.is_dir():
            raise FileNotFoundError(f"Local E5 model does not exist: {model_path}")
        if profile_id not in {"nq-dense8-v1", "nq-minilm-rrf8-v1"}:
            raise ValueError(f"Unknown retrieval profile: {profile_id}")
        if candidate_k < 1:
            raise ValueError("candidate_k must be positive")
        if profile_id == "nq-minilm-rrf8-v1" and reranker_path is None:
            raise ValueError("MiniLM profile requires a local reranker")
        self.asset_identity = asset_identity(index_path, corpus_path, model_path)
        self.profile_id = profile_id
        self.candidate_k = candidate_k
        faiss.omp_set_num_threads(threads)
        self.faiss = faiss
        self.threads = threads
        torch.set_num_threads(threads)
        logging.info("Loading FAISS index: %s", index_path)
        self.index = faiss.read_index(str(index_path))
        self.corpus = JsonlCorpus(corpus_path)
        self.document_count = len(self.corpus)
        if self.index.ntotal != self.document_count:
            raise ValueError(
                "FAISS vector count and corpus row count do not match: "
                f"index={self.index.ntotal}, corpus={self.document_count}"
            )
        if self.index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise ValueError("Expected the Search-R1 E5 inner-product index")
        self.encoder = E5SkillEmbedder(model_path)
        dimension = len(self.encoder.encode(["retriever readiness"], query=True)[0])
        if dimension != self.index.d:
            raise ValueError("E5 embedding dimension does not match FAISS index")
        self.lock = threading.Lock()
        self.reranker = None
        if reranker_path is not None:
            if profile_id != "nq-minilm-rrf8-v1":
                raise ValueError("reranker is only available in the MiniLM profile")
            from .passage_reranker import PassageReranker
            self.reranker = PassageReranker(reranker_path, device="cpu", fusion="rrf")
        profile = {"profile_id": profile_id, "candidate_k": candidate_k if self.reranker else None,
                   "reranker_sha256": self.reranker.fingerprint if self.reranker else None,
                   "dimension": dimension, "metric": "inner_product", "dtype": "float32"}
        identity = {"assets": self.asset_identity, "profile": profile}
        self.health_identity = {
            "profile_id": profile_id,
            "asset_identity": self.asset_identity,
            "identity_sha256": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
        }
        logging.info("Ready: %s documents, dimension %s", self.document_count, dimension)

    def search(self, query: str, top_k: int):
        return self.search_batch([query], top_k)[0]

    def search_batch(self, queries: list[str], top_k: int):
        if not queries:
            return []
        if top_k < 1:
            raise ValueError("top_k must be positive")
        # The long-lived batcher is the only caller in normal service operation.
        with self.lock:
            # OpenMP settings are thread-local; HTTP requests run in new threads.
            self.faiss.omp_set_num_threads(self.threads)
            vectors = np.asarray(self.encoder.encode(queries, query=True), dtype=np.float32)
            if vectors.shape != (len(queries), self.index.d):
                raise ValueError("E5 encoder returned an invalid batch shape")
            requested = max(top_k * 2, top_k) if self.reranker is None else max(self.candidate_k, top_k)
            scores, rows = self.index.search(vectors, min(requested, self.document_count))
            groups = []
            for query, row_scores, row_ids in zip(queries, scores, rows, strict=True):
                hits = []
                seen = set()
                for row, score in zip(row_ids, row_scores, strict=True):
                    row = int(row)
                    if not 0 <= row < self.document_count:
                        continue
                    source = self.corpus[row]
                    original_id = source.get("id", str(row))
                    key = str(original_id)
                    if key in seen:
                        continue
                    seen.add(key)
                    document = {field: source[field] for field in ("id", "title", "text", "contents")
                                if field in source}
                    document["corpus_id"] = self.asset_identity["corpus"]["sha256"]
                    document["faiss_row"] = row
                    hits.append({"document": document, "score": float(score),
                                 "faiss_row": row, "dense_rank": len(hits) + 1})
                if self.reranker is not None:
                    hits = self.reranker.rerank(query, hits, top_k)
                groups.append(hits[:top_k])
            return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--profile", choices=("nq-dense8-v1", "nq-minilm-rrf8-v1"),
                        default="nq-dense8-v1")
    parser.add_argument("--candidate-k", type=int, default=128)
    parser.add_argument("--reranker", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-wait-ms", type=int, default=20)
    parser.add_argument("--max-pending", type=int, default=128)
    parser.add_argument("--timeout-s", type=float, default=240.0)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    index = DenseRetrievalIndex(args.index, args.corpus, args.model, args.threads,
                                profile_id=args.profile, candidate_k=args.candidate_k,
                                reranker_path=args.reranker)
    with RetrievalServer((args.host, args.port), RetrievalHandler) as server:
        server.index = index
        server.batcher = RetrievalBatcher(index, batch_size=args.batch_size,
                                          max_wait_ms=args.max_wait_ms,
                                          max_pending=args.max_pending,
                                          timeout_s=args.timeout_s)
        server.serve_forever()


if __name__ == "__main__":
    main()
