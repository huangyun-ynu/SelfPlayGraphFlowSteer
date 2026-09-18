"""Search-R1 Wiki-18 E5/FAISS retrieval, using the project's offline E5 encoder."""

from __future__ import annotations

import argparse
import array
import json
import logging
import mmap
import threading
from pathlib import Path

import numpy as np

from .retrieval_service import RetrievalHandler, RetrievalServer
from .skills import E5SkillEmbedder


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

    def __init__(self, index_path: Path, corpus_path: Path, model_path: Path, threads: int = 4):
        import faiss
        import torch

        if not model_path.is_dir():
            raise FileNotFoundError(f"Local E5 model does not exist: {model_path}")
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
        logging.info("Ready: %s documents, dimension %s", self.document_count, dimension)

    def search(self, query: str, top_k: int):
        # Bound model/index work when the 24 rollout slots call concurrently.
        with self.lock:
            # OpenMP settings are thread-local; HTTP requests run in new threads.
            self.faiss.omp_set_num_threads(self.threads)
            vectors = np.asarray(self.encoder.encode([query], query=True), dtype=np.float32)
            scores, rows = self.index.search(vectors, min(top_k, self.document_count))
            return [
                {"document": self.corpus[int(row)], "score": float(score)}
                for row, score in zip(rows[0], scores[0], strict=True)
                if 0 <= row < self.document_count
            ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    index = DenseRetrievalIndex(args.index, args.corpus, args.model, args.threads)
    with RetrievalServer((args.host, args.port), RetrievalHandler) as server:
        server.index = index
        server.serve_forever()


if __name__ == "__main__":
    main()
