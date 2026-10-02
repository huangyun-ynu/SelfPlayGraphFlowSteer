"""Reference-blind cross-encoder passage ranking using local model weights."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def select_reranked(hits: list[dict], scores: list[float], top_k: int,
                    *, dense_keep: int = 0, fusion: str = "cross_encoder") -> list[dict]:
    if len(hits) != len(scores) or any(not math.isfinite(s) for s in scores):
        raise ValueError("Invalid reranker scores")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if dense_keep < 0:
        raise ValueError("dense_keep must be non-negative")
    if fusion not in {"cross_encoder", "rrf"}:
        raise ValueError("Unknown ranking fusion")
    ranking_scores = scores
    if fusion == "rrf":
        ranks = {i: rank for rank, i in enumerate(
            sorted(range(len(hits)), key=lambda i: -scores[i]), 1)}
        ranking_scores = [1 / (60 + i + 1) + 1 / (60 + ranks[i]) for i in range(len(hits))]
    # Stable ties retain the dense ranking. The original similarity stays available.
    kept = min(dense_keep, top_k, len(hits))
    order = list(range(kept)) + sorted(range(kept, len(hits)), key=lambda i: -ranking_scores[i])[:top_k - kept]
    return [
        {**hits[i], "dense_score": hits[i]["score"], "dense_rank": i + 1,
         "score": float(ranking_scores[i]), "reranker_score": float(scores[i])}
        for i in order
    ]


class PassageReranker:
    def __init__(self, model_path: Path, *, device: str = "cpu", batch_size: int = 32,
                 dense_keep: int = 0, fusion: str = "rrf"):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if not model_path.is_dir():
            raise FileNotFoundError(f"Local reranker does not exist: {model_path}")
        if batch_size < 1:
            raise ValueError("reranker batch_size must be positive")
        self.torch = torch
        self.device = device
        self.batch_size = batch_size
        self.dense_keep = dense_keep
        self.fusion = fusion
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, local_files_only=True, use_safetensors=True,
        ).to(device).eval()
        if self.model.config.num_labels != 1:
            raise ValueError("Expected a single relevance score per passage")
        # Content identity, not a path alias, invalidates frozen evidence caches.
        digests = {}
        for name in ("model.safetensors", "config.json", "tokenizer.json",
                     "tokenizer_config.json", "special_tokens_map.json", "vocab.txt",
                     "sentencepiece.bpe.model"):
            path = model_path / name
            if path.is_file():
                with path.open("rb") as source:
                    digests[name] = hashlib.file_digest(source, "sha256").hexdigest()
        self.fingerprint = hashlib.sha256(json.dumps(digests, sort_keys=True).encode()).hexdigest()

    def score(self, question: str, hits: list[dict]) -> list[float]:
        # This interface deliberately accepts only a question and public passages.
        scores = []
        for start in range(0, len(hits), self.batch_size):
            documents = [hit["document"] for hit in hits[start:start + self.batch_size]]
            texts = [str(d.get("contents", d.get("text", ""))) for d in documents]
            pairs = self.tokenizer(
                [question] * len(texts), texts, padding=True, truncation=True,
                max_length=512, return_tensors="pt",
            ).to(self.device)
            with self.torch.inference_mode():
                scores.extend(self.model(**pairs).logits.reshape(-1).float().cpu().tolist())
        return scores

    def rerank(self, question: str, hits: list[dict], top_k: int) -> list[dict]:
        return select_reranked(hits, self.score(question, hits), top_k,
                               dense_keep=self.dense_keep, fusion=self.fusion)
