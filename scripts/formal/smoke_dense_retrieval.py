"""Exercise real E5 + FAISS + HTTP + Worker tool on an explicitly synthetic corpus."""
import argparse
import json
import threading
from pathlib import Path

import faiss
import numpy as np

from selfplay_graph_flowsteer.agent_tools import SearchServiceTool
from selfplay_graph_flowsteer.dense_retrieval_service import DenseRetrievalIndex
from selfplay_graph_flowsteer.retrieval_service import RetrievalHandler, RetrievalServer
from selfplay_graph_flowsteer.skills import E5SkillEmbedder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    documents = [
        {"id": "smoke-1", "contents": "Hamlet\nHamlet is a tragedy written by William Shakespeare."},
        {"id": "smoke-2", "contents": "Paris\nParis is the capital of France."},
        {"id": "smoke-3", "contents": "Water\nWater has the chemical formula H2O."},
    ]
    corpus_path = args.output / "synthetic.jsonl"
    corpus_path.write_text("".join(json.dumps(doc) + "\n" for doc in documents))
    encoder = E5SkillEmbedder(args.model)
    vectors = np.asarray(encoder.encode([doc["contents"] for doc in documents], query=False), dtype=np.float32)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    index_path = args.output / "synthetic.index"
    faiss.write_index(index, str(index_path))
    backend = DenseRetrievalIndex(index_path, corpus_path, args.model)
    with RetrievalServer(("127.0.0.1", 0), RetrievalHandler) as server:
        server.index = backend
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            tool = SearchServiceTool(service_url=f"http://127.0.0.1:{server.server_port}/retrieve", top_k=5)
            result = json.loads(tool.execute({"query": "Who wrote Hamlet?"}))
            hits = result["result"][0]
            assert len(hits) == 3, "top_k > corpus size must not return invalid FAISS rows"
            assert hits[0]["document"]["id"] == "smoke-1", result
            report = {"scope": "synthetic plumbing smoke only; not benchmark evidence", "result": result}
            (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2))
        finally:
            server.shutdown()
            thread.join()


if __name__ == "__main__":
    main()
