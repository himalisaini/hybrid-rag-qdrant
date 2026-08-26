import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.bm25_index import build_bm25
from rag.retrieval import bm25_search, dense_search, hybrid_search

STRATEGY = "structure"
QUERY = "How do I make a query parameter optional?"
K = 5

bm25, chunks = build_bm25(STRATEGY)


def show(label, results):
    print(f"\n--- {label} ---")
    for rank, r in enumerate(results, start=1):
        print(f"{rank}. score={r.score:.4f}  {r.chunk.chunk_id}")


show("BM25-only", bm25_search(bm25, chunks, QUERY, k=K))
show("Dense-only", dense_search(STRATEGY, QUERY, k=K))
show("Hybrid (RRF)", hybrid_search(STRATEGY, bm25, chunks, QUERY, k=K))
