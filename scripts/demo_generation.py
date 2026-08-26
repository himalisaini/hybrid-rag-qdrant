import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.bm25_index import build_bm25
from rag.generation import generate_answer
from rag.retrieval import bm25_search, dense_search, fuse_rrf

STRATEGY = "structure"
QUERY = "How do I make a query parameter optional?"
K = 5

bm25, chunks = build_bm25(STRATEGY)
bm25_results = bm25_search(bm25, chunks, QUERY, k=20)
dense_results = dense_search(STRATEGY, QUERY, k=20)
top_results = fuse_rrf(bm25_results, dense_results, k=K)

print("Retrieved chunks (hybrid):")
for r in top_results:
    print(f"  {r.chunk.chunk_id}")

print("\n--- Generated answer ---\n")
print(generate_answer(QUERY, top_results))
