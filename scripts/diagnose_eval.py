import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.bm25_index import build_bm25
from rag.eval import K, POOL_SIZE, _hit, load_eval_set
from rag.retrieval import bm25_search, dense_search, fuse_rrf

STRATEGY = "fixed"

eval_set = load_eval_set()
bm25, chunks = build_bm25(STRATEGY)

for item in eval_set:
    question, correct_doc_id = item["question"], item["doc_id"]

    bm25_pool = bm25_search(bm25, chunks, question, k=POOL_SIZE)
    dense_pool = dense_search(STRATEGY, question, k=POOL_SIZE)
    hybrid_topk = fuse_rrf(bm25_pool, dense_pool, k=K)

    dense_ok = _hit(dense_pool[:K], correct_doc_id)
    hybrid_ok = _hit(hybrid_topk[:K], correct_doc_id)

    if dense_ok and not hybrid_ok:
        print(f"\nFLIP (dense hit, hybrid miss): {question!r}")
        print(f"  correct doc_id: {correct_doc_id}")
        print("  dense top-5:")
        for r in dense_pool[:K]:
            print(f"    {r.chunk.doc_id}  (score={r.score:.4f})")
        print("  hybrid top-5:")
        for r in hybrid_topk[:K]:
            print(f"    {r.chunk.doc_id}  (score={r.score:.4f})")
        print("  bm25 top-5 (raw, for reference):")
        for r in bm25_pool[:K]:
            print(f"    {r.chunk.doc_id}  (score={r.score:.4f})")
