"""Recall@k eval harness: for each question, is a chunk from the known-correct
source document among the top-k results for each retrieval method?
"""

import json
from pathlib import Path

from rag.bm25_index import build_bm25
from rag.reranking import rerank
from rag.retrieval import SearchResult, bm25_search, dense_search, fuse_rrf

EVAL_SET_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_set.json"
K = 5
POOL_SIZE = 20
RERANK_POOL_SIZE = 10  # smaller than POOL_SIZE: each rerank call costs a rate-limited batch per ~6-17 candidates
DENSE_WEIGHT = 2.0  # dense outperformed bm25 by a wide margin in the unweighted eval; test discounting bm25's vote


def load_eval_set() -> list[dict]:
    return json.loads(EVAL_SET_PATH.read_text())


def _hit(results: list[SearchResult], correct_doc_id: str) -> bool:
    return any(r.chunk.doc_id == correct_doc_id for r in results)


def _dedup_by_chunk_id(*result_lists: list[SearchResult]) -> list[SearchResult]:
    seen: dict[str, SearchResult] = {}
    for results in result_lists:
        for r in results:
            seen.setdefault(r.chunk.chunk_id, r)
    return list(seen.values())


def run_eval(strategy: str, k: int = K, pool_size: int = POOL_SIZE) -> dict[str, float]:
    eval_set = load_eval_set()
    bm25, chunks = build_bm25(strategy)

    methods = ["bm25", "dense", "hybrid", "hybrid_weighted", "rerank"]
    hits = {m: 0 for m in methods}
    for item in eval_set:
        question, correct_doc_id = item["question"], item["doc_id"]

        bm25_pool = bm25_search(bm25, chunks, question, k=pool_size)
        dense_pool = dense_search(strategy, question, k=pool_size)
        hybrid_topk = fuse_rrf(bm25_pool, dense_pool, k=k)
        hybrid_weighted_topk = fuse_rrf(bm25_pool, dense_pool, k=k, dense_weight=DENSE_WEIGHT)

        candidates = _dedup_by_chunk_id(bm25_pool[:RERANK_POOL_SIZE], dense_pool[:RERANK_POOL_SIZE])
        reranked_topk = rerank(question, candidates, k=k)

        if _hit(bm25_pool[:k], correct_doc_id):
            hits["bm25"] += 1
        if _hit(dense_pool[:k], correct_doc_id):
            hits["dense"] += 1
        if _hit(hybrid_topk, correct_doc_id):
            hits["hybrid"] += 1
        if _hit(hybrid_weighted_topk, correct_doc_id):
            hits["hybrid_weighted"] += 1
        if _hit(reranked_topk, correct_doc_id):
            hits["rerank"] += 1

    n = len(eval_set)
    return {method: count / n for method, count in hits.items()}
