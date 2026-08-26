"""Three retrieval methods over one strategy's index: BM25-only, dense-only, and hybrid (RRF)."""

from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from rag.bm25_index import tokenize
from rag.chunking import Chunk
from rag.embeddings import embed_texts
from rag.indexing import collection_name, get_client

RRF_K = 60


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


def _chunk_from_payload(payload: dict) -> Chunk:
    return Chunk(
        chunk_id=payload["chunk_id"],
        doc_id=payload["doc_id"],
        section=payload["section"],
        title=payload["title"],
        heading=payload["heading"],
        text=payload["text"],
        token_count=payload["token_count"],
        strategy=payload["strategy"],
    )


def bm25_search(bm25: BM25Okapi, chunks: list[Chunk], query: str, k: int = 10) -> list[SearchResult]:
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(zip(chunks, scores), key=lambda pair: -pair[1])[:k]
    return [SearchResult(chunk=c, score=float(s)) for c, s in ranked]


def dense_search(strategy: str, query: str, k: int = 10) -> list[SearchResult]:
    qvec = embed_texts([query], input_type="query")[0]
    client = get_client()
    hits = client.query_points(collection_name=collection_name(strategy), query=qvec, limit=k).points
    return [SearchResult(chunk=_chunk_from_payload(h.payload), score=h.score) for h in hits]


def fuse_rrf(
    bm25_results: list[SearchResult],
    dense_results: list[SearchResult],
    k: int = 10,
    bm25_weight: float = 1.0,
    dense_weight: float = 1.0,
) -> list[SearchResult]:
    """Combine two already-computed ranked lists via (optionally weighted) Reciprocal Rank Fusion.

    Takes result lists rather than a query, so a caller that needs BM25-only,
    dense-only, AND hybrid results for the same query (e.g. the eval harness)
    can compute each ranked list exactly once and reuse it in all three places
    -- important because a dense search costs a rate-limited API call.

    Equal weights (the default) reproduce standard RRF, where both rankers get
    an equal vote regardless of which one is actually more reliable for a given
    query. bm25_weight/dense_weight let a caller discount a signal it trusts
    less -- e.g. dense_weight=2.0 halves BM25's relative influence on the fused
    score without ignoring it outright.
    """
    weights = {"bm25": bm25_weight, "dense": dense_weight}
    rrf_scores: dict[str, float] = {}
    chunk_lookup: dict[str, Chunk] = {}
    for method, ranked_list in (("bm25", bm25_results), ("dense", dense_results)):
        for rank, result in enumerate(ranked_list, start=1):
            cid = result.chunk.chunk_id
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + weights[method] / (RRF_K + rank)
            chunk_lookup[cid] = result.chunk

    top_ids = sorted(rrf_scores, key=lambda cid: -rrf_scores[cid])[:k]
    return [SearchResult(chunk=chunk_lookup[cid], score=rrf_scores[cid]) for cid in top_ids]


def hybrid_search(
    strategy: str,
    bm25: BM25Okapi,
    chunks: list[Chunk],
    query: str,
    k: int = 10,
    pool_size: int = 20,
) -> list[SearchResult]:
    """Convenience one-shot version for standalone use (e.g. a CLI demo). Runs
    both searches itself -- prefer fuse_rrf directly when you already have
    bm25_results and dense_results from elsewhere."""
    bm25_results = bm25_search(bm25, chunks, query, k=pool_size)
    dense_results = dense_search(strategy, query, k=pool_size)
    return fuse_rrf(bm25_results, dense_results, k=k)
