"""Cross-encoder reranking with Voyage's rerank-2.5: scores each (query, chunk)
pair jointly, unlike BM25/dense which score the query and each chunk independently.
This is what catches cases like a chunk matching on a coincidental shared word
("custom") despite being about a completely different topic.

A single rerank call sends every candidate's full text in one request, which for
a pool of ~30 chunks can alone exceed Voyage's free-tier 10K-tokens/minute cap
regardless of call spacing -- so candidates are split with the same token-budget
batching used for embedding, and per-batch scores are merged afterward. This is
safe because Voyage's reranker scores each (query, document) pair independently
(pointwise), not relative to the other documents in the same request.
"""

import os

import voyageai
from dotenv import load_dotenv

from rag.embeddings import _make_batches
from rag.rate_limit import wait_for_rate_limit
from rag.retrieval import SearchResult

load_dotenv()

MODEL = "rerank-2.5"

_client = None


def _get_client() -> voyageai.Client:
    global _client
    if _client is None:
        _client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    return _client


def rerank(query: str, candidates: list[SearchResult], k: int = 5) -> list[SearchResult]:
    if not candidates:
        return []

    client = _get_client()
    texts = [c.chunk.text for c in candidates]
    batches = _make_batches(texts)

    all_scored: list[SearchResult] = []
    offset = 0
    for batch in batches:
        wait_for_rate_limit()
        result = client.rerank(query, batch, model=MODEL, top_k=len(batch))
        for r in result.results:
            chunk = candidates[offset + r.index].chunk
            all_scored.append(SearchResult(chunk=chunk, score=r.relevance_score))
        offset += len(batch)

    all_scored.sort(key=lambda r: -r.score)
    return all_scored[:k]
