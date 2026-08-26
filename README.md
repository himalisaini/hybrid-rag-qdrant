# Hybrid Retrieval RAG Pipeline

RAG over FastAPI's docs (154 pages), built to actually measure retrieval quality instead of just wiring an LLM to a vector store and calling it done.

## What it does

- **Chunking**: two strategies — fixed-size (512 tokens, 50 overlap) and structure-aware (splits on markdown headers).
- **Indexing**: dense vectors in Qdrant (Voyage embeddings) + a BM25 keyword index, one pair per chunking strategy.
- **Retrieval**: BM25-only, dense-only, and hybrid via Reciprocal Rank Fusion.
- **Generation**: retrieved chunks go to an LLM with a prompt that forces every claim to cite its source chunk.
- **Eval**: 34 hand-written Q&A pairs with known-correct sources, scored with recall@5 across every method × both chunking strategies.

## What I found

Naive hybrid (equal-weight RRF) actually *lost* to plain dense retrieval — 91% vs 100% recall@5. Dug into why: RRF gives BM25 and dense an equal vote no matter which one's actually right for a given query, and on one question BM25 confidently latched onto a coincidental shared word ("custom") in a totally unrelated doc, dragging the correct answer out of the fused top-5.

| method | fixed chunks | structure chunks |
|---|---|---|
| BM25 only | 82% | 79% |
| dense only | 100% | 94% |
| hybrid (RRF, equal weight) | 91% | 91% |
| hybrid (RRF, dense weighted 2x) | 94% | 91% |
| **rerank (cross-encoder, after RRF)** | **100%** | **100%** |

Weighting dense higher clawed back some of the gap. Adding a reranking stage — a cross-encoder that reads the query and chunk *together* instead of combining two independently-computed scores — closed it completely. Makes sense: it's the only method that can tell "these share a word" apart from "these are actually about the same thing."

## Stack

Python · Qdrant (embedded, no Docker) · Voyage AI (embeddings + reranking) · `rank_bm25` · Gemini for generation.

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env   # add your API keys
python scripts/build_index.py   # builds both indexes, ~50 min on free-tier rate limits
python scripts/run_eval.py      # runs the full eval, ~70 min, same reason
```

Everything's rate-limited to survive Voyage's no-billing free tier (3 req/min), which is why the two big scripts are slow. `scripts/demo_retrieval.py` and `scripts/demo_generation.py` are faster one-off ways to poke at it.
