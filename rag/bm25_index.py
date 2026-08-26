"""BM25 (sparse/keyword) index, rebuilt in-memory from the deterministic chunker each run."""

import re

from rank_bm25 import BM25Okapi

from rag.chunking import Chunk, chunk_corpus
from rag.corpus import load_corpus

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def build_bm25(strategy: str) -> tuple[BM25Okapi, list[Chunk]]:
    docs = load_corpus()
    chunks = chunk_corpus(docs, strategy)
    tokenized_corpus = [tokenize(c.text) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, chunks
