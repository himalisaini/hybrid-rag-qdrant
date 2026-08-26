"""Embed chunks with Voyage AI and load them into a Qdrant (embedded, local-disk) collection."""

from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm

from rag.chunking import Chunk, chunk_corpus
from rag.corpus import load_corpus
from rag.embeddings import EMBED_DIM, embed_texts

QDRANT_PATH = Path(__file__).resolve().parent.parent / "qdrant_data"

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(path=str(QDRANT_PATH))
    return _client


def collection_name(strategy: str) -> str:
    return f"chunks_{strategy}"


def build_index(strategy: str) -> list[Chunk]:
    docs = load_corpus()
    chunks = chunk_corpus(docs, strategy)

    client = get_client()
    name = collection_name(strategy)
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )

    texts = [c.text for c in chunks]
    vectors = embed_texts(texts, input_type="document")

    points = [
        PointStruct(
            id=i,
            vector=vector,
            payload={
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "section": c.section,
                "title": c.title,
                "heading": c.heading,
                "text": c.text,
                "token_count": c.token_count,
                "strategy": c.strategy,
            },
        )
        for i, (c, vector) in enumerate(zip(chunks, vectors))
    ]

    for i in tqdm(range(0, len(points), 64), desc=f"upserting {name}"):
        client.upsert(collection_name=name, points=points[i : i + 64])

    return chunks
