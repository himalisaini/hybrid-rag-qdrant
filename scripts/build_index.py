import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.bm25_index import build_bm25
from rag.indexing import build_index, collection_name, get_client

for strategy in ["fixed", "structure"]:
    print(f"\n=== indexing strategy: {strategy} ===")

    t0 = time.time()
    chunks = build_index(strategy)
    print(f"Qdrant: upserted {len(chunks)} points into '{collection_name(strategy)}' in {time.time() - t0:.1f}s")

    bm25, bm25_chunks = build_bm25(strategy)
    assert len(bm25_chunks) == len(chunks)
    print(f"BM25:   built index over {len(bm25_chunks)} chunks")

print("\n=== collection summary ===")
client = get_client()
for strategy in ["fixed", "structure"]:
    info = client.get_collection(collection_name(strategy))
    print(f"{collection_name(strategy)}: {info.points_count} points")
