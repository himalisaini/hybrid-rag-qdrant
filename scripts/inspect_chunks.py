import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.corpus import load_corpus
from rag.chunking import chunk_corpus

docs = load_corpus()

for strategy in ["fixed", "structure"]:
    chunks = chunk_corpus(docs, strategy)
    counts = [c.token_count for c in chunks]
    print(f"\n=== strategy: {strategy} ===")
    print(f"total chunks:   {len(chunks)}")
    print(f"avg tokens:     {sum(counts) // len(chunks)}")
    print(f"min/max tokens: {min(counts)} / {max(counts)}")

print("\n\n=== side-by-side on tutorial/query-params.md ===")
target = next(d for d in docs if d.doc_id == "tutorial/query-params.md")

print("\n--- FIXED-SIZE chunks (size=512, overlap=50) ---")
for c in chunk_corpus([target], "fixed"):
    print(f"\n[{c.chunk_id}]  ({c.token_count} tokens)")
    print(c.text[:300].replace("\n", " ") + ("..." if len(c.text) > 300 else ""))

print("\n\n--- STRUCTURE-AWARE chunks ---")
for c in chunk_corpus([target], "structure"):
    print(f"\n[{c.chunk_id}]  heading={c.heading!r}  ({c.token_count} tokens)")
    print(c.text[:300].replace("\n", " ") + ("..." if len(c.text) > 300 else ""))
