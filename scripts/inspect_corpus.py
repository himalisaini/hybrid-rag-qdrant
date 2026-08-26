import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tiktoken
from rag.corpus import load_corpus

docs = load_corpus()
enc = tiktoken.get_encoding("cl100k_base")

token_counts = [len(enc.encode(d.text)) for d in docs]
sections = sorted(set(d.section for d in docs))

print(f"documents:      {len(docs)}")
print(f"sections:       {sections}")
print(f"total tokens:   {sum(token_counts):,}")
print(f"avg tokens/doc: {sum(token_counts) // len(docs)}")
print(f"min/max tokens: {min(token_counts)} / {max(token_counts)}")

print("\n5 longest documents:")
for d, t in sorted(zip(docs, token_counts), key=lambda x: -x[1])[:5]:
    print(f"  {t:6d} tokens  {d.doc_id}")
