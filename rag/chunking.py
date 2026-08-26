"""Two chunking strategies over a Document: fixed-size w/ overlap, and structure-aware (markdown headers)."""

import re
from dataclasses import dataclass

import tiktoken

from rag.corpus import Document

ENC = tiktoken.get_encoding("cl100k_base")

HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


@dataclass
class Chunk:
    chunk_id: str      # e.g. "tutorial/query-params.md::fixed::0"
    doc_id: str          # which source file this came from
    section: str          # top-level folder, inherited from the Document
    title: str             # doc title, inherited from the Document
    heading: str             # nearest markdown heading (structure strategy only; "" for fixed)
    text: str
    token_count: int
    strategy: str       # "fixed" or "structure"


def fixed_size_chunks(doc: Document, size: int = 512, overlap: int = 50) -> list[Chunk]:
    tokens = ENC.encode(doc.text)
    step = size - overlap
    chunks = []
    start = 0
    idx = 0
    while start < len(tokens):
        end = min(start + size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}::fixed::{idx}",
                doc_id=doc.doc_id,
                section=doc.section,
                title=doc.title,
                heading="",
                text=ENC.decode(chunk_tokens),
                token_count=len(chunk_tokens),
                strategy="fixed",
            )
        )
        idx += 1
        if end == len(tokens):
            break
        start += step
    return chunks


def structure_aware_chunks(doc: Document, min_tokens: int = 40) -> list[Chunk]:
    text = doc.text
    matches = list(HEADER_RE.finditer(text))

    if not matches:
        return [
            Chunk(
                chunk_id=f"{doc.doc_id}::struct::0",
                doc_id=doc.doc_id,
                section=doc.section,
                title=doc.title,
                heading=doc.title,
                text=text,
                token_count=len(ENC.encode(text)),
                strategy="structure",
            )
        ]

    # Slice the raw text between consecutive header positions -- each slice is
    # "one heading + everything until the next heading, of any level".
    raw_sections = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        heading = m.group(2).split("{")[0].strip()
        raw_sections.append((heading, text[start:end].strip()))

    # Merge sections smaller than min_tokens into the previous chunk so a lone
    # short heading (e.g. just a code snippet placeholder) doesn't become its
    # own near-empty, low-signal chunk.
    merged: list[list] = []
    for heading, section_text in raw_sections:
        token_count = len(ENC.encode(section_text))
        if merged and token_count < min_tokens:
            merged[-1][1] += "\n\n" + section_text
            merged[-1][2] += token_count
        else:
            merged.append([heading, section_text, token_count])

    return [
        Chunk(
            chunk_id=f"{doc.doc_id}::struct::{idx}",
            doc_id=doc.doc_id,
            section=doc.section,
            title=doc.title,
            heading=heading,
            text=section_text,
            token_count=token_count,
            strategy="structure",
        )
        for idx, (heading, section_text, token_count) in enumerate(merged)
    ]


def chunk_corpus(documents: list[Document], strategy: str) -> list[Chunk]:
    fn = fixed_size_chunks if strategy == "fixed" else structure_aware_chunks
    return [chunk for doc in documents for chunk in fn(doc)]
