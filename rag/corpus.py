"""Load the raw corpus: one Document per markdown file under corpus/docs/."""

from dataclasses import dataclass
from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus" / "docs"

# release-notes.md is a 200k-token changelog, not prose documentation -- it would
# dominate chunk counts and skew eval, so it's excluded from the corpus.
EXCLUDED_FILES = {"release-notes.md"}


@dataclass
class Document:
    doc_id: str       # relative path, e.g. "tutorial/query-params.md"
    section: str       # top-level folder, e.g. "tutorial"
    title: str          # first markdown heading, falls back to filename
    text: str            # raw markdown content


def _extract_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").split("{")[0].strip()
    return fallback


def load_corpus(corpus_dir: Path = CORPUS_DIR) -> list[Document]:
    documents = []
    for path in sorted(corpus_dir.rglob("*.md")):
        if path.name in EXCLUDED_FILES:
            continue
        rel_path = path.relative_to(corpus_dir)
        doc_id = str(rel_path)
        section = rel_path.parts[0] if len(rel_path.parts) > 1 else "root"
        text = path.read_text(encoding="utf-8")
        title = _extract_title(text, fallback=path.stem)
        documents.append(Document(doc_id=doc_id, section=section, title=title, text=text))
    return documents


if __name__ == "__main__":
    docs = load_corpus()
    print(f"Loaded {len(docs)} documents")
    print(f"Sections: {sorted(set(d.section for d in docs))}")
    print(f"Sample: {docs[0].doc_id!r} -> title={docs[0].title!r}")
