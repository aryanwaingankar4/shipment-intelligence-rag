"""Structure-aware chunking.

Documents use `=== HEADER ===` lines to mark sections (one product, one shipment,
one SOP section). We keep each section intact as one chunk and only sub-split
sections longer than `max_chars`, so retrieval returns complete records.
"""

import hashlib
import re
from collections import Counter
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from shipment_rag.models import Chunk, Document

_HEADER_RE = re.compile(r"^={3,}\s*(.*?)\s*={3,}", re.MULTILINE)
_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "in", "for", "is", "are", "be", "at", "by", "with",
    "from", "this", "that", "must", "any", "all", "not", "no", "if", "as", "per", "within",
    "product", "shipment",
}


def load_documents(docs_path: Path) -> list[Document]:
    docs = []
    for path in sorted(Path(docs_path).glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            docs.append(Document(text=text, source=path.name))
    return docs


def split_into_sections(text: str) -> list[str]:
    parts = re.split(r"(?=^={3,}.*$)", text, flags=re.MULTILINE)
    return [p.strip() for p in parts if p.strip()]


def section_name_of(block: str) -> str:
    m = _HEADER_RE.match(block)
    return m.group(1).strip() if m else "GENERAL"


def extract_keywords(text: str, k: int = 6) -> str:
    words = [w for w in re.findall(r"[a-z0-9\-]{3,}", text.lower()) if w not in _STOPWORDS]
    # Chroma metadata can't hold lists, so store as a comma string.
    return ", ".join(w for w, _ in Counter(words).most_common(k))


def chunk_id(source: str, text: str) -> str:
    """Content-addressed ID: identical text -> identical ID, so upserts are idempotent."""
    digest = hashlib.sha1(text.strip().lower().encode("utf-8")).hexdigest()[:16]
    return f"{source}:{digest}"


def _splitter(size: int, overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=min(overlap, size // 2),
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_document(
    doc: Document, max_chars: int = 700, overlap: int = 80, section_headers: bool = True
) -> list[Chunk]:
    """section_headers: repeat a section's `=== HEADER ===` line on every sub-chunk, so a
    piece that lost its header (e.g. the CAPA half of an incident) still says which
    incident/product/shipment it belongs to — for both BM25 and the cross-encoder."""
    candidates: list[tuple[str, str]] = []
    for section in split_into_sections(doc.text):
        name = section_name_of(section)
        if len(section) <= max_chars:
            candidates.append((section, name))
            continue
        first = section.splitlines()[0]
        header = first if section_headers and _HEADER_RE.match(first) else ""
        budget = max_chars - len(header) - 1 if header else max_chars
        for piece in _splitter(budget, overlap).split_text(section):
            piece = piece.strip()
            if header and not piece.startswith(header):
                piece = f"{header}\n{piece}"
            candidates.append((piece, name))

    chunks, seen = [], set()
    for i, (text, section) in enumerate(candidates):
        cid = chunk_id(doc.source, text)
        if cid in seen:  # exact duplicate inside one document
            continue
        seen.add(cid)
        chunks.append(
            Chunk(
                id=cid,
                text=text,
                metadata={
                    "source": doc.source,
                    "doc_name": Path(doc.source).stem,
                    "section_name": section,
                    "chunk_index": i,
                    "chunk_total": len(candidates),
                    "char_count": len(text),
                    "word_count": len(text.split()),
                    "keywords": extract_keywords(text),
                },
            )
        )
    return chunks
