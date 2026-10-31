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


def chunk_document(doc: Document, max_chars: int = 700, overlap: int = 80) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars, chunk_overlap=overlap, separators=["\n\n", "\n", ". ", " ", ""]
    )
    candidates: list[tuple[str, str]] = []
    for section in split_into_sections(doc.text):
        name = section_name_of(section)
        if len(section) <= max_chars:
            candidates.append((section, name))
        else:
            candidates.extend((piece.strip(), name) for piece in splitter.split_text(section))

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
