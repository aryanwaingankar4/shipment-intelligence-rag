"""Incremental indexing into Chroma.

Chunk IDs are content hashes, so a re-run only embeds chunks that are new,
deletes chunks whose text disappeared, and leaves unchanged chunks alone.
Safe to run repeatedly (e.g. on every deploy or from a cron job).
"""

import logging
import time
from dataclasses import dataclass

from shipment_rag.config import Settings
from shipment_rag.embeddings import Embedder
from shipment_rag.ingestion.chunker import chunk_document, load_documents

log = logging.getLogger(__name__)
BATCH = 64


@dataclass
class IngestReport:
    documents: int
    chunks: int
    added: int
    deleted: int
    unchanged: int
    seconds: float


def ingest(
    settings: Settings, collection, embedder: Embedder, rebuild: bool = False
) -> IngestReport:
    t0 = time.perf_counter()
    docs = load_documents(settings.docs_path)
    if not docs:
        raise FileNotFoundError(f"no .txt documents found in {settings.docs_path}")

    chunks = [
        c for d in docs for c in chunk_document(d, settings.chunk_max_chars, settings.chunk_overlap)
    ]
    # the same text can appear in two files; keep the first occurrence
    by_id = {}
    for c in chunks:
        by_id.setdefault(c.id, c)

    if rebuild:
        existing_ids: set[str] = set()
        all_ids = collection.get(include=[])["ids"]
        if all_ids:
            collection.delete(ids=all_ids)
    else:
        existing_ids = set(collection.get(include=[])["ids"])

    stale = sorted(existing_ids - by_id.keys())
    if stale:
        collection.delete(ids=stale)

    kept = [c for cid, c in by_id.items() if cid in existing_ids]
    if kept:  # refresh metadata cheaply (no re-embedding)
        collection.update(ids=[c.id for c in kept], metadatas=[c.metadata for c in kept])

    new = [c for cid, c in by_id.items() if cid not in existing_ids]
    for start in range(0, len(new), BATCH):
        batch = new[start : start + BATCH]
        collection.upsert(
            ids=[c.id for c in batch],
            documents=[c.text for c in batch],
            metadatas=[c.metadata for c in batch],
            embeddings=embedder.encode([c.text for c in batch]),
        )

    report = IngestReport(
        documents=len(docs),
        chunks=len(by_id),
        added=len(new),
        deleted=len(stale) if not rebuild else 0,
        unchanged=len(kept),
        seconds=round(time.perf_counter() - t0, 2),
    )
    log.info("ingest complete", extra=report.__dict__)
    return report
