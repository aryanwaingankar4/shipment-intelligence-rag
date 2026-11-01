from pathlib import Path

from shipment_rag.ingestion.chunker import (
    chunk_document,
    load_documents,
    section_name_of,
    split_into_sections,
)
from shipment_rag.models import Document

DOCS = Path(__file__).resolve().parents[1] / "data" / "docs"

SAMPLE = """=== PRODUCT MANUAL ===
Intro line.

=== PRODUCT TL-COLD-9001 ===
Storage: 2°C to 8°C

=== PRODUCT TL-DRY-4402 ===
Storage: 15°C to 25°C
"""


def test_split_keeps_header_with_body():
    sections = split_into_sections(SAMPLE)
    assert len(sections) == 3
    assert sections[1].startswith("=== PRODUCT TL-COLD-9001 ===")
    assert "2°C to 8°C" in sections[1]


def test_section_name():
    assert section_name_of("=== SHIPMENT SH-2024-003 ===\nbody") == "SHIPMENT SH-2024-003"
    assert section_name_of("no header here") == "GENERAL"


def test_chunk_ids_are_deterministic_and_unique():
    doc = Document(text=SAMPLE, source="x.txt")
    a, b = chunk_document(doc), chunk_document(doc)
    assert [c.id for c in a] == [c.id for c in b]
    assert len({c.id for c in a}) == len(a)


def test_long_sections_are_sub_split_and_keep_section_name():
    body = "=== BIG SECTION ===\n" + "\n".join(f"Rule {i}: keep cold." for i in range(200))
    chunks = chunk_document(Document(text=body, source="big.txt"), max_chars=300, overlap=30)
    assert len(chunks) > 1
    assert all(len(c.text) <= 300 for c in chunks)
    assert {c.metadata["section_name"] for c in chunks} == {"BIG SECTION"}


def test_sub_chunks_repeat_their_section_header():
    body = "=== INCIDENT INC-2024-01 ===\n" + "\n".join(f"Line {i}: detail." for i in range(80))
    doc = Document(text=body, source="inc.txt")
    with_headers = chunk_document(doc, max_chars=300, overlap=30)
    assert len(with_headers) > 1
    assert all(c.text.startswith("=== INCIDENT INC-2024-01 ===") for c in with_headers)
    assert all(len(c.text) <= 300 for c in with_headers)

    without = chunk_document(doc, max_chars=300, overlap=30, section_headers=False)
    assert not all(c.text.startswith("=== INCIDENT") for c in without)


def test_knowledge_base_uses_fictional_brand_only():
    # D1: the docs must not reference the real company TraceLink
    for doc in load_documents(DOCS):
        assert "tracelink" not in doc.text.lower(), doc.source
    manual = (DOCS / "product_manual.txt").read_text(encoding="utf-8")
    assert "quality@nexachain-biomed.com" in manual


def test_repo_knowledge_base_chunks_cleanly():
    docs = load_documents(DOCS)
    assert {d.source for d in docs} >= {"product_manual.txt", "shipment_log.txt"}
    chunks = [c for d in docs for c in chunk_document(d)]
    # every shipment record must land in its own chunk
    for sid in range(1, 11):
        tag = f"SH-2024-{sid:03d} ==="
        assert any(tag in c.text for c in chunks), tag
