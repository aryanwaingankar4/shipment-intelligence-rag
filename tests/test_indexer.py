import hashlib
import uuid

import chromadb
import pytest

from shipment_rag.config import Settings
from shipment_rag.ingestion.indexer import ingest


class FakeEmbedder:
    def __init__(self):
        self.encoded = 0

    def encode(self, texts):
        self.encoded += len(texts)
        return [[b / 255 for b in hashlib.sha256(t.encode()).digest()[:8]] for t in texts]


@pytest.fixture
def collection():
    client = chromadb.EphemeralClient()
    return client.create_collection(f"t{uuid.uuid4().hex}", metadata={"hnsw:space": "cosine"})


def write(path, text):
    path.write_text(text, encoding="utf-8")


def test_ingest_is_incremental(tmp_path, collection):
    write(tmp_path / "a.txt", "=== A1 ===\nalpha\n\n=== A2 ===\nbeta\n")
    write(tmp_path / "b.txt", "=== B1 ===\ngamma\n")
    settings = Settings(_env_file=None, docs_path=tmp_path)
    emb = FakeEmbedder()

    first = ingest(settings, collection, emb)
    assert (first.added, first.deleted, collection.count()) == (3, 0, 3)

    # unchanged re-run embeds nothing
    second = ingest(settings, collection, emb)
    assert (second.added, second.unchanged, emb.encoded) == (0, 3, 3)

    # edit one section, delete a file
    write(tmp_path / "a.txt", "=== A1 ===\nalpha\n\n=== A2 ===\nbeta v2\n")
    (tmp_path / "b.txt").unlink()
    third = ingest(settings, collection, emb)
    assert (third.added, third.deleted, third.unchanged) == (1, 2, 1)
    assert collection.count() == 2


def test_rebuild_wipes_collection(tmp_path, collection):
    write(tmp_path / "a.txt", "=== A1 ===\nalpha\n")
    settings = Settings(_env_file=None, docs_path=tmp_path)
    ingest(settings, collection, FakeEmbedder())
    report = ingest(settings, collection, FakeEmbedder(), rebuild=True)
    assert report.added == 1 and collection.count() == 1


def test_empty_docs_dir_fails_loudly(tmp_path, collection):
    with pytest.raises(FileNotFoundError):
        ingest(Settings(_env_file=None, docs_path=tmp_path), collection, FakeEmbedder())
