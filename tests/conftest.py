import pytest

from shipment_rag.models import RetrievedChunk
from shipment_rag.pipeline import RAGPipeline


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.collection = FakeCollection(len(chunks))

    def retrieve(self, query):
        return self.chunks


class FakeCollection:
    def __init__(self, n):
        self.n = n

    def count(self):
        return self.n


class FakeLLM:
    def __init__(self, reply="TL-COLD-9001 is stored at 2°C to 8°C [1].", exc=None):
        self.reply, self.exc, self.calls = reply, exc, []

    def generate(self, system, prompt):
        self.calls.append((system, prompt))
        if self.exc:
            raise self.exc
        return self.reply

    def generate_stream(self, system, prompt):
        self.calls.append((system, prompt))
        if self.exc:
            raise self.exc
        words = self.reply.split(" ")
        yield from (w if i == 0 else f" {w}" for i, w in enumerate(words))


@pytest.fixture
def chunk():
    return RetrievedChunk(
        text="=== PRODUCT TL-COLD-9001 ===\nStorage condition: 2°C to 8°C",
        source="product_manual.txt",
        section="PRODUCT TL-COLD-9001",
        embed_score=0.71,
        rerank_score=7.2,
    )


@pytest.fixture
def make_pipeline():
    def _make(chunks, llm=None):
        return RAGPipeline(FakeRetriever(chunks), llm or FakeLLM())

    return _make
