"""Domain objects shared across ingestion, retrieval, pipeline and API."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Document:
    text: str
    source: str  # filename, e.g. "shipment_log.txt"


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    source: str
    section: str
    embed_score: float | None  # cosine similarity 0..1; None if only BM25 found it
    rerank_score: float  # cross-encoder logit


@dataclass
class RAGAnswer:
    answer: str
    sources: list[RetrievedChunk]  # shown to the user; empty when refused
    refused: bool
    latency_ms: int
    # passages given to the LLM, in prompt order: what [n] citations refer to
    context: list[RetrievedChunk] = field(default_factory=list)
    llm_called: bool = False  # False when refused at retrieval
