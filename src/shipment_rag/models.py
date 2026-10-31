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
    embed_score: float  # cosine similarity, 0..1
    rerank_score: float  # cross-encoder logit


@dataclass
class RAGAnswer:
    answer: str
    sources: list[RetrievedChunk]
    refused: bool
    latency_ms: int
