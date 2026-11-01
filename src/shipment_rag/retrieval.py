"""Retrieval: candidate generation (vector, BM25, or both fused with RRF), then
cross-encoder rerank and an irrelevance threshold.

    vector  : bi-encoder top-k from Chroma
    bm25    : lexical top-k (good at exact IDs like SH-2024-003, BATCH-FRZ-2403)
    hybrid  : both lists merged with reciprocal rank fusion, then reranked
"""

import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Literal

from shipment_rag.embeddings import Embedder, Reranker
from shipment_rag.models import RetrievedChunk

log = logging.getLogger(__name__)

RetrievalMode = Literal["vector", "bm25", "hybrid"]

_TOKEN = re.compile(r"[a-z0-9]+(?:[-_./@][a-z0-9]+)*")
_SPLIT = re.compile(r"[-_./@]")
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "did", "do", "does", "for",
    "from", "had", "has", "have", "how", "i", "if", "in", "into", "is", "it", "its", "me",
    "my", "of", "on", "or", "our", "should", "so", "than", "that", "the", "their", "them",
    "then", "there", "these", "this", "those", "to", "was", "were", "what", "when", "where",
    "which", "who", "whom", "why", "will", "with", "would", "you", "your",
})


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens; compound IDs are kept whole *and* split into parts, so
    'SH-2024-003' matches exactly while 'shipment 2024' still overlaps."""
    out = []
    for tok in _TOKEN.findall(text.lower()):
        if tok in _STOPWORDS:
            continue
        out.append(tok)
        if _SPLIT.search(tok):
            out.extend(p for p in _SPLIT.split(tok) if p and p not in _STOPWORDS)
    return out


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """score(d) = sum over rankings of 1 / (k + rank(d)), rank starting at 1."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, 1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


class BM25Index:
    def __init__(self, ids: list[str], texts: list[str], metadatas: list[dict]):
        from rank_bm25 import BM25Okapi

        self.ids = ids
        self.docs = {i: (t, m or {}) for i, t, m in zip(ids, texts, metadatas, strict=True)}
        self._bm25 = BM25Okapi([tokenize(t) for t in texts]) if ids else None

    @classmethod
    def from_collection(cls, collection) -> "BM25Index":
        data = collection.get(include=["documents", "metadatas"])
        return cls(data["ids"], data["documents"], data["metadatas"])

    def __len__(self) -> int:
        return len(self.ids)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        terms = tokenize(query)
        if self._bm25 is None or not terms:
            return []
        scores = self._bm25.get_scores(terms)
        ranked = sorted(zip(self.ids, scores, strict=True), key=lambda x: x[1], reverse=True)
        return [(i, float(s)) for i, s in ranked[:k] if s > 0]


@dataclass
class Candidate:
    id: str
    text: str
    meta: dict
    embed_score: float | None = None
    found_by: set[str] = field(default_factory=set)


@dataclass
class RetrievalTrace:
    """What happened inside one retrieval, for evaluation and debugging."""

    candidates: list[Candidate]  # what the reranker saw, in fused order
    results: list[RetrievedChunk]  # final answer context
    top_score: float | None = None  # best rerank score before the threshold


class Retriever:
    def __init__(
        self,
        collection,
        embedder: Embedder,
        reranker: Reranker,
        retrieve_k: int = 8,
        final_k: int = 3,
        min_rerank_score: float | None = None,
        mode: RetrievalMode = "hybrid",
        bm25_k: int = 8,
        rerank_k: int = 12,
        rrf_k: int = 60,
    ):
        self.collection = collection
        self.embedder = embedder
        self.reranker = reranker
        self.retrieve_k = retrieve_k
        self.final_k = final_k
        self.min_rerank_score = min_rerank_score
        self.mode = mode
        self.bm25_k = bm25_k
        self.rerank_k = rerank_k
        self.rrf_k = rrf_k
        self._bm25: BM25Index | None = None
        self._bm25_lock = threading.Lock()

    # --- candidate generation ---

    def _bm25_index(self) -> BM25Index:
        # Rebuild when the collection size changes (e.g. after a re-ingest).
        # Chroma stays the single source of truth; BM25 is a derived in-memory index.
        count = self.collection.count()
        with self._bm25_lock:
            if self._bm25 is None or len(self._bm25) != count:
                self._bm25 = BM25Index.from_collection(self.collection)
                log.info("bm25 index built", extra={"chunks": len(self._bm25)})
            return self._bm25

    def _vector_candidates(self, query: str) -> list[Candidate]:
        res = self.collection.query(
            query_embeddings=self.embedder.encode([query]),
            n_results=self.retrieve_k,
            include=["documents", "metadatas", "distances"],
        )
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        return [
            Candidate(i, d, m or {}, embed_score=round(1.0 - float(dist), 4), found_by={"vector"})
            for i, d, m, dist in zip(ids, docs, metas, dists, strict=True)
        ]

    def _bm25_candidates(self, query: str) -> list[Candidate]:
        index = self._bm25_index()
        out = []
        for doc_id, _score in index.search(query, self.bm25_k):
            text, meta = index.docs[doc_id]
            out.append(Candidate(doc_id, text, meta, found_by={"bm25"}))
        return out

    def candidates(self, query: str) -> list[Candidate]:
        if self.mode == "vector":
            return self._vector_candidates(query)[: self.rerank_k]
        if self.mode == "bm25":
            return self._bm25_candidates(query)[: self.rerank_k]

        vec, lex = self._vector_candidates(query), self._bm25_candidates(query)
        by_id: dict[str, Candidate] = {c.id: c for c in vec}
        for c in lex:
            if c.id in by_id:
                by_id[c.id].found_by |= c.found_by
            else:
                by_id[c.id] = c
        fused = reciprocal_rank_fusion([[c.id for c in vec], [c.id for c in lex]], k=self.rrf_k)
        return [by_id[i] for i, _ in fused[: self.rerank_k]]

    # --- rerank + threshold ---

    def trace(self, query: str) -> RetrievalTrace:
        cands = self.candidates(query)
        if not cands:
            return RetrievalTrace([], [])
        threshold = self.min_rerank_score
        try:
            scores = self.reranker.score(query, [c.text for c in cands])
        except Exception:  # degrade gracefully to candidate order
            log.exception("rerank failed; falling back to candidate order")
            scores = [float(-i) for i in range(len(cands))]
            threshold = None  # placeholder scores aren't cross-encoder logits

        ranked = sorted(zip(cands, scores, strict=True), key=lambda x: x[1], reverse=True)
        results = [
            RetrievedChunk(
                text=c.text,
                source=c.meta.get("source", "unknown"),
                section=c.meta.get("section_name", ""),
                embed_score=c.embed_score,
                rerank_score=round(float(s), 4),
            )
            for c, s in ranked
        ]
        top = results[0].rerank_score
        if threshold is not None:
            results = [r for r in results if r.rerank_score >= threshold]
        return RetrievalTrace(cands, results[: self.final_k], top_score=top)

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        return self.trace(query).results
