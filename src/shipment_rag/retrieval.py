"""Two-stage retrieval: bi-encoder recall from Chroma, cross-encoder precision rerank."""

import logging

from shipment_rag.embeddings import Embedder, Reranker
from shipment_rag.models import RetrievedChunk

log = logging.getLogger(__name__)


class Retriever:
    def __init__(
        self,
        collection,
        embedder: Embedder,
        reranker: Reranker,
        retrieve_k: int = 8,
        final_k: int = 3,
        min_rerank_score: float | None = None,
    ):
        self.collection = collection
        self.embedder = embedder
        self.reranker = reranker
        self.retrieve_k = retrieve_k
        self.final_k = final_k
        self.min_rerank_score = min_rerank_score

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        res = self.collection.query(
            query_embeddings=self.embedder.encode([query]),
            n_results=self.retrieve_k,
            include=["documents", "metadatas", "distances"],
        )
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        if not docs:
            return []

        try:
            scores = self.reranker.score(query, docs)
        except Exception:  # degrade gracefully to embedding order
            log.exception("rerank failed; falling back to embedding similarity")
            scores = [1.0 - d for d in dists]

        ranked = sorted(
            zip(docs, metas, dists, scores, strict=True), key=lambda x: x[3], reverse=True
        )
        results = [
            RetrievedChunk(
                text=doc,
                source=(meta or {}).get("source", "unknown"),
                section=(meta or {}).get("section_name", ""),
                embed_score=round(1.0 - float(dist), 4),
                rerank_score=round(float(score), 4),
            )
            for doc, meta, dist, score in ranked
        ]
        if self.min_rerank_score is not None:
            results = [r for r in results if r.rerank_score >= self.min_rerank_score]
        return results[: self.final_k]
