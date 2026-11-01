"""Hybrid retrieval with fake vector store / models (no torch, no network)."""

import pytest

from shipment_rag.retrieval import BM25Index, Retriever, reciprocal_rank_fusion, tokenize

DOCS = {
    "a": "=== SHIPMENT SH-2024-003 ===\nBatch: BATCH-CRY-2401 | Max temp 11.2°C | QUARANTINED",
    "b": "=== SHIPMENT SH-2024-006 ===\nBatch: BATCH-CRY-2405 | Max temp 7.4°C | DELIVERED",
    "c": "=== SECTION 2 — TEMPERATURE MONITORING ===\nCold chain above 10°C: quarantine.",
    "d": "=== VENDOR V-003 — ArcticLine ===\nContact person: Rohan Mehta",
}


class FakeCollection:
    """Vector search deliberately returns a fixed, ID-blind order (like a bi-encoder that
    sees all shipment records as near-identical)."""

    def __init__(self, vector_order):
        self.vector_order = vector_order

    def count(self):
        return len(DOCS)

    def get(self, include):
        ids = list(DOCS)
        return {"ids": ids, "documents": [DOCS[i] for i in ids],
                "metadatas": [{"source": f"{i}.txt", "section_name": i} for i in ids]}

    def query(self, query_embeddings, n_results, include):
        ids = self.vector_order[:n_results]
        return {"ids": [ids], "documents": [[DOCS[i] for i in ids]],
                "metadatas": [[{"source": f"{i}.txt", "section_name": i} for i in ids]],
                "distances": [[0.2 + 0.1 * n for n in range(len(ids))]]}


class FakeEmbedder:
    def encode(self, texts):
        return [[0.0] * 4 for _ in texts]


class OverlapReranker:
    """Scores by shared tokens, so an exact ID match wins; off-topic gets -11."""

    def __init__(self, fail=False):
        self.fail = fail

    def score(self, query, texts):
        if self.fail:
            raise RuntimeError("reranker down")
        q = set(tokenize(query))
        return [float(len(q & set(tokenize(t)))) or -11.0 for t in texts]


def retriever(mode, vector_order=("c", "b", "a", "d"), **kw):
    defaults = dict(retrieve_k=2, final_k=2, min_rerank_score=-6.0, bm25_k=2, rerank_k=4)
    defaults.update(kw)
    return Retriever(FakeCollection(list(vector_order)), FakeEmbedder(),
                     kw.pop("reranker", None) or OverlapReranker(), mode=mode,
                     **{k: v for k, v in defaults.items() if k != "reranker"})


def test_tokenize_keeps_ids_whole_and_split():
    toks = tokenize("Which batch was in SH-2024-006? Contact quality@nexachain-biomed.com")
    assert "sh-2024-006" in toks and {"sh", "2024", "006"} <= set(toks)
    assert "quality@nexachain-biomed.com" in toks and "nexachain" in toks
    assert "which" not in toks and "was" not in toks  # stopwords


def test_rrf_rewards_agreement_and_rank():
    fused = dict(reciprocal_rank_fusion([["x", "y", "z"], ["y", "w"]], k=60))
    assert fused["y"] > fused["x"] > fused["z"]  # y is in both lists
    assert fused["x"] == pytest.approx(1 / 61)


def test_bm25_finds_exact_id():
    idx = BM25Index(list(DOCS), list(DOCS.values()), [{}] * len(DOCS))
    assert idx.search("SH-2024-006 batch", k=1)[0][0] == "b"
    assert idx.search("the of and", k=3) == []  # stopwords only -> no lexical signal


def test_vector_mode_misses_id_that_hybrid_recovers():
    q = "Which batch was shipped in SH-2024-006?"
    # vector's top-2 is c, b... make it miss b entirely
    vec = retriever("vector", vector_order=("c", "a", "b", "d"))
    assert "BATCH-CRY-2405" not in " ".join(r.text for r in vec.retrieve(q))

    hyb = retriever("hybrid", vector_order=("c", "a", "b", "d"))
    tr = hyb.trace(q)
    assert any(c.id == "b" and "bm25" in c.found_by for c in tr.candidates)
    assert "BATCH-CRY-2405" in tr.results[0].text
    assert tr.results[0].embed_score is None  # found by BM25 only


def test_hybrid_merges_duplicates_from_both_retrievers():
    tr = retriever("hybrid", vector_order=("b", "a", "c", "d")).trace("SH-2024-006 batch")
    ids = [c.id for c in tr.candidates]
    assert len(ids) == len(set(ids))
    assert next(c for c in tr.candidates if c.id == "b").found_by == {"vector", "bm25"}


def test_threshold_refuses_off_topic_and_reports_top_score():
    tr = retriever("hybrid").trace("capital of France")
    assert tr.results == [] and tr.top_score == -11.0


def test_reranker_failure_falls_back_without_threshold():
    r = retriever("hybrid", reranker=OverlapReranker(fail=True))
    assert len(r.retrieve("SH-2024-006 batch")) == 2


def test_bm25_index_rebuilds_when_collection_grows():
    r = retriever("bm25")
    r.retrieve("SH-2024-006")
    first = r._bm25
    r.collection.count = lambda: len(DOCS) + 1  # simulate a re-ingest
    r._bm25_index()
    assert r._bm25 is not first
