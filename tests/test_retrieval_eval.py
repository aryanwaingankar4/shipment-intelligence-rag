import json
from dataclasses import dataclass

import pytest

from shipment_rag.evaluation.retrieval_eval import (
    Case,
    comparison_table,
    evaluate,
    load_cases,
)
from shipment_rag.models import RetrievedChunk
from shipment_rag.retrieval import Candidate, RetrievalTrace


def chunk(text, score=5.0):
    return RetrievedChunk(text=text, source="s.txt", section="", embed_score=0.5,
                          rerank_score=score)


@dataclass
class ScriptedRetriever:
    """Returns pre-baked traces per question."""

    traces: dict
    mode: str = "hybrid"
    final_k: int = 3
    min_rerank_score: float = -6.0

    def trace(self, q):
        return self.traces[q]


def test_load_cases_kinds(tmp_path):
    f = tmp_path / "e.json"
    f.write_text(json.dumps([
        {"question": "a", "expect_any": ["x"], "tags": ["id"]},
        {"question": "b", "expect_refusal": True},
        {"question": "c", "expect_refusal": True, "kind": "near_domain"},
    ]), encoding="utf-8")
    kinds = [c.kind for c in load_cases(f)]
    assert kinds == ["in_scope", "off_domain", "near_domain"]


def test_expect_all_requires_every_keyword():
    c = Case("q", "in_scope", expect_all=["sh-2024-010", "anita rao"])
    assert not c.is_relevant("SH-2024-004 accepted by Dr. Anita Rao")
    assert c.is_relevant("SH-2024-010 accepted by Dr. Anita Rao")


def test_metrics():
    cases = [
        Case("hit1", "in_scope", expect_any=["gold"], tags=["id"]),
        Case("hit2", "in_scope", expect_any=["gold"]),
        Case("miss", "in_scope", expect_any=["gold"]),
        Case("off", "off_domain"),
        Case("near", "near_domain"),
    ]
    gold, junk = Candidate("g", "gold", {}), Candidate("j", "junk", {})
    traces = {
        "hit1": RetrievalTrace([gold], [chunk("gold")], 5.0),
        "hit2": RetrievalTrace([junk, gold], [chunk("junk"), chunk("gold")], 3.0),
        "miss": RetrievalTrace([junk], [], -7.0),  # refused: below threshold
        "off": RetrievalTrace([junk], [], -11.0),
        "near": RetrievalTrace([junk], [chunk("junk", -2.0)], -2.0),
    }
    rep = evaluate(ScriptedRetriever(traces), cases)
    assert rep.hit_at_k() == pytest.approx(2 / 3)
    assert rep.hit_at_k("id") == 1.0
    assert rep.mrr() == pytest.approx((1 + 0.5 + 0) / 3)
    assert rep.candidate_recall() == pytest.approx(2 / 3)
    assert rep.false_refusals() == 1
    assert rep.refusal_rate("off_domain") == 1.0
    assert rep.refusal_rate("near_domain") == 0.0
    assert rep.gate_score() == pytest.approx(3 / 4)  # near-domain not gated
    assert rep.margin()["max_off_domain_top"] == -11.0
    assert "| hit@3 — all in-scope (n=3) | 67% |" in comparison_table([rep])
