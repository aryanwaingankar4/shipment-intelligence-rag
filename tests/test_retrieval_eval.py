import json
from dataclasses import dataclass

import pytest

from shipment_rag.evaluation.retrieval_eval import (
    Case,
    CaseResult,
    EvalReport,
    assign_splits,
    choose_threshold,
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
    assert "| hit@3 — all in-scope | 2/3 (67%) |" in comparison_table([rep])


def test_object_format_default_tags_and_dedup(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps([{"question": "Q1", "expect_any": ["x"]}]),
                                     encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps({
        "default_tags": ["paraphrase", "user"],
        "cases": [{"question": "q1 ", "expect_any": ["x"]},  # duplicate of Q1
                  {"question": "Q2", "expect_any": ["y"]}],
    }), encoding="utf-8")
    cases = load_cases(tmp_path / "a.json", tmp_path / "b.json", tmp_path / "missing.json")
    assert [c.question for c in cases] == ["Q1", "Q2"]
    assert cases[1].tags == ["paraphrase", "user"]


def _cases(n, kind="in_scope", tags=()):
    return [Case(f"{kind} {t} {i}", kind, expect_any=["x"], tags=list(tags))
            for i in range(n) for t in [tags[0] if tags else "plain"]]


def test_splits_are_stratified_deterministic_and_respect_explicit_split():
    cases = _cases(10, tags=("id",)) + _cases(6, "off_domain") + _cases(5, "near_domain")
    cases[0].split = "test"
    assign_splits(cases)
    again = assign_splits([Case(c.question, c.kind, tags=c.tags) for c in cases[1:]])
    assert [c.split for c in cases[1:]] == [c.split for c in again]
    assert cases[0].split == "test"
    off = [c.split for c in cases if c.kind == "off_domain"]
    assert off.count("tune") == off.count("test") == 3
    near = [c.split for c in cases if c.kind == "near_domain"]
    assert abs(near.count("tune") - near.count("test")) == 1


def _scored(kind, score, in_candidates=True):
    return CaseResult(Case("q", kind), rank=None, in_candidates=in_candidates,
                      refused=False, top_score=score, top_source="-")


def test_choose_threshold_takes_widest_gap_with_fewest_errors():
    rows = [_scored("in_scope", s) for s in (4.0, -2.0, -9.0)]
    rows += [_scored("in_scope", -12.0, in_candidates=False)]  # unanswerable: ignored
    rows += [_scored("off_domain", s) for s in (-11.0, -11.5)]
    t, info = choose_threshold(EvalReport("hybrid", 3, None, rows))
    assert t == -10.0 and info["tune_errors"] == 0 and info["margin"] == 1.0


def test_choose_threshold_minimises_errors_when_scores_overlap():
    rows = [_scored("in_scope", s) for s in (2.0, -10.0)]
    rows += [_scored("off_domain", s) for s in (-9.0, -12.0)]
    _, info = choose_threshold(EvalReport("hybrid", 3, None, rows))
    assert info["tune_errors"] == 1
