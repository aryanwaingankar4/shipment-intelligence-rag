"""Retrieval evaluation (no LLM calls).

Case types in the JSON eval file:
  in-scope      {"question", "expect_any": [...]} or {"expect_all": [...]}, optional "tags"
                relevant chunk = contains any (or all) of the keywords, case-insensitive
  off-domain    {"question", "expect_refusal": true}
                nothing should clear the rerank threshold (refused before the LLM)
  near-domain   {"question", "expect_refusal": true, "kind": "near_domain"}
                on-topic but unanswerable (unknown ID, missing fact). Retrieval *will*
                often find related chunks; the LLM must refuse. Reported, not gated.

Metrics (in-scope): hit@k, MRR@k over the final top-k, candidate recall (a relevant chunk
reached the reranker at all), false refusals. Gate score = (in-scope hits + off-domain
refusals) / (in-scope + off-domain cases).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Case:
    question: str
    kind: str  # in_scope | off_domain | near_domain
    expect_any: list[str] = field(default_factory=list)
    expect_all: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def is_relevant(self, text: str) -> bool:
        t = text.lower()
        if self.expect_all:
            return all(k.lower() in t for k in self.expect_all)
        return any(k.lower() in t for k in self.expect_any)


@dataclass
class CaseResult:
    case: Case
    rank: int | None  # 1-based rank of first relevant final chunk
    in_candidates: bool
    refused: bool
    top_score: float | None
    top_source: str

    @property
    def passed(self) -> bool:
        if self.case.kind == "in_scope":
            return self.rank is not None
        return self.refused


@dataclass
class EvalReport:
    mode: str
    k: int
    threshold: float | None
    rows: list[CaseResult]

    def _rows(self, kind: str, tag: str | None = None) -> list[CaseResult]:
        return [r for r in self.rows if r.case.kind == kind and (tag is None or tag in r.case.tags)]

    def hit_at_k(self, tag: str | None = None) -> float:
        rows = self._rows("in_scope", tag)
        return sum(r.rank is not None for r in rows) / len(rows) if rows else float("nan")

    def mrr(self, tag: str | None = None) -> float:
        rows = self._rows("in_scope", tag)
        return sum(1 / r.rank for r in rows if r.rank) / len(rows) if rows else float("nan")

    def candidate_recall(self) -> float:
        rows = self._rows("in_scope")
        return sum(r.in_candidates for r in rows) / len(rows) if rows else float("nan")

    def false_refusals(self) -> int:
        return sum(r.refused for r in self._rows("in_scope"))

    def refusal_rate(self, kind: str) -> float:
        rows = self._rows(kind)
        return sum(r.refused for r in rows) / len(rows) if rows else float("nan")

    def gate_score(self) -> float:
        gated = self._rows("in_scope") + self._rows("off_domain")
        return sum(r.passed for r in gated) / len(gated) if gated else float("nan")

    def margin(self) -> dict[str, float | None]:
        def _scores(rows):
            return [r.top_score for r in rows if r.top_score is not None]

        hits = _scores([r for r in self._rows("in_scope") if r.in_candidates])
        off = _scores(self._rows("off_domain"))
        near = _scores(self._rows("near_domain"))
        return {
            "min_in_scope_top": min(hits) if hits else None,
            "max_off_domain_top": max(off) if off else None,
            "max_near_domain_top": max(near) if near else None,
        }


def load_cases(path: Path) -> list[Case]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = []
    for c in raw:
        kind = c.get("kind", "off_domain") if c.get("expect_refusal") else "in_scope"
        cases.append(
            Case(
                question=c["question"],
                kind=kind,
                expect_any=c.get("expect_any", []),
                expect_all=c.get("expect_all", []),
                tags=c.get("tags", []),
            )
        )
    return cases


def evaluate(retriever, cases: list[Case]) -> EvalReport:
    rows = []
    for case in cases:
        tr = retriever.trace(case.question)
        rank = next(
            (i for i, r in enumerate(tr.results, 1) if case.is_relevant(r.text)), None
        )
        rows.append(
            CaseResult(
                case=case,
                rank=rank if case.kind == "in_scope" else None,
                in_candidates=any(case.is_relevant(c.text) for c in tr.candidates),
                refused=not tr.results,
                top_score=tr.top_score,
                top_source=tr.results[0].source if tr.results else "-",
            )
        )
    return EvalReport(retriever.mode, retriever.final_k, retriever.min_rerank_score, rows)


def _pct(x: float) -> str:
    return "n/a" if x != x else f"{x:.0%}"


def _num(x: float | None) -> str:
    return "n/a" if x is None else f"{x:+.2f}"


def comparison_table(reports: list[EvalReport]) -> str:
    k = reports[0].k
    n_in = len(reports[0]._rows("in_scope"))
    n_id = len(reports[0]._rows("in_scope", "id"))
    n_para = len(reports[0]._rows("in_scope", "paraphrase"))
    n_off = len(reports[0]._rows("off_domain"))
    n_near = len(reports[0]._rows("near_domain"))
    metrics = [
        (f"hit@{k} — all in-scope (n={n_in})", lambda r: _pct(r.hit_at_k())),
        (f"hit@{k} — ID-heavy (n={n_id})", lambda r: _pct(r.hit_at_k("id"))),
        (f"hit@{k} — paraphrased (n={n_para})", lambda r: _pct(r.hit_at_k("paraphrase"))),
        (f"MRR@{k} — all in-scope", lambda r: f"{r.mrr():.3f}"),
        (f"MRR@{k} — ID-heavy", lambda r: f"{r.mrr('id'):.3f}"),
        (f"MRR@{k} — paraphrased", lambda r: f"{r.mrr('paraphrase'):.3f}"),
        ("candidate recall (relevant chunk reached reranker)",
         lambda r: _pct(r.candidate_recall())),
        ("false refusals (in-scope refused)", lambda r: str(r.false_refusals())),
        (f"off-domain refused (n={n_off})", lambda r: _pct(r.refusal_rate("off_domain"))),
        (f"near-domain refused at retrieval (n={n_near}, informational)",
         lambda r: _pct(r.refusal_rate("near_domain"))),
        ("min top score, answerable in-scope", lambda r: _num(r.margin()["min_in_scope_top"])),
        ("max top score, off-domain", lambda r: _num(r.margin()["max_off_domain_top"])),
        ("**gate score**", lambda r: f"**{_pct(r.gate_score())}**"),
    ]
    head = "| Metric | " + " | ".join(r.mode for r in reports) + " |"
    sep = "|---|" + "---|" * len(reports)
    body = [f"| {name} | " + " | ".join(fn(r) for r in reports) + " |" for name, fn in metrics]
    return "\n".join([head, sep, *body])


def case_table(reports: list[EvalReport]) -> str:
    head = "| # | Question | kind | " + " | ".join(f"{r.mode} rank / top" for r in reports) + " |"
    sep = "|---|---|---|" + "---|" * len(reports)
    lines = [head, sep]
    for i, row0 in enumerate(reports[0].rows, 1):
        cells = []
        for rep in reports:
            r = rep.rows[i - 1]
            if r.case.kind == "in_scope":
                status = f"{r.rank}" if r.rank else ("refused" if r.refused else "miss")
            else:
                status = "refused" if r.refused else "answered"
            cells.append(f"{status} / {_num(r.top_score)}")
        q = row0.case.question.replace("|", "\\|")
        lines.append(f"| {i} | {q} | {row0.case.kind} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
