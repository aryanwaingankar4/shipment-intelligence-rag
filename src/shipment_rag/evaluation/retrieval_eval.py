"""Retrieval evaluation (no LLM calls).

Eval files are JSON: either a list of cases, or an object
{"default_tags": [...], "cases": [...]} whose tags are added to every case.

Case types:
  in-scope      {"question", "expect_any": [...]} or {"expect_all": [...]}, optional "tags"
                relevant chunk = contains any (or all) of the keywords, case-insensitive
  off-domain    {"question", "expect_refusal": true}
                nothing should clear the rerank threshold (refused before the LLM)
  near-domain   {"question", "expect_refusal": true, "kind": "near_domain"}
                on-topic but unanswerable (unknown ID, missing fact). Retrieval *will*
                often find related chunks; the LLM must refuse. Reported, not gated.

Splits: every case is deterministically assigned to "tune" or "test" (stratified by kind
and slice, ordered by a hash of the question), unless it sets "split" itself. Thresholds
are chosen on "tune" and reported on "test" so they aren't tuned on their own test set.

Metrics (in-scope): hit@k, MRR@k over the final top-k, candidate recall (a relevant chunk
reached the reranker at all), false refusals. Gate score = (in-scope hits + off-domain
refusals) / (in-scope + off-domain cases).
"""

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

SLICES = ("id", "paraphrase")  # tags reported separately


@dataclass
class Case:
    question: str
    kind: str  # in_scope | off_domain | near_domain
    expect_any: list[str] = field(default_factory=list)
    expect_all: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    split: str = ""  # tune | test

    def is_relevant(self, text: str) -> bool:
        t = text.lower()
        if self.expect_all:
            return all(k.lower() in t for k in self.expect_all)
        return any(k.lower() in t for k in self.expect_any)

    @property
    def stratum(self) -> str:
        return self.kind + ":" + next((s for s in SLICES if s in self.tags), "other")


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

    def hits(self, tag: str | None = None) -> tuple[int, int]:
        rows = self._rows("in_scope", tag)
        return sum(r.rank is not None for r in rows), len(rows)

    def hit_at_k(self, tag: str | None = None) -> float:
        h, n = self.hits(tag)
        return h / n if n else float("nan")

    def mrr(self, tag: str | None = None) -> float:
        rows = self._rows("in_scope", tag)
        return sum(1 / r.rank for r in rows if r.rank) / len(rows) if rows else float("nan")

    def candidate_hits(self) -> tuple[int, int]:
        rows = self._rows("in_scope")
        return sum(r.in_candidates for r in rows), len(rows)

    def candidate_recall(self) -> float:
        h, n = self.candidate_hits()
        return h / n if n else float("nan")

    def false_refusals(self) -> int:
        return sum(r.refused for r in self._rows("in_scope"))

    def refusals(self, kind: str) -> tuple[int, int]:
        rows = self._rows(kind)
        return sum(r.refused for r in rows), len(rows)

    def refusal_rate(self, kind: str) -> float:
        h, n = self.refusals(kind)
        return h / n if n else float("nan")

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


def _stable_hash(text: str) -> str:
    return hashlib.sha1(text.strip().lower().encode("utf-8")).hexdigest()


def assign_splits(cases: list[Case]) -> list[Case]:
    """Stratified, deterministic tune/test split: within each (kind, slice) stratum, order
    by question hash and alternate tune, test, tune, ... Explicit "split" values win."""
    groups: dict[str, list[Case]] = defaultdict(list)
    for c in cases:
        if not c.split:
            groups[c.stratum].append(c)
    for group in groups.values():
        for i, c in enumerate(sorted(group, key=lambda c: _stable_hash(c.question))):
            c.split = "tune" if i % 2 == 0 else "test"
    return cases


def _parse(raw: dict, default_tags: list[str]) -> Case:
    kind = raw.get("kind", "off_domain") if raw.get("expect_refusal") else "in_scope"
    return Case(
        question=raw["question"],
        kind=kind,
        expect_any=raw.get("expect_any", []),
        expect_all=raw.get("expect_all", []),
        tags=sorted(set(raw.get("tags", [])) | set(default_tags)),
        split=raw.get("split", ""),
    )


def load_cases(*paths: Path, split: str = "all") -> list[Case]:
    cases = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        items, tags = (data, []) if isinstance(data, list) else (
            data.get("cases", []), data.get("default_tags", []))
        cases += [_parse(c, tags) for c in items]
    seen, unique = set(), []
    for c in cases:  # the same question in two files counts once
        key = _stable_hash(c.question)
        if key not in seen:
            seen.add(key)
            unique.append(c)
    assign_splits(unique)
    return unique if split == "all" else [c for c in unique if c.split == split]


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


def choose_threshold(report: EvalReport) -> tuple[float, dict]:
    """Pick MIN_RERANK_SCORE from an *unthresholded* report (normally the tune split).

    Errors at threshold t: answerable in-scope questions (relevant chunk reached the
    reranker) whose top score < t, plus off-domain questions whose top score >= t.
    Candidates are the midpoints between consecutive observed scores; pick the one with
    the fewest errors, breaking ties by the widest margin to the nearest observed score.
    """
    keep = [r.top_score for r in report.rows
            if r.case.kind == "in_scope" and r.in_candidates and r.top_score is not None]
    drop = [r.top_score for r in report.rows
            if r.case.kind == "off_domain" and r.top_score is not None]
    if not keep or not drop:
        raise ValueError("need answerable in-scope and off-domain cases to choose a threshold")
    scores = sorted(set(keep + drop))
    candidates = [(a + b) / 2 for a, b in zip(scores, scores[1:], strict=False)]
    candidates += [scores[0] - 1.0, scores[-1] + 1.0]

    def errors(t):
        return sum(s < t for s in keep) + sum(s >= t for s in drop)

    def margin(t):
        return min(abs(t - s) for s in scores)

    best = min(candidates, key=lambda t: (errors(t), -margin(t)))
    info = {
        "threshold": round(best, 2),
        "tune_errors": errors(best),
        "margin": round(margin(best), 2),
        "weakest_answerable": min(keep),
        "strongest_off_domain": max(drop),
        "n_answerable": len(keep),
        "n_off_domain": len(drop),
    }
    return round(best, 2), info


def _pct(x: float) -> str:
    return "n/a" if x != x else f"{x:.0%}"


def _frac(h: int, n: int) -> str:
    return "n/a" if not n else f"{h}/{n} ({h / n:.0%})"


def _num(x: float | None) -> str:
    return "n/a" if x is None else f"{x:+.2f}"


def comparison_table(reports: list[EvalReport]) -> str:
    k = reports[0].k
    metrics = [
        (f"hit@{k} — all in-scope", lambda r: _frac(*r.hits())),
        (f"hit@{k} — ID-heavy", lambda r: _frac(*r.hits("id"))),
        (f"hit@{k} — paraphrased", lambda r: _frac(*r.hits("paraphrase"))),
        (f"hit@{k} — owner-written paraphrases", lambda r: _frac(*r.hits("user"))),
        (f"MRR@{k} — all in-scope", lambda r: f"{r.mrr():.3f}"),
        (f"MRR@{k} — ID-heavy", lambda r: f"{r.mrr('id'):.3f}"),
        (f"MRR@{k} — paraphrased", lambda r: f"{r.mrr('paraphrase'):.3f}"),
        ("candidate recall (relevant chunk reached reranker)",
         lambda r: _frac(*r.candidate_hits())),
        ("false refusals (in-scope refused)", lambda r: str(r.false_refusals())),
        ("off-domain refused", lambda r: _frac(*r.refusals("off_domain"))),
        ("near-domain refused at retrieval (informational)",
         lambda r: _frac(*r.refusals("near_domain"))),
        ("min top score, answerable in-scope", lambda r: _num(r.margin()["min_in_scope_top"])),
        ("max top score, off-domain", lambda r: _num(r.margin()["max_off_domain_top"])),
        ("**gate score**", lambda r: f"**{_pct(r.gate_score())}**"),
    ]
    head = "| Metric | " + " | ".join(r.mode for r in reports) + " |"
    sep = "|---|" + "---|" * len(reports)
    body = [f"| {name} | " + " | ".join(fn(r) for r in reports) + " |" for name, fn in metrics]
    return "\n".join([head, sep, *body])


def case_table(reports: list[EvalReport]) -> str:
    head = ("| # | Question | kind | split | "
            + " | ".join(f"{r.mode} rank / top" for r in reports) + " |")
    sep = "|---|---|---|---|" + "---|" * len(reports)
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
        lines.append(f"| {i} | {q} | {row0.case.kind} | {row0.case.split} | "
                     + " | ".join(cells) + " |")
    return "\n".join(lines)
