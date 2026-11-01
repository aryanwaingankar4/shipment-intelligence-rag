"""Generation evaluation: answer quality of the full RAG pipeline, scored against a golden set.

Per question:
- Deterministic (no LLM): key-fact recall, citation validity ([n] points at a passage that
  exists), whether a cited passage comes from an expected source file.
- One LLM-judge call with a fixed rubric and JSON output: refusal detection, correctness
  against the reference answer (0/1/2), unsupported claims (faithfulness), and whether each
  cited passage supports the sentence citing it.
Questions refused at retrieval (no LLM answer) are scored by rule, without a judge call.

Why a custom judge instead of RAGAS: one cached call per answer fits the Gemini free tier
(RAGAS faithfulness makes several calls per answer), no langchain/datasets dependency stack,
and a rubric small enough to read. The judge itself is checked against hand-made flawed
answers (`judge_calibration.json`) before its scores are trusted.
"""

import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from shipment_rag.llm import LLMUnavailableError, QuotaExhaustedError

log = logging.getLogger(__name__)

JUDGE_SYSTEM = (
    "You are a strict evaluator of answers written by a retrieval-augmented assistant for a "
    "pharmaceutical supply-chain company. Judge only against the numbered passages and the "
    "reference answer you are given; never use outside knowledge."
)

EXPECTED = {
    "answerable": "answer the question from the passages",
    "refusal": "refuse: the documents do not contain this information",
    "trap": ("say that the documents do not contain the specific information asked for "
             "(related facts from the passages are fine); inventing it is wrong"),
}

JUDGE_PROMPT = """Question: {question}
Expected behaviour: {expected}
Reference answer: {reference}

Passages the assistant was given:
{passages}

Assistant's answer:
{answer}

Score the answer:
- is_refusal: true if the answer declines or says the information is not available.
- correctness, compared with the reference answer:
  2 = contains the reference's key facts and contradicts nothing (anything the reference
      marks "Optional:" is extra context, not required; never penalise its absence);
  1 = partly correct, or misses an important fact;
  0 = wrong, contradicts the reference, answers when it should refuse, or refuses when it
      should answer. For "refuse" questions a refusal scores 2. For trap questions, 2 needs
      a clear statement that the specific information is not in the documents; inventing
      it scores 0.
- unsupported_claims: every factual claim in the answer that the passages do not support
  (saying that information is unavailable is not a claim). Empty list if none.
- faithful: true if and only if unsupported_claims is empty.
- citations: "all_supported" if every [n] marker points to a passage that supports the
  sentence it is attached to; "some_unsupported" if any marker does not (or points to a
  passage that does not exist); "no_citations" if the answer has no [n] markers.
- reason: one or two sentences explaining the scores."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "is_refusal": {"type": "boolean"},
        "correctness": {"type": "integer", "enum": [0, 1, 2]},
        "unsupported_claims": {"type": "array", "items": {"type": "string"}},
        "faithful": {"type": "boolean"},
        "citations": {"type": "string",
                      "enum": ["all_supported", "some_unsupported", "no_citations"]},
        "reason": {"type": "string"},
    },
    "required": ["is_refusal", "correctness", "unsupported_claims", "faithful",
                 "citations", "reason"],
}

_CITATION = re.compile(r"\[(\d+)\]")


@dataclass
class GoldenItem:
    id: str
    type: str  # answerable | refusal | trap
    question: str
    reference_answer: str
    key_facts: list[list[str]] = field(default_factory=list)  # each fact: alternatives
    expected_sources: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


@dataclass
class Verdict:
    is_refusal: bool
    correctness: int
    unsupported_claims: list[str]
    faithful: bool
    citations: str
    reason: str
    by_rule: bool = False  # True when scored without a judge call


@dataclass
class GenResult:
    item: GoldenItem
    answer: str = ""
    sources: list[str] = field(default_factory=list)  # source file of passage [1], [2], ...
    retrieval_refused: bool = False
    cited: list[int] = field(default_factory=list)
    key_fact_hits: int = 0
    verdict: Verdict | None = None
    error: str = ""

    @property
    def citations_valid(self) -> bool:
        return all(1 <= n <= len(self.sources) for n in self.cited)

    @property
    def cited_expected_source(self) -> bool:
        cited = {self.sources[n - 1] for n in self.cited if 1 <= n <= len(self.sources)}
        return bool(cited & set(self.item.expected_sources))


def load_golden(path: Path) -> list[GoldenItem]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [GoldenItem(**{k: v for k, v in r.items() if k in GoldenItem.__dataclass_fields__})
            for r in raw]


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).lower()
    t = t.replace("−", "-").replace("–", "-")
    return re.sub(r"\s+°", "°", t)


def key_fact_hits(answer: str, facts: list[list[str]]) -> int:
    a = normalize(answer)
    return sum(any(normalize(alt) in a for alt in fact) for fact in facts)


def citations_in(answer: str) -> list[int]:
    return [int(n) for n in _CITATION.findall(answer)]


def format_passages(passages: list[tuple[str, str]]) -> str:
    if not passages:
        return "(none: retrieval found nothing relevant)"
    return "\n\n".join(f"[{i}] (source: {src})\n{text}" for i, (src, text) in
                       enumerate(passages, 1))


def judge(judge_llm, question, item_type, reference, passages, answer) -> Verdict:
    prompt = JUDGE_PROMPT.format(
        question=question, expected=EXPECTED[item_type], reference=reference,
        passages=format_passages(passages), answer=answer,
    )
    raw = judge_llm.generate(JUDGE_SYSTEM, prompt, json_schema=JUDGE_SCHEMA)
    data = json.loads(raw)
    return Verdict(**{k: data[k] for k in JUDGE_SCHEMA["required"]})


def rule_verdict(item: GoldenItem) -> Verdict:
    """Refused at retrieval: nothing was generated, so score it without a judge."""
    score = {"refusal": 2, "trap": 1, "answerable": 0}[item.type]
    return Verdict(True, score, [], True, "no_citations",
                   "refused at retrieval (no passage cleared the reranker threshold)",
                   by_rule=True)


def run(pipeline, judge_llm, items: list[GoldenItem], max_consecutive_errors: int = 3
        ) -> list[GenResult]:
    results, consecutive, stop_reason = [], 0, ""
    for item in items:
        res = GenResult(item)
        results.append(res)
        if stop_reason:
            res.error = f"not run ({stop_reason}; re-run resumes from cache)"
            continue
        try:
            out = pipeline.answer(item.question)
            res.answer = out.answer
            res.sources = [c.source for c in out.context]  # [n] refers to context[n-1]
            res.cited = citations_in(out.answer)
            res.key_fact_hits = key_fact_hits(out.answer, item.key_facts)
            if not out.llm_called:
                res.retrieval_refused = True
                res.verdict = rule_verdict(item)
            else:
                passages = [(c.source, c.text) for c in out.context]
                res.verdict = judge(judge_llm, item.question, item.type,
                                    item.reference_answer, passages, out.answer)
            consecutive = 0
        except QuotaExhaustedError as exc:
            # Keep going: cached answers/judgments are still served and retrieval refusals
            # need no LLM; uncached calls fail instantly without an API request.
            res.error = f"not run: {exc} (re-run resumes from cache)"[:200]
        except (LLMUnavailableError, json.JSONDecodeError, KeyError) as exc:
            res.error = f"{type(exc).__name__}: {exc}"[:200]
            consecutive += 1
            log.warning("generation eval item failed", extra={"id": item.id, "error": res.error})
            if consecutive >= max_consecutive_errors:
                stop_reason = "stopped after repeated LLM errors"
    return results


def calibration_passed(cal: list[dict]) -> bool:
    """Every case scored and every planted flaw caught."""
    return bool(cal) and all(
        not c.get("error") and c["checks"] and all(ok for _, ok in c["checks"]) for c in cal
    )


def run_calibration(judge_llm, cases: list[dict]) -> list[dict]:
    """Score hand-made answers with known flaws; report where the judge agrees."""
    out, stopped = [], False
    for c in cases:
        if stopped:
            out.append({"id": c["id"], "flaw": c["flaw"], "checks": [],
                        "error": "not run (daily quota exhausted)"})
            continue
        passages = [(p["source"], p["text"]) for p in c["passages"]]
        item_type = "refusal" if c["expect"].get("is_refusal") else (
            "trap" if "trap" in c["flaw"] else "answerable")
        try:
            v = judge(judge_llm, c["question"], item_type, c["reference_answer"],
                      passages, c["answer"])
        except (LLMUnavailableError, json.JSONDecodeError, KeyError) as exc:
            out.append({"id": c["id"], "flaw": c["flaw"], "checks": [], "error": str(exc)[:200]})
            stopped = isinstance(exc, QuotaExhaustedError)
            continue
        checks = []
        e = c["expect"]
        if "correctness_max" in e:
            want = e["correctness_max"]
            checks.append((f"correctness ≤ {want}", v.correctness <= want))
        if "correctness_min" in e:
            want = e["correctness_min"]
            checks.append((f"correctness ≥ {want}", v.correctness >= want))
        if "faithful" in e:
            checks.append((f"faithful = {e['faithful']}", v.faithful == e["faithful"]))
        if "citations_support" in e:
            want = "all_supported" if e["citations_support"] else "some_unsupported"
            checks.append((f"citations = {want}", v.citations == want))
        if "is_refusal" in e:
            checks.append((f"is_refusal = {e['is_refusal']}", v.is_refusal == e["is_refusal"]))
        out.append({"id": c["id"], "flaw": c["flaw"], "checks": checks, "verdict": v})
    return out


# ---------------- aggregation + report ----------------

def _f(h: int, n: int) -> str:
    return "n/a" if not n else f"{h}/{n} ({h / n:.0%})"


def summarize(results: list[GenResult]) -> dict:
    done = [r for r in results if r.verdict]
    ans = [r for r in done if r.item.type == "answerable"]
    ref = [r for r in done if r.item.type == "refusal"]
    trap = [r for r in done if r.item.type == "trap"]
    generated = [r for r in done if not r.verdict.is_refusal]
    answered_ans = [r for r in ans if not r.verdict.is_refusal]
    with_cites = [r for r in answered_ans if r.cited]
    facts_total = sum(len(r.item.key_facts) for r in ans)
    return {
        "n_total": len(results), "n_scored": len(done),
        "n_errors": sum(bool(r.error) for r in results),
        "answer_correct": (sum(r.verdict.correctness == 2 for r in ans), len(ans)),
        "answer_partial": (sum(r.verdict.correctness == 1 for r in ans), len(ans)),
        "mean_correctness": (sum(r.verdict.correctness for r in ans) / (2 * len(ans))
                             if ans else float("nan")),
        "key_facts": (sum(r.key_fact_hits for r in ans), facts_total),
        "wrongly_refused": (sum(r.verdict.is_refusal for r in ans), len(ans)),
        "faithful": (sum(r.verdict.faithful for r in generated), len(generated)),
        "has_citation": (len(with_cites), len(answered_ans)),
        "citations_valid": (sum(r.citations_valid for r in with_cites), len(with_cites)),
        "cited_expected_source": (sum(r.cited_expected_source for r in with_cites),
                                  len(with_cites)),
        "citations_supported": (sum(r.verdict.citations == "all_supported" for r in with_cites),
                                len(with_cites)),
        "refused_correctly": (sum(r.verdict.is_refusal for r in ref), len(ref)),
        "refused_at_retrieval": (sum(r.retrieval_refused for r in ref), len(ref)),
        "trap_handled": (sum(r.verdict.correctness == 2 for r in trap), len(trap)),
        "trap_invented": (sum(r.verdict.correctness == 0 for r in trap), len(trap)),
    }


def summary_table(s: dict) -> str:
    rows = [
        ("**Answerable questions**", ""),
        ("fully correct (judge: 2/2)", _f(*s["answer_correct"])),
        ("partly correct (1/2)", _f(*s["answer_partial"])),
        ("mean correctness", f"{s['mean_correctness']:.0%}"),
        ("key facts present in the answer (string match)", _f(*s["key_facts"])),
        ("wrongly refused", _f(*s["wrongly_refused"])),
        ("**Faithfulness** (no unsupported claims, all generated answers)", _f(*s["faithful"])),
        ("**Citations** (answered answerable questions)", ""),
        ("answer has ≥1 [n] citation", _f(*s["has_citation"])),
        ("all citation numbers valid", _f(*s["citations_valid"])),
        ("cites a passage from an expected source file", _f(*s["cited_expected_source"])),
        ("judge: every cited passage supports its sentence", _f(*s["citations_supported"])),
        ("**Should refuse** (unanswerable / off-domain)", ""),
        ("refused", _f(*s["refused_correctly"])),
        ("of which refused at retrieval (no LLM answer call)", _f(*s["refused_at_retrieval"])),
        ("**Traps** (documents lack the specific figure)", ""),
        ("handled: says the information isn't in the documents", _f(*s["trap_handled"])),
        ("invented the figure", _f(*s["trap_invented"])),
    ]
    return "\n".join(["| Metric | Result |", "|---|---|"] + [f"| {a} | {b} |" for a, b in rows])


def _cell(text: str, n: int = 140) -> str:
    t = " ".join(text.split()).replace("|", "\\|")
    return t if len(t) <= n else t[: n - 1] + "…"


def per_question_table(results: list[GenResult]) -> str:
    lines = ["| ID | type | question | correct | faithful | citations | key facts | note |",
             "|---|---|---|---|---|---|---|---|"]
    for r in results:
        if not r.verdict:
            lines.append(f"| {r.item.id} | {r.item.type} | {_cell(r.item.question, 60)} "
                         f"| – | – | – | – | {_cell(r.error, 80)} |")
            continue
        v = r.verdict
        cites = v.citations if not r.cited or r.citations_valid else "invalid index"
        note = "refused at retrieval" if v.by_rule else ("refused" if v.is_refusal else "")
        if v.unsupported_claims:
            note = "unsupported: " + "; ".join(v.unsupported_claims)
        facts = f"{r.key_fact_hits}/{len(r.item.key_facts)}" if r.item.key_facts else "–"
        lines.append(f"| {r.item.id} | {r.item.type} | {_cell(r.item.question, 60)} "
                     f"| {v.correctness}/2 | {'yes' if v.faithful else '**no**'} | {cites} "
                     f"| {facts} | {_cell(note, 90)} |")
    return "\n".join(lines)


def failures_section(results: list[GenResult]) -> str:
    bad = [r for r in results if r.verdict and (r.verdict.correctness < 2 or not r.verdict.faithful
                                                or r.verdict.citations == "some_unsupported")]
    if not bad:
        return "None."
    out = []
    for r in bad:
        out.append(f"**{r.item.id}: {r.item.question}**\n\n"
                   f"- Answer: {_cell(r.answer, 400)}\n"
                   f"- Reference: {_cell(r.item.reference_answer, 300)}\n"
                   f"- Judge: correctness {r.verdict.correctness}/2, "
                   f"faithful {r.verdict.faithful}, citations {r.verdict.citations}. "
                   f"{_cell(r.verdict.reason, 300)}\n")
    return "\n".join(out)


def calibration_table(cal: list[dict]) -> str:
    lines = ["| Case | Planted flaw | Judge checks |", "|---|---|---|"]
    for c in cal:
        if c.get("error"):
            lines.append(f"| {c['id']} | {c['flaw']} | error: {_cell(c['error'], 80)} |")
            continue
        checks = ", ".join(f"{name} {'✅' if ok else '❌'}" for name, ok in c["checks"])
        lines.append(f"| {c['id']} | {c['flaw']} | {checks} |")
    passed = sum(ok for c in cal for _, ok in c["checks"])
    total = sum(len(c["checks"]) for c in cal)
    scored = sum(not c.get("error") for c in cal)
    return "\n".join(lines) + (f"\n\nJudge agreement on planted flaws: **{passed}/{total}** "
                               f"checks ({scored}/{len(cal)} cases scored).")
