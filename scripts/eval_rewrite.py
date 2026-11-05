"""Does query rewriting help retrieval? Same retriever, questions as written vs rewritten.

    python scripts/eval_rewrite.py --model qwen2.5:7b --report <path>

Part 1, follow-ups (data/eval/followups.json): can the conversation be resolved?
Part 2, every retrieval eval question rewritten with no conversation, which is the paraphrase fix
(D5), and whether it damages anything that already worked.

Uses a local Ollama model and retrieval only, so no Gemini quota. Not part of pytest. Rewrites are
not cached, so the timings are real.
"""

import argparse
import json
import logging
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

from shipment_rag.cli import DEFAULT_EVAL_FILES
from shipment_rag.config import get_settings
from shipment_rag.evaluation.retrieval_eval import (
    _parse,
    comparison_table,
    evaluate,
    load_cases,
)
from shipment_rag.factory import build_retriever, build_structured_router
from shipment_rag.llm import OllamaLLM
from shipment_rag.rewrite import QueryRewriter


class FailureCounter(logging.Handler):
    """Counts rewrites that fell back to the question as written, which would flatter nothing
    and hide a broken rewriter if they went unreported."""

    def __init__(self):
        super().__init__(logging.WARNING)
        self.count = 0

    def emit(self, record):
        self.count += 1


class TimedRewriter:
    def __init__(self, rewriter: QueryRewriter):
        self.rewriter = rewriter
        self.seconds: list[float] = []

    def rewrite(self, question: str, history: list[dict] | None = None) -> str:
        t0 = time.perf_counter()
        out = self.rewriter.rewrite(question, history)
        self.seconds.append(time.perf_counter() - t0)
        return out


class RewritingRetriever:
    """The real retriever behind a rewrite, for evaluate()."""

    def __init__(self, base, rewriter: TimedRewriter, mode: str):
        self.base, self.rewriter, self.mode = base, rewriter, mode
        self.final_k, self.min_rerank_score = base.final_k, base.min_rerank_score
        self.rewrites: dict[str, str] = {}

    def trace(self, question: str):
        self.rewrites[question] = self.rewriter.rewrite(question)
        return self.base.trace(self.rewrites[question])


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _rank(case, trace) -> int | None:
    return next((i for i, r in enumerate(trace.results, 1)
                 if case.is_relevant(r.text, r.source)), None)


def followup_part(raw_cases: list[dict], retriever, rewriter: TimedRewriter, router) -> str:
    rows, n_ok, n_raw, n_new, n_terms, n_table, n_table_want = [], 0, 0, 0, 0, 0, 0
    for raw in raw_cases:
        case = _parse(raw, [])
        rewrite = rewriter.rewrite(raw["question"], raw["history"])
        before, after = retriever.trace(raw["question"]), retriever.trace(rewrite)
        if case.kind == "in_scope":
            was, now = _rank(case, before) is not None, _rank(case, after) is not None
        else:
            was, now = not before.results, not after.results
        low = rewrite.lower()
        terms = (all(t.lower() in low for t in raw.get("rewrite_must_contain", []))
                 and not any(t.lower() in low for t in raw.get("rewrite_must_not_contain", [])))
        table = ""
        if "expect_table_contains" in raw:
            n_table_want += 1
            found = router.answer(rewrite) if router else None
            good = found is not None and raw["expect_table_contains"] in found.answer
            n_table += good
            table = f" table: {'yes' if good else 'no'}"
        n_raw += was
        n_new += now
        n_terms += terms
        n_ok += now and terms
        rows.append(f"| {raw['id']} | {_cell(raw['question'])} | {_cell(rewrite)} | "
                    f"{'yes' if terms else '**no**'} | {'hit' if was else 'miss'} | "
                    f"{'hit' if now else '**miss**'}{table} |")
        print(f"followup {raw['id']} terms={terms} before={was} after={now}{table} "
              f"rewrite={rewrite!r}", flush=True)
    n = len(raw_cases)
    summary = "\n".join([
        "| Follow-up metric | Result |", "|---|---|",
        f"| Right passage in top 3 (or refused, F11), question as written | {n_raw}/{n} |",
        f"| Right passage in top 3 (or refused, F11), rewritten | {n_new}/{n} |",
        f"| Rewrite names what it should and nothing it shouldn't | {n_terms}/{n} |",
        f"| Both of the above | {n_ok}/{n} |",
        f"| Reached the shipment table after rewriting | {n_table}/{n_table_want} |",
    ])
    table = ["| # | Follow-up | Rewritten to | Terms | As written | Rewritten |",
             "|---|---|---|---|---|---|", *rows]
    return summary + "\n\n" + "\n".join(table)


def retrieval_part(files: list[str], retriever, rewriter: TimedRewriter) -> str:
    cases = load_cases(*map(Path, files))
    wrapped = RewritingRetriever(retriever, rewriter, "hybrid + rewrite")
    before = evaluate(retriever, cases)
    after = evaluate(wrapped, cases)
    changes = []
    for b, a in zip(before.rows, after.rows, strict=True):
        if b.passed != a.passed and b.case.kind != "near_domain":
            rewrite = wrapped.rewrites[b.case.question]
            changes.append(f"| {_cell(b.case.question)} | {_cell(rewrite)} "
                           f"| {'pass' if b.passed else 'fail'} | "
                           f"{'pass' if a.passed else '**fail**'} |")
            print(f"changed: {b.case.question[:60]} {b.passed} -> {a.passed}", flush=True)
    change_table = "\n".join(["| Question | Rewritten to | As written | Rewritten |",
                              "|---|---|---|---|", *changes]) if changes else "None."
    return (f"{comparison_table([replace(before, mode='as written'), after])}\n\n"
            f"### Questions whose result changed\n\n{change_table}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--followups", default="data/eval/followups.json")
    parser.add_argument("--file", action="append", help="retrieval eval file, repeatable")
    parser.add_argument("--followups-only", action="store_true")
    parser.add_argument("--report")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    settings = get_settings()
    failures = FailureCounter()
    logging.getLogger("shipment_rag.rewrite").addHandler(failures)
    llm = OllamaLLM(model=args.model, base_url=settings.ollama_base_url, max_output_tokens=128,
                    num_ctx=2048, timeout_s=settings.rewrite_timeout_s)
    llm.check_model()
    rewriter = TimedRewriter(QueryRewriter(llm, max_messages=settings.rewrite_max_messages))
    retriever = build_retriever(settings)
    router = build_structured_router(settings)

    raw_cases = json.loads(Path(args.followups).read_text(encoding="utf-8"))
    part1 = followup_part(raw_cases, retriever, rewriter, router)
    part2 = ("not run (--followups-only)" if args.followups_only
             else retrieval_part(args.file or DEFAULT_EVAL_FILES, retriever, rewriter))
    secs = rewriter.seconds
    report = (
        f"# Query rewriting eval\n\nRewriter `{args.model}` (local via Ollama, digest "
        f"{llm.digest}, temperature 0). Retrieval: {retriever.mode}, top-{retriever.final_k}, "
        f"MIN_RERANK_SCORE={retriever.min_rerank_score}. {len(secs)} rewrites, median "
        f"{statistics.median(secs):.1f} s, slowest {max(secs):.1f} s on this machine; "
        f"{failures.count} fell back to the question as written.\n\n"
        "The follow-up cases and the rewrite prompt were committed before this first run and "
        "nothing was tuned on them. The threshold was chosen earlier on questions as written.\n\n"
        f"## Follow-up questions\n\n{part1}\n\n"
        f"## Every retrieval eval question, rewritten with no conversation\n\n{part2}\n"
    )
    print("\n" + report)
    if args.report:
        Path(args.report).write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
