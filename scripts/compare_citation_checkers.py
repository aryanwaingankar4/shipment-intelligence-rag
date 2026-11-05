"""Compare citation-support checkers on the hand-labelled calibration cases.

Every cited sentence in data/eval/judge_calibration.json has a known answer, so this reports
accuracy, wrong citations caught, false alarms and speed for each checker.

    python scripts/compare_citation_checkers.py --llm qwen2.5:7b \
        --nli cross-encoder/nli-deberta-v3-small --report <path>

Calls are not cached, so the timings are real. Needs Ollama running unless --holdout is given;
uses no Gemini quota.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from shipment_rag.evaluation.citation_check import (
    LLMSupportChecker,
    NLISupportChecker,
    cited_claims,
)
from shipment_rag.llm import OllamaLLM


def labelled_claims(cases: list[dict]) -> list[dict]:
    rows = []
    for case in cases:
        want = case["expect"].get("citation_claims")
        if want is None:
            continue
        claims = cited_claims(case["answer"])
        if len(claims) != len(want):
            sys.exit(f"{case['id']}: {len(claims)} cited sentences found, {len(want)} labelled")
        for claim, expected in zip(claims, want, strict=True):
            cited = "\n\n".join(case["passages"][n - 1]["text"] for n in claim.markers)
            rows.append({"case": case["id"], "claim": claim.claim, "passage": cited,
                         "expected": expected})
    return rows


def run_checker(checker, rows: list[dict]) -> tuple[list[dict], float]:
    t0 = time.perf_counter()
    if isinstance(checker, NLISupportChecker):
        checker.supports("warm up", "warm up")  # load time is reported, not counted per check
    load = time.perf_counter() - t0
    out = []
    for row in rows:
        t0 = time.perf_counter()
        got, detail = checker.supports(row["passage"], row["claim"])
        out.append({**row, "got": got, "detail": detail, "seconds": time.perf_counter() - t0})
        print(f"{checker.name} {row['case']} expected={row['expected']} got={got} "
              f"{out[-1]['seconds']:.1f}s", flush=True)
    return out, load


def summary_row(name: str, res: list[dict], load: float) -> str:
    right = sum(r["got"] == r["expected"] for r in res)
    bad = [r for r in res if not r["expected"]]
    good = [r for r in res if r["expected"]]
    caught = sum(not r["got"] for r in bad)
    alarms = sum(not r["got"] for r in good)
    secs = [r["seconds"] for r in res]
    return (f"| {name} | {right}/{len(res)} | {caught}/{len(bad)} | {alarms}/{len(good)} | "
            f"{statistics.median(secs):.2f} s | {sum(secs):.1f} s | {load:.1f} s | "
            f"{'yes' if right == len(res) else '**no**'} |")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--file", default="data/eval/judge_calibration.json")
    parser.add_argument("--llm", default="qwen2.5:7b")
    parser.add_argument("--nli", default="cross-encoder/nli-deberta-v3-small")
    parser.add_argument("--min-entailment", type=float, default=0.9)
    parser.add_argument("--holdout", action="store_true",
                        help="fresh cases: run only the chosen NLI checker, nothing is tuned here")
    parser.add_argument("--report")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    rows = labelled_claims(json.loads(Path(args.file).read_text(encoding="utf-8")))
    checkers = [NLISupportChecker(args.nli, args.min_entailment)]
    if not args.holdout:
        llm = OllamaLLM(args.llm)
        llm.check_model()
        checkers.append(LLMSupportChecker(llm))
    results = [(c.name, *run_checker(c, rows)) for c in checkers]

    summary = "\n".join(
        ["| Checker | Correct | Wrong citations caught | False alarms | Median per check "
         "| Total | Load | Passes |", "|---|---|---|---|---|---|---|---|"]
        + [summary_row(name, res, load) for name, res, load in results]
    )
    head = "| Case | Cited sentence | Should be supported | " + " | ".join(
        name for name, _, _ in results) + " |"
    detail = [head, "|---|---|---|" + "---|" * len(results)]
    for i, row in enumerate(rows):
        cells = []
        for _, res, _ in results:
            r = res[i]
            cells.append(f"{r['got']} {'✅' if r['got'] == r['expected'] else '❌'}")
        claim = row["claim"].replace("|", "\\|")
        detail.append(f"| {row['case']} | {claim} | {row['expected']} | {' | '.join(cells)} |")
    report = (f"# Citation checker comparison\n\n{summary}\n\n"
              f"{len(rows)} cited sentences from `{Path(args.file).as_posix()}`. A checker passes "
              "only if it gets all of them right. "
              + ("These cases were written and committed before this run and nothing was tuned on "
                 "them. They leave out claims that stretch a passage (calibration case C4), an "
                 "accepted limit. The set is small: a pass rules out obvious failure, it is not a "
                 "measured accuracy.\n\n" if args.holdout else
                 "This set is small and was also used to choose the checker, so treat a pass as "
                 "necessary, not as a measured accuracy.\n\n")
              + "## Per sentence\n\n" + "\n".join(detail) + "\n")
    print("\n" + report)
    if args.report:
        Path(args.report).write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
