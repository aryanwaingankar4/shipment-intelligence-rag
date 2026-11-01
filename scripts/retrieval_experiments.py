"""Stage 1 retrieval experiments (no LLM calls): reranker ablation and threshold sweep.

    python scripts/retrieval_experiments.py --out data/eval/reports/stage1_experiments.md

Uses the index configured by CHROMA_PERSIST_PATH / CHROMA_HOST (run `shipment-rag ingest`
first) and the eval set in data/eval/retrieval_eval.json.
"""

import argparse
from pathlib import Path

from shipment_rag.config import get_settings
from shipment_rag.embeddings import Reranker
from shipment_rag.evaluation.retrieval_eval import evaluate, load_cases
from shipment_rag.factory import build_retriever

RERANKERS = {
    "MiniLM-L6": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "MiniLM-L12": "cross-encoder/ms-marco-MiniLM-L-12-v2",
}
THRESHOLDS = [-6.0, -9.0, -10.0, -10.5, -10.9, -11.0, None]


class CandidateOrder:
    """'No reranker': keep the candidate order (RRF order for hybrid)."""

    def score(self, query, texts):
        return [float(len(texts) - i) for i in range(len(texts))]


def row(label, rep) -> str:
    return (
        f"| {label} | {rep.hit_at_k():.0%} | {rep.hit_at_k('id'):.0%} | "
        f"{rep.hit_at_k('paraphrase'):.0%} | {rep.mrr():.3f} | {rep.candidate_recall():.0%} | "
        f"{rep.false_refusals()} | {rep.refusal_rate('off_domain'):.0%} |"
    )


HEAD = (
    "| Config | hit@3 all | hit@3 ID | hit@3 paraphrased | MRR@3 | candidate recall | "
    "false refusals | off-domain refused |\n|---|---|---|---|---|---|---|---|"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="data/eval/retrieval_eval.json")
    ap.add_argument("--out", default="data/eval/reports/stage1_experiments.md")
    args = ap.parse_args()

    cases = load_cases(Path(args.file))
    r = build_retriever(get_settings())
    out = ["# Stage 1 retrieval experiments", "", f"Eval set: `{args.file}`.", ""]

    out += ["## Does the reranker help? (no threshold)", "", HEAD]
    for mode in ("vector", "bm25", "hybrid"):
        r.mode, r.min_rerank_score = mode, None
        r.reranker = CandidateOrder()
        out.append(row(f"{mode}, no rerank", evaluate(r, cases)))
        r.reranker = Reranker(RERANKERS["MiniLM-L6"])
        out.append(row(f"{mode} + MiniLM-L6 rerank", evaluate(r, cases)))

    out += ["", "## Threshold sweep (hybrid)", "", HEAD]
    r.mode = "hybrid"
    for name, model in RERANKERS.items():
        r.reranker = Reranker(model)
        for thr in THRESHOLDS:
            r.min_rerank_score = thr
            out.append(row(f"{name}, MIN_RERANK_SCORE={thr}", evaluate(r, cases)))
        r.min_rerank_score = None
        rep = evaluate(r, cases)
        ins = sorted(x.top_score for x in rep.rows
                     if x.case.kind == "in_scope" and x.in_candidates and x.top_score is not None)
        off = [x.top_score for x in rep.rows if x.case.kind == "off_domain" and x.top_score]
        out.append(f"| *{name}: weakest answerable in-scope top score {ins[0]:+.2f}, "
                   f"strongest off-domain {max(off):+.2f}* | | | | | | | |")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
