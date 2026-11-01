"""Command line entry point: `shipment-rag {ingest,ask,eval}`."""

import argparse
import json
import sys
from pathlib import Path

from shipment_rag.config import get_settings
from shipment_rag.llm import ModelNotFoundError
from shipment_rag.logging_config import configure_logging


def cmd_ingest(args) -> int:
    from shipment_rag.embeddings import Embedder
    from shipment_rag.ingestion.indexer import ingest
    from shipment_rag.vectorstore import get_collection

    settings = get_settings()
    report = ingest(
        settings, get_collection(settings), Embedder(settings.embedding_model), rebuild=args.rebuild
    )
    print(json.dumps(report.__dict__, indent=2))
    return 0


def cmd_ask(args) -> int:
    from shipment_rag.factory import build_pipeline

    pipeline = build_pipeline(get_settings())
    pipeline.llm.check_model()
    result = pipeline.answer(args.question)
    print(result.answer)
    for c in result.sources:
        print(f"  - {c.source} [{c.section}] rerank={c.rerank_score:+.2f}")
    return 0


def cmd_models(args) -> int:
    from shipment_rag.factory import build_llm

    for name in build_llm(get_settings()).list_models():
        print(name)
    return 0


def cmd_eval(args) -> int:
    """Retrieval-only evaluation: no LLM calls, so it is free to run in CI."""
    from shipment_rag.evaluation.retrieval_eval import (
        case_table,
        comparison_table,
        evaluate,
        load_cases,
    )
    from shipment_rag.factory import build_retriever

    settings = get_settings()
    base = build_retriever(settings)
    modes = ["vector", "bm25", "hybrid"] if args.mode == "all" else [args.mode or base.mode]
    cases = load_cases(Path(args.file))

    reports = []
    for mode in modes:
        base.mode = mode  # same collection, models and threshold; only candidate generation varies
        reports.append(evaluate(base, cases))

    for rep in reports:
        for r in rep.rows:
            if not r.passed and r.case.kind != "near_domain":
                print(f"FAIL [{rep.mode}] {r.case.question[:70]} (top {r.top_source}, "
                      f"score {r.top_score})")
    table = comparison_table(reports)
    print("\n" + table)

    if args.report:
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            f"# Retrieval eval — `{Path(args.file).as_posix()}`\n\n"
            f"Final top-{settings.final_k}, MIN_RERANK_SCORE={settings.min_rerank_score}, "
            f"retrieve_k={settings.retrieve_k}, bm25_k={settings.bm25_k}, "
            f"rerank_k={settings.rerank_k}, rrf_k={settings.rrf_k}, "
            f"CHUNK_SECTION_HEADERS={settings.chunk_section_headers}, "
            f"reranker `{settings.reranker_model}`.\n\n"
            f"## Summary\n\n{table}\n\n## Per question\n\n"
            f"Cell = rank of first relevant chunk in the final top-{settings.final_k} "
            "(or miss / refused) / best rerank score before the threshold.\n\n"
            f"{case_table(reports)}\n",
            encoding="utf-8",
        )
        print(f"\nreport written to {out}")

    gate = reports[-1].gate_score()
    print(f"\ngate ({reports[-1].mode}): {gate:.0%} (min {args.min_score:.0%})")
    return 0 if gate >= args.min_score else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="shipment-rag")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest", help="index documents from DOCS_PATH into Chroma")
    p.add_argument("--rebuild", action="store_true", help="wipe the collection first")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("ask", help="ask one question end to end (uses Gemini)")
    p.add_argument("question")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("models", help="list Gemini models available to GEMINI_API_KEY")
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("eval", help="retrieval quality check (no LLM calls)")
    p.add_argument("--file", default="data/eval/retrieval_eval.json")
    p.add_argument("--min-score", type=float, default=0.9, help="gate on the last mode run")
    p.add_argument("--mode", choices=["vector", "bm25", "hybrid", "all"],
                   help="default: RETRIEVAL_MODE; 'all' compares the three (gate uses hybrid)")
    p.add_argument("--report", help="write a markdown report to this path")
    p.set_defaults(func=cmd_eval)

    args = parser.parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level, json_output=False)
    try:
        return args.func(args)
    except ModelNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
