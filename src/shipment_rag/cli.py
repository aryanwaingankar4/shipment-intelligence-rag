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
    from shipment_rag.factory import build_retriever

    retriever = build_retriever(get_settings())
    cases = json.loads(Path(args.file).read_text(encoding="utf-8"))
    passed = 0
    for case in cases:
        hits = retriever.retrieve(case["question"])
        if case.get("expect_refusal"):
            ok = not hits
        else:
            ok = any(kw.lower() in h.text.lower() for h in hits for kw in case["expect_any"])
        passed += ok
        top = f"{hits[0].source} {hits[0].rerank_score:+.2f}" if hits else "no chunks"
        print(f"{'PASS' if ok else 'FAIL'} | {case['question'][:60]:60s} | {top}")
    score = passed / len(cases)
    print(f"\nretrieval score: {passed}/{len(cases)} = {score:.0%} "
          f"(threshold {args.min_score:.0%})")
    return 0 if score >= args.min_score else 1


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
    p.add_argument("--min-score", type=float, default=0.9)
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
