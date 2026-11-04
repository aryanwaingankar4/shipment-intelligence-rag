"""Write each golden answer's cited sentences and passages, ready to be labelled by hand.

    python scripts/prepare_citation_labels.py --out data/eval/citation_labels_todo.json

Blind on purpose: this never loads the NLI checker and never reads the eval report, so the file
holds no verdicts. Answers come from the eval cache only: a miss is listed, never sent to Gemini.
The output has the same shape as the citation holdouts, so once expect.citation_claims is filled
in, scripts/compare_citation_checkers.py --holdout --file <it> reports NLI's agreement.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from shipment_rag.config import Settings, get_settings
from shipment_rag.evaluation.citation_check import cited_claims
from shipment_rag.evaluation.generation_eval import load_golden
from shipment_rag.evaluation.llm_cache import CachedLLM
from shipment_rag.factory import build_llm, build_pipeline


class NotCached(LookupError):
    pass


class CacheOnlyLLM(CachedLLM):
    """Serves stored answers and refuses to call the model."""

    def generate(self, system: str, prompt: str, json_schema: dict | None = None) -> str:
        path = self.cache_dir
        key = self._key(system, prompt, json_schema)
        path = path / key[:2] / f"{key}.json"
        if not path.exists():
            raise NotCached(key)
        return json.loads(path.read_text(encoding="utf-8"))["text"]


def eval_gen_settings(settings: Settings) -> Settings:
    """The same answers eval-gen scored: on the document path, with questions as written."""
    return settings.model_copy(update={"structured_queries": False, "rewrite_provider": "off"})


def prepare(pipeline, items) -> tuple[list[dict], list[str], list[str]]:
    """(cases to label, ids with no cached answer, ids answered with no citation to label)."""
    cases, missing, uncited = [], [], []
    for item in items:
        try:
            result = pipeline.answer(item.question)
        except NotCached:
            missing.append(item.id)
            continue
        if not result.llm_called:
            continue  # refused at retrieval: nothing was generated
        claims = cited_claims(result.answer)
        if not claims:
            uncited.append(item.id)
            continue
        cases.append({
            "id": item.id,
            "type": item.type,
            "question": item.question,
            "answer": result.answer,
            "passages": [{"source": c.source, "text": c.text} for c in result.context],
            "claims": [{"markers": c.markers, "claim": c.claim} for c in claims],
            "expect": {"citation_claims": [None] * len(claims)},
        })
    return cases, missing, uncited


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--file", default="data/eval/golden_qa.json")
    parser.add_argument("--out", default="data/eval/citation_labels_todo.json")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.getLogger("shipment_rag.audit").setLevel(logging.CRITICAL)  # misses are expected here

    settings = eval_gen_settings(get_settings())
    llm = CacheOnlyLLM(build_llm(settings), settings.eval_cache_dir, "answer")
    pipeline = build_pipeline(settings, llm=llm)
    cases, missing, uncited = prepare(pipeline, load_golden(Path(args.file)))
    Path(args.out).write_text(json.dumps(cases, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"answers to label: {len(cases)}; cited sentences: "
          f"{sum(len(c['claims']) for c in cases)}; no citation: {uncited or 'none'}; "
          f"no cached answer yet: {missing or 'none'}")
    print(f"written to {args.out}")
    return 0 if cases else 1


if __name__ == "__main__":
    raise SystemExit(main())
