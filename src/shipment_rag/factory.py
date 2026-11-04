"""Wires concrete components together from Settings (the only place that does)."""

import logging

from shipment_rag.config import Settings
from shipment_rag.embeddings import Embedder, Reranker
from shipment_rag.evaluation.citation_check import LLMSupportChecker, NLISupportChecker
from shipment_rag.llm import GeminiLLM, OllamaLLM
from shipment_rag.pipeline import RAGPipeline
from shipment_rag.retrieval import Retriever
from shipment_rag.rewrite import QueryRewriter
from shipment_rag.structured.router import StructuredRouter
from shipment_rag.structured.store import ShipmentStore
from shipment_rag.vectorstore import get_collection

log = logging.getLogger(__name__)


def build_retriever(settings: Settings, collection=None) -> Retriever:
    return Retriever(
        collection=collection if collection is not None else get_collection(settings),
        embedder=Embedder(settings.embedding_model),
        reranker=Reranker(settings.reranker_model),
        retrieve_k=settings.retrieve_k,
        final_k=settings.final_k,
        min_rerank_score=settings.min_rerank_score,
        mode=settings.retrieval_mode,
        bm25_k=settings.bm25_k,
        rerank_k=settings.rerank_k,
        rrf_k=settings.rrf_k,
    )


def build_llm(settings: Settings) -> GeminiLLM:
    if settings.gemini_api_key is None or not settings.gemini_api_key.get_secret_value():
        raise RuntimeError("GEMINI_API_KEY is not set")
    return GeminiLLM(
        api_key=settings.gemini_api_key.get_secret_value(),
        model=settings.gemini_model,
        temperature=settings.llm_temperature,
        max_output_tokens=settings.llm_max_output_tokens,
        max_retries=settings.llm_max_retries,
        timeout_s=settings.llm_timeout_s,
        thinking_level=settings.llm_thinking_level,
        thinking_budget=settings.llm_thinking_budget,
    )


def build_judge_llm(settings: Settings) -> GeminiLLM | OllamaLLM:
    """LLM-as-judge for eval-gen: deterministic (temperature 0), on Gemini or a local model."""
    if settings.judge_provider == "ollama":
        if not settings.judge_model:
            raise RuntimeError("JUDGE_PROVIDER=ollama needs JUDGE_MODEL, e.g. qwen2.5:7b")
        return OllamaLLM(model=settings.judge_model, base_url=settings.ollama_base_url)
    if settings.gemini_api_key is None or not settings.gemini_api_key.get_secret_value():
        raise RuntimeError("GEMINI_API_KEY is not set")
    return GeminiLLM(
        api_key=settings.gemini_api_key.get_secret_value(),
        model=settings.judge_model or settings.gemini_model,
        temperature=0.0,
        max_output_tokens=4096,
        max_retries=settings.llm_max_retries,
        timeout_s=max(settings.llm_timeout_s, 120.0),  # judge thinks longer
        thinking_level=settings.judge_thinking_level,
    )


def build_citation_checker(settings: Settings, llm) -> LLMSupportChecker | NLISupportChecker:
    """Per-citation support check for eval-gen. llm is only used by the 'llm' checker."""
    if settings.citation_checker == "nli":
        return NLISupportChecker(settings.nli_model, settings.nli_min_entailment)
    return LLMSupportChecker(llm)


def build_structured_router(settings: Settings) -> StructuredRouter | None:
    """None if turned off or the log can't be loaded: the document path still answers."""
    if not settings.structured_queries:
        return None
    try:
        return StructuredRouter(ShipmentStore.from_file(settings.docs_path / "shipment_log.txt"))
    except (OSError, ValueError) as exc:
        log.warning("shipment table unavailable, using documents only", extra={"error": str(exc)})
        return None


def build_rewriter(settings: Settings) -> QueryRewriter | None:
    """None when rewriting is off. Short limits: a person is waiting for this call."""
    if settings.rewrite_provider == "off":
        return None
    if settings.rewrite_provider == "ollama":
        if not settings.rewrite_model:
            raise RuntimeError("REWRITE_PROVIDER=ollama needs REWRITE_MODEL, e.g. qwen2.5:7b")
        llm = OllamaLLM(model=settings.rewrite_model, base_url=settings.ollama_base_url,
                        max_output_tokens=128, num_ctx=2048,
                        timeout_s=settings.rewrite_timeout_s)
    else:
        if settings.gemini_api_key is None or not settings.gemini_api_key.get_secret_value():
            raise RuntimeError("GEMINI_API_KEY is not set")
        llm = GeminiLLM(api_key=settings.gemini_api_key.get_secret_value(),
                        model=settings.rewrite_model or settings.gemini_model,
                        temperature=0.0, max_output_tokens=512, max_retries=1,
                        timeout_s=settings.rewrite_timeout_s, thinking_level="low")
    return QueryRewriter(llm, max_messages=settings.rewrite_max_messages)


def build_pipeline(
    settings: Settings, retriever: Retriever | None = None, llm=None
) -> RAGPipeline:
    return RAGPipeline(
        retriever=retriever or build_retriever(settings),
        llm=llm or build_llm(settings),
        company=settings.company_name,
        assistant=settings.assistant_name,
        structured=build_structured_router(settings),
        rewriter=build_rewriter(settings),
        rewrite_every_question=settings.rewrite_every_question,
    )
