"""Wires concrete components together from Settings (the only place that does)."""

from shipment_rag.config import Settings
from shipment_rag.embeddings import Embedder, Reranker
from shipment_rag.llm import GeminiLLM, OllamaLLM
from shipment_rag.pipeline import RAGPipeline
from shipment_rag.retrieval import Retriever
from shipment_rag.vectorstore import get_collection


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


def build_pipeline(
    settings: Settings, retriever: Retriever | None = None, llm=None
) -> RAGPipeline:
    return RAGPipeline(
        retriever=retriever or build_retriever(settings),
        llm=llm or build_llm(settings),
        company=settings.company_name,
        assistant=settings.assistant_name,
    )
