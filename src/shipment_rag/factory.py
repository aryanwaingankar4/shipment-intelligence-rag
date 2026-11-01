"""Wires concrete components together from Settings (the only place that does)."""

from shipment_rag.config import Settings
from shipment_rag.embeddings import Embedder, Reranker
from shipment_rag.llm import GeminiLLM
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


def build_pipeline(settings: Settings, retriever: Retriever | None = None) -> RAGPipeline:
    return RAGPipeline(
        retriever=retriever or build_retriever(settings),
        llm=build_llm(settings),
        company=settings.company_name,
        assistant=settings.assistant_name,
    )
