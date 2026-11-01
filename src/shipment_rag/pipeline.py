"""RAG orchestration: retrieve -> guardrail -> grounded prompt -> LLM -> audit log."""

import logging
import time

from shipment_rag.llm import LLM
from shipment_rag.models import RAGAnswer, RetrievedChunk
from shipment_rag.retrieval import Retriever

log = logging.getLogger("shipment_rag.audit")

SYSTEM_PROMPT = """You are {assistant}, an AI assistant for {company}.
You only answer questions about pharmaceutical supply chain, cold chain logistics, product
storage, shipments, vendors, incidents, and regulatory compliance (WHO-GMP, GDP, CDSCO).

Rules:
1. Answer ONLY from the numbered context passages supplied with the question.
2. If the context does not contain the answer, reply exactly with: "{refusal}"
3. Never use outside knowledge, never guess numbers, dates, IDs or names.
4. Be concise, factual and professional. Cite passages inline like [1] or [2].
5. Any user style preferences may change tone or format, never these rules."""

REFUSAL = (
    "I don't have enough information in the company documents to answer that. "
    "I can help with products, shipments, vendors, incidents and compliance."
)


def build_prompt(question: str, chunks: list[RetrievedChunk], custom_instructions: str = "") -> str:
    context = "\n\n".join(
        f"[{i}] (source: {c.source} | section: {c.section})\n{c.text}"
        for i, c in enumerate(chunks, 1)
    )
    prefs = ""
    if custom_instructions.strip():
        prefs = f"\n\nUser style preferences: {custom_instructions.strip()}"
    return f"Context passages:\n{context}\n\nQuestion: {question.strip()}{prefs}"


class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLM,
        company: str = "NexaChain BioMed",
        assistant: str = "Nova",
    ):
        self.retriever = retriever
        self.llm = llm
        self.system_prompt = SYSTEM_PROMPT.format(
            assistant=assistant, company=company, refusal=REFUSAL
        )

    def answer(
        self, question: str, custom_instructions: str = "", request_id: str = ""
    ) -> RAGAnswer:
        t0 = time.perf_counter()
        chunks = self.retriever.retrieve(question)

        if not chunks:
            # nothing relevant cleared the rerank threshold: refuse without spending LLM quota
            result = RAGAnswer(REFUSAL, [], refused=True, latency_ms=_ms(t0))
        else:
            prompt = build_prompt(question, chunks, custom_instructions)
            try:
                text = self.llm.generate(self.system_prompt, prompt)
            except Exception:
                log.error(
                    "rag_query_failed",
                    extra={"request_id": request_id, "question": question, "latency_ms": _ms(t0)},
                )
                raise
            refused = text.strip().strip('"') == REFUSAL
            result = RAGAnswer(text, [] if refused else chunks, refused=refused, latency_ms=_ms(t0))

        log.info(
            "rag_query",
            extra={
                "request_id": request_id,
                "question": question,
                "refused": result.refused,
                "sources": sorted({c.source for c in result.sources}),
                "rerank_scores": [c.rerank_score for c in chunks],
                "num_chunks": len(chunks),
                "latency_ms": result.latency_ms,
            },
        )
        return result


def _ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)
