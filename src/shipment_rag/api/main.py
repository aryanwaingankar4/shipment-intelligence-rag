"""HTTP API for the RAG pipeline.

Run:  uvicorn shipment_rag.api.main:create_app --factory --host 0.0.0.0 --port 8080
Models are loaded once per worker process at startup, not per request.
"""

import hmac
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from shipment_rag import __version__
from shipment_rag.config import Settings, get_settings
from shipment_rag.llm import LLMUnavailableError
from shipment_rag.logging_config import configure_logging
from shipment_rag.pipeline import RAGPipeline

log = logging.getLogger("shipment_rag.api")


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    custom_instructions: str = Field("", max_length=500)


class SourceOut(BaseModel):
    source: str
    section: str
    rerank_score: float
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    refused: bool
    sources: list[SourceOut]
    latency_ms: int
    request_id: str


def create_app(settings: Settings | None = None, pipeline: RAGPipeline | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if pipeline is not None:
            app.state.pipeline = pipeline
        else:
            from shipment_rag.factory import build_pipeline

            app.state.pipeline = build_pipeline(settings)
            # load model weights now so the first user request isn't slow
            app.state.pipeline.retriever.embedder.warm_up()
            app.state.pipeline.retriever.reranker.warm_up()
        log.info("api started", extra={"version": __version__, "env": settings.app_env})
        yield

    app = FastAPI(title="Shipment Intelligence RAG", version=__version__, lifespan=lifespan)

    if settings.cors_origin_list:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origin_list,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = request_id
        t0 = time.perf_counter()
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        if request.url.path not in ("/health", "/ready"):
            log.info(
                "http_request",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "latency_ms": int((time.perf_counter() - t0) * 1000),
                },
            )
        return response

    def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
        keys = settings.api_key_set
        if not keys:
            return  # auth disabled (dev)
        if not x_api_key or not any(hmac.compare_digest(x_api_key, k) for k in keys):
            raise HTTPException(status_code=401, detail="invalid or missing API key")

    @app.get("/health", tags=["ops"])
    def health():
        """Liveness: the process is up."""
        return {"status": "ok", "version": __version__}

    @app.get("/ready", tags=["ops"])
    def ready(request: Request):
        """Readiness: models loaded and the vector index is reachable and non-empty."""
        try:
            count = request.app.state.pipeline.retriever.collection.count()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"vector store unavailable: {exc}") from exc
        if count == 0:
            raise HTTPException(status_code=503, detail="index is empty; run ingestion")
        return {"status": "ready", "chunks": count}

    # sync handler -> FastAPI runs it in a threadpool, so model inference doesn't block the loop
    @app.post("/v1/query", response_model=QueryResponse, dependencies=[Depends(require_api_key)])
    def query(body: QueryRequest, request: Request):
        rid = request.state.request_id
        try:
            result = request.app.state.pipeline.answer(
                body.question, body.custom_instructions, request_id=rid
            )
        except LLMUnavailableError as exc:
            raise HTTPException(
                status_code=503, detail="The assistant is temporarily unavailable."
            ) from exc
        return QueryResponse(
            answer=result.answer,
            refused=result.refused,
            sources=[
                SourceOut(
                    source=c.source,
                    section=c.section,
                    rerank_score=c.rerank_score,
                    snippet=c.text[:240],
                )
                for c in result.sources
            ],
            latency_ms=result.latency_ms,
            request_id=rid,
        )

    return app
