"""Central configuration. Every tunable comes from environment variables (or a
local .env file), so the same image runs in dev, staging and prod unchanged."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- runtime ---
    app_env: str = "dev"
    log_level: str = "INFO"
    log_json: bool = True

    # --- knowledge base ---
    docs_path: Path = Path("data/docs")
    chunk_max_chars: int = 700
    chunk_overlap: int = 80
    # repeat the section header on sub-split chunks (see chunker.chunk_document)
    chunk_section_headers: bool = True

    # answer exact shipment questions (counts, lists, field lookups) from a table, no LLM call
    structured_queries: bool = True

    # --- vector store: set CHROMA_HOST to use a Chroma server, else embedded on disk ---
    chroma_host: str | None = None
    chroma_port: int = 8000
    chroma_persist_path: Path = Path("data/chroma_db")
    collection_name: str = "supply_chain"

    # --- models ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- retrieval ---
    # vector = embeddings only; bm25 = lexical only; hybrid = both, fused with RRF
    retrieval_mode: Literal["vector", "bm25", "hybrid"] = "hybrid"
    retrieve_k: int = Field(8, ge=1, le=50)  # vector candidates
    bm25_k: int = Field(8, ge=1, le=50)  # lexical candidates
    rerank_k: int = Field(12, ge=1, le=100)  # fused candidates sent to the cross-encoder
    rrf_k: int = Field(60, ge=1)  # RRF damping constant (60 is the standard value)
    final_k: int = Field(3, ge=1, le=10)
    # ms-marco cross-encoder returns logits; chunks scoring below this are treated
    # as irrelevant so out-of-scope questions are refused without calling the LLM.
    # Chosen by `shipment-rag tune-threshold` on the *tune* split only
    # (data/eval/reports/stage1_threshold_holdout.md): weakest answerable in-scope top score
    # -9.24, strongest off-domain -11.25 -> -10.25. On the held-out test split: 0 false
    # refusals, 3/3 off-domain refused. It filters obvious junk to save LLM calls; it is NOT
    # the refusal guarantee — the grounded prompt is.
    min_rerank_score: float = -10.25

    # --- LLM ---
    gemini_api_key: SecretStr | None = None
    gemini_model: str = "gemini-3.5-flash-lite"
    llm_temperature: float = 0.1
    # thinking tokens count against this on Gemini 3.x, so leave headroom
    llm_max_output_tokens: int = 2048
    # Gemini 3.x thinking control; "" = model default. Grounded Q&A needs little reasoning.
    llm_thinking_level: Literal["", "minimal", "low", "medium", "high"] = "low"
    # Gemini 2.5 models use a token budget instead; when set, it overrides the level.
    llm_thinking_budget: int | None = None
    llm_max_retries: int = 4
    # Sent to Gemini as the server-side deadline: 30 s produced 504 DEADLINE_EXCEEDED on
    # gemini-3.8-flash during the Stage 2 eval runs.
    llm_timeout_s: float = 60.0

    # --- follow-up questions: rewrite them into standalone questions ---
    # off = questions are used as written; ollama = local model, no quota; gemini = one extra
    # Gemini call per rewrite. A rewriter that fails falls back to the question as written.
    rewrite_provider: Literal["off", "ollama", "gemini"] = "off"
    rewrite_model: str = ""  # gemini: empty = same as GEMINI_MODEL; ollama: required
    rewrite_timeout_s: float = 60.0
    rewrite_max_messages: int = 6  # recent chat messages shown to the rewriter
    # Also rewrite questions with no history, into document wording (one more LLM call each).
    rewrite_every_question: bool = False

    # --- generation eval (shipment-rag eval-gen) ---
    judge_provider: Literal["gemini", "ollama"] = "gemini"
    judge_model: str = ""  # gemini: empty = same as GEMINI_MODEL; ollama: required
    ollama_base_url: str = "http://localhost:11434"
    judge_thinking_level: Literal["", "minimal", "low", "medium", "high"] = "medium"  # gemini
    citation_checker: Literal["llm", "nli"] = "nli"  # llm = the judge model, one call per citation
    nli_model: str = "cross-encoder/nli-deberta-v3-small"
    nli_min_entailment: float = 0.9
    eval_rpm: float = 8.0  # max uncached Gemini calls per minute (a local judge is not limited)
    eval_cache_dir: Path = Path("data/eval/.cache")

    # --- API ---
    # Comma-separated list of accepted X-API-Key values. Empty = auth disabled (dev only).
    api_keys: str = ""
    cors_origins: str = ""
    max_question_chars: int = 1000

    company_name: str = "NexaChain BioMed"
    assistant_name: str = "Nova"

    @field_validator("llm_thinking_budget", mode="before")
    @classmethod
    def _empty_budget_is_unset(cls, v):
        return None if v == "" else v

    @property
    def api_key_set(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
