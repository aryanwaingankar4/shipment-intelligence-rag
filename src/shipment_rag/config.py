"""Central configuration. Every tunable comes from environment variables (or a
local .env file), so the same image runs in dev, staging and prod unchanged."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
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

    # --- vector store: set CHROMA_HOST to use a Chroma server, else embedded on disk ---
    chroma_host: str | None = None
    chroma_port: int = 8000
    chroma_persist_path: Path = Path("data/chroma_db")
    collection_name: str = "supply_chain"

    # --- models ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- retrieval ---
    retrieve_k: int = Field(8, ge=1, le=50)
    final_k: int = Field(3, ge=1, le=10)
    # ms-marco cross-encoder returns logits; chunks scoring below this are treated
    # as irrelevant so out-of-scope questions are refused without calling the LLM.
    min_rerank_score: float = -6.0

    # --- LLM ---
    gemini_api_key: SecretStr | None = None
    gemini_model: str = "gemini-2.5-flash"
    llm_temperature: float = 0.1
    llm_max_output_tokens: int = 512
    llm_max_retries: int = 4
    llm_timeout_s: float = 30.0

    # --- API ---
    # Comma-separated list of accepted X-API-Key values. Empty = auth disabled (dev only).
    api_keys: str = ""
    cors_origins: str = ""
    max_question_chars: int = 1000

    company_name: str = "NexaChain BioMed"
    assistant_name: str = "Nova"

    @property
    def api_key_set(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
