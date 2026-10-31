# Shipment Intelligence RAG

A document-grounded AI assistant for pharmaceutical cold-chain operations. Ask about
products, shipments, temperature breaches, vendors, incidents and GDP compliance, and get
answers that cite the company documents and refuse anything outside them.

![Home](assets/Home_page.png)

## Architecture

```
            ┌────────────┐   HTTP    ┌──────────────────────────────┐       ┌───────────┐
 user ────▶ │ Streamlit  │ ────────▶ │ FastAPI  /v1/query           │ ────▶ │  Gemini   │
            │ UI (ui/)   │  X-API-Key│  retrieve → rerank → prompt  │       │  (LLM)    │
            └────────────┘           └──────────────┬───────────────┘       └───────────┘
                                                    │ query
                                             ┌──────▼──────┐      ┌───────────────────┐
                                             │  ChromaDB   │ ◀─── │ ingest job        │
                                             │  (server)   │      │ data/docs/*.txt   │
                                             └─────────────┘      └───────────────────┘
```

| Stage | Choice |
|---|---|
| Chunking | Section-aware (`=== HEADER ===`), sub-split only if > 700 chars, content-hash IDs |
| Embeddings | `all-MiniLM-L6-v2` (384-d, normalized) |
| Vector DB | ChromaDB 1.5.9 (cosine) — embedded locally, server mode in Docker |
| Rerank | `cross-encoder/ms-marco-MiniLM-L-6-v2`, top-8 → top-3, irrelevance threshold |
| LLM | Gemini (`gemini-2.5-flash` default) via `google-genai`, retries with backoff |
| Guardrails | Refuse without an LLM call when nothing relevant is retrieved; strict grounded system prompt |

## Quick start (Docker — recommended)

```bash
cp .env.example .env          # then set GEMINI_API_KEY
docker compose up --build
```

- UI: http://localhost:8501 (demo / demo123)
- API docs: http://localhost:8080/docs

Re-index after editing anything in `data/docs/`: `docker compose run --rm ingest`

## Local development (no Docker)

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cpu torch
uv pip install --python .venv -e ".[dev]" -r ui/requirements.txt

# Windows: .venv\Scripts\activate   |  macOS/Linux: source .venv/bin/activate
shipment-rag ingest                     # builds data/chroma_db
shipment-rag eval                       # retrieval quality check, no LLM calls
shipment-rag ask "Why was SH-2024-007 rejected?"

uvicorn shipment_rag.api.main:create_app --factory --port 8080
streamlit run ui/streamlit_app.py
```

Tests and lint: `pytest` and `ruff check src tests ui`.

## API

```http
POST /v1/query
X-API-Key: <key>            # required when API_KEYS is set
{"question": "Which shipments breached temperature?", "custom_instructions": ""}
```

Response: `answer`, `refused`, `sources[] {source, section, rerank_score, snippet}`,
`latency_ms`, `request_id`. Ops endpoints: `GET /health` (liveness), `GET /ready`
(index reachable and non-empty).

## Layout

```
src/shipment_rag/   config, ingestion/, embeddings, vectorstore, retrieval, llm, pipeline, api/, cli
ui/                 Streamlit front end (separate image)
data/docs/          knowledge base (source of truth for the index)
data/eval/          retrieval evaluation set
tests/              unit + API tests (no models / network needed)
notebooks/          original Colab prototype, kept for reference
```

All configuration is via environment variables — see `.env.example`.
Roadmap, decisions and pending work: [PROJECT_STATUS.md](PROJECT_STATUS.md).
