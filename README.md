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
| LLM | Gemini (`gemini-3.8-flash` default, low thinking level) via `google-genai`, retries with backoff, fails fast if the model is unavailable |
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

Tests and lint: `pytest` and `ruff check src tests ui scripts`.

Secrets: enable the pre-commit scan once per clone with `git config core.hooksPath .githooks`
(it also runs in CI). Real keys only ever go in `.env`, which is gitignored.
`shipment-rag models` lists the Gemini models your key can use.

## Example queries

| Question | Expected behaviour |
|---|---|
| What is the storage temperature for TL-COLD-9001? | 2 °C to 8 °C, below 60 % RH, dark conditions |
| What is the shelf life of StableCap Dry Powder? | 36 months from date of manufacture |
| Why was shipment SH-2024-007 rejected? | Rose above −10 °C in transit; dry ice sublimated (insufficient quantity, wrong case) |
| Within how many hours must CDSCO be notified of a recall? | 72 hours of a confirmed recall |
| What is the capital of France? | Refused before calling the LLM (no passage clears the reranker threshold) |

<p>
  <img src="assets/chat1.png" alt="Chat example 1" width="49%">
  <img src="assets/chat2.png" alt="Chat example 2" width="49%">
</p>

*Screenshots are from the Colab prototype (Phase 3).*

## Results

Retrieval is evaluated without any LLM calls (`shipment-rag eval`), so it runs in CI on
every PR. A case passes if an expected keyword appears in the final top-3 passages; an
out-of-scope case passes if nothing clears the reranker threshold.

| Eval set | Method | Score |
|---|---|---|
| `data/eval/retrieval_eval.json` (10 in-scope + 2 out-of-scope) | vector top-8 → cross-encoder rerank → threshold → top-3 | **12/12** |
| `data/eval/notebook_eval_8.json` (the prototype's 8 questions, recovered from the notebook) | notebook method: vector top-3, no rerank | 8/8 |
| same | current pipeline | 8/8 |

- The prototype README reported **7/8** on that 8-question set; it does **not** reproduce on
  the current documents and index (8/8), so it isn't used as a before/after number.
- Both sets are saturated: plain vector search already solves them, so they can't show what
  the reranker or future changes add. Known failures (a recall miss on an escalation-email
  question, and a correct passage scored just under the refusal threshold) are tracked in
  [PROJECT_STATUS.md](PROJECT_STATUS.md) and drive the next stage: hybrid search plus a harder eval set.

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
data/eval/          retrieval evaluation sets
assets/             screenshots
tests/              unit + API tests (no models / network needed)
notebooks/          original Colab prototype, kept for reference
```

All configuration is via environment variables — see `.env.example`.
Roadmap, decisions and pending work: [PROJECT_STATUS.md](PROJECT_STATUS.md).
