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
| Chunking | Section-aware (`=== HEADER ===`), sub-split only if > 700 chars, section header repeated on every sub-chunk, content-hash IDs |
| Embeddings | `all-MiniLM-L6-v2` (384-d, normalized) |
| Vector DB | ChromaDB 1.5.9 (cosine) — embedded locally, server mode in Docker |
| Candidates | **Hybrid** (`RETRIEVAL_MODE=hybrid`): vector top-8 + BM25 top-8 with an ID-aware tokenizer, fused by reciprocal rank fusion → 12 candidates |
| Rerank | `cross-encoder/ms-marco-MiniLM-L-6-v2` → top-3, junk filter at `MIN_RERANK_SCORE=-10.9` |
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

Retrieval is evaluated without any LLM calls (`shipment-rag eval --mode all`), so it runs
in CI on every PR. Eval set: `data/eval/retrieval_eval.json`, **58 questions**: 47 in-scope
(20 ID-heavy such as `SH-2024-006` / `BATCH-FRZ-2403` / `V-003`, 10 paraphrased in everyday
wording, 17 other), 6 off-domain, and 5 near-domain (on-topic but unanswerable, e.g. a
non-existent shipment ID). A question is a hit if a passage containing the expected facts
is in the final top-3.

### Retrieval: before → after (same 58-question set, same index and models)

| Metric | Phase 4 baseline<br>vector, threshold −6 | Stage 1<br>hybrid + section headers, −10.9 |
|---|---|---|
| hit@3, all in-scope (n=47) | 74% | **91%** |
| hit@3, ID-heavy (n=20) | 90% | **100%** |
| hit@3, paraphrased (n=10) | 30% | **70%** |
| MRR@3 | 0.734 | **0.858** |
| candidate recall (right chunk reached the reranker) | 94% | **98%** |
| in-scope questions wrongly refused | 8 | **0** |
| off-domain questions refused before the LLM (n=6) | 100% | 100% |

What each change contributed (full tables in `data/eval/reports/`):

| Step | hit@3 | MRR@3 | Why |
|---|---|---|---|
| Phase 4 baseline (vector, −6) | 74% | 0.734 | |
| + hybrid BM25 / RRF | 79% | 0.777 | exact IDs; candidate recall 94% → 98% |
| + section header on every sub-chunk | 81% | 0.777 | split sections no longer lose their incident / product ID |
| + threshold −6 → −10.9 | **91%** | **0.858** | paraphrased questions score low on the cross-encoder and were being refused |

Findings worth knowing:

- **Vector and BM25 fail on different questions.** BM25 is perfect on IDs but finds nothing
  for "how often do we get the exterminators in?"; vector handles paraphrases but misses
  `SH-2024-006`. Hybrid gets the right chunk to the reranker 98 % of the time
  (vector 94 %, BM25 87 %).
- **The cross-encoder is now the bottleneck on paraphrases.** It lifts MRR (0.766 → 0.858)
  and ID accuracy, but on paraphrased questions it does slightly *worse* than plain RRF
  order (70 % vs 80 %). A larger `MiniLM-L12` reranker didn't help (89 % hit@3).
- **The refusal threshold is a cost saver, not a guarantee.** The gap between the weakest
  answerable question (−9.97) and the strongest off-domain one (−11.14) is small, and it
  was tuned on this same set. Unanswerable on-topic questions (e.g. "status of
  SH-2024-011?") score high and are left to the grounded prompt to refuse.
- **Still failing:** 2 paraphrases ranked outside the top 3, 1 paraphrase with no relevant
  candidate at all, and "which vendor had the lowest on-time rate?", which needs all 4
  vendors at once (planned: structured query path).

The prototype's own 8 questions (`data/eval/notebook_eval_8.json`, recovered from the
notebook) score 8/8 with every method; the 7/8 in the prototype README doesn't reproduce,
so it isn't used as a "before" number.

Reproduce: `shipment-rag ingest && shipment-rag eval --mode all --report report.md`, and
`python scripts/retrieval_experiments.py` for the ablation and threshold sweep.

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
data/eval/          evaluation sets; reports/ holds generated eval reports
scripts/            secret scanner, retrieval experiments
assets/             screenshots
tests/              unit + API tests (no models / network needed)
notebooks/          original Colab prototype, kept for reference
```

All configuration is via environment variables — see `.env.example`.
Roadmap, decisions and pending work: [PROJECT_STATUS.md](PROJECT_STATUS.md).
