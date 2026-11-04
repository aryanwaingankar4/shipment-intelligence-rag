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
| Rerank | `cross-encoder/ms-marco-MiniLM-L-6-v2` → top-3, junk filter at `MIN_RERANK_SCORE=-10.25` (chosen on a held-out split) |
| LLM | Gemini (`gemini-3.8-flash` default, low thinking level) via `google-genai`, retries with backoff, fails fast if the model is unavailable |
| Exact questions | Counts, lists, totals and field lookups on shipments come from a read-only SQLite table, with no LLM call. Rule-based router; anything it doesn't fully understand goes to the documents |
| Follow-ups | The client sends recent messages and a small LLM call rewrites the question to stand alone. The answer model never sees the conversation, so it can still only answer from passages |
| Streaming | Answers arrive over server-sent events while they are written (`POST /v1/query/stream`) |
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
| How many vials of TL-COLD-9001 were shipped in total? | 650 vials across 3 shipments, computed from the shipment table with no LLM call |

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

### Where the gains come from (all else equal)

Final configuration: hybrid retrieval, section headers on sub-chunks, `MIN_RERANK_SCORE=-10.25`.
Each row switches **one** thing back and keeps everything else at the final setting.

| Change (final ← alternative) | hit@3 all (47) | hit@3 ID-heavy (20) | hit@3 paraphrased (10) | MRR@3 | candidate recall |
|---|---|---|---|---|---|
| **Final** | **42/47** | **20/20** | **6/10** | **0.851** | **46/47** |
| hybrid ← vector-only | 40/47 | 19/20 | 6/10 | 0.809 | 44/47 |
| hybrid ← BM25-only | 40/47 | 20/20 | 4/10 | 0.830 | 41/47 |
| section headers ← off | 41/47 | 19/20 | 6/10 | 0.851 | 46/47 |
| cutoff −10.25 ← −6 | 38/47 | 20/20 | 3/10 | 0.777 | 46/47 |

- **Hybrid vs vector** (only `RETRIEVAL_MODE` differs): +2 questions (40 → 42), MRR
  0.809 → 0.851, candidate recall 44 → 46 of 47. The gain is on exact IDs; paraphrases are
  the same (6/10).
- **The cutoff fix is the largest single change**: −6 wrongly refused 8 answerable
  questions; −10.25 refuses 1, and that one has no relevant candidate anyway. Paraphrases
  go from 3/10 to 6/10.
- **Section headers**: +1 question (an incident whose CAPA half had lost its ID).

### Baseline → final (bundles all three changes)

| Metric | Phase 4 baseline (vector, cutoff −6) | Final |
|---|---|---|
| hit@3, all in-scope | 35/47 (74%) | **42/47 (89%)** |
| hit@3, ID-heavy | 18/20 | **20/20** |
| hit@3, paraphrased | 3/10 | **6/10** |
| MRR@3 | 0.734 | **0.851** |
| candidate recall | 44/47 | **46/47** |
| answerable questions wrongly refused | 8 | **1** |
| off-domain refused before the LLM | 6/6 | 6/6 |

### How the cutoff was chosen (held out)

`shipment-rag tune-threshold` splits the eval set into a fixed, stratified **tune** half
(30 questions) and **test** half (28) and picks the cutoff on the tune half only: the
midpoint that minimises errors (answerable questions below it plus off-domain questions
above it). Tune half: weakest answerable −9.24, strongest off-domain −11.25 → **−10.25**.

On the held-out test half: **0/23 answerable questions refused, 3/3 off-domain refused**.
Caveats: the test half has only 3 off-domain questions, and by chance it contains none of
the retrieval failures (all 4 fell in the tune half), so its 23/23 hit@3 says nothing about
retrieval quality; it only checks the cutoff. An earlier value of −10.9, picked on the full
set, scored one more hit (43/47). That difference is what tuning on your own test set buys
you, which is why it wasn't kept.

### Owner-written paraphrases (independent check)

10 questions written by the project owner in their own words, without looking at the
documents (`data/eval/user_paraphrases.json`, each with expected source files; report:
`data/eval/reports/stage1_e_with_owner_paraphrases.md`):

| | vector | BM25 | hybrid |
|---|---|---|---|
| hit@3, owner questions | 7/10 | 4/10 | **8/10** |
| hit@3, all 57 in-scope questions | 47/57 | 44/57 | **50/57** |

Both hybrid misses have the right passage among the candidates, but the cross-encoder
ranks it low: "how many days can a frozen item last?" (the shelf-life passage ranks 6th
of 12; "last" ≠ "shelf life") and "which company delivery is fast?" (a comparison across
all vendors, ranked 10th). One question is a deliberate trap: no fine for storage-rule
violations exists in the documents, so the right answer is to say so; the generation eval
checks that. Re-tuning the cutoff with the owner's tune-half questions included chose
−10.25 again.

### Findings worth knowing

- **Vector and BM25 fail on different questions.** BM25 is perfect on IDs but finds nothing
  for "how often do we get the exterminators in?"; vector handles paraphrases but misses
  `SH-2024-006`. Hybrid gets the right chunk to the reranker for 46/47 questions
  (vector 44, BM25 41).
- **The cross-encoder is now the bottleneck on paraphrases.** With no cutoff it lifts MRR
  (0.766 → 0.858) and ID accuracy, but on paraphrases it does slightly *worse* than plain
  RRF order (7/10 vs 8/10). A larger `MiniLM-L12` reranker was mixed: same hit@3, better
  MRR (0.887 vs 0.858), worse on IDs (19/20).
- **The cutoff is a cost saver, not a guarantee.** Paraphrased questions score almost as
  low as off-topic ones, and on-topic unanswerables ("status of SH-2024-011?") score high.
  The grounded prompt does the real refusing; the next stage measures it.
- **Still failing:** "which vendor had the lowest on-time rate?" (needs all 4 vendors at
  once; planned structured query path) and 4 paraphrases (planned: LLM query rewriting).

The prototype's own 8 questions (`data/eval/notebook_eval_8.json`, recovered from the
notebook) score 8/8 with every method; the 7/8 in the prototype README doesn't reproduce,
so it isn't used as a "before" number.

Reproduce (reports in `data/eval/reports/`):
`shipment-rag ingest`, then `shipment-rag eval --mode all --report report.md`,
`shipment-rag tune-threshold`, and `python scripts/retrieval_experiments.py`.
Owner-written paraphrases go in `data/eval/user_paraphrases.json` and are included automatically.

### Exact shipment questions (no LLM)

A language model is the wrong tool for "how many" or "which shipments": retrieval only returns the
top 3 passages, so an answer can't see all 10 records. Those questions now go to a table built from
`data/docs/shipment_log.txt`, and the API and UI show which path answered.

- **Safe by construction:** neither users nor the LLM write SQL. There is a fixed set of query
  templates with bound parameters, and the connection refuses anything but reads.
- **Conservative router:** plain rules, and it only claims a question if it understands every word.
  "How many shipments went to Tokyo?" and "breach rate in Q3" go to the documents rather than being
  answered as if the unknown part weren't there.

| Check (offline, in `pytest`) | Result |
|---|---|
| Table totals vs the log's own summary line (10 total, 7 delivered, 1 quarantined, 1 rejected, 1 delayed, 2 breaches) | match |
| `data/eval/structured_eval.json`: table questions answered correctly | 29/29 |
| Same file: questions that must stay on the document path | 19/19 |
| All 102 existing eval questions: claimed by the table | 10, all correct when checked against the log |

`structured_eval.json` was written alongside the router, so 29/29 and 19/19 are regression tests, not
an independent measure. The 102 existing questions predate the router, which makes them the better
check that it doesn't over-claim. It has one data source so far: only the shipment log is a table.

### Running the heavy evals

The local judge needs about 6 GB of free RAM. Without that, `notebooks/colab_eval_runner.ipynb` runs the
same evals on Colab's free GPU and gives back a zip of reports and saved answers.

### Follow-up questions

"Who accepted it?" can't be searched for. With `REWRITE_PROVIDER` set, it is rewritten into a standalone
question first, and that is what the table, retrieval and the answer model see. If the rewrite fails, or
adds a shipment ID or product code nobody mentioned, the question is used as written. A question with no
conversation makes no extra call and produces exactly the same prompt as before.

| First run: 14 follow-ups in `data/eval/followups.json`, local `qwen2.5:7b` | Result |
|---|---|
| Right passage in the top 3 (or correctly refused), question as written | 10/14 |
| Same, rewritten | 13/14 |
| Rewrite names what it should and nothing it shouldn't | 14/14 |

These cases were written alongside the code, so this shows the mechanism works rather than measuring
its accuracy. They and the prompt were committed before the first run and not tuned afterwards. Details and the one miss:
`data/eval/reports/stage4_rewrite.md`. **Not measured yet:** whether rewriting also fixes the paraphrase
weakness above, because that part of the run was stopped when the machine ran out of memory.

## API

```http
POST /v1/query
X-API-Key: <key>            # required when API_KEYS is set
{"question": "Which shipments breached temperature?", "custom_instructions": "",
 "history": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

`history` is optional, up to 20 messages, oldest first. The client keeps it: the server stores no
conversations. `POST /v1/query/stream` takes the same body and sends server-sent events: `start` {route,
rewritten_question, request_id}, then `delta` {text} pieces, then `done` with the full response below, or
`error` {detail} instead of `done`. The status is already 200 by the time a model fails, so only `done`
means the answer is complete.

Response: `answer`, `refused`, `route` (`documents` or `shipment_table`), `sources[] {source,
section, rerank_score, snippet}`, `structured_query {template, params, rows}` (table answers only),
`rewritten_question` (when a rewrite changed what was searched for), `latency_ms`, `request_id`. Ops
endpoints: `GET /health` (liveness), `GET /ready`
(index reachable and non-empty).

## Layout

```
src/shipment_rag/   config, ingestion/, embeddings, vectorstore, retrieval, structured/, rewrite,
                    llm, pipeline, api/, cli
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
