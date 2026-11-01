# Project Status — Shipment Intelligence RAG

> Living document. Update it at the end of every phase: what was built, what changed,
> what's verified, and what's next. Newest phase at the top of the history section.

| | |
|---|---|
| **Current version** | 0.3.0 |
| **Current phase** | Stage 1 done (hybrid search) — awaiting review → next: Stage 2 generation eval |
| **Branches** | `phase-4-productionization` → PR #1 (open, CI green); `stage-1-hybrid-search` (on top of it, local) |
| **Last updated** | 2026-11-01 |
| **Owner** | Aryan Waingankar |
| **Repo** | github.com/aryanwaingankar4/shipment-intelligence-rag |

---

## 1. What this project is

An AI assistant for a (fictional) pharmaceutical distributor. Staff ask natural-language
questions about products, shipments, temperature breaches, vendors, incidents and GDP
compliance; the system retrieves the relevant passages from company documents and has an
LLM answer **only** from those passages, with citations, refusing anything out of scope.

**Knowledge base** (`data/docs/`, 5 documents → 38 chunks):

| File | Contents |
|---|---|
| `product_manual.txt` | 5 SKUs (cold chain, ambient, frozen) — storage, shelf life, packaging, excursion rules |
| `compliance_guide.txt` | GDP SOP: documentation, temperature monitoring, warehouse, delays, recalls, penalties |
| `shipment_log.txt` | 10 FY2024 shipments with temps, status, breaches |
| `vendor_list.txt` | 4 vendors with SLAs, on-time rates, ratings, escalation matrix |
| `incident_report_2024.txt` | RCA + CAPA for the 2 temperature breaches |

## 2. Current architecture (v0.3.0)

```
Streamlit UI (ui/)  ──HTTP + X-API-Key──▶  FastAPI (/v1/query)  ──▶  Gemini (google-genai)
                                               │
                        ┌── MiniLM embed → Chroma top-8 ──┐
                 query ─┤                                 ├─ RRF → 12 → cross-encoder → top-3
                        └── BM25 (in-memory) top-8 ───────┘          (junk filter −10.9)
                                               │
                                        ChromaDB server  ◀──  ingest job (incremental)
```

| Layer | Tech | Notes |
|---|---|---|
| Config | pydantic-settings | Every tunable is an env var (`.env.example`) |
| Chunking | custom section splitter + LangChain recursive splitter | Content-hash chunk IDs; section header repeated on sub-chunks (`CHUNK_SECTION_HEADERS`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | 384-d, normalized, baked into image |
| Vector DB | ChromaDB 1.5.9 | Embedded (dev) or HTTP server (Docker/prod) |
| Lexical | `rank-bm25` (BM25Okapi), ID-aware tokenizer | Built in memory from Chroma at first query, rebuilt when the chunk count changes |
| Fusion | Reciprocal rank fusion (k=60) | `RETRIEVAL_MODE=vector\|bm25\|hybrid` (default hybrid) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | `MIN_RERANK_SCORE=-10.9` filters obvious junk; not a refusal guarantee |
| LLM | Gemini `gemini-3.8-flash`, thinking level `low`, 2048 max output tokens | tenacity retries on 408/429/5xx, 30 s timeout, warns on `MAX_TOKENS`, fails fast on model 404 |
| API | FastAPI + uvicorn | `/health`, `/ready`, `/v1/query`, request IDs, JSON logs |
| UI | Streamlit | Thin client, HTML-escaped rendering, sources shown |
| Packaging | Docker (API+ingest image, UI image), docker-compose | CPU torch, non-root, healthchecks |
| CI | GitHub Actions | ruff → pytest → ingest + retrieval eval gate → docker builds |

### How to run

- Docker: `cp .env.example .env` (set `GEMINI_API_KEY`) → `docker compose up --build` → UI :8501, API :8080/docs
- Local: see README "Local development"
- Re-index after editing docs: `docker compose run --rm ingest` (or `shipment-rag ingest`)

---

## 3. Phase history

### Work plan (agreed 2026-11-01)

Goal: a live demo plus measurable ML improvements for AI/ML placement interviews.
Deliberately **out of scope for now**: Prometheus/Grafana, OpenTelemetry, multi-tenancy,
queue-based ingestion.

| Stage | Scope | Status |
|---|---|---|
| 0 | Git cleanup + Phase 4 close-out (D1 rename, live checks) | ✅ done (Docker check pending, see §4) |
| 0.5 | Fixes from the owner's live check: truncation, model default + 404, secret scanning | ✅ done |
| 1 | Hybrid search (BM25 + RRF), bigger eval set, vector-vs-hybrid table | ✅ done, awaiting review |
| 2 | Generation eval (`eval-gen`: faithfulness, correctness, citations), nightly workflow | ⬜ |
| 3 | Structured query path for aggregate shipment questions + router | ⬜ |
| 4 | SSE streaming + follow-up questions (query rewriting) | ⬜ |
| 5 | Real login, rate limiting, GHCR images, VM + Caddy deploy, CD | ⬜ (ask before spending money) |

### Stage 1 — Hybrid search (v0.3.0) — ✅ complete, 2026-11-01

**Built**
- `RETRIEVAL_MODE=vector|bm25|hybrid` (default **hybrid**). BM25 via `rank-bm25` with an
  ID-aware tokenizer (`SH-2024-003` is indexed whole *and* as `sh`/`2024`/`003`; emails
  too). The BM25 index is derived from Chroma (single source of truth) and rebuilt when the
  chunk count changes. Vector top-8 and BM25 top-8 are fused with reciprocal rank fusion
  (k=60); the top 12 go to the cross-encoder.
- **Section headers on sub-chunks** (`CHUNK_SECTION_HEADERS=true`): found while debugging.
  The INC-2024-01 section is > 700 chars, so its CAPA half became a chunk starting
  "Disposition: …" with no incident ID. Neither BM25 nor the reranker could link it to
  INC-2024-01, so hybrid surfaced the *other* half instead.
- Eval set 12 → **58** questions: +20 ID-heavy (shipments, batches, SKUs, vendor codes,
  deviation/destruction records), +10 paraphrased in everyday wording, +6 off-domain incl.
  a prompt-injection attempt, +5 near-domain unanswerables (non-existent `SH-2024-011`,
  `TL-COLD-9999`, FY2023). `expect_all` for answers that appear in several records.
- `shipment-rag eval --mode all --report …`: hit@3, MRR@3, candidate recall, false
  refusals, refusal rates and score margins per mode; per-question tables. CI runs it and
  uploads the report as an artifact.
- `scripts/retrieval_experiments.py`: reranker ablation + threshold sweep, committed as
  `data/eval/reports/stage1_experiments.md`.

**Before → after** (same 58 questions, same index and models; reports in `data/eval/reports/`)

| Metric | Phase 4 baseline (vector, −6) | + hybrid | + section headers | + threshold −10.9 (final) |
|---|---|---|---|---|
| hit@3, all in-scope (47) | 74% | 79% | 81% | **91%** |
| hit@3, ID-heavy (20) | 90% | 95% | 100% | **100%** |
| hit@3, paraphrased (10) | 30% | 30% | 30% | **70%** |
| MRR@3 | 0.734 | 0.777 | 0.777 | **0.858** |
| candidate recall | 94% | 98% | 98% | **98%** |
| false refusals | 8 | 8 | 8 | **0** |
| off-domain refused | 100% | 100% | 100% | 100% |

Final config, vector vs hybrid (only `RETRIEVAL_MODE` differs): hit@3 87% → 91%,
ID-heavy 95% → 100%, MRR 0.816 → 0.858, candidate recall 94% → 98%.

**What the experiments showed**
1. *The first 48-question run was misleading.* BM25 alone matched hybrid on hit@3 and beat it
   on MRR. The cause was the eval set: questions written from the documents share their
   vocabulary, which favours lexical search. Adding a paraphrase slice exposed the
   difference. BM25 gets 40% on paraphrases and finds no candidates at all for some;
   vector gets 70–80%.
2. *Hybrid's value is candidate recall* (vector 94%, BM25 87%, hybrid 98%): the two
   retrievers fail on different questions.
3. *The cross-encoder is the bottleneck for paraphrases.* With no threshold it raises MRR
   (RRF order 0.766 → 0.858) and ID accuracy (95% → 100%), but scores colloquial questions
   very low (−9 to −10.7) and does slightly worse than plain RRF order on them (70% vs 80%).
   `ms-marco-MiniLM-L12` was not better (89% hit@3, ID-heavy 95%), so L6 stays.
4. *The −6 / −9 thresholds were wrongly refusing real questions.* At −6, 8 of 47 in-scope
   questions were refused. Swept −6 … −11: −10.9 is the highest value with zero false
   refusals that still refuses all off-domain questions before the LLM. The margin on the
   off-domain side is only ~0.24, and it was tuned on the same set it is reported on.
5. One small regression from section headers: `TL-COLD-9001` storage moved from rank 1 to
   rank 2 (both halves of that product section now carry its ID). Still in the top 3.

**Still failing (final config):** "Which vendor had the lowest on-time rate?" needs all four
vendor records (Stage 3); 2 paraphrases ranked outside the top 3 and 1 with no relevant
candidate.

### Stage 0.5 — Live-check fixes (v0.2.2) — ✅ complete, 2026-11-01

Triggered by the owner's live run with `GEMINI_MODEL=gemini-3.8-flash`.

1. **Truncated answers.** Root cause: `LLM_MAX_OUTPUT_TOKENS=512`, and on Gemini 3.x
   thinking tokens are charged against that same budget. Fix: default raised to **2048**,
   new `LLM_THINKING_LEVEL` (default **`low`**; grounded Q&A needs little reasoning) and
   `LLM_THINKING_BUDGET` for 2.5-series models. A `finish_reason == MAX_TOKENS` now logs a
   warning with thought vs answer token counts and a hint. Note: a local `.env` copied
   from the old template still pinned 512; that override was the actual cause in the run.
2. **Model default + fail fast.** Default `GEMINI_MODEL=gemini-3.8-flash` (confirmed present
   in the live model list). `check_model()` (a metadata call, no generation quota) runs at
   API startup and before `shipment-rag ask`; a 404 raises `ModelNotFoundError` with
   instructions instead of failing every request. New `shipment-rag models` lists the
   models the key can use. D3 updated.
3. **Secret scanning.** The Stage 0 incident (an `AQ.`-format key in `.env.example`) slipped
   past a grep that only knew `AIza…`. New dependency-free `scripts/scan_secrets.py`:
   known formats (Google `AIza`/`AQ.`, GitHub, `sk-` style, AWS, Slack, private keys) plus
   generic high-entropy detection (secret-named assignments and long mixed-case tokens),
   masked output, exact-match allowlist (`.secret-scan-allow`). It runs in CI on all tracked
   files, as a unit test, and as an opt-in pre-commit hook (`.githooks/pre-commit`).

### Stage 0 — Git cleanup + Phase 4 close-out (v0.2.1) — ✅ complete, 2026-11-01

- Remote URL no longer embeds a token; `gh auth login` used (done by owner).
- Inspected divergence: `origin/main` has 7 web-UI commits (screenshots in `assets/`
  under clean names, a 336-line Colab-era README, an empty `assets/placeholder.md`);
  local `main` has 1 commit adding the **byte-identical** screenshots under `screenshots/`.
- **D4 resolved:** rebased the branch onto `origin/main`, dropping local commit `aa4e065`
  (duplicate screenshots). README now uses `assets/*.png`; example queries carried over from
  the remote README; empty `assets/placeholder.md` deleted. Remote history untouched;
  merged via PR after owner review.
- **Notebook baseline recovered:** the prototype's 8 retrieval questions (Cell 7) are now
  `data/eval/notebook_eval_8.json`. On the current index they score **8/8 with the
  notebook's own method** (vector top-3, no rerank) and **8/8 with the current pipeline**.
  The 7/8 in the old README does not reproduce, so it's labelled as notebook-reported, not
  used as a before/after. Both existing sets are saturated, which Stage 1 must fix.
- All Phase 4 work committed on branch `phase-4-productionization`; audited that no `.env`,
  `.venv`, Chroma DB, model weights, logs or secrets are tracked. Added `.gitattributes`
  (LF line endings, so files used inside Linux containers aren't converted to CRLF on Windows).
- **D1 resolved:** "TraceLink BioMed" → "NexaChain BioMed" across `data/docs/*` incl. email
  domain `@nexachain-biomed.com` (13 occurrences). SKU codes (`TL-COLD-9001` …) kept as
  product IDs. New unit test guards against the old name returning.
- Incremental re-index after the rename: 12 chunks replaced, 26 untouched.

**Found during close-out (baseline cases for Stage 1):**

| Question | Problem |
|---|---|
| "Which email should I use to escalate a temperature breach to the QA head?" | **Recall miss** — the `VENDOR ESCALATION MATRIX` chunk is not in the vector top-8 at all; breach/incident chunks crowd it out |
| "What is the quality contact email for CoA requests?" | Correct chunk is rank 1 but scores −6.54, **below `MIN_RERANK_SCORE` −6.0** → would be wrongly refused. The Phase 4 margin check only covered strong in-scope questions |

These were kept out of the CI-gated set for now so the original 12 remain a clean
before/after comparison; they go into the Stage 1 expanded eval.

### Phase 4 — Productionization (v0.2.0) — ✅ complete, 2026-11-01

**Goal:** turn the Colab notebook into a codebase that deploys, scales and is maintained
like a real service.

What was done:
- Split the 2,100-line notebook into a package `src/shipment_rag/` with one responsibility
  per module (config, chunker, indexer, embeddings, vectorstore, retrieval, llm, pipeline, api, cli).
- Moved the knowledge base out of Python strings into `data/docs/*.txt`.
- New **FastAPI service**: models loaded once per process, typed request/response, liveness and
  readiness probes, optional API-key auth, request-ID propagation, structured JSON logs.
- **Streamlit UI is now a separate service** that calls the API (UI and API scale independently).
- **Incremental ingestion**: re-running only embeds new/changed chunks, deletes removed ones.
- **Docker + docker-compose**: Chroma server, one-shot ingest job, API, UI; CPU-only torch;
  model weights baked in; non-root users; healthchecks; UI gets no LLM secret.
- **Tests** (unit + API + indexer integration with in-memory Chroma) — no models or network needed.
- **CI** on GitHub Actions including a retrieval-quality gate that fails the build if accuracy drops.
- Original notebook preserved at `notebooks/shipment_chatbot_colab.py` for reference.

Bugs / weaknesses fixed from the prototype:

| # | Problem in notebook | Fix |
|---|---|---|
| 1 | Out-of-scope guard was dead code: retrieval always returns top-k, so `if not chunks` never fired and every question cost an LLM call | Cross-encoder score threshold (`MIN_RERANK_SCORE`); refuse before calling the LLM |
| 2 | Day-3 retrieval read `meta["section"]` but chunks stored `section_name` → section always blank | Reads `section_name`; sections shown in prompt and API |
| 3 | HTML injection: user and model text inserted into `unsafe_allow_html` unescaped | `html.escape` on all dynamic content |
| 4 | Two `log_query` versions with different CSV schemas writing to the same file | Structured JSON audit log (`rag_query` events) to stdout |
| 5 | Every ingestion deleted and rebuilt the whole collection | Content-hash IDs + incremental upsert/delete |
| 6 | `google-generativeai` SDK is deprecated | Migrated to `google-genai` with native `system_instruction` |
| 7 | Streamlit app duplicated the pipeline (and lacked the retry logic) | Single pipeline behind the API |
| 8 | Hard-coded Drive paths, demo credentials, model names | Environment-based config |
| 9 | Deployment = Colab + ngrok with CORS/XSRF disabled; dies when the Colab session ends | Containers that run anywhere |

### Phase 3 — LLM integration + UI + demo deploy (Colab "Day 3") — ✅ complete

- Gemini 2.5 Flash via Colab Secrets, retry with exponential backoff on 429/connection resets.
- `query_rag()` grounded prompt + "I don't know" refusal string.
- 3-question quota-lite eval (factual, multi-doc, out-of-scope).
- Streamlit app "NexaChain BioMed / Nova": login, home dashboard, personalization, chat.
- Launched from Colab through an ngrok tunnel. Screenshots in `screenshots/`.

### Phase 2 — Retrieval pipeline (Colab "Day 2") — ✅ complete

- 5 richer documents with `===` section headers.
- Section-aware chunking (max 700 chars, 80 overlap) + rich metadata.
- MiniLM embeddings, ChromaDB persisted on Google Drive, MD5 dedup.
- 8-question retrieval test (keyword in top-3).
- Cross-encoder rerank (top-8 → top-3); CSV query logging.

### Phase 1 — Setup + mock data (Colab "Day 1") — ✅ complete

- Colab + Drive folder structure, library install, 3 initial mock documents.

---

## 4. Verification

### Stage 1 (v0.3.0) — 2026-11-01

| Check | Result |
|---|---|
| `ruff check src tests ui scripts` | ✅ clean |
| `pytest` | ✅ 58/58 (new: 8 hybrid-retrieval tests on a fake store, 3 eval tests — case kinds, `expect_all`, metrics — and 1 section-header chunker test) |
| `shipment-rag eval --mode all` (58 q), final config | ✅ gate 92% (hybrid) ≥ 90%; vector 89%, BM25 87% |
| `shipment-rag eval --file data/eval/notebook_eval_8.json` | ✅ 8/8 |
| Ablation reports regenerated from committed code | ✅ `data/eval/reports/stage1_{a,b,c}_*.md`, `stage1_experiments.md` |
| `scan_secrets.py --all` | ✅ clean |
| Determinism: no-headers run repeated | ✅ identical metrics on both runs |

### Stage 0.5 (v0.2.2) — 2026-11-01

| Check | Result |
|---|---|
| `ruff check src tests ui scripts` | ✅ clean |
| `pytest` | ✅ 46/46 (new: 7 LLM-client tests with a stubbed SDK client, 18 secret-scanner tests) |
| Live `shipment-rag ask "Why was SH-2024-007 rejected?"` on `gemini-3.8-flash` | ✅ complete answer (no truncation), grounded, cited [1]/[2]; sources: incident report + 2 shipment-log sections; 2 transient errors retried automatically |
| Live model 404 (`GEMINI_MODEL=gemini-9.9-nonexistent`) | ✅ exits with the `ModelNotFoundError` message, no retries |
| `shipment-rag models` | ✅ lists `gemini-3.8-flash` |
| `scan_secrets.py --all` on tracked files | ✅ clean (1 known non-secret allowlisted: notebook Drive file ID) |
| Scanner catches the Stage 0 key format on a real `.env` | ✅ flagged, masked |

### Stage 0 (v0.2.1) — 2026-11-01

| Check | Result |
|---|---|
| `ruff check src tests ui` | ✅ clean |
| `pytest` | ✅ 20/20 (new: brand-name guard) |
| Re-ingest after rename | ✅ 38 chunks; 12 replaced / 26 unchanged |
| `shipment-rag eval` (12 cases) | ✅ 12/12 = 100 % — unchanged by rename, top rerank scores within ±0.2 of before |
| Notebook 8-question set, notebook method (vector top-3) | 8/8 (old README claimed 7/8 — not reproducible) |
| Notebook 8-question set, current pipeline | 8/8 |
| Staged-file audit (no `.env`, `.venv`, `chroma_db`, weights, logs, secrets) | ✅ |
| Rebase onto `origin/main`: diff vs pre-rebase branch = screenshots location + README path only | ✅ |
| Live Gemini answer (`shipment-rag ask …`) | ⏸ owner to run (no key on this machine) |
| `docker compose up --build` + `/ready` + one query | ⏸ owner to run (Docker Desktop not running) |

### Phase 4 (v0.2.0)

Run locally on 2026-11-01 (Windows 11, Python 3.12, CPU):

| Check | Result |
|---|---|
| `ruff check src tests ui` | ✅ clean |
| `pytest` (19 tests: chunker, pipeline, API, indexer on in-memory Chroma) | ✅ 19/19 pass, ~1 s |
| Real ingest (5 docs) | ✅ 38 chunks in 8.3 s; re-run → 0 added / 38 unchanged (incremental works) |
| `shipment-rag eval` (10 in-scope + 2 out-of-scope, real models) | ✅ 12/12 = 100 % |
| Guardrail margin | Off-topic questions score ≈ −11; in-scope top chunks ≥ −3 → threshold −6.0 has headroom |
| API with real models: `/health`, `/ready` (38 chunks), missing key → 401, off-topic → refused in ~0.3 s without an LLM call, Gemini failure → clean 503 with no internal details leaked | ✅ |
| Structured JSON audit + request logs with matching `request_id` | ✅ |
| Streamlit UI (AppTest, stubbed API): bad login rejected, login, chat, `<script>` escaped, sources shown | ✅ |
| `docker compose config` | ✅ valid (chroma, ingest, api, ui) |

**Not yet verified:**
- ⚠️ A real Gemini answer through the new `google-genai` client (no API key on this machine). First thing to do: put the key in `.env` and run `shipment-rag ask "Why was SH-2024-007 rejected?"`.
- ⚠️ `docker compose up --build` (Docker Desktop wasn't running). CI builds both images on push.

---

## 5. Open decisions

| # | Decision | Recommendation |
|---|---|---|
| D1 | ~~Brand mismatch (docs "TraceLink BioMed" vs UI "NexaChain BioMed")~~ | ✅ **Resolved in Stage 0** — docs renamed to NexaChain BioMed |
| D2 | Where to host first | See Phase 5 options below. Cheapest credible path: one small VM + docker compose + Caddy (HTTPS). |
| D4 | ~~Reconcile with `origin/main`~~ | ✅ **Resolved in Stage 0**: rebased, `aa4e065` dropped, PR to `main` open for owner review |
| D5 | **Fixing the paraphrase weakness** (70% hit@3; cross-encoder scores colloquial questions very low) | Options, cheapest first: (a) LLM query rewriting into document vocabulary, which fits Stage 4 (it already adds rewriting for follow-ups) but costs one LLM call per question; (b) a stronger reranker such as `BAAI/bge-reranker-base` (~280M params, about 3× the image weight and CPU latency), which needs owner approval as a heavier model; (c) more/better embeddings. Recommendation: (a) in Stage 4, measured on the paraphrase slice. |
| D3 | Gemini model lifetime | Default is now **`gemini-3.8-flash`** (Stage 0.5). Google retires models regularly: run `shipment-rag models` before each release. A retired model now fails fast at startup with a clear message instead of failing every request. |

---

## 6. Pending work / roadmap

Priority: **P0** = before anyone outside the team uses it, **P1** = next, **P2** = later.

### Phase 5 — Security + first real deployment (next)
- [x] **P0** Remove the GitHub token embedded in the local git remote URL; use `gh auth login` instead. *(done by owner, Stage 0)*
- [ ] **P0** Confirm the Gemini key exposed in Stage 0 (sha256 prefix `67023945746a`) is **deleted** in AI Studio, not just superseded, and that `.env` holds the new key.
- [ ] **P1** Enable the secret-scan pre-commit hook locally: `git config core.hooksPath .githooks`.
- [ ] **P0** Replace the Streamlit demo login with real auth (OIDC/SSO via an auth proxy such as oauth2-proxy, or Streamlit's native `st.login`).
- [ ] **P0** Set `API_KEYS` / `API_KEY` in every non-dev environment; store secrets in a secret manager, not `.env` on disk.
- [ ] **P0** HTTPS in front of UI and API (Caddy / cloud load balancer). Don't expose Chroma publicly.
- [ ] **P1** Push images to a registry (GHCR) from CI on `main`, tagged with the git SHA.
- [ ] **P1** Pick a host and deploy:
  - *Portfolio/demo*: single VM (GCP e2-medium / AWS Lightsail 4 GB) running `docker compose` + Caddy.
  - *Company-style*: API + UI on Cloud Run / ECS Fargate / Azure Container Apps (stateless, autoscaling); vector DB as a managed or dedicated service (Chroma Cloud, Qdrant Cloud, or pgvector on managed Postgres); ingest as a scheduled job.
- [ ] **P1** CD workflow: deploy on tag, smoke-test `/ready` + one query, auto-rollback on failure.
- [ ] **P1** Rate limiting per API key (Redis-backed, e.g. `slowapi` or at the gateway) to protect Gemini quota.

### Phase 6 — Observability + reliability
- [ ] **P1** Metrics: Prometheus `/metrics` (latency histograms per stage, refusal rate, LLM errors, token usage) + Grafana dashboard.
- [ ] **P1** Tracing: OpenTelemetry spans for embed / vector query / rerank / LLM.
- [ ] **P1** Persist the audit log (`rag_query` events) to a database or log store for analytics and compliance audit.
- [ ] **P2** LLM fallback model (e.g. `gemini-2.5-flash-lite`) when the primary returns 429/5xx.
- [ ] **P2** Response cache for repeated questions (keyed on normalized question + index version).
- [ ] **P2** Alerting on error-rate and p95 latency SLOs.

### Phase 7 — Scale
- [ ] **P1** Lock file for reproducible builds (`uv lock` / pip-tools).
- [ ] **P2** Ingestion as a queue-driven worker (upload → queue → embed) instead of a one-shot job; support PDF/DOCX (pypdf already proven in Phase 2).
- [ ] **P2** Stream answers (SSE) for faster perceived latency.
- [ ] **P2** Move embeddings/reranker to a dedicated inference service or GPU if QPS grows; or use a hosted embedding API.
- [ ] **P2** Multi-tenant: one collection (or metadata filter) per customer, tenant ID from auth.
- [ ] **P2** Load test (k6/Locust) and document capacity per API replica.

### Phase 8 — Answer quality
- [ ] **P1** Generation eval: grow `data/eval/` into a golden Q&A set and score faithfulness / answer correctness (e.g. RAGAS or LLM-as-judge), run nightly — not on every PR, to save quota.
- [x] **P1** Tune `MIN_RERANK_SCORE` on a larger set of in-scope vs out-of-scope questions. *(Stage 1: −10.9)*
- [ ] **P1** Re-validate `MIN_RERANK_SCORE` on held-out questions (Stage 2 golden set); it was tuned on the reported set.
- [x] **P2** Hybrid search (BM25 + vectors). *(Stage 1)*
- [ ] **P1** Paraphrase robustness (D5): query rewriting and/or stronger reranker, measured on the paraphrase slice.
- [ ] **P2** BM25 index is rebuilt in memory per API process from the full collection; fine for thousands of chunks, revisit (e.g. Chroma full-text search or a dedicated index) beyond that.
- [ ] **P2** Metadata filters (product code, shipment ID, date) extracted from the question.
- [ ] **P2** Structured data path: shipment log as a table (SQL tool) for aggregate questions ("how many breaches in Q1?").
- [ ] **P2** Conversation memory (follow-up questions) with query rewriting.

### Phase 9 — Product features
- [ ] **P2** Clickable citations that open the source section.
- [ ] **P2** Thumbs up/down feedback stored with the request ID, feeding the eval set.
- [ ] **P2** Admin page: upload/replace documents, trigger re-index, view usage.
- [ ] **P2** Role-based access (QA vs warehouse vs vendor views).

---

## 7. Decision log

| Date | Decision | Why |
|---|---|---|
| 2026-11-01 | Default `RETRIEVAL_MODE=hybrid` over BM25-only | Equal on hit@3 overall but BM25 collapses on paraphrases (40% vs 70%) and has the lowest candidate recall (87% vs 98%) |
| 2026-11-01 | BM25 derived from Chroma in memory (rank-bm25), not a second store | One source of truth, no new service; corpus is tiny |
| 2026-11-01 | RRF to choose reranker candidates instead of score blending | Rank-based, no score calibration between BM25 and cosine needed; standard k=60 |
| 2026-11-01 | Repeat section header on sub-chunks | Fixed an ID-linking failure (INC-2024-01 CAPA) at the cost of one rank-1→2 swap |
| 2026-11-01 | Keep `ms-marco-MiniLM-L-6` reranker | L-12 was not better on this set (89% vs 91% hit@3, ID 95% vs 100%) |
| 2026-11-01 | `MIN_RERANK_SCORE` −6.0 → −10.9 and documented as a cost filter | −6 wrongly refused 8/47 in-scope questions; the grounded prompt is the real refusal mechanism |
| 2026-11-01 | Add a paraphrase slice to the eval | Document-derived questions were biased towards lexical overlap and hid hybrid's value |
| 2026-11-01 | Default `gemini-3.8-flash`, thinking level `low`, 2048 output tokens | Thinking tokens share the output budget on 3.x; grounded answers need little reasoning, so a low level keeps latency and cost down while 2048 leaves headroom |
| 2026-11-01 | Validate the model at startup with `models.get` | Costs no generation quota; turns a silent per-request failure into one clear startup error |
| 2026-11-01 | In-repo, dependency-free secret scanner instead of gitleaks/detect-secrets | No new dependency or binary; known formats plus entropy covers this repo; easy to explain. Revisit if the repo grows |
| 2026-11-01 | Rename brand to NexaChain BioMed in docs; keep `TL-` SKU codes | Persona and knowledge agree; no real company referenced. SKUs are IDs used across shipments/eval |
| 2026-11-01 | Rebase onto `origin/main` instead of merge; drop duplicate-screenshot commit | Linear history; remote already had identical images under cleaner names |
| 2026-11-01 | Don't present the notebook's 7/8 as a "before" number | Different eval set and not reproducible (re-run gives 8/8); comparisons must be on the same set and the same index |
| 2026-11-01 | Keep newly found failing retrieval cases out of the CI gate until Stage 1 | Keeps the original 12-case set as a clean baseline; failures are documented, not hidden |
| 2026-11-01 | `.gitattributes` with `eol=lf` | Repo developed on Windows, run in Linux containers |
| 2026-11-01 | Separate UI and API services | Independent scaling and deploys; the API can serve other clients (Slack bot, mobile) |
| 2026-11-01 | Chroma in server mode for Docker/prod, embedded for dev | Multiple API replicas need a shared index; dev stays zero-setup |
| 2026-11-01 | Content-hash chunk IDs | Idempotent, incremental ingestion; safe to run on every deploy |
| 2026-11-01 | Bake model weights into the image | No runtime downloads, faster cold starts, reproducible |
| 2026-11-01 | Retrieval-only eval gate in CI | Catches retrieval regressions on every PR without spending LLM quota |
| 2026-11-01 | Reranker score threshold as out-of-scope guardrail | Refuses obvious off-topic questions with zero LLM cost |
| Phase 3 | Gemini 2.5 Flash, temperature 0.1 | Free tier for prototyping, low temperature for factual answers |
| Phase 2 | Section-aware chunking | Keeps each product/shipment/SOP record intact → higher retrieval precision |
