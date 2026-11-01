# MIN_RERANK_SCORE: tuned on the tune split, reported on the test split

Mode `hybrid`, reranker `cross-encoder/ms-marco-MiniLM-L-6-v2`, CHUNK_SECTION_HEADERS=True. Tune split: 30 questions, test split: 28.

Rule: the midpoint between observed top scores that minimises errors on the tune split (answerable in-scope questions below it + off-domain questions above it), ties broken by the widest margin.

Tune result: `{'threshold': -10.25, 'tune_errors': 0, 'margin': 1.0, 'weakest_answerable': -9.2434, 'strongest_off_domain': -11.2476, 'n_answerable': 23, 'n_off_domain': 3}`

## Held-out test split

| Metric | chosen -10.25 | configured -10.25 | no threshold |
|---|---|---|---|
| hit@3 — all in-scope | 23/23 (100%) | 23/23 (100%) | 23/23 (100%) |
| hit@3 — ID-heavy | 10/10 (100%) | 10/10 (100%) | 10/10 (100%) |
| hit@3 — paraphrased | 5/5 (100%) | 5/5 (100%) | 5/5 (100%) |
| hit@3 — owner-written paraphrases | n/a | n/a | n/a |
| MRR@3 — all in-scope | 0.935 | 0.935 | 0.935 |
| MRR@3 — ID-heavy | 0.900 | 0.900 | 0.900 |
| MRR@3 — paraphrased | 0.900 | 0.900 | 0.900 |
| candidate recall (relevant chunk reached reranker) | 23/23 (100%) | 23/23 (100%) | 23/23 (100%) |
| false refusals (in-scope refused) | 0 | 0 | 0 |
| off-domain refused | 3/3 (100%) | 3/3 (100%) | 0/3 (0%) |
| near-domain refused at retrieval (informational) | 0/2 (0%) | 0/2 (0%) | 0/2 (0%) |
| min top score, answerable in-scope | -9.97 | -9.97 | -9.97 |
| max top score, off-domain | -11.14 | -11.14 | -11.14 |
| **gate score** | **100%** | **100%** | **88%** |
