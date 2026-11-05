# Generation eval — 2026-11-05

Golden set `data/eval/golden_qa.json` (26 questions). Answers: `gemini-3.8-flash` (thinking low, temperature 0.1). Judge: `qwen2.5:7b` (local via Ollama, digest 845dbda0ea48), temperature 0, rubric `4fd3b947`. Citations: NLI `cross-encoder/nli-deberta-v3-small` (entailment >= 0.9), one citation at a time. Retrieval: hybrid, MIN_RERANK_SCORE=-10.25, top-3.

> **Partial run:** 22 of 26 questions not scored (LLM errors / quota). Re-running resumes from the cache.

## Summary

| Metric | Result |
|---|---|
| **Answerable questions** |  |
| fully correct (judge: 2/2) | 4/4 (100%) |
| partly correct (1/2) | 0/4 (0%) |
| mean correctness | 100% |
| key facts present in the answer (string match) | 9/9 (100%) |
| wrongly refused | 0/4 (0%) |
| **Faithfulness** (no unsupported claims, all generated answers) | 4/4 (100%) |
| **Citations** (answered answerable questions) |  |
| answer has ≥1 [n] citation | 4/4 (100%) |
| all citation numbers valid | 4/4 (100%) |
| cites a passage from an expected source file | 4/4 (100%) |
| every cited passage supports its sentence | 3/4 (75%) |
| **Should refuse** (unanswerable / off-domain) |  |
| refused | n/a |
| of which refused at retrieval (no LLM answer call) | n/a |
| **Traps** (documents lack the specific figure) |  |
| handled: says the information isn't in the documents | n/a |
| invented the figure | n/a |

## Judge calibration

Hand-written answers with planted flaws (`data/eval/judge_calibration.json`), scored the same way:

| Case | Planted flaw | Judge checks |
|---|---|---|
| C1 | wrong number | correctness ≤ 0 ✅, faithful = False ✅, citation 1 supported = False ✅ |
| C2 | invented fact | faithful = False ✅, citation 1 supported = True ✅, citation 2 supported = False ✅ |
| C3 | citation points to the wrong passage | correctness ≥ 2 ✅, citation 1 supported = False ✅ |
| C4 | trap answered with an invented fine | correctness ≤ 0 ✅, faithful = False ✅; known limit, not gated: citation 1 supported = False ❌ |
| C5 | none (control) | correctness ≥ 2 ✅, faithful = True ✅, citation 1 supported = True ✅ |
| C6 | none (correct refusal) | correctness ≥ 2 ✅, is_refusal = True ✅ |
| C7 | citation points to a sibling vendor | correctness ≥ 2 ✅, faithful = True ✅, citation 1 supported = False ✅ |
| C8 | one right citation, one wrong | correctness ≥ 2 ✅, faithful = True ✅, citation 1 supported = True ✅, citation 2 supported = False ✅ |
| C9 | citation points to the other incident | correctness ≥ 2 ✅, faithful = True ✅, citation 1 supported = False ✅ |
| C10 | none (control: correct citation to passage 2) | correctness ≥ 2 ✅, faithful = True ✅, citation 1 supported = True ✅ |
| C11 | none (control: paraphrased, two correct citations) | correctness ≥ 2 ✅, faithful = True ✅, citation 1 supported = True ✅, citation 2 supported = True ✅ |
| C12 | none (control: fact from a vendor record) | correctness ≥ 2 ✅, faithful = True ✅, citation 1 supported = True ✅ |

Judge agreement on planted flaws: **35/35** checks (12/12 cases scored).

## Per question

| ID | type | question | correct | faithful | citations | key facts | note |
|---|---|---|---|---|---|---|---|
| G01 | answerable | What are the storage conditions for ImmunoChill Vaccine Via… | 2/2 | yes | all_supported | 2/2 |  |
| G02 | answerable | Which documents must travel with every shipment? | 2/2 | yes | some_unsupported | 4/4 |  |
| G03 | answerable | What is ArcticLine's delivery window for dry-ice frozen shi… | 2/2 | yes | all_supported | 1/1 |  |
| G04 | answerable | What has to happen if an ambient product stays above 30°C f… | 2/2 | yes | all_supported | 2/2 |  |
| G05 | answerable | How often is temperature mapping of the cold rooms repeated? | – | – | – | – | LLMUnavailableError: 503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This … |
| G06 | answerable | Within how many days of delivery can a product be returned? | – | – | – | – | LLMUnavailableError: 503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This … |
| G07 | answerable | What quality rating does Deccan Pharma Logistics have? | – | – | – | – | LLMUnavailableError: 503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This … |
| G08 | answerable | What temperatures were recorded for SH-2024-004? | – | – | – | – | not run (stopped after repeated LLM errors; re-run resumes from cache) |
| G09 | answerable | Which product and batch were shipped in SH-2024-008, and wh… | – | – | – | – | not run (stopped after repeated LLM errors; re-run resumes from cache) |
| G10 | answerable | Who is the contact person for vendor V-004, and what is the… | – | – | – | – | not run (stopped after repeated LLM errors; re-run resumes from cache) |
| G11 | answerable | Which deviation report was raised for SH-2024-003? | – | – | – | – | not run (stopped after repeated LLM errors; re-run resumes from cache) |
| G12 | answerable | Why was SH-2024-003 quarantined, and what corrective action… | – | – | – | – | not run (stopped after repeated LLM errors; re-run resumes from cache) |
| G13 | answerable | Which vendor handled the rejected frozen shipment, and what… | – | – | – | – | not run (stopped after repeated LLM errors; re-run resumes from cache) |
| G14 | answerable | A cold-chain consignment arrived at 11°C. What must we do, … | – | – | – | – | not run (stopped after repeated LLM errors; re-run resumes from cache) |
| G15 | answerable | What were the two temperature breaches in FY2024, and what … | – | – | – | – | not run (stopped after repeated LLM errors; re-run resumes from cache) |
| G16 | answerable | A vendor shipment is more than 48 hours late. Who do we esc… | – | – | – | – | not run (stopped after repeated LLM errors; re-run resumes from cache) |
| G17 | answerable | How long do the frozen sample kits stay usable? | – | – | – | – | not run (stopped after repeated LLM errors; re-run resumes from cache) |
| G18 | answerable | Do we have to keep the old logger calibration paperwork? | – | – | – | – | not run (stopped after repeated LLM errors; re-run resumes from cache) |
| G19 | answerable | Our truck broke down halfway to the customer. What now? | – | – | – | – | not run (stopped after repeated LLM errors; re-run resumes from cache) |
| G20 | answerable | Which vendor has the best on-time delivery rate? | – | – | – | – | not run (stopped after repeated LLM errors; re-run resumes from cache) |
| G21 | refusal | What is the status of shipment SH-2024-011? | – | – | – | – | not run (stopped after repeated LLM errors; re-run resumes from cache) |
| G22 | refusal | What temperature should insulin pens be stored at? | – | – | – | – | not run (stopped after repeated LLM errors; re-run resumes from cache) |
| G23 | refusal | Who is the CEO of NexaChain BioMed? | – | – | – | – | not run (stopped after repeated LLM errors; re-run resumes from cache) |
| G24 | refusal | What's the capital of France? | – | – | – | – | not run (stopped after repeated LLM errors; re-run resumes from cache) |
| G25 | trap | What fine do we pay if we break the storage rules? | – | – | – | – | not run (stopped after repeated LLM errors; re-run resumes from cache) |
| G26 | trap | What was the temperature breach rate in Q3 FY2024? | – | – | – | – | not run (stopped after repeated LLM errors; re-run resumes from cache) |

## Answers that lost points

**G02: Which documents must travel with every shipment?**

- Answer: Every shipment must include the following documents: - Tax Invoice [1] - Certificate of Analysis (CoA) [1] - Temperature log [1] - Batch/expiry information [1] Additionally, a signed Delivery Order is mandatory before dispatch [1]. For refrigerated items, the driver must sign the Cold Chain Handover Form [1].
- Reference: A tax invoice, the Certificate of Analysis (CoA), the temperature log, and batch/expiry information.
- Judge: correctness 2/2, faithful True, citations some_unsupported. The answer correctly identifies all the required documents and does not include any unsupported claims.
- Unsupported citation [1]: Additionally, a signed Delivery Order is mandatory before dispatch. (entailment 0.009)


## Cost

Answer model `gemini-3.8-flash`: 0 API calls, 4 from cache; judge `qwen2.5:7b` (local via Ollama, digest 845dbda0ea48): 0 calls, 16 from cache; citations NLI `cross-encoder/nli-deberta-v3-small` (entailment >= 0.9): 0 LLM calls, 0 from cache. Tokens this run: answers {'calls': 0, 'prompt_tokens': 0, 'output_tokens': 0, 'thought_tokens': 0, 'seconds': 0.0, 'max_seconds': 0.0}, judge {'calls': 0, 'prompt_tokens': 0, 'output_tokens': 0, 'thought_tokens': 0, 'seconds': 0.0, 'max_seconds': 0.0}.

## Caveats

- The judge is an LLM and makes mistakes. The calibration table above checks it only on the flaw types planted there, so spot-check verdicts in the per-question table, especially with a small local judge.
- Small sample: each slice has only a handful of questions, so treat single questions as anecdotes and look at the per-question table.
- Key-fact recall is a strict string match and undercounts paraphrased facts; the judge's correctness score is the primary metric.
