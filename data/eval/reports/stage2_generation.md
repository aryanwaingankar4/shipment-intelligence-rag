# Generation eval — 2026-11-02

Golden set `data/eval/golden_qa.json` (26 questions). Answers: `gemini-3.8-flash` (thinking low, temperature 0.1). Judge: `qwen2.5:7b` (local via Ollama, digest 845dbda0ea48), temperature 0, rubric `4fd3b947`. Citations: NLI `cross-encoder/nli-deberta-v3-small` (entailment >= 0.9), one citation at a time. Retrieval: hybrid, MIN_RERANK_SCORE=-10.25, top-3.

> **Partial run:** 21 of 26 questions not scored (LLM errors / quota). Re-running resumes from the cache.

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
| every cited passage supports its sentence | 2/4 (50%) |
| **Should refuse** (unanswerable / off-domain) |  |
| refused | 1/1 (100%) |
| of which refused at retrieval (no LLM answer call) | 1/1 (100%) |
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
| G04 | answerable | What has to happen if an ambient product stays above 30°C f… | 2/2 | yes | some_unsupported | 2/2 |  |
| G05 | answerable | How often is temperature mapping of the cold rooms repeated? | – | – | – | – | not run: Gemini quota exhausted for gemini-3.8-flash (HTTP 429); resets in ~18.… |
| G06 | answerable | Within how many days of delivery can a product be returned? | – | – | – | – | not run: Gemini quota exhausted for gemini-3.8-flash (HTTP 429); resets in ~18.… |
| G07 | answerable | What quality rating does Deccan Pharma Logistics have? | – | – | – | – | not run: Gemini quota exhausted for gemini-3.8-flash (HTTP 429); resets in ~18.… |
| G08 | answerable | What temperatures were recorded for SH-2024-004? | – | – | – | – | not run: Gemini quota exhausted for gemini-3.8-flash (HTTP 429); resets in ~18.… |
| G09 | answerable | Which product and batch were shipped in SH-2024-008, and wh… | – | – | – | – | not run: Gemini quota exhausted for gemini-3.8-flash (HTTP 429); resets in ~18.… |
| G10 | answerable | Who is the contact person for vendor V-004, and what is the… | – | – | – | – | not run: Gemini quota exhausted for gemini-3.8-flash (HTTP 429); resets in ~18.… |
| G11 | answerable | Which deviation report was raised for SH-2024-003? | – | – | – | – | not run: Gemini quota exhausted for gemini-3.8-flash (HTTP 429); resets in ~18.… |
| G12 | answerable | Why was SH-2024-003 quarantined, and what corrective action… | – | – | – | – | not run: Gemini quota exhausted for gemini-3.8-flash (HTTP 429); resets in ~18.… |
| G13 | answerable | Which vendor handled the rejected frozen shipment, and what… | – | – | – | – | not run: Gemini quota exhausted for gemini-3.8-flash (HTTP 429); resets in ~18.… |
| G14 | answerable | A cold-chain consignment arrived at 11°C. What must we do, … | – | – | – | – | not run: Gemini quota exhausted for gemini-3.8-flash (HTTP 429); resets in ~18.… |
| G15 | answerable | What were the two temperature breaches in FY2024, and what … | – | – | – | – | not run: Gemini quota exhausted for gemini-3.8-flash (HTTP 429); resets in ~18.… |
| G16 | answerable | A vendor shipment is more than 48 hours late. Who do we esc… | – | – | – | – | not run: Gemini quota exhausted for gemini-3.8-flash (HTTP 429); resets in ~18.… |
| G17 | answerable | How long do the frozen sample kits stay usable? | – | – | – | – | not run: Gemini quota exhausted for gemini-3.8-flash (HTTP 429); resets in ~18.… |
| G18 | answerable | Do we have to keep the old logger calibration paperwork? | – | – | – | – | not run: Gemini quota exhausted for gemini-3.8-flash (HTTP 429); resets in ~18.… |
| G19 | answerable | Our truck broke down halfway to the customer. What now? | – | – | – | – | not run: Gemini quota exhausted for gemini-3.8-flash (HTTP 429); resets in ~18.… |
| G20 | answerable | Which vendor has the best on-time delivery rate? | – | – | – | – | not run: Gemini quota exhausted for gemini-3.8-flash (HTTP 429); resets in ~18.… |
| G21 | refusal | What is the status of shipment SH-2024-011? | – | – | – | – | not run: Gemini quota exhausted for gemini-3.8-flash (HTTP 429); resets in ~18.… |
| G22 | refusal | What temperature should insulin pens be stored at? | – | – | – | – | not run: Gemini quota exhausted for gemini-3.8-flash (HTTP 429); resets in ~18.… |
| G23 | refusal | Who is the CEO of NexaChain BioMed? | – | – | – | – | not run: Gemini quota exhausted for gemini-3.8-flash (HTTP 429); resets in ~18.… |
| G24 | refusal | What's the capital of France? | 2/2 | yes | no_citations | – | refused at retrieval |
| G25 | trap | What fine do we pay if we break the storage rules? | – | – | – | – | not run: Gemini quota exhausted for gemini-3.8-flash (HTTP 429); resets in ~18.… |
| G26 | trap | What was the temperature breach rate in Q3 FY2024? | – | – | – | – | not run: Gemini quota exhausted for gemini-3.8-flash (HTTP 429); resets in ~18.… |

## Answers that lost points

**G02: Which documents must travel with every shipment?**

- Answer: Every shipment must include the following documents: - Tax Invoice [1] - Certificate of Analysis (CoA) [1] - Temperature log [1] - Batch/expiry information [1] Additionally, a signed Delivery Order is mandatory before dispatch [1]. For refrigerated items, the driver must sign the Cold Chain Handover Form [1].
- Reference: A tax invoice, the Certificate of Analysis (CoA), the temperature log, and batch/expiry information.
- Judge: correctness 2/2, faithful True, citations some_unsupported. The answer correctly identifies all the required documents and does not include any unsupported claims.
- Unsupported citation [1]: Tax Invoice (entailment 0.001)
- Unsupported citation [1]: Certificate of Analysis (CoA) (entailment 0.001)
- Unsupported citation [1]: Temperature log (entailment 0.000)
- Unsupported citation [1]: Batch/expiry information (entailment 0.001)
- Unsupported citation [1]: Additionally, a signed Delivery Order is mandatory before dispatch. (entailment 0.002)
- Unsupported citation [1]: For refrigerated items, the driver must sign the Cold Chain Handover Form. (entailment 0.001)

**G04: What has to happen if an ambient product stays above 30°C for 4 hours?**

- Answer: If an ambient product exceeds 30°C for 4 or more continuous hours, it must be reported to the warehouse supervisor, and QA issues a deviation report [1].
- Reference: It must be reported to the warehouse supervisor, and QA issues a deviation report (rule: above 30°C for 4 or more continuous hours).
- Judge: correctness 2/2, faithful True, citations some_unsupported. The answer correctly identifies the reporting requirements for ambient products that exceed 30°C for 4 or more continuous hours, as stated in the passages. It does not include any unsupported claims.
- Unsupported citation [1]: If an ambient product exceeds 30°C for 4 or more continuous hours, it must be reported to the warehouse supervisor, and QA issues a deviation report. (entailment 0.002)


## Cost

Answer model `gemini-3.8-flash`: 0 API calls, 4 from cache; judge `qwen2.5:7b` (local via Ollama, digest 845dbda0ea48): 4 calls, 12 from cache; citations NLI `cross-encoder/nli-deberta-v3-small` (entailment >= 0.9): 0 LLM calls, 0 from cache. Tokens this run: answers {'calls': 0, 'prompt_tokens': 0, 'output_tokens': 0, 'thought_tokens': 0, 'seconds': 0.0, 'max_seconds': 0.0}, judge {'calls': 4, 'prompt_tokens': 3053, 'output_tokens': 276, 'thought_tokens': 0, 'seconds': 335.2, 'max_seconds': 95.7}.

## Caveats

- The judge is an LLM and makes mistakes. The calibration table above checks it only on the flaw types planted there, so spot-check verdicts in the per-question table, especially with a small local judge.
- Small sample: each slice has only a handful of questions, so treat single questions as anecdotes and look at the per-question table.
- Key-fact recall is a strict string match and undercounts paraphrased facts; the judge's correctness score is the primary metric.
