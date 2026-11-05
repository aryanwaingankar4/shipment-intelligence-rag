# Generation eval — 2026-11-05

Golden set `data/eval/golden_qa.json` (26 questions). Answers: `gemini-3.5-flash-lite` (thinking low, temperature 0.1). Judge: `qwen2.5:7b` (local via Ollama, digest 845dbda0ea48), temperature 0, rubric `4fd3b947`. Citations: NLI `cross-encoder/nli-deberta-v3-small` (entailment >= 0.9), one citation at a time. Retrieval: hybrid, MIN_RERANK_SCORE=-10.25, top-3.

## Summary

| Metric | Result |
|---|---|
| **Answerable questions** |  |
| fully correct (judge: 2/2) | 20/20 (100%) |
| partly correct (1/2) | 0/20 (0%) |
| mean correctness | 100% |
| key facts present in the answer (string match) | 38/38 (100%) |
| wrongly refused | 0/20 (0%) |
| **Faithfulness** (no unsupported claims, all generated answers) | 20/20 (100%) |
| **Citations** (answered answerable questions) |  |
| answer has ≥1 [n] citation | 18/20 (90%) |
| all citation numbers valid | 18/18 (100%) |
| cites a passage from an expected source file | 18/18 (100%) |
| every cited passage supports its sentence | 14/18 (78%) |
| **Should refuse** (unanswerable / off-domain) |  |
| refused | 4/4 (100%) |
| of which refused at retrieval (no LLM answer call) | 1/4 (25%) |
| **Traps** (documents lack the specific figure) |  |
| handled: says the information isn't in the documents | 2/2 (100%) |
| invented the figure | 0/2 (0%) |

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
| G02 | answerable | Which documents must travel with every shipment? | 2/2 | yes | all_supported | 4/4 |  |
| G03 | answerable | What is ArcticLine's delivery window for dry-ice frozen shi… | 2/2 | yes | all_supported | 1/1 |  |
| G04 | answerable | What has to happen if an ambient product stays above 30°C f… | 2/2 | yes | no_citations | 2/2 |  |
| G05 | answerable | How often is temperature mapping of the cold rooms repeated? | 2/2 | yes | all_supported | 1/1 |  |
| G06 | answerable | Within how many days of delivery can a product be returned? | 2/2 | yes | all_supported | 1/1 |  |
| G07 | answerable | What quality rating does Deccan Pharma Logistics have? | 2/2 | yes | all_supported | 1/1 |  |
| G08 | answerable | What temperatures were recorded for SH-2024-004? | 2/2 | yes | all_supported | 2/2 |  |
| G09 | answerable | Which product and batch were shipped in SH-2024-008, and wh… | 2/2 | yes | all_supported | 3/3 |  |
| G10 | answerable | Who is the contact person for vendor V-004, and what is the… | 2/2 | yes | all_supported | 2/2 |  |
| G11 | answerable | Which deviation report was raised for SH-2024-003? | 2/2 | yes | no_citations | 1/1 |  |
| G12 | answerable | Why was SH-2024-003 quarantined, and what corrective action… | 2/2 | yes | all_supported | 3/3 |  |
| G13 | answerable | Which vendor handled the rejected frozen shipment, and what… | 2/2 | yes | all_supported | 2/2 |  |
| G14 | answerable | A cold-chain consignment arrived at 11°C. What must we do, … | 2/2 | yes | all_supported | 2/2 |  |
| G15 | answerable | What were the two temperature breaches in FY2024, and what … | 2/2 | yes | some_unsupported | 3/3 |  |
| G16 | answerable | A vendor shipment is more than 48 hours late. Who do we esc… | 2/2 | yes | all_supported | 2/2 |  |
| G17 | answerable | How long do the frozen sample kits stay usable? | 2/2 | yes | some_unsupported | 1/1 |  |
| G18 | answerable | Do we have to keep the old logger calibration paperwork? | 2/2 | yes | some_unsupported | 1/1 |  |
| G19 | answerable | Our truck broke down halfway to the customer. What now? | 2/2 | yes | all_supported | 2/2 |  |
| G20 | answerable | Which vendor has the best on-time delivery rate? | 2/2 | yes | some_unsupported | 2/2 |  |
| G21 | refusal | What is the status of shipment SH-2024-011? | 2/2 | yes | no_citations | – | refused |
| G22 | refusal | What temperature should insulin pens be stored at? | 2/2 | yes | no_citations | – | refused |
| G23 | refusal | Who is the CEO of NexaChain BioMed? | 2/2 | yes | no_citations | – | refused |
| G24 | refusal | What's the capital of France? | 2/2 | yes | no_citations | – | refused at retrieval |
| G25 | trap | What fine do we pay if we break the storage rules? | 2/2 | yes | no_citations | – | refused |
| G26 | trap | What was the temperature breach rate in Q3 FY2024? | 2/2 | yes | no_citations | – | refused |

## Answers that lost points

**G15: What were the two temperature breaches in FY2024, and what did they have in common?**

- Answer: Based on the company documents, the two temperature breaches recorded in FY2024 were shipments SH-2024-003 and SH-2024-007 [3]. They had the following in common: * Both breaches involved temperature-sensitive products on multi-day transit [1]. * Both shared the common theme of thermal packaging not being matched to the actual transit duration [1].
- Reference: SH-2024-003 (cold chain, reached 11.2°C, quarantined) and SH-2024-007 (frozen, rose to -8°C, rejected). Both were temperature-sensitive products on multi-day transit whose thermal packaging was not matched to the actual transit duration.
- Judge: correctness 2/2, faithful True, citations some_unsupported. The answer correctly identifies the commonalities between the two breaches as stated in the passages, without adding unsupported claims or contradicting the reference answer.
- Unsupported citation [3]: Based on the company documents, the two temperature breaches recorded in FY2024 were shipments SH-2024-003 and SH-2024-007. (entailment 0.194)
- Unsupported citation [1]: They had the following in common Both breaches involved temperature-sensitive products on multi-day transit. (entailment 0.291)

**G17: How long do the frozen sample kits stay usable?**

- Answer: Based on the company documents, the FreezeGuard Biological Sample Kit (Product TL-FROZ-7700) has a shelf life of 18 months from the date of manufacture [1].
- Reference: 18 months shelf life from manufacture (TL-FROZ-7700, FreezeGuard Biological Sample Kit). Optional: must not be refrozen once thawed.
- Judge: correctness 2/2, faithful True, citations some_unsupported. The answer correctly identifies the 18-month shelf life from the date of manufacture for the FreezeGuard Biological Sample Kit as stated in the passage. It does not contradict the reference answer and includes all necessary information.
- Unsupported citation [1]: Based on the company documents, the FreezeGuard Biological Sample Kit (Product TL-FROZ-7700) has a shelf life of 18 months from the date of manufacture. (entailment 0.036)

**G18: Do we have to keep the old logger calibration paperwork?**

- Answer: Yes, logger calibration certificates are retained for 3 years [1].
- Reference: Yes: logger calibration certificates are retained for 3 years. Optional: loggers are calibrated every 6 months by an accredited lab.
- Judge: correctness 2/2, faithful True, citations some_unsupported. The answer correctly states that logger calibration certificates are retained for 3 years, which is accurate according to the passage provided. It does not contradict the reference answer and includes the key fact without adding unnecessary information.
- Unsupported citation [1]: Yes, logger calibration certificates are retained for 3 years. (entailment 0.539)

**G20: Which vendor has the best on-time delivery rate?**

- Answer: Based on the provided company documents, MedSupply India Pvt Ltd (vendor V-001) has the best on-time delivery rate at 96% [2]. (Deccan Pharma Logistics has 91% [1] and ArcticLine Cold Logistics has 88% [3]).
- Reference: MedSupply India Pvt Ltd (V-001), with 96% in FY2024 (SafeMove 94%, Deccan 91%, ArcticLine 88%).
- Judge: correctness 2/2, faithful True, citations some_unsupported. The answer correctly identifies MedSupply India Pvt Ltd (V-001) with the highest on-time delivery rate of 96% in FY2024, as per the provided passages. It also correctly compares this rate with the other vendors mentioned.
- Unsupported citation [2]: Based on the provided company documents, MedSupply India Pvt Ltd (vendor V-001) has the best on-time delivery rate at 96%. (entailment 0.005)
- Unsupported citation [1, 3]: (Deccan Pharma Logistics has 91% and ArcticLine Cold Logistics has 88%). (entailment 0.505)


## Cost

Answer model `gemini-3.5-flash-lite`: 5 API calls, 20 from cache; judge `qwen2.5:7b` (local via Ollama, digest 845dbda0ea48): 5 calls, 32 from cache; citations NLI `cross-encoder/nli-deberta-v3-small` (entailment >= 0.9): 0 LLM calls, 0 from cache. Tokens this run: answers {'calls': 5, 'prompt_tokens': 3444, 'output_tokens': 251, 'thought_tokens': 310, 'seconds': 73.4, 'max_seconds': 56.9}, judge {'calls': 5, 'prompt_tokens': 4143, 'output_tokens': 361, 'thought_tokens': 0, 'seconds': 15.2, 'max_seconds': 3.8}.

## Caveats

- The judge is an LLM and makes mistakes. The calibration table above checks it only on the flaw types planted there, so spot-check verdicts in the per-question table, especially with a small local judge.
- Small sample: each slice has only a handful of questions, so treat single questions as anecdotes and look at the per-question table.
- Key-fact recall is a strict string match and undercounts paraphrased facts; the judge's correctness score is the primary metric.
