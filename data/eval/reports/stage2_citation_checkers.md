# Citation checker comparison

| Checker | Correct | Wrong citations caught | False alarms | Median per check | Total | Load | Passes |
|---|---|---|---|---|---|---|---|
| NLI `cross-encoder/nli-deberta-v3-small` | 12/14 | 5/7 | 0/7 | 0.08 s | 1.2 s | 168.6 s | **no** |
| LLM `qwen2.5:7b` | 11/14 | 6/7 | 2/7 | 18.17 s | 280.4 s | 0.0 s | **no** |

14 cited sentences from `data/eval/judge_calibration.json`. A checker passes only if it gets all of them right. This set is small and was also used to choose the checker, so treat a pass as necessary, not as a measured accuracy.

## Per sentence

| Case | Cited sentence | Should be supported | NLI `cross-encoder/nli-deberta-v3-small` | LLM `qwen2.5:7b` |
|---|---|---|---|---|
| C1 | ArcticLine's delivery window for dry-ice frozen shipments is 48 hours. | False | False ✅ | False ✅ |
| C2 | SH-2024-003 reached 11.2°C, above the 10°C limit for 3 hours, because the PCM packs were not fully conditioned. | True | True ✅ | False ❌ |
| C2 | The driver was suspended and the transporter was fined Rs. 50,000. | False | False ✅ | False ✅ |
| C3 | The contact person for V-004 is Imran Shaikh (ops@safemove.in). | False | False ✅ | False ✅ |
| C4 | Breaking the storage rules carries a fine of Rs. 5,000. | False | True ❌ | True ❌ |
| C5 | ArcticLine (V-003) delivers dry-ice frozen shipments within 36 hours. | True | True ✅ | True ✅ |
| C7 | Deccan Pharma Logistics had an on-time delivery rate of 91% in FY2024. | False | False ✅ | False ✅ |
| C8 | A cold chain product above 10°C must be quarantined immediately and QA notified within 2 hours. | True | True ✅ | True ✅ |
| C8 | The shipment log must be updated within 24 hours of dispatch. | False | True ❌ | False ✅ |
| C9 | The SH-2024-003 breach happened because the PCM cold packs were not fully conditioned before loading, and a highway diversion lengthened transit. | False | False ✅ | False ✅ |
| C10 | The contact person for V-004 is Imran Shaikh (ops@safemove.in). | True | True ✅ | False ❌ |
| C11 | SH-2024-007 was rejected because too little dry ice was loaded for the 2-day journey, so it ran out before delivery. | True | True ✅ | True ✅ |
| C11 | Frozen products are automatically rejected once they rise above -10°C. | True | True ✅ | True ✅ |
| C12 | MedSupply India delivers within 48 hours inside Maharashtra and within 72 hours outside it. | True | True ✅ | True ✅ |
