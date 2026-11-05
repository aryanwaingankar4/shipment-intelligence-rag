# Citation checker comparison

| Checker | Correct | Wrong citations caught | False alarms | Median per check | Total | Load | Passes |
|---|---|---|---|---|---|---|---|
| NLI `cross-encoder/nli-deberta-v3-small` (entailment >= 0.9) | 6/6 | 3/3 | 0/3 | 0.09 s | 0.6 s | 8.6 s | yes |

6 cited sentences from `data/eval/citation_holdout.json`. A checker passes only if it gets all of them right. These cases were written and committed before this run and nothing was tuned on them. They leave out claims that stretch a passage (calibration case C4), an accepted limit. The set is small: a pass rules out obvious failure, it is not a measured accuracy.

## Per sentence

| Case | Cited sentence | Should be supported | NLI `cross-encoder/nli-deberta-v3-small` (entailment >= 0.9) |
|---|---|---|---|
| F1 | The customer must be notified of any delay exceeding 24 hours. | False | False ✅ |
| F2 | StableCap Dry Powder has a shelf life of 36 months from the date of manufacture. | False | False ✅ |
| F3 | SH-2024-004 was accepted by Dr. Anita Rao. | False | False ✅ |
| F4 | Products can only be returned for 30 days after they are delivered. | True | True ✅ |
| F5 | FreezeGuard kits must travel on dry ice and regular ice must never be used. | True | True ✅ |
| F6 | SH-2024-003 was going to Fortis Hospital Hyderabad and was quarantined after exceeding 10°C for 3 hours. | True | True ✅ |
