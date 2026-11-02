# Citation checker comparison

| Checker | Correct | Wrong citations caught | False alarms | Median per check | Total | Load | Passes |
|---|---|---|---|---|---|---|---|
| NLI `cross-encoder/nli-deberta-v3-small` (entailment >= 0.9) | 12/12 | 5/5 | 0/7 | 1.82 s | 23.3 s | 33.7 s | yes |

12 cited sentences from `data/eval/citation_holdout_full.json`. A checker passes only if it gets all of them right. These cases were written and committed before this run and nothing was tuned on them. They leave out claims that stretch a passage (calibration case C4), an accepted limit. The set is small: a pass rules out obvious failure, it is not a measured accuracy.

## Per sentence

| Case | Cited sentence | Should be supported | NLI `cross-encoder/nli-deberta-v3-small` (entailment >= 0.9) |
|---|---|---|---|
| H1 | Temperature mapping of each cold room is repeated every 2 years. | False | False ✅ |
| H2 | ImmunoChill Vaccine Vials have a shelf life of 12 months from the date of manufacture. | False | False ✅ |
| H3 | Anil Deshmukh is the contact person for MedSupply India. | False | False ✅ |
| H4 | The SH-2024-007 rejection was caused by insufficient dry ice loaded for a 2-day transit. | False | False ✅ |
| H5 | The requirements include FIFO is mandatory for all product storage | True | True ✅ |
| H5 | The requirements include The customer must be notified of any delay exceeding 24 hours | False | False ✅ |
| H6 | Additionally, QA must reassess product integrity before acceptance for any delay exceeding 48 hours. | True | True ✅ |
| H7 | The handling rules for StableCap Dry Powder are Keep it away from moisture | True | True ✅ |
| H7 | The handling rules for StableCap Dry Powder are Do not open it until the point of use | True | True ✅ |
| H8 | The CAPA revised the dry ice quantity formula to 1.5x transit hours and made extended-duration cases mandatory for transit over 24 hours. | True | True ✅ |
| H9 | ArcticLine is based in Mumbai and had an on-time delivery rate of 88% in FY2024. | True | True ✅ |
| H10 | A delay over 48 hours must be escalated to the QA head at quality@nexachain-biomed.com, with a written deviation. | True | True ✅ |
