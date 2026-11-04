# Query rewriting eval (partial)

> **Partial run, 2026-11-05.** The machine ran critically low on memory, so the run was stopped before Part 2 finished and the script never wrote its report. These rows were recovered from its log, which did not record the rewrite text or timings (it does now). **Part 2, which measures the paraphrase fix (D5), has no result.**

Rewriter `qwen2.5:7b` (local via Ollama, temperature 0). Retrieval: hybrid, top-3, MIN_RERANK_SCORE=-10.25. No Gemini calls. The cases and the rewrite prompt were committed in `96b37f7`, before this first run, and nothing was tuned on them.

## Part 1: follow-up questions (`data/eval/followups.json`)

| Metric | Result |
|---|---|
| Right passage in top 3 (or refused, F11), question as written | 10/14 |
| Right passage in top 3 (or refused, F11), rewritten | 13/14 |
| Rewrite names what it should and nothing it shouldn't | 14/14 |
| Reached the shipment table after rewriting | 1/1 |

| # | Follow-up | Terms | As written | Rewritten |
|---|---|---|---|---|
| F01 | What corrective action was decided? | yes | miss | **miss** |
| F02 | Which vendor handled it? | yes | hit | hit |
| F03 | How long is its shelf life? | yes | miss | hit |
| F04 | And for TL-FROZ-7700? | yes | hit | hit |
| F05 | What's their on-time delivery rate? | yes | hit | hit |
| F06 | What was the root cause of the second one? | yes | hit | hit |
| F07 | And the other one? | yes | miss | hit |
| F08 | What if the delay was caused by the truck breaking down? | yes | hit | hit |
| F09 | And for not updating the shipment log on time? | yes | hit | hit |
| F10 | How many days does a customer have to return delivered goods? | yes | hit | hit |
| F11 | What is the capital of France? | yes | hit | hit |
| F12 | And how many were rejected? | yes | hit | hit (table: yes) |
| F13 | Who accepted it? | yes | miss | hit |
| F14 | Who supplies it? | yes | hit | hit |

Read with care: the cases were written alongside the code and there are only 14. A vague question can also find the right passage by luck, which is why as-written scores 10/14. F01 is a real miss: the rewrite named the right shipment and retrieval still missed the passage. It has been left as it is, not tuned.

## Part 2: every retrieval eval question, rewritten with no conversation

Not finished: no result.
