# Stage 1 retrieval experiments

Eval set: `data/eval/retrieval_eval.json`.

## Does the reranker help? (no threshold)

| Config | hit@3 all | hit@3 ID | hit@3 paraphrased | MRR@3 | candidate recall | false refusals | off-domain refused |
|---|---|---|---|---|---|---|---|
| vector, no rerank | 81% | 70% | 80% | 0.674 | 94% | 0 | 0% |
| vector + MiniLM-L6 rerank | 87% | 95% | 70% | 0.816 | 94% | 0 | 0% |
| bm25, no rerank | 83% | 100% | 40% | 0.794 | 87% | 2 | 67% |
| bm25 + MiniLM-L6 rerank | 85% | 100% | 40% | 0.830 | 87% | 2 | 67% |
| hybrid, no rerank | 91% | 95% | 80% | 0.766 | 98% | 0 | 0% |
| hybrid + MiniLM-L6 rerank | 91% | 100% | 70% | 0.858 | 98% | 0 | 0% |

## Threshold sweep (hybrid)

| Config | hit@3 all | hit@3 ID | hit@3 paraphrased | MRR@3 | candidate recall | false refusals | off-domain refused |
|---|---|---|---|---|---|---|---|
| MiniLM-L6, MIN_RERANK_SCORE=-6.0 | 81% | 100% | 30% | 0.777 | 98% | 8 | 100% |
| MiniLM-L6, MIN_RERANK_SCORE=-9.0 | 85% | 100% | 40% | 0.819 | 98% | 4 | 100% |
| MiniLM-L6, MIN_RERANK_SCORE=-10.0 | 87% | 100% | 50% | 0.840 | 98% | 1 | 100% |
| MiniLM-L6, MIN_RERANK_SCORE=-10.5 | 91% | 100% | 70% | 0.858 | 98% | 1 | 100% |
| MiniLM-L6, MIN_RERANK_SCORE=-10.9 | 91% | 100% | 70% | 0.858 | 98% | 0 | 100% |
| MiniLM-L6, MIN_RERANK_SCORE=-11.0 | 91% | 100% | 70% | 0.858 | 98% | 0 | 100% |
| MiniLM-L6, MIN_RERANK_SCORE=None | 91% | 100% | 70% | 0.858 | 98% | 0 | 0% |
| *MiniLM-L6: weakest answerable in-scope top score -9.97, strongest off-domain -11.14* | | | | | | | |
| MiniLM-L12, MIN_RERANK_SCORE=-6.0 | 83% | 95% | 40% | 0.816 | 98% | 7 | 100% |
| MiniLM-L12, MIN_RERANK_SCORE=-9.0 | 85% | 95% | 40% | 0.837 | 98% | 5 | 100% |
| MiniLM-L12, MIN_RERANK_SCORE=-10.0 | 87% | 95% | 50% | 0.858 | 98% | 3 | 100% |
| MiniLM-L12, MIN_RERANK_SCORE=-10.5 | 89% | 95% | 60% | 0.879 | 98% | 1 | 100% |
| MiniLM-L12, MIN_RERANK_SCORE=-10.9 | 89% | 95% | 60% | 0.879 | 98% | 0 | 100% |
| MiniLM-L12, MIN_RERANK_SCORE=-11.0 | 89% | 95% | 60% | 0.879 | 98% | 0 | 100% |
| MiniLM-L12, MIN_RERANK_SCORE=None | 91% | 95% | 70% | 0.887 | 98% | 0 | 0% |
| *MiniLM-L12: weakest answerable in-scope top score -10.32, strongest off-domain -11.16* | | | | | | | |
