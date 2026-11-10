# RetailIQ C8 Post-Fix Evaluation

Questions: 25. LLM calls: 0. Original freeze manifest was preserved; its backend hash differs because this is the authorized post-fix evaluation.

| Metric | Result |
|---|---:|
| Filtering accuracy | 1.0000 |
| Numerical accuracy | 1.0000 |
| Ranking accuracy | 1.0000 |
| Mixed accuracy | 1.0000 |
| Semantic P@3 | 0.2667 |
| Semantic P@5 | 0.2400 |
| Semantic R@3 | 0.0068 |
| Semantic R@5 | 0.0103 |
| Semantic MRR@5 | 0.2667 |

Semantic relevance uses each question's complete, frozen Product Name substring set; low recall is expected when that set is much larger than five rows.

## Deterministic failures

None.
