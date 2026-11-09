# RetailIQ C8 Independent Evaluation

Questions: 25. LLM calls: 0. Backend freeze verified before evaluation.

| Metric | Result |
|---|---:|
| Filtering accuracy | 0.8000 |
| Numerical accuracy | 1.0000 |
| Ranking accuracy | 0.6000 |
| Mixed accuracy | 1.0000 |
| Semantic P@3 | 0.2667 |
| Semantic P@5 | 0.2400 |
| Semantic R@3 | 0.0068 |
| Semantic R@5 | 0.0103 |
| Semantic MRR@5 | 0.2667 |

Semantic relevance uses each question's complete, frozen Product Name substring set; low recall is expected when that set is much larger than five rows.

## Deterministic failures

C8I2-FIL-004, C8I2-RANK-002, C8I2-RANK-003
