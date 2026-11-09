# RetailIQ C8 Independent Evaluation

Questions: 25. LLM calls: 0. Backend freeze verified before evaluation.

| Metric | Result |
|---|---:|
| Filtering accuracy | 0.8000 |
| Numerical accuracy | 1.0000 |
| Ranking accuracy | 0.8000 |
| Mixed accuracy | 0.8000 |
| Semantic P@3 | 0.5333 |
| Semantic P@5 | 0.5600 |
| Semantic R@3 | 0.0126 |
| Semantic R@5 | 0.0219 |
| Semantic MRR@5 | 0.5000 |

Semantic relevance uses each question's complete, frozen Product Name substring set; low recall is expected when that set is much larger than five rows.

## Deterministic failures

C8-FIL-003, C8-RANK-003, C8-MIX-002
