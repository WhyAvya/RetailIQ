# RetailIQ Benchmark B2 Baseline

B2 evaluated only the 40 B1 development questions. The 20 held-out questions were not executed. ChromaDB was read only and OpenRouter was not called.

## Baseline metrics

| Evaluation | Questions | P@3 | P@5 | R@3 | R@5 | MRR@5 | Deterministic accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Semantic retrieval | 13 | 0.0513 | 0.0308 | 0.0308 | 0.0308 | 0.0769 | n/a |
| Mixed retrieval | 7 | 0.4762 | 0.5429 | 0.0051 | 0.0094 | 0.6071 | 0.0000 |
| All retrieval-scored | 20 | 0.2000 | 0.2100 | 0.0218 | 0.0233 | 0.2625 | n/a |
| Metadata exact filters | 10 | n/a | n/a | n/a | n/a | n/a | 0.6000 |
| Numerical analytics | 10 | n/a | n/a | n/a | n/a | n/a | 0.3000 |

Metadata supported-only exact accuracy was 1.0000 across 6 supported questions. Numerical supported-only accuracy was 1.0000 across 3 supported questions.

At least one relevant row appeared in the top five for 6 of 20 retrieval-scored questions; 0 achieved full Recall@5.

## Failure types

- `missing_relevant_rows`: 20 question(s); examples: B1-SEM-001, B1-SEM-002, B1-SEM-003
- `irrelevant_rows_in_top_5`: 18 question(s); examples: B1-SEM-001, B1-SEM-002, B1-SEM-003
- `unsupported_or_incomplete_calculation`: 14 question(s); examples: B1-NUM-003, B1-NUM-004, B1-NUM-005
- `numeric_ranking_mismatch`: 13 question(s); examples: B1-SEM-001, B1-SEM-002, B1-SEM-003
- `incorrect_sub_category`: 5 question(s); examples: B1-SEM-005, B1-SEM-009, B1-SEM-010
- `incorrect_region`: 4 question(s); examples: B1-SEM-003, B1-SEM-009, B1-SEM-013
- `unsupported_filter_dimension`: 4 question(s); examples: B1-META-005, B1-META-006, B1-META-009
- `product_keyword_mismatch`: 3 question(s); examples: B1-SEM-006, B1-SEM-007, B1-SEM-008

## Recommendations

- Improve semantic ranking for exact top-k numeric comparisons while preserving the current benchmark and held-out split.
- Add explicit metadata filtering support for State, City, Segment, and Ship Mode before evaluating those questions again.
- Expand deterministic analytics intents for averages, medians, quantities, filtered counts, and mixed retrieval-plus-aggregation questions.
- Evaluate LLM-generated answer quality separately; B2 measures no generated answers.

## Interpretation boundary

Retrieval metrics score Chroma row ranking. Metadata and numerical scores test current deterministic capabilities. No LLM-generated answer was produced or assessed, so these results must not be interpreted as answer-quality scores.

Complete per-question rankings, record-set results, routed intents, failure examples, and ground-truth comparisons are in `benchmark_b2_baseline.json`.
