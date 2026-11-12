# RetailIQ Phase 5 Live Evaluation

Final assessment: **PASS**. Five pipeline queries were attempted against the canonical 9,994-row upload.

| Category | Route | Correct | Grounded | Citations | Hallucination | Latency (ms) | Error |
|---|---|---:|---:|---|---:|---:|---|
| semantic | semantic_rag | true | true | valid | false | 44554.12 |  |
| numerical | analytics | true | true | not_applicable_exact_csv_calculation | false | 216.08 |  |
| filtering | analytics | true | true | not_applicable_exact_csv_calculation | false | 190.74 |  |
| ranking | analytics | true | true | not_applicable_exact_csv_calculation | false | 174.54 |  |
| mixed | analytics | true | true | not_applicable_exact_csv_calculation | false | 372.89 |  |

## Execution summary

- LLM invocations: 1 (expected: semantic only).
- Provider/model: OpenRouter / `nvidia/nemotron-3-ultra-550b-a55b:free`.
- Average latency: 9101.67 ms; maximum: 44554.12 ms.
- Frozen benchmark/report hashes unchanged: true.

## C8 baseline versus post-fix

| Metric | Baseline | Post-fix |
|---|---:|---:|
| Filtering accuracy | 80% | 100% |
| Numerical accuracy | 100% | 100% |
| Ranking accuracy | 60% | 100% |
| Mixed accuracy | 100% | 100% |

The post-fix values above come from `c8_post_fix_evaluation.json`; they are not the original independent baseline.

## Retrieval metrics and limitations

| Metric | Value |
|---|---:|
| P@3 | 0.2667 |
| P@5 | 0.2400 |
| R@3 | 0.0068 |
| R@5 | 0.0103 |
| MRR@5 | 0.2667 |

Recall@K is intrinsically capped by retrieving only 3 or 5 rows when each frozen substring relevance set can contain many rows, but values near 1% still mean the result is not exhaustive. P@3/P@5 show that most top results were not in the strict frozen relevance sets, and MRR@5 means the first relevant result was often late or absent. The semantic answer must therefore be treated as a small set of approximate matches, not a complete catalog result. These metrics cover five semantic questions, so they are also high-variance and narrow in scope.

## Assessment rationale

- 5/5 live pipeline results passed independent CSV/evidence validation.
- The LLM boundary recorded 1 real invocation(s); structured and mixed routes made no LLM calls.
- All tracked frozen artifact hashes were unchanged before and after execution: True.
