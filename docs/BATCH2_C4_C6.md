# RetailIQ Batch 2: C4, C5, and C6

## C4 hybrid integration

The live `answer_question` pipeline now uses the validated Batch 1 planner.
Exact filters, calculations, grouping, and rankings run through the Pandas
executor. Pure semantic questions retain the existing MiniLM/Chroma/Nemotron
path. Mixed questions execute exact calculations with Pandas and use the
metadata-filtered retriever only when a semantic product term is present.

The API response modes and schemas are unchanged. Ambiguous, predictive, and
unsupported requests return `unsupported` without an LLM call. The original
12-row benchmark has integration coverage for deterministic results and
grounded semantic sources.

## C5 development evaluation

The integrated system evaluated the fixed 40 development questions. Compared
with B2, exact filtering improved from 0.60 to 1.00 and numerical correctness
from 0.30 to 1.00. Exact ranking order, exact ranking set, and mixed correctness
were all 1.00. Raw MiniLM retrieval metrics were byte-for-byte unchanged from
B2, so deterministic improvements are not presented as embedding gains.

Reports:

- `evaluation/reports/hybrid_c5_development.json`
- `evaluation/reports/hybrid_c5_development.md`

## C6 held-out evaluation

The implementation was frozen after C5 and the 20 held-out questions were run
once. Results were 0.60 exact filtering, 0.00 numerical, 0.8571 exact ranking
order/set, and 0.3333 mixed correctness. Raw retrieval P@5 was 0.12, R@5 was
0.0632, and MRR@5 was 0.3333.

Held-out failures expose unsupported discount and shipping-duration predicates,
grouped “each region/category/year” output, state/sub-category aggregate
ranking, and customer-plus-order multi-grain analysis. They were recorded and
were not used to alter the frozen implementation.

Reports:

- `evaluation/reports/hybrid_c6_held_out.json`
- `evaluation/reports/hybrid_c6_held_out.md`

## Real Nemotron sample

Three capped requests were made to
`nvidia/nemotron-3-ultra-550b-a55b:free`. All three completed. Manual
claim-to-evidence review measured 0.75 grounding/faithfulness, a 0.25
unsupported-claim rate, and 0.8889 answer relevance. These are real live-answer
scores; historical fixture-only results are excluded.

Reports and review provenance:

- `evaluation/reports/hybrid_live_nemotron_raw.json`
- `evaluation/reports/hybrid_live_nemotron_evaluation.json`
- `evaluation/reports/hybrid_live_nemotron_evaluation.md`
- `evaluation/fixtures/hybrid_live_nemotron_annotations.json`

Phase 6 deployment is outside this batch.
