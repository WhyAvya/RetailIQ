# Phase 5 Round 3: Answer Quality Evaluation

Round 3 separates evaluation results by provenance so fixture scores cannot be mistaken for live model performance.

## Human-review framework

Each reviewed answer contains the question and answer text, retrieved evidence rows, reviewer identity, a list of claims, and a relevance review. Every claim is labeled as factual or non-factual. Factual claims are marked supported or unsupported and may reference only rows that were actually retrieved.

- **Faithfulness** = supported factual claims / all factual claims.
- **Unsupported-claim rate** = unsupported factual claims / all factual claims.
- **Answer relevance** uses a human rubric from 0 to 3 and is normalized by dividing by 3:
  - `0`: does not answer the question.
  - `1`: addresses a small part of the question or omits most required information.
  - `2`: substantially answers the question but contains a meaningful omission or distraction.
  - `3`: directly and completely addresses the question.

Non-factual statements are excluded from claim denominators. An answer with no factual claims receives faithfulness `1.0` and unsupported-claim rate `0.0`; relevance still captures that such an answer may not be useful. Aggregate faithfulness and unsupported rate are micro-averaged across factual claims. Relevance is averaged across reviewed answers.

`fixtures/answer_annotations.json` contains deliberately constructed examples for automated scoring tests. Their scores are labeled `test_fixture_only` and are not RetailIQ LLM performance.

## Actual numerical verification

The five numerical benchmark questions are loaded from the unchanged Round 1 benchmark. `backend.analytics.calculate_kpis` calculates the corresponding metrics from the versioned 12-row CSV:

| Metric | Verified value |
| --- | ---: |
| Total revenue | 45,150 |
| Total profit | 5,750 |
| Distinct orders | 11 |
| Average order value | 4,104.55 |
| Profit margin | 12.74% |

All five match the benchmark after applying its documented display precision, producing numerical correctness `1.0`.

## Live LLM status

Live OpenRouter answer evaluation is `pending` because the provider currently returns HTTP 429. No live answer was replaced with a mock, and the report keeps faithfulness, answer relevance, and unsupported-claim rate as `null` until reviewed live answers exist.

## Reproduction

Generate the answer-quality report without calling OpenRouter:

```powershell
.\venv\Scripts\python.exe -m evaluation.answer_evaluation
```

Run the focused tests:

```powershell
.\venv\Scripts\python.exe -m pytest -q tests/test_answer_evaluation.py tests/test_evaluation_api.py
```

The read-only `GET /api/evaluation/report` endpoint combines the saved Round 2 retrieval report and Round 3 answer report for the React Evaluation page. It does not run retrieval, analytics, or an LLM during the request.

## Limitations

- No actual semantic answer-quality score is available until live answers can be generated and human-reviewed.
- Claim boundaries and support labels require human judgment; the framework validates annotations but does not automate that judgment.
- The numerical check covers the five benchmark KPIs on the 12-row sample only.
- The dashboard displays saved reports and does not trigger or edit evaluations.
