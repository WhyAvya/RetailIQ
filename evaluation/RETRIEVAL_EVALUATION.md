# Phase 5 Round 2: Retrieval Evaluation

The evaluator reads the validated 20-question benchmark but sends only its 15 `semantic` questions to the existing `retrieve_semantic_matches` service. The five numerical questions are excluded because they are answered by deterministic Pandas analytics, not vector retrieval.

Each question is queried once at the largest requested K. The same ranking is then sliced for K=3 and K=5. The evaluator checks that every match belongs to benchmark upload `6377aeff-8a64-47ac-9748-28e0b0523d56` and uses the existing 1-based `source_row` metadata. Duplicate chunks referring to one source row retain only their first ranked occurrence.

## Metrics

For relevant row set `R`, ranked unique retrieved rows `L`, and cutoff `K`:

- `Precision@K = |L[:K] intersect R| / K`
- `Recall@K = |L[:K] intersect R| / |R|`
- `Reciprocal Rank = 1 / rank of the first relevant row`, or zero when none is retrieved within the evaluated ranking
- `MRR` is the mean reciprocal rank across all 15 semantic questions. With K values 3 and 5, the evaluated ranking ends at 5, so this run's MRR cutoff is 5.

Precision always uses fixed K as its denominator. If Chroma returns fewer than K unique source rows, the missing ranks count as non-relevant. Empty results score zero. A question with no relevant labels also scores zero, although Round 1 validation prevents that case for this benchmark.

Expected retrieval-service failures are recorded per question with empty rankings and zero metrics; they are never replaced with invented results. Aggregate metrics include those failures as zero, and `status` becomes `completed_with_failures`. The report records the failure count so these zeros cannot be mistaken for successful retrieval quality.

## Execution

From the project root:

```powershell
.\venv\Scripts\python.exe -m evaluation.retrieval_evaluation --k-values 3 5
```

This reads the existing persistent Chroma collection and embedding model. It does not regenerate, replace, or delete vectors. The JSON output is written to `evaluation/reports/retrieval_evaluation.json` and includes benchmark identity, model and collection names, settings, aggregate metrics, per-question rankings, similarity scores, and failures.

## Limitations

- The benchmark has only 15 semantic questions over 12 source rows, so results are diagnostic rather than statistically broad.
- Relevance is binary and manually grounded in exact CSV predicates; no graded relevance is used.
- Several questions have more relevant rows than K, which caps achievable Recall@K.
- Metrics evaluate row retrieval only. They do not evaluate generated answer quality or OpenRouter behavior.
- Scores depend on the existing `all-MiniLM-L6-v2` model, stored chunks, and Chroma version.

## Recorded Round 2 run

The real run on 2026-11-08 evaluated all 15 semantic questions with no per-question retrieval failures:

- Precision@3: `0.4444444444444444`
- Precision@5: `0.3466666666666667`
- Recall@3: `0.5584126984126985`
- Recall@5: `0.6668253968253969`
- MRR: `0.5888888888888889`

The local model weights were already cached. The model library attempted optional Hugging Face metadata requests, which the sandbox network blocked, but it successfully continued from the cached commit and completed every query. No vectors were regenerated and no retrieval result was fabricated.
