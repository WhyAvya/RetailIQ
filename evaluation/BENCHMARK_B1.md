# RetailIQ Benchmark B1

Benchmark B1 prepares the complete 9,994-row Superstore dataset and a fixed
60-question evaluation set. It does not call OpenRouter, run questions through
the RAG pipeline, change models, or alter previous evaluation artifacts.

## Artifacts

- `retailiq_benchmark_b1.json` contains all 60 questions, expected answers,
  structured ground truth, exact relevance definitions, applicable source-row
  sets, and fixed development/held-out labels.
- `reports/benchmark_b1_dataset_coverage.json` records the Chroma audit for
  upload `11111111-1111-1111-1111-111111111111`.
- `benchmark_b1.py` reproducibly calculates the benchmark with Pandas and
  provides the missing-rows-only Chroma preparation logic.
- `scripts/prepare_benchmark_b1.py` reruns the coverage check and artifact
  generation. If rows are absent, it embeds and upserts only absent source
  rows. It refuses to proceed when an existing row has a conflicting chunk ID
  or when duplicate source rows are present.

The original `data/sample_superstore.csv`, the 12-row sample, and all prior
evaluation files remain unchanged.

## Dataset and ground truth

The original CSV is read directly with Pandas using CP-1252 encoding. It has
9,994 data rows and SHA-256
`58fb59c63089e564d835e4aabe19d371757de1137a5bf1888243ae386c4b7a97`.
CSV data rows use 1-based numbering with the header excluded.

Each product-discovery question defines an exact predicate, ordering,
tie-breaker, and top-k. Each metadata question includes every matching source
row. Each mixed question includes every filtered row needed by its aggregate.
Numerical questions declare all 9,994 rows as their source scope and leave the
retrieval-row list empty because they are full structured aggregates.

Ground truth is calculated from DataFrame filters, groupings, and aggregations,
not generated text. Validation rebuilds every answer and row judgment from the
CSV and compares the complete artifact.

## Fixed split

| Type | Development | Held-out test | Total |
|---|---:|---:|---:|
| Semantic/product discovery | 13 | 7 | 20 |
| Metadata filtering | 10 | 5 | 15 |
| Numerical | 10 | 5 | 15 |
| Mixed | 7 | 3 | 10 |
| **Total** | **40** | **20** | **60** |

Held-out questions must not be used for tuning.

## Validation

Run only B1 preparation when the local Chroma store needs to be audited:

```powershell
.\venv\Scripts\python.exe -m scripts.prepare_benchmark_b1
```

Run the focused validation suite:

```powershell
.\venv\Scripts\python.exe -m pytest -q tests/test_benchmark_b1.py
```
