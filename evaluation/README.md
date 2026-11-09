# RetailIQ Phase 5 Evaluation Benchmark

This directory contains the Round 1 ground truth for the 12-row RetailIQ sample. It does not call ChromaDB, embeddings, OpenRouter, or the RAG pipeline.

## Files

- `retailiq_12row_benchmark.json` contains exactly 15 semantic retrieval questions and 5 numerical analytics questions.
- `data/retailiq_sample_12_rows.csv` is a version-controlled, byte-identical evaluation copy of upload `6377aeff-8a64-47ac-9748-28e0b0523d56`. Its SHA-256 is `029ad4b1efb2e65d96fa5ae8586f46024e20cf828202c11b97de85a047eaef8b`.
- `benchmark_validation.py` checks benchmark structure, counts, IDs, labels, source-row bounds, source identity, and the row-numbering declaration.

The retained original upload under `backend/storage/uploads/` was read only and was not modified.

## Ground-truth method

1. The CSV was loaded directly with Pandas and checked to contain 12 rows and 11 columns.
2. Source rows follow the Phase 4 convention in `backend/rag_preparation.py`: the first CSV data row is row 1 and the header is excluded.
3. Semantic row sets were produced with explicit DataFrame predicates. Threshold terms are stated in the question and notes, such as `Profit >= 800` for high profit and `Sales < 700` for low sales.
4. Expected semantic answers were written from the matching CSV cells. Tests confirm that each named product/customer and requested sales, profit, quantity, category, or date value occurs in the referenced rows.
5. Numerical answers were independently calculated from all 12 rows: `SUM(Sales)`, `SUM(Profit)`, distinct order count, revenue divided by distinct orders, and profit divided by revenue times 100.
6. Neither ChromaDB rankings nor generated model answers were used to establish relevance or correctness.

Numerical entries intentionally have empty `relevant_source_rows`: they are full-dataset calculations rather than retrieval questions.

## Validation

Run the focused validation tests from the project root:

```powershell
.\venv\Scripts\python.exe -m pytest -q tests/test_evaluation_benchmark.py
```

The tests also ensure duplicate IDs, incorrect type labels, out-of-range rows, and incorrect question counts are rejected. Retrieval metrics and an evaluation dashboard are deliberately outside Round 1.
