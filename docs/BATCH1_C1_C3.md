# RetailIQ Batch 1: C1, C2, and C3

Batch 1 adds isolated planning, structured analytics, and metadata-filtered
retrieval components. They are not connected to the live RAG pipeline or API.

## Metadata audit and enrichment

The original full-dataset Chroma records already had valid upload IDs, source
rows, regions, categories, sub-categories, order IDs, and ISO order/ship dates.
All 9,994 records were missing typed sales, profit, quantity, discount, product
identity/name, numeric date fields, and several useful exact dimensions.

`backend/chroma_metadata.py` verifies deterministic chunk identity and upload
isolation before using Chroma's metadata-only `update` operation. The completed
run added the missing fields to 9,994 records. It supplied no documents or
embeddings, and the collection retained exactly 9,994 unique IDs and source
rows. The before/after audit is saved in
`evaluation/reports/batch1_chroma_metadata_audit.json`.

## Query planning and Pandas execution

`backend/query_planner.py` produces typed, allowlisted plans for:

- pure semantic retrieval;
- region, category, sub-category, inclusive date, sales, and profit filters;
- counts, sums, means, medians, and profit margin;
- ascending or descending top-N ranking by sales, profit, or quantity;
- row, order, and product grains;
- grouped calculations by region, category, sub-category, or order year;
- quoted product-name terms combined with structured constraints.

Predictive, causal, and correlation requests are unsupported. Rankings without
a metric or grain, and calculations without a required metric or grain, are
marked ambiguous. The planner never produces executable Python or SQL.

`backend/structured_analytics.py` executes valid plans with explicit Pandas
operations. It applies only known columns and operators, groups orders and
products before grain-sensitive averages/counts/rankings, and refuses invalid
or semantic-only plans.

## Metadata-filtered retrieval

`backend/metadata_filtered_retrieval.py` translates validated constraints into
Chroma `where` clauses that always include the upload ID. ISO dates are mapped
to numeric epoch-day metadata for safe range comparisons. MiniLM similarity is
then used only within matching records. With no structured constraints, the
module uses only the upload predicate and preserves existing semantic ranking.

This module is intentionally not imported or called by `backend/rag_pipeline.py`.
Integration, fallbacks, reranking, and evaluation improvements remain Batch 2
work.
