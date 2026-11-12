# RetailIQ Semantic Retrieval Optimization

Assessment: **MEANINGFULLY_BETTER**. The frozen C8 benchmark hash remained unchanged.

## Before and after

| Metric | Baseline | Optimized | Delta |
|---|---:|---:|---:|
| Precision@3 | 0.2667 | 1.0000 | +0.7333 |
| Precision@5 | 0.2400 | 1.0000 | +0.7600 |
| Recall@3 | 0.0068 | 0.1090 | +0.1021 |
| Recall@5 | 0.0103 | 0.1816 | +0.1713 |
| MRR@5 | 0.2667 | 1.0000 | +0.7333 |

## Root causes

- Embedding each full row alongside many unrelated attributes likely dilutes the short product-name terms used by C8.
- Vector-only top-5 retrieval misses exact markers, speakers, and batteries terms; widening to 30 still found zero relevant battery rows.
- Frozen relevance is row-level exact substring matching, so repeated products create sets of 9 to 127 rows while top-K remains 3 or 5.
- The labels are narrower than some query wording (for example, 'related to printers') and include substring artifacts such as 'Ideal Clamps' for 'lamp'.

## Implemented changes

- Retrieve up to 30 MiniLM candidates instead of only the requested K.
- Union those candidates with up to 30 lightweight lexical matches over product name, sub-category, and category.
- Rerank the union with 70% lexical field coverage and 30% cosine similarity, then return the existing top-K schema.
- Cache searchable text for up to eight uploads and invalidate it after reindexing.

No embedding text or stored vectors changed, so no reindex or index-version migration was required.

## Per-query results

| Query | P@5 before | P@5 after | R@5 before | R@5 after | MRR@5 before | MRR@5 after | Regression |
|---|---:|---:|---:|---:|---:|---:|---:|
| C8I2-SEM-001 | 0.0000 | 1.0000 | 0.0000 | 0.5556 | 0.0000 | 1.0000 | false |
| C8I2-SEM-002 | 0.0000 | 1.0000 | 0.0000 | 0.0769 | 0.0000 | 1.0000 | false |
| C8I2-SEM-003 | 1.0000 | 1.0000 | 0.0439 | 0.0439 | 1.0000 | 1.0000 | false |
| C8I2-SEM-004 | 0.2000 | 1.0000 | 0.0079 | 0.0394 | 0.3333 | 1.0000 | false |
| C8I2-SEM-005 | 0.0000 | 1.0000 | 0.0000 | 0.1923 | 0.0000 | 1.0000 | false |

## Latency and indexing

- Baseline average including first query: 950.37 ms.
- Optimized average including first cache fill: 920.70 ms.
- Baseline warm average: 446.84 ms.
- Optimized warm average: 325.09 ms.
- Additional indexing overhead: 0 ms; existing embeddings are reused.
- Citation/source-row validation: all returned rows were valid.

## Remaining limitations

- Recall@5 remains bounded by five returned rows and large row-level relevance sets; it cannot reach 1.0 for these labels.
- Lexical matching favors explicit catalog terms and does not replace semantic understanding for paraphrases.
- The first query for an upload pays a cache-fill cost; subsequent measured queries are near baseline latency.
- The frozen five-query semantic set is small and lexically oriented, so broader production relevance still needs monitoring.
