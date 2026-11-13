import assert from 'node:assert/strict'
import test from 'node:test'
import {
  c8ComparisonRows,
  citationOutcome,
  formatLatency,
  formatMetric,
  retrievalOutcome,
} from './evaluationView.js'

test('evaluation metrics format verified values and pending states', () => {
  assert.equal(formatMetric(0.444444), '44.4%')
  assert.equal(formatMetric(0.588889, 'decimal'), '0.589')
  assert.equal(formatMetric(null), 'Pending')
})

test('retrieval outcomes distinguish complete, partial, and missed rankings', () => {
  assert.deepEqual(retrievalOutcome({ relevant_source_rows: [1, 7], retrieved_source_rows: [1, 7, 3] }), { label: 'All relevant found', tone: 'emerald' })
  assert.deepEqual(retrievalOutcome({ relevant_source_rows: [1, 7], retrieved_source_rows: [1, 3] }), { label: '1 of 2 found', tone: 'amber' })
  assert.deepEqual(retrievalOutcome({ relevant_source_rows: [1, 7], retrieved_source_rows: [2, 3] }), { label: 'No top-5 match', tone: 'red' })
})

test('C8 comparison rows keep baseline and optimized metrics separate', () => {
  const rows = c8ComparisonRows({
    precision_at_3: { baseline: 0.2667, optimized: 1, delta: 0.7333 },
    reciprocal_rank_at_5: { baseline: 0.2667, optimized: 1, delta: 0.7333 },
  })
  assert.deepEqual(rows[0], {
    label: 'Precision@3',
    style: 'percent',
    baseline: 0.2667,
    optimized: 1,
    delta: 0.7333,
  })
  assert.equal(rows[4].label, 'MRR@5')
  assert.equal(rows[4].style, 'decimal')
})

test('live evaluation presentation distinguishes latency and citation applicability', () => {
  assert.equal(formatLatency(190.74), '191 ms')
  assert.equal(formatLatency(44554.12), '44.55 s')
  assert.equal(citationOutcome('valid', true), 'Valid')
  assert.equal(citationOutcome('not_applicable_exact_csv_calculation', true), 'Not required')
})
