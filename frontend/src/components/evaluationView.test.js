import assert from 'node:assert/strict'
import test from 'node:test'
import { formatMetric, retrievalOutcome } from './evaluationView.js'

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
