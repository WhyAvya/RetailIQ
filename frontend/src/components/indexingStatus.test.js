import assert from 'node:assert/strict'
import test from 'node:test'
import { formatIndexingTime, indexingStatusText, isRagReady } from './indexingStatus.js'

test('RAG is enabled for ready and legacy processed uploads', () => {
  assert.equal(isRagReady({ status: 'ready' }), true)
  assert.equal(isRagReady({ status: 'processed' }), true)
  for (const status of ['pending', 'indexing', 'failed']) {
    assert.equal(isRagReady({ status }), false)
  }
})

test('indexing states have user-facing progress text', () => {
  assert.match(indexingStatusText('pending'), /waiting/i)
  assert.match(indexingStatusText('indexing'), /indexing/i)
  assert.match(indexingStatusText('failed'), /failed/i)
  assert.match(indexingStatusText('ready'), /ready/i)
})

test('indexing status formats estimated minutes and seconds', () => {
  assert.equal(formatIndexingTime(125), '2m 05s')
  assert.match(
    indexingStatusText({ status: 'indexing', estimated_seconds_remaining: 125 }),
    /2m 05s remaining/,
  )
  assert.match(
    indexingStatusText({ status: 'pending', estimated_seconds_remaining: 90 }),
    /1m 30s/,
  )
})
