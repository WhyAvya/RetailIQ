import assert from 'node:assert/strict'
import test from 'node:test'
import { indexingStatusText, isRagReady } from './indexingStatus.js'

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
