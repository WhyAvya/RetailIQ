import assert from 'node:assert/strict'
import test from 'node:test'
import { chatReducer, initialChatState } from './chatState.js'

test('switching datasets clears messages and pending state', () => {
  const populated = {
    datasetId: 'first',
    pending: true,
    messages: [{ id: '1', question: 'Old dataset question' }],
  }
  assert.deepEqual(chatReducer(populated, { type: 'dataset_changed', datasetId: 'second' }), {
    datasetId: 'second',
    pending: false,
    messages: [],
  })
})

test('duplicate request starts are ignored and clear removes the conversation', () => {
  const pending = chatReducer(initialChatState, { type: 'request_started' })
  assert.equal(chatReducer(pending, { type: 'request_started' }), pending)
  const withMessage = chatReducer(pending, {
    type: 'request_succeeded',
    message: { id: '1', question: 'Revenue?', answer: '45,150' },
  })
  assert.equal(withMessage.messages.length, 1)
  assert.deepEqual(chatReducer(withMessage, { type: 'clear' }).messages, [])
})

test('a failed request leaves loading state and stores its friendly error', () => {
  const pending = { ...initialChatState, datasetId: 'selected', pending: true }
  const failed = chatReducer(pending, {
    type: 'request_failed',
    message: { id: 'error', question: 'Ask AI', error: 'Temporarily unavailable' },
  })
  assert.equal(failed.pending, false)
  assert.equal(failed.messages[0].error, 'Temporarily unavailable')
})
