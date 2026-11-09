import assert from 'node:assert/strict'
import test from 'node:test'
import { ApiError, askRetailIQ, fetchEvaluationReport, friendlyRagError } from './retailiqApi.js'

test('askRetailIQ sends the selected upload id and question', async () => {
  const originalFetch = globalThis.fetch
  let request
  globalThis.fetch = async (url, options) => {
    request = { url, options }
    return new Response(JSON.stringify({ mode: 'analytics', answer: 'Total revenue is 45,150.', sources: [] }), { status: 200, headers: { 'Content-Type': 'application/json' } })
  }
  try {
    const result = await askRetailIQ('upload-123', 'What is total revenue?', 3)
    assert.equal(JSON.parse(request.options.body).upload_id, 'upload-123')
    assert.equal(JSON.parse(request.options.body).top_k, 3)
    assert.equal(result.mode, 'analytics')
    assert.deepEqual(result.sources, [])
  } finally {
    globalThis.fetch = originalFetch
  }
})

test('askRetailIQ preserves semantic source rows and similarity scores', async () => {
  const originalFetch = globalThis.fetch
  globalThis.fetch = async () => new Response(JSON.stringify({
    mode: 'semantic_rag',
    answer: 'The strongest accessory examples are in the cited rows.',
    sources: [
      { chunk_id: 'one', source_row: 1, similarity_score: 0.291889, metadata: { category: 'Technology' }, document_excerpt: 'Row 1' },
      { chunk_id: 'seven', source_row: 7, similarity_score: 0.280409, metadata: { category: 'Technology' }, document_excerpt: 'Row 7' },
      { chunk_id: 'ten', source_row: 10, similarity_score: 0.268301, metadata: { category: 'Technology' }, document_excerpt: 'Row 10' },
    ],
  }), { status: 200, headers: { 'Content-Type': 'application/json' } })
  try {
    const result = await askRetailIQ('retail-upload', 'Which accessory orders were profitable?', 3)
    assert.deepEqual(result.sources.map((source) => source.source_row), [1, 7, 10])
    assert.deepEqual(result.sources.map((source) => source.similarity_score), [0.291889, 0.280409, 0.268301])
  } finally {
    globalThis.fetch = originalFetch
  }
})

test('rate limits and unindexed datasets receive user-friendly messages', () => {
  assert.equal(
    friendlyRagError(new ApiError('The language model is temporarily rate limited.', 503)),
    'The AI model is temporarily unavailable due to rate limits. Please try again later.',
  )
  assert.equal(
    friendlyRagError(new ApiError('The selected dataset has not been indexed for retrieval.', 404)),
    'This dataset has not been indexed for AI search yet.',
  )
})

test('fetchEvaluationReport loads saved actual and pending evaluation states', async () => {
  const originalFetch = globalThis.fetch
  let requestUrl
  globalThis.fetch = async (url) => {
    requestUrl = url
    return new Response(JSON.stringify({
      retrieval: { result_kind: 'actual_chromadb', aggregate_metrics: { precision_at_3: 0.444444 } },
      answer_quality: { live_llm_evaluation: { status: 'pending', faithfulness: null } },
    }), { status: 200, headers: { 'Content-Type': 'application/json' } })
  }
  try {
    const result = await fetchEvaluationReport()
    assert.match(requestUrl, /\/api\/evaluation\/report$/)
    assert.equal(result.retrieval.result_kind, 'actual_chromadb')
    assert.equal(result.answer_quality.live_llm_evaluation.status, 'pending')
    assert.equal(result.answer_quality.live_llm_evaluation.faithfulness, null)
  } finally {
    globalThis.fetch = originalFetch
  }
})
