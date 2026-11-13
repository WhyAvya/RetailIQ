import assert from 'node:assert/strict'
import test from 'node:test'
import {
  ApiError,
  askRetailIQ,
  fetchEvaluationReport,
  fetchCurrencyMetadata,
  fetchIndexStatus,
  friendlyRagError,
  retryIndexing,
} from './retailiqApi.js'

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
      optimized_c8_retrieval: { semantic_questions_evaluated: 5 },
      structured_evaluation: { questions_evaluated: 20 },
      phase5_live: { queries_completed: 5, llm_invocation_count: 1 },
      answer_quality: {
        verified_live_measurement: { answers_evaluated: 3, faithfulness: 0.75 },
        live_llm_evaluation: { status: 'pending', faithfulness: null },
      },
    }), { status: 200, headers: { 'Content-Type': 'application/json' } })
  }
  try {
    const result = await fetchEvaluationReport()
    assert.match(requestUrl, /\/api\/evaluation\/report$/)
    assert.equal(result.retrieval.result_kind, 'actual_chromadb')
    assert.equal(result.optimized_c8_retrieval.semantic_questions_evaluated, 5)
    assert.equal(result.structured_evaluation.questions_evaluated, 20)
    assert.equal(result.phase5_live.llm_invocation_count, 1)
    assert.equal(result.answer_quality.verified_live_measurement.faithfulness, 0.75)
    assert.equal(result.answer_quality.live_llm_evaluation.status, 'pending')
    assert.equal(result.answer_quality.live_llm_evaluation.faithfulness, null)
  } finally {
    globalThis.fetch = originalFetch
  }
})

test('index status and retry use the upload-scoped endpoints', async () => {
  const originalFetch = globalThis.fetch
  const requests = []
  globalThis.fetch = async (url, options = {}) => {
    requests.push({ url, method: options.method || 'GET' })
    return new Response(JSON.stringify({ state: 'pending', ready: false }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    })
  }
  try {
    await fetchIndexStatus('upload-123')
    await retryIndexing('upload-123')
    assert.match(requests[0].url, /\/api\/uploads\/upload-123\/index-status$/)
    assert.equal(requests[0].method, 'GET')
    assert.match(requests[1].url, /\/api\/uploads\/upload-123\/index$/)
    assert.equal(requests[1].method, 'POST')
  } finally {
    globalThis.fetch = originalFetch
  }
})

test('currency metadata uses the selected upload-scoped endpoint', async () => {
  const originalFetch = globalThis.fetch
  let requestUrl
  globalThis.fetch = async (url) => {
    requestUrl = url
    return new Response(JSON.stringify({
      upload_id: 'upload-123',
      currency: { status: 'detected', detected_code: 'USD', detected_codes: ['USD'] },
    }), { status: 200, headers: { 'Content-Type': 'application/json' } })
  }
  try {
    const result = await fetchCurrencyMetadata('upload-123')
    assert.match(requestUrl, /\/api\/analytics\/upload-123\/currency$/)
    assert.equal(result.currency.detected_code, 'USD')
  } finally {
    globalThis.fetch = originalFetch
  }
})
