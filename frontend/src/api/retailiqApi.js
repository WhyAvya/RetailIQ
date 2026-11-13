const API_BASE_URL = import.meta.env?.VITE_API_BASE_URL || 'http://localhost:8000'
let csrfToken = ''

export class ApiError extends Error {
  constructor(message, status = 0) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function requestJson(path, options = {}) {
  const timeoutSignal = AbortSignal.timeout(45000)
  const signal = options.signal
    ? AbortSignal.any([options.signal, timeoutSignal])
    : timeoutSignal

  try {
    const cookieCsrfToken = (typeof document === 'undefined' ? '' : document.cookie)
      .split('; ')
      .find((item) => item.startsWith('retailiq_csrf='))
      ?.split('=')[1]
    const headers = new Headers(options.headers || {})
    const unsafe = !['GET', 'HEAD'].includes((options.method || 'GET').toUpperCase())
    if (unsafe && !csrfToken && !cookieCsrfToken && typeof document !== 'undefined') {
      const bootstrap = await fetch(`${API_BASE_URL}/api/session`, { credentials: 'include' })
      const session = await bootstrap.json().catch(() => ({}))
      if (!bootstrap.ok || !session.csrf_token) throw new ApiError('A secure session could not be started.', bootstrap.status)
      csrfToken = session.csrf_token
    }
    if (unsafe && (csrfToken || cookieCsrfToken)) {
      headers.set('X-CSRF-Token', csrfToken || decodeURIComponent(cookieCsrfToken))
    }
    const response = await fetch(`${API_BASE_URL}${path}`, {
      ...options,
      headers,
      credentials: 'include',
      signal,
    })
    const data = await response.json().catch(() => ({}))
    if (!response.ok) {
      throw new ApiError(
        typeof data.detail === 'string' ? data.detail : 'The request could not be completed.',
        response.status,
      )
    }
    return data
  } catch (error) {
    if (error instanceof ApiError) throw error
    if (error.name === 'TimeoutError') {
      throw new ApiError('The request timed out. Please try again.', 408)
    }
    if (error.name === 'AbortError') throw error
    throw new ApiError('RetailIQ could not reach the backend. Please check that it is running.')
  }
}

export async function fetchUploads(options = {}) {
  const data = await requestJson('/api/uploads', options)
  return Array.isArray(data.uploads) ? data.uploads : []
}

export function fetchEvaluationReport(options = {}) {
  return requestJson('/api/evaluation/report', options)
}

export function fetchCurrencyMetadata(uploadId, options = {}) {
  return requestJson(`/api/analytics/${uploadId}/currency`, options)
}

export function uploadCsv(file, options = {}) {
  const body = new FormData()
  body.append('file', file)
  return requestJson('/api/upload', { ...options, method: 'POST', body })
}

export function fetchDatasetPreview(uploadId, options = {}) {
  return requestJson(`/api/uploads/${uploadId}/preview`, options)
}

export function fetchDashboard(uploadId, query = '', options = {}) {
  return requestJson(`/api/analytics/${uploadId}/dashboard${query ? `?${query}` : ''}`, options)
}

export function fetchFilterOptions(uploadId, options = {}) {
  return requestJson(`/api/analytics/${uploadId}/filter-options`, options)
}

export function clearSessionData(options = {}) {
  return requestJson('/api/session', { ...options, method: 'DELETE' }).finally(() => { csrfToken = '' })
}

export function fetchIndexStatus(uploadId, options = {}) {
  return requestJson(`/api/uploads/${uploadId}/index-status`, options)
}

export function retryIndexing(uploadId, options = {}) {
  return requestJson(`/api/uploads/${uploadId}/index`, {
    ...options,
    method: 'POST',
  })
}

export function askRetailIQ(uploadId, question, topK = 3, options = {}) {
  return requestJson('/api/rag/ask', {
    ...options,
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ upload_id: uploadId, question, top_k: topK }),
  })
}

export function friendlyRagError(error) {
  const detail = error?.message || ''
  if (error?.status === 503 && detail.toLowerCase().includes('rate limit')) {
    return 'The AI model is temporarily unavailable due to rate limits. Please try again later.'
  }
  if (error?.status === 404 && detail.toLowerCase().includes('not been indexed')) {
    return 'This dataset has not been indexed for AI search yet.'
  }
  if (error?.status === 404) return 'The selected dataset could not be found.'
  if (error?.status === 504 || error?.status === 408) {
    return 'The AI request timed out. Please try again later.'
  }
  if (error?.status === 503 && detail.toLowerCase().includes('not configured')) {
    return 'The AI model is not configured on the backend.'
  }
  return detail || 'RetailIQ could not answer this question.'
}
