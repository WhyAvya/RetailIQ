export function isRagReady(upload) {
  return upload?.status === 'ready' || upload?.status === 'processed'
}

export function formatIndexingTime(seconds) {
  const remaining = Math.max(0, Math.ceil(Number(seconds) || 0))
  const minutes = Math.floor(remaining / 60)
  const remainder = remaining % 60
  return `${minutes}m ${String(remainder).padStart(2, '0')}s`
}

export function indexingStatusText(upload) {
  const status = typeof upload === 'string' ? upload : upload?.status
  if (status === 'ready') return 'Ready for AI questions'
  if (status === 'processed') return 'Ready (legacy upload)'
  if (status === 'failed') return 'Indexing failed'
  if (status === 'indexing') {
    if (Number.isFinite(upload?.estimated_seconds_remaining)) {
      return `Indexing dataset for AI search... About ${formatIndexingTime(upload.estimated_seconds_remaining)} remaining`
    }
    return 'Indexing dataset for AI search...'
  }
  if (status === 'pending' && Number.isFinite(upload?.estimated_seconds_remaining)) {
    return `Waiting to start AI indexing... Estimated time ${formatIndexingTime(upload.estimated_seconds_remaining)}`
  }
  return 'Waiting to start AI indexing...'
}
