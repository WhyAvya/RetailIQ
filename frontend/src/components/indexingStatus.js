export function isRagReady(upload) {
  return upload?.status === 'ready' || upload?.status === 'processed'
}

export function indexingStatusText(status) {
  if (status === 'ready') return 'Ready for AI questions'
  if (status === 'processed') return 'Ready (legacy upload)'
  if (status === 'failed') return 'Indexing failed'
  if (status === 'indexing') return 'Indexing dataset for AI search...'
  return 'Waiting to start AI indexing...'
}
