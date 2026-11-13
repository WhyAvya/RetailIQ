export function formatMetric(value, style = 'percent') {
  if (typeof value !== 'number' || !Number.isFinite(value)) return 'Pending'
  return style === 'decimal' ? value.toFixed(3) : `${(value * 100).toFixed(1)}%`
}

export function retrievalOutcome(question) {
  if (question.failure) return { label: 'Failed', tone: 'red' }
  const relevant = new Set(question.relevant_source_rows || [])
  const retrieved = (question.retrieved_source_rows || []).slice(0, 5)
  const hits = retrieved.filter((row) => relevant.has(row)).length
  if (hits === 0) return { label: 'No top-5 match', tone: 'red' }
  if (hits === relevant.size) return { label: 'All relevant found', tone: 'emerald' }
  return { label: `${hits} of ${relevant.size} found`, tone: 'amber' }
}

export function c8ComparisonRows(metrics = {}) {
  return [
    ['Precision@3', 'precision_at_3', 'percent'],
    ['Precision@5', 'precision_at_5', 'percent'],
    ['Recall@3', 'recall_at_3', 'percent'],
    ['Recall@5', 'recall_at_5', 'percent'],
    ['MRR@5', 'reciprocal_rank_at_5', 'decimal'],
  ].map(([label, key, style]) => ({ label, style, ...metrics[key] }))
}

export function formatLatency(value) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return 'Unavailable'
  return value >= 1000 ? `${(value / 1000).toFixed(2)} s` : `${Math.round(value)} ms`
}

export function citationOutcome(status, valid) {
  if (status === 'valid' && valid) return 'Valid'
  if (status?.startsWith('not_applicable')) return 'Not required'
  return valid ? 'Valid' : 'Failed'
}
