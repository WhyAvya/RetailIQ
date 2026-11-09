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
