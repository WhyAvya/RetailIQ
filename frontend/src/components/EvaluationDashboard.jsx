import { useEffect, useState } from 'react'
import { fetchEvaluationReport } from '../api/retailiqApi.js'
import { formatMetric, retrievalOutcome } from './evaluationView.js'

const METRICS = [
  ['Precision@3', 'precision_at_3', 'percent'],
  ['Precision@5', 'precision_at_5', 'percent'],
  ['Recall@3', 'recall_at_3', 'percent'],
  ['Recall@5', 'recall_at_5', 'percent'],
  ['MRR@5', 'mrr', 'decimal'],
]

const TONE_CLASSES = {
  emerald: 'border-emerald-200 bg-emerald-50 text-emerald-700',
  amber: 'border-amber-200 bg-amber-50 text-amber-800',
  red: 'border-red-200 bg-red-50 text-red-700',
}

const STATUS_CLASSES = {
  Actual: 'bg-emerald-50 text-emerald-700',
  Pending: 'bg-amber-100 text-amber-800',
  Fixture: 'bg-slate-200 text-slate-600',
}

function MetricCard({ label, value, style = 'percent', status = 'Actual' }) {
  return (
    <article className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex items-start justify-between gap-3">
        <p className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">{label}</p>
        <span className={`rounded-full px-2 py-1 text-[10px] font-semibold uppercase ${STATUS_CLASSES[status]}`}>{status}</span>
      </div>
      <p className="mt-3 text-2xl font-semibold tracking-tight text-slate-950">{formatMetric(value, style)}</p>
    </article>
  )
}

function RowChips({ rows, relevantRows }) {
  const relevant = new Set(relevantRows)
  if (!rows.length) return <span className="text-xs text-slate-400">None</span>
  return (
    <span className="flex flex-wrap gap-1.5">
      {rows.map((row) => (
        <span key={row} className={`rounded-md border px-2 py-1 text-xs font-medium ${relevant.has(row) ? 'border-emerald-200 bg-emerald-50 text-emerald-700' : 'border-slate-200 bg-slate-50 text-slate-500'}`}>
          {row}
        </span>
      ))}
    </span>
  )
}

function RetrievalQuestion({ item }) {
  const outcome = retrievalOutcome(item)
  return (
    <details className="rounded-xl border border-slate-200 bg-white">
      <summary className="cursor-pointer list-none p-4 sm:p-5">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div className="min-w-0">
            <p className="text-xs font-semibold text-emerald-700">{item.id}</p>
            <p className="mt-1 text-sm font-medium leading-6 text-slate-800">{item.question}</p>
          </div>
          <span className={`w-fit shrink-0 rounded-full border px-2.5 py-1 text-xs font-semibold ${TONE_CLASSES[outcome.tone]}`}>
            {outcome.label}
          </span>
        </div>
      </summary>
      <div className="grid gap-4 border-t border-slate-200 bg-slate-50/60 p-4 sm:grid-cols-2 sm:p-5">
        <div>
          <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">Ground-truth rows</p>
          <RowChips rows={item.relevant_source_rows} relevantRows={item.relevant_source_rows} />
        </div>
        <div>
          <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">Retrieved rows</p>
          <RowChips rows={item.retrieved_source_rows} relevantRows={item.relevant_source_rows} />
        </div>
        <div className="sm:col-span-2 flex flex-wrap gap-x-5 gap-y-2 border-t border-slate-200 pt-4 text-xs text-slate-600">
          <span>P@3 <strong className="text-slate-800">{formatMetric(item.metrics.precision_at_3)}</strong></span>
          <span>R@3 <strong className="text-slate-800">{formatMetric(item.metrics.recall_at_3)}</strong></span>
          <span>P@5 <strong className="text-slate-800">{formatMetric(item.metrics.precision_at_5)}</strong></span>
          <span>R@5 <strong className="text-slate-800">{formatMetric(item.metrics.recall_at_5)}</strong></span>
          <span>RR <strong className="text-slate-800">{formatMetric(item.metrics.reciprocal_rank, 'decimal')}</strong></span>
        </div>
      </div>
    </details>
  )
}

function EvaluationDashboard() {
  const [report, setReport] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const controller = new AbortController()
    fetchEvaluationReport({ signal: controller.signal })
      .then(setReport)
      .catch((requestError) => {
        if (requestError.name !== 'AbortError') setError(requestError.message)
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false)
      })
    return () => controller.abort()
  }, [])

  if (loading) return <p role="status" className="rounded-xl border border-slate-200 bg-white p-6 text-sm text-slate-500">Loading evaluation results...</p>
  if (error) return <p role="alert" className="rounded-xl border border-red-200 bg-red-50 p-5 text-sm text-red-700">{error}</p>

  const retrieval = report.retrieval
  const answer = report.answer_quality
  const live = answer.live_llm_evaluation
  const fixtures = answer.fixture_validation
  const numerical = answer.numerical_correctness

  return (
    <section aria-labelledby="evaluation-heading" className="space-y-8">
      <header className="rounded-2xl border border-slate-200 bg-white p-6 shadow-[0_18px_55px_-30px_rgba(15,23,42,0.25)] sm:p-8">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-emerald-700">Phase 5 evaluation</p>
            <h1 id="evaluation-heading" className="mt-2 text-3xl font-semibold tracking-tight text-slate-950">Retrieval and answer quality</h1>
            <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-600">Read-only results from the saved benchmark reports. Actual system measurements, deterministic checks, fixtures, and pending live evaluation are labeled separately.</p>
          </div>
          <span className="w-fit rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1.5 text-xs font-semibold text-emerald-700">15 semantic questions</span>
        </div>
      </header>

      <section aria-labelledby="retrieval-results-heading">
        <div className="mb-4">
          <p className="text-xs font-semibold uppercase tracking-[0.15em] text-emerald-700">Actual ChromaDB results</p>
          <h2 id="retrieval-results-heading" className="mt-1 text-xl font-semibold text-slate-900">Retrieval metrics</h2>
          <p className="mt-1 text-sm text-slate-500">Measured against verified 1-based source-row labels. Failures: {retrieval.failure_count}.</p>
        </div>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
          {METRICS.map(([label, key, style]) => <MetricCard key={key} label={label} value={retrieval.aggregate_metrics[key]} style={style} />)}
        </div>
      </section>

      <section aria-labelledby="question-results-heading">
        <h2 id="question-results-heading" className="text-xl font-semibold text-slate-900">Per-question retrieval</h2>
        <p className="mt-1 text-sm text-slate-500">Expand a question to compare ranked rows with its verified ground truth.</p>
        <div className="mt-4 space-y-3">
          {retrieval.questions.map((item) => <RetrievalQuestion key={item.id} item={item} />)}
        </div>
      </section>

      <section aria-labelledby="answer-quality-heading" className="space-y-5">
        <div>
          <h2 id="answer-quality-heading" className="text-xl font-semibold text-slate-900">Answer quality</h2>
          <p className="mt-1 text-sm text-slate-500">Claim support and relevance require reviewed live answers; deterministic numerical answers are verified separately.</p>
        </div>

        <div className="rounded-2xl border border-amber-200 bg-amber-50 p-5 sm:p-6">
          <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
            <h3 className="font-semibold text-amber-950">Live LLM evaluation pending</h3>
            <span className="w-fit rounded-full bg-amber-100 px-2.5 py-1 text-xs font-semibold uppercase text-amber-800">Pending</span>
          </div>
          <p className="mt-2 text-sm leading-6 text-amber-900">{live.reason}</p>
          <div className="mt-4 grid gap-3 sm:grid-cols-3">
            <MetricCard label="Faithfulness" value={live.faithfulness} status="Pending" />
            <MetricCard label="Answer relevance" value={live.answer_relevance} status="Pending" />
            <MetricCard label="Unsupported claims" value={live.unsupported_claim_rate} status="Pending" />
          </div>
        </div>

        <div className="rounded-2xl border border-slate-200 bg-white p-5 sm:p-6">
          <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <h3 className="font-semibold text-slate-900">Numerical correctness</h3>
              <p className="mt-1 text-sm text-slate-500">Actual deterministic analytics pipeline versus benchmark values.</p>
            </div>
            <span className="w-fit rounded-full bg-emerald-100 px-2.5 py-1 text-xs font-semibold text-emerald-800">{numerical.correct_answers}/{numerical.questions_evaluated} correct</span>
          </div>
          <div className="mt-4 overflow-x-auto rounded-xl border border-slate-200">
            <table className="min-w-full divide-y divide-slate-200 text-left text-xs">
              <thead className="bg-slate-50 text-slate-600"><tr><th className="px-3 py-2.5">Metric</th><th className="px-3 py-2.5">Expected</th><th className="px-3 py-2.5">Actual</th><th className="px-3 py-2.5">Result</th></tr></thead>
              <tbody className="divide-y divide-slate-100 text-slate-600">
                {numerical.questions.map((item) => (
                  <tr key={item.id}><td className="px-3 py-2.5 font-medium text-slate-800">{item.metric.replaceAll('_', ' ')}</td><td className="px-3 py-2.5">{item.expected_value}</td><td className="px-3 py-2.5">{typeof item.actual_value === 'number' ? item.actual_value.toFixed(item.display_decimals) : 'Unavailable'}</td><td className="px-3 py-2.5 font-semibold text-emerald-700">{item.correct ? 'Verified' : 'Mismatch'}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="rounded-2xl border border-slate-200 bg-slate-100/70 p-5 sm:p-6">
          <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <h3 className="font-semibold text-slate-900">Scoring framework fixtures</h3>
              <p className="mt-1 text-sm text-slate-600">{fixtures.warning}</p>
            </div>
            <span className="w-fit rounded-full bg-slate-200 px-2.5 py-1 text-xs font-semibold uppercase text-slate-600">Fixture only</span>
          </div>
          <div className="mt-4 grid gap-3 sm:grid-cols-3">
            <MetricCard label="Faithfulness" value={fixtures.faithfulness} status="Fixture" />
            <MetricCard label="Answer relevance" value={fixtures.answer_relevance} status="Fixture" />
            <MetricCard label="Unsupported claims" value={fixtures.unsupported_claim_rate} status="Fixture" />
          </div>
        </div>
      </section>
    </section>
  )
}

export default EvaluationDashboard
