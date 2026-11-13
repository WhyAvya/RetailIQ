import { useEffect, useState } from 'react'
import { fetchEvaluationReport } from '../api/retailiqApi.js'
import {
  c8ComparisonRows,
  citationOutcome,
  formatLatency,
  formatMetric,
  retrievalOutcome,
} from './evaluationView.js'
import Reveal from './Reveal.jsx'

const GENERAL_METRICS = [
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
  Actual: 'bg-emerald-100 text-emerald-800',
  Live: 'bg-emerald-100 text-emerald-800',
  Deterministic: 'bg-blue-100 text-blue-800',
  Fixture: 'bg-slate-200 text-slate-600',
}

function MetricCard({ label, value, style = 'percent', status = 'Actual', displayValue }) {
  return (
    <article className="motion-surface rounded-xl border border-slate-200 bg-white p-4 shadow-[0_12px_30px_-24px_rgba(6,78,59,0.45)]">
      <div className="flex items-start justify-between gap-3">
        <p className="text-xs font-semibold uppercase tracking-[0.1em] text-slate-500">{label}</p>
        <span className={`rounded-full px-2 py-1 text-[10px] font-semibold uppercase ${STATUS_CLASSES[status]}`}>
          {status}
        </span>
      </div>
      <p className="mt-3 break-words text-2xl font-semibold tracking-tight text-slate-950">
        {displayValue ?? formatMetric(value, style)}
      </p>
    </article>
  )
}

function SectionHeading({ id, eyebrow, title, children }) {
  return (
    <div className="mb-4">
      <p className="text-xs font-semibold uppercase tracking-[0.15em] text-emerald-700">{eyebrow}</p>
      <h2 id={id} className="mt-1 text-xl font-semibold text-slate-900">{title}</h2>
      {children && <p className="mt-1 max-w-3xl text-sm leading-6 text-slate-600">{children}</p>}
    </div>
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
      <div className="grid gap-4 border-t border-slate-200 bg-[#F5F8F3] p-4 sm:grid-cols-2 sm:p-5">
        <div>
          <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">Ground-truth rows</p>
          <RowChips rows={item.relevant_source_rows} relevantRows={item.relevant_source_rows} />
        </div>
        <div>
          <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">Retrieved rows</p>
          <RowChips rows={item.retrieved_source_rows} relevantRows={item.relevant_source_rows} />
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

  if (loading) return <p role="status" className="rounded-xl border border-slate-200 bg-white p-6 text-sm text-slate-500">Loading verified evaluation results...</p>
  if (error) return <p role="alert" className="rounded-xl border border-red-200 bg-red-50 p-5 text-sm text-red-700">{error}</p>

  const retrieval = report.retrieval
  const optimized = report.optimized_c8_retrieval
  const structured = report.structured_evaluation
  const phase5 = report.phase5_live
  const answer = report.answer_quality
  const liveAnswer = answer.verified_live_measurement
  const fixture = answer.fixture_validation
  const pending = answer.live_llm_evaluation
  const numerical = answer.numerical_correctness

  return (
    <section aria-labelledby="evaluation-heading" className="space-y-10">
      <Reveal as="header" revealKey="evaluation-header" className="rounded-2xl border border-emerald-950/10 bg-white p-6 shadow-[0_18px_55px_-30px_rgba(6,78,59,0.28)] sm:p-8">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-emerald-700">Verified reports</p>
            <h1 id="evaluation-heading" className="mt-2 text-3xl font-semibold tracking-tight text-slate-950">RetailIQ evaluation results</h1>
            <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-600">
              Results are shown by benchmark so differently scoped tests are not combined. Actual system measurements, deterministic checks, fixtures, and pending attempts are labeled separately.
            </p>
          </div>
          <span className="w-fit rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1.5 text-xs font-semibold text-emerald-800">Saved report evidence</span>
        </div>
        <dl className="mt-6 grid gap-3 border-t border-slate-200 pt-5 text-xs sm:grid-cols-2 lg:grid-cols-4">
          <div><dt className="font-semibold text-slate-800">Precision</dt><dd className="mt-1 leading-5 text-slate-500">Share of returned rows that are relevant.</dd></div>
          <div><dt className="font-semibold text-slate-800">Recall</dt><dd className="mt-1 leading-5 text-slate-500">Share of all labeled relevant rows retrieved.</dd></div>
          <div><dt className="font-semibold text-slate-800">MRR</dt><dd className="mt-1 leading-5 text-slate-500">How early the first relevant result appears.</dd></div>
          <div><dt className="font-semibold text-slate-800">Faithfulness</dt><dd className="mt-1 leading-5 text-slate-500">Share of reviewed answer claims supported by evidence.</dd></div>
        </dl>
      </Reveal>

      <Reveal as="section" revealKey="evaluation-general" aria-labelledby="general-retrieval-heading">
        <SectionHeading id="general-retrieval-heading" eyebrow="General semantic retrieval" title="15-question benchmark">
          Actual ChromaDB retrieval measured against verified source-row labels. This benchmark is separate from C8.
        </SectionHeading>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
          {GENERAL_METRICS.map(([label, key, style]) => <MetricCard key={key} label={label} value={retrieval.aggregate_metrics[key]} style={style} />)}
        </div>
        <details className="mt-4 rounded-xl border border-slate-200 bg-white">
          <summary className="cursor-pointer px-4 py-3 text-sm font-semibold text-slate-700 sm:px-5">
            Review all {retrieval.semantic_questions_evaluated} retrieval questions ({retrieval.failure_count} failures)
          </summary>
          <div className="space-y-3 border-t border-slate-200 bg-[#F5F8F3] p-4 sm:p-5">
            {retrieval.questions.map((item) => <RetrievalQuestion key={item.id} item={item} />)}
          </div>
        </details>
      </Reveal>

      <Reveal as="section" revealKey="evaluation-c8" aria-labelledby="c8-retrieval-heading">
        <SectionHeading id="c8-retrieval-heading" eyebrow="Optimized C8 retrieval" title="Five-question baseline vs optimized">
          The frozen C8 labels and top-K settings are unchanged. Large row-level relevance sets limit Recall@K even when the returned rows are relevant.
        </SectionHeading>
        <div className="overflow-x-auto rounded-xl border border-slate-200 bg-white shadow-sm">
          <table className="min-w-full divide-y divide-slate-200 text-left text-sm">
            <thead className="bg-[#F5F8F3] text-xs uppercase tracking-wide text-slate-600"><tr><th className="px-4 py-3">Metric</th><th className="px-4 py-3">Baseline</th><th className="px-4 py-3">Optimized</th><th className="px-4 py-3">Change</th></tr></thead>
            <tbody className="divide-y divide-slate-100">
              {c8ComparisonRows(optimized.metrics).map((row) => (
                <tr key={row.label}><th className="px-4 py-3 font-semibold text-slate-800">{row.label}</th><td className="px-4 py-3 text-slate-600">{formatMetric(row.baseline, row.style)}</td><td className="px-4 py-3 font-semibold text-emerald-800">{formatMetric(row.optimized, row.style)}</td><td className="px-4 py-3 text-emerald-700">+{formatMetric(row.delta, row.style)}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-2 text-xs text-slate-500">{optimized.semantic_questions_evaluated} questions; source-row citations valid: {optimized.citations_valid ? 'yes' : 'no'}.</p>
      </Reveal>

      <Reveal as="section" revealKey="evaluation-structured" aria-labelledby="structured-heading">
        <SectionHeading id="structured-heading" eyebrow="Structured query evaluation" title="Post-fix deterministic accuracy">
          These queries run through the Pandas analytics path without LLM calls. Accuracy is reported independently for each C8 category.
        </SectionHeading>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {Object.entries(structured.metrics).map(([category, value]) => (
            <MetricCard key={category} label={category} value={value} status="Deterministic" displayValue={`${formatMetric(value)} · ${structured.question_count_by_category[category]} queries`} />
          ))}
        </div>
        <p className="mt-2 text-xs text-slate-500">{structured.questions_evaluated} structured questions total; {structured.llm_calls} LLM calls.</p>
      </Reveal>

      <Reveal as="section" revealKey="evaluation-phase5" aria-labelledby="phase5-heading">
        <SectionHeading id="phase5-heading" eyebrow="Phase 5 live evaluation" title="Five-query pipeline check">
          A small live check across semantic, numerical, filtering, ranking, and mixed routes. Five correct cases verify these cases only; they do not establish universal accuracy.
        </SectionHeading>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
          <MetricCard label="Correct" status="Live" displayValue={`${phase5.correct_queries}/${phase5.queries_completed}`} />
          <MetricCard label="Required citations" status="Live" displayValue={`${phase5.valid_citation_checks}/${phase5.citation_checks} valid`} />
          <MetricCard label="Actual Nemotron calls" status="Live" displayValue={String(phase5.llm_invocation_count)} />
          <MetricCard label="Average latency" status="Live" displayValue={formatLatency(phase5.latency.average_ms)} />
          <MetricCard label="Maximum latency" status="Live" displayValue={formatLatency(phase5.latency.max_ms)} />
        </div>
        <p className="mt-3 text-xs leading-5 text-slate-500">Provider: {phase5.provider}. Model: {phase5.model}. Provider errors: {phase5.provider_error_count}. Exact Pandas calculations did not require row citations.</p>
        <div className="mt-4 overflow-x-auto rounded-xl border border-slate-200 bg-white">
          <table className="min-w-[760px] divide-y divide-slate-200 text-left text-xs">
            <thead className="bg-[#F5F8F3] uppercase tracking-wide text-slate-600"><tr><th className="px-3 py-3">Route</th><th className="px-3 py-3">Question</th><th className="px-3 py-3">Result</th><th className="px-3 py-3">Citation</th><th className="px-3 py-3">LLM</th><th className="px-3 py-3">Latency</th></tr></thead>
            <tbody className="divide-y divide-slate-100 text-slate-600">
              {phase5.queries.map((item) => (
                <tr key={`${item.category}-${item.question}`}><td className="px-3 py-3 font-semibold capitalize text-slate-800">{item.category}</td><td className="max-w-md px-3 py-3 leading-5">{item.question}</td><td className="px-3 py-3 font-semibold text-emerald-700">{item.correct ? 'Correct' : 'Incorrect'}</td><td className="px-3 py-3">{citationOutcome(item.citation_status, item.citations_valid)}</td><td className="px-3 py-3">{item.llm_invoked ? 'Nemotron' : 'None'}</td><td className="px-3 py-3">{formatLatency(item.latency_ms)}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      </Reveal>

      <Reveal as="section" revealKey="evaluation-answer-quality" aria-labelledby="answer-quality-heading">
        <SectionHeading id="answer-quality-heading" eyebrow="Answer quality" title="Live evidence, fixtures, and pending work">
          Live scores come from manual claim/evidence review of three real Nemotron answers. Fixture scores validate the scorer only and are not model performance.
        </SectionHeading>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          <MetricCard label="Faithfulness" value={liveAnswer.faithfulness} status="Live" />
          <MetricCard label="Answer relevance" value={liveAnswer.answer_relevance} status="Live" />
          <MetricCard label="Unsupported claims" value={liveAnswer.unsupported_claim_rate} status="Live" />
        </div>
        <p className="mt-2 text-xs leading-5 text-slate-500">{liveAnswer.answers_evaluated} live answers reviewed with {liveAnswer.model}. {liveAnswer.score_provenance}</p>

        <div className="mt-5 grid gap-4 lg:grid-cols-2">
          <article className="motion-surface rounded-xl border border-slate-200 bg-slate-100/70 p-5">
            <div className="flex items-center justify-between gap-3"><h3 className="font-semibold text-slate-900">Scoring fixtures</h3><span className="rounded-full bg-slate-200 px-2.5 py-1 text-[10px] font-semibold uppercase text-slate-600">Fixture only</span></div>
            <p className="mt-2 text-xs leading-5 text-slate-600">{fixture.warning}</p>
            <dl className="mt-4 grid grid-cols-3 gap-2 text-xs"><div><dt className="text-slate-500">Faithfulness</dt><dd className="mt-1 font-semibold text-slate-800">{formatMetric(fixture.faithfulness)}</dd></div><div><dt className="text-slate-500">Relevance</dt><dd className="mt-1 font-semibold text-slate-800">{formatMetric(fixture.answer_relevance)}</dd></div><div><dt className="text-slate-500">Unsupported</dt><dd className="mt-1 font-semibold text-slate-800">{formatMetric(fixture.unsupported_claim_rate)}</dd></div></dl>
          </article>
          <article className="motion-surface rounded-xl border border-amber-200 bg-amber-50 p-5">
            <div className="flex items-center justify-between gap-3"><h3 className="font-semibold text-amber-950">Earlier unscored attempt</h3><span className="rounded-full bg-amber-100 px-2.5 py-1 text-[10px] font-semibold uppercase text-amber-800">Pending</span></div>
            <p className="mt-2 text-xs leading-5 text-amber-900">{pending.reason} This older attempt produced no scores; the separate three-answer live review above is the available measured evidence.</p>
          </article>
        </div>

        <details className="mt-4 rounded-xl border border-slate-200 bg-white">
          <summary className="cursor-pointer px-4 py-3 text-sm font-semibold text-slate-700">Earlier deterministic numerical sample: {numerical.correct_answers}/{numerical.questions_evaluated} correct</summary>
          <div className="overflow-x-auto border-t border-slate-200">
            <table className="min-w-full divide-y divide-slate-200 text-left text-xs"><thead className="bg-[#F5F8F3] text-slate-600"><tr><th className="px-3 py-2.5">Metric</th><th className="px-3 py-2.5">Expected</th><th className="px-3 py-2.5">Actual</th><th className="px-3 py-2.5">Result</th></tr></thead><tbody className="divide-y divide-slate-100 text-slate-600">{numerical.questions.map((item) => <tr key={item.id}><td className="px-3 py-2.5 font-medium text-slate-800">{item.metric.replaceAll('_', ' ')}</td><td className="px-3 py-2.5">{item.expected_value}</td><td className="px-3 py-2.5">{typeof item.actual_value === 'number' ? item.actual_value.toFixed(item.display_decimals) : 'Unavailable'}</td><td className="px-3 py-2.5 font-semibold text-emerald-700">{item.correct ? 'Verified' : 'Mismatch'}</td></tr>)}</tbody></table>
          </div>
        </details>
      </Reveal>
    </section>
  )
}

export default EvaluationDashboard
