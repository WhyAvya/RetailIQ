import { formatMonetaryValue } from './currency.js'
import Reveal from './Reveal.jsx'

const KPI_CARDS = [
  { key: 'total_revenue', label: 'Total revenue', decimals: 2, monetary: true },
  { key: 'total_profit', label: 'Total profit', decimals: 2, monetary: true },
  { key: 'number_of_orders', label: 'Orders', decimals: 0 },
  { key: 'unique_customers', label: 'Unique customers', decimals: 0 },
  { key: 'average_order_value', label: 'Average order value', decimals: 2, monetary: true },
  { key: 'profit_margin', label: 'Profit margin', decimals: 2, suffix: '%' },
]

function formatMetric(value, decimals, suffix = '') {
  if (typeof value !== 'number' || !Number.isFinite(value)) return 'Unavailable'
  return `${value.toLocaleString(undefined, {
    minimumFractionDigits: 0,
    maximumFractionDigits: decimals,
  })}${suffix}`
}

function KpiDashboard({ uploadId, result, error, isLoading, currency }) {
  const currentAnalytics = result?.upload_id === uploadId
    ? { filename: result.filename, ...result.summary }
    : null

  return (
    <section
      aria-labelledby="analytics-heading"
      className="rounded-2xl border border-slate-200 bg-white p-5 shadow-[0_18px_55px_-30px_rgba(15,23,42,0.25)] sm:p-7"
    >
      <div>
        <p className="text-xs font-semibold uppercase tracking-[0.18em] text-emerald-700">
          Retail analytics
        </p>
        <h2 id="analytics-heading" className="mt-1 text-xl font-semibold text-slate-900">
          KPI summary
        </h2>
        <p className="mt-1.5 text-sm leading-6 text-slate-500">
          {currentAnalytics
            ? `Calculated from ${currentAnalytics.filename}.`
            : 'Select a dataset above to calculate its retail KPIs.'}
        </p>
      </div>

      {!uploadId && (
        <div className="mt-5 rounded-xl border border-dashed border-slate-300 bg-slate-50 px-5 py-8 text-center text-sm text-slate-500">
          No dataset is selected.
        </div>
      )}

      {isLoading && (
        <div className="mt-5 grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3" aria-label="Loading analytics">
          {KPI_CARDS.map((card) => (
            <div key={card.key} className="h-28 animate-pulse rounded-xl bg-slate-100" />
          ))}
        </div>
      )}

      {error && (
        <p role="alert" className="mt-5 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          {error}
        </p>
      )}

      {currentAnalytics && (
        <>
          <div className="mt-5 grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3">
            {KPI_CARDS.map((card, index) => {
              const metric = currentAnalytics.kpis[card.key]
              const mixedCurrencyBlocked = card.monetary && currency?.metadata?.status === 'multiple'
              const isAvailable = Boolean(metric?.available) && !mixedCurrencyBlocked
              return (
                <Reveal key={card.key} revealKey={`kpi-${card.key}`} delay={index * 45} className="h-full">
                  <article
                    className={`motion-surface h-full rounded-xl border p-4 sm:p-5 ${
                      isAvailable
                        ? 'border-slate-200 bg-gradient-to-br from-white to-[#F5F8F3] shadow-sm'
                        : 'border-amber-200 bg-amber-50/60'
                    }`}
                  >
                  <p className="text-xs font-medium text-slate-500">{card.label}</p>
                  <p className={`mt-2 break-words text-xl font-semibold sm:text-2xl ${
                    isAvailable ? 'text-slate-900' : 'text-amber-800'
                  }`}>
                    {isAvailable
                      ? card.monetary
                        ? formatMonetaryValue(metric.value, currency?.selection?.code || '', true)
                        : formatMetric(metric.value, card.decimals, card.suffix)
                      : 'Unavailable'}
                  </p>
                  {!isAvailable && (
                    <p className="mt-2 text-xs leading-5 text-amber-700">
                      {mixedCurrencyBlocked
                        ? 'Multiple currencies cannot be aggregated without filtering to one currency.'
                        : metric?.reason || 'This metric cannot be calculated.'}
                    </p>
                  )}
                  </article>
                </Reveal>
              )
            })}
          </div>

          {currentAnalytics.warnings?.length > 0 && (
            <div className="mt-5 rounded-xl border border-amber-200 bg-amber-50 p-4">
              <h3 className="text-sm font-semibold text-amber-900">Data-quality notes</h3>
              <ul className="mt-2 space-y-1 text-xs leading-5 text-amber-800">
                {currentAnalytics.warnings.map((warning, index) => (
                  <li key={`${warning}-${index}`}>• {warning}</li>
                ))}
              </ul>
            </div>
          )}
        </>
      )}
    </section>
  )
}

export default KpiDashboard
