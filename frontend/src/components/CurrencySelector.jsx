import { CURRENCY_OPTIONS } from './currency.js'

function CurrencySelector({ currency, onChange }) {
  if (!currency) return null
  const metadata = currency.metadata
  const detectedLabel = metadata?.detected_codes?.length
    ? metadata.detected_codes.join(', ')
    : 'none'

  return (
    <section className="rounded-xl border border-slate-200 bg-white p-4" aria-labelledby="currency-heading">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <h2 id="currency-heading" className="text-sm font-semibold text-slate-800">Display currency</h2>
            <span className={`rounded-full px-2 py-0.5 text-[11px] font-semibold ${
              currency.selection.source === 'detected'
                ? 'bg-emerald-100 text-emerald-700'
                : currency.selection.source === 'user'
                  ? 'bg-blue-100 text-blue-700'
                  : 'bg-slate-100 text-slate-600'
            }`}>
              {currency.selection.source === 'detected'
                ? `Detected from ${metadata?.column}`
                : currency.selection.source === 'user'
                  ? 'User selected'
                  : 'Not specified by dataset'}
            </span>
          </div>
          <p className="mt-1 text-xs leading-5 text-slate-500">
            Formatting only—RetailIQ never converts or changes stored values.
          </p>
        </div>
        <label className="text-xs font-medium text-slate-600" htmlFor="currency-selector">
          Currency
          <select
            id="currency-selector"
            value={currency.selection.code}
            onChange={(event) => onChange(event.target.value)}
            className="mt-1 block w-full min-w-48 rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 outline-none focus:border-emerald-500 focus:ring-2 focus:ring-emerald-100"
          >
            <option value="">Unspecified</option>
            {CURRENCY_OPTIONS.map((code) => <option key={code} value={code}>{code}</option>)}
          </select>
        </label>
      </div>
      {currency.error && <p role="alert" className="mt-3 text-xs text-red-700">{currency.error}</p>}
      {metadata?.status === 'multiple' && (
        <p role="alert" className="mt-3 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs leading-5 text-amber-800">
          Multiple currencies were detected ({detectedLabel}). Monetary KPIs, charts, and aggregate AI answers are hidden until the dataset is filtered to one currency. Selecting a display currency does not filter or convert rows.
        </p>
      )}
      {metadata?.unsupported_values?.length > 0 && (
        <p className="mt-3 text-xs text-amber-700">
          Unrecognized currency values: {metadata.unsupported_values.join(', ')}.
        </p>
      )}
    </section>
  )
}

export default CurrencySelector
