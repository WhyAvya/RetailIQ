function FilterNotice({ option }) {
  if (!option || option.available) return null
  return <p className="mt-1 text-xs text-amber-700">{option.reason}</p>
}

function DashboardFilters({
  uploadId,
  options,
  optionsError,
  isLoadingOptions,
  draft,
  applied,
  validationError,
  filterContext,
  onChange,
  onApply,
  onReset,
}) {
  const activeCount = Object.values(applied).filter(Boolean).length
  const regions = options?.regions
  const categories = options?.categories
  const dateRange = options?.date_range

  return (
    <section
      aria-labelledby="filters-heading"
      className="rounded-2xl border border-slate-200 bg-white p-5 shadow-[0_18px_55px_-30px_rgba(15,23,42,0.25)] sm:p-7"
    >
      <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-emerald-700">
            Dashboard controls
          </p>
          <h2 id="filters-heading" className="mt-1 text-xl font-semibold text-slate-900">
            Filter analytics
          </h2>
          <p className="mt-1.5 text-sm leading-6 text-slate-500">
            Region, category, and date filters apply to every KPI and chart together.
          </p>
        </div>
        {filterContext && (
          <p className="rounded-lg bg-slate-100 px-3 py-2 text-xs font-medium text-slate-600">
            {filterContext.rows_after.toLocaleString()} of{' '}
            {filterContext.rows_before.toLocaleString()} rows
          </p>
        )}
      </div>

      {!uploadId && (
        <div className="mt-5 rounded-xl border border-dashed border-slate-300 bg-slate-50 px-5 py-7 text-center text-sm text-slate-500">
          Select a dataset to enable dashboard filters.
        </div>
      )}

      {uploadId && (
        <form className="mt-5" onSubmit={onApply}>
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <label className="text-sm font-medium text-slate-700">
              Region
              <select
                value={draft.region}
                onChange={(event) => onChange('region', event.target.value)}
                disabled={isLoadingOptions || !regions?.available}
                className="mt-1.5 w-full rounded-lg border border-slate-300 bg-white px-3 py-2.5 text-sm disabled:bg-slate-100 disabled:text-slate-400"
              >
                <option value="">All regions</option>
                {regions?.values.map((region) => (
                  <option key={region} value={region}>{region}</option>
                ))}
              </select>
              <FilterNotice option={regions} />
            </label>

            <label className="text-sm font-medium text-slate-700">
              Category
              <select
                value={draft.category}
                onChange={(event) => onChange('category', event.target.value)}
                disabled={isLoadingOptions || !categories?.available}
                className="mt-1.5 w-full rounded-lg border border-slate-300 bg-white px-3 py-2.5 text-sm disabled:bg-slate-100 disabled:text-slate-400"
              >
                <option value="">All categories</option>
                {categories?.values.map((category) => (
                  <option key={category} value={category}>{category}</option>
                ))}
              </select>
              <FilterNotice option={categories} />
            </label>

            <label className="text-sm font-medium text-slate-700">
              Start date
              <input
                type="date"
                value={draft.startDate}
                min={dateRange?.min || undefined}
                max={dateRange?.max || undefined}
                onChange={(event) => onChange('startDate', event.target.value)}
                disabled={isLoadingOptions || !dateRange?.available}
                className="mt-1.5 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm disabled:bg-slate-100 disabled:text-slate-400"
              />
              <FilterNotice option={dateRange} />
            </label>

            <label className="text-sm font-medium text-slate-700">
              End date
              <input
                type="date"
                value={draft.endDate}
                min={dateRange?.min || undefined}
                max={dateRange?.max || undefined}
                onChange={(event) => onChange('endDate', event.target.value)}
                disabled={isLoadingOptions || !dateRange?.available}
                className="mt-1.5 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm disabled:bg-slate-100 disabled:text-slate-400"
              />
            </label>
          </div>

          {(validationError || optionsError) && (
            <p role="alert" className="mt-4 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
              {validationError || optionsError}
            </p>
          )}

          <div className="mt-5 flex flex-wrap items-center gap-3">
            <button
              type="submit"
              disabled={isLoadingOptions}
              className="rounded-lg bg-emerald-700 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-emerald-800 disabled:cursor-not-allowed disabled:bg-slate-300"
            >
              Apply filters
            </button>
            <button
              type="button"
              onClick={onReset}
              className="rounded-lg border border-slate-300 bg-white px-4 py-2.5 text-sm font-semibold text-slate-700 transition hover:bg-slate-50"
            >
              Clear filters
            </button>
            <span className="text-xs text-slate-500">
              {activeCount ? `${activeCount} active filter${activeCount === 1 ? '' : 's'}` : 'No active filters'}
            </span>
          </div>

          {filterContext?.warnings?.length > 0 && (
            <ul className="mt-4 rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-xs leading-5 text-amber-800">
              {filterContext.warnings.map((warning) => <li key={warning}>{warning}</li>)}
            </ul>
          )}
        </form>
      )}
    </section>
  )
}

export default DashboardFilters
