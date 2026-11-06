function formatStatistic(value) {
  if (value === null) return 'N/A'
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: 2 })
}

function DatasetReport({ profile, preprocessing }) {
  return (
    <div className="mt-6 space-y-5 border-t border-slate-200 pt-6">
      <div>
        <p className="text-xs font-semibold uppercase tracking-[0.16em] text-emerald-700">
          Data quality report
        </p>
        <h3 className="mt-1 text-lg font-semibold text-slate-900">Dataset profile</h3>
      </div>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
        <div className="rounded-xl bg-slate-50 p-3.5">
          <p className="text-xs text-slate-500">Rows</p>
          <p className="mt-1 text-xl font-semibold text-slate-900">
            {profile.rows.toLocaleString()}
          </p>
        </div>
        <div className="rounded-xl bg-slate-50 p-3.5">
          <p className="text-xs text-slate-500">Columns</p>
          <p className="mt-1 text-xl font-semibold text-slate-900">
            {profile.columns.toLocaleString()}
          </p>
        </div>
        <div className="col-span-2 rounded-xl bg-slate-50 p-3.5 sm:col-span-1">
          <p className="text-xs text-slate-500">Duplicate rows</p>
          <p className="mt-1 text-xl font-semibold text-slate-900">
            {profile.duplicate_rows.toLocaleString()}
          </p>
        </div>
      </div>

      <div>
        <h4 className="mb-2 text-sm font-semibold text-slate-800">Column quality</h4>
        <div className="overflow-x-auto rounded-xl border border-slate-200">
          <table className="min-w-full divide-y divide-slate-200 text-left text-sm">
            <thead className="bg-slate-50 text-xs uppercase tracking-wide text-slate-500">
              <tr>
                <th className="px-4 py-3 font-medium">Column</th>
                <th className="px-4 py-3 font-medium">Type</th>
                <th className="px-4 py-3 text-right font-medium">Missing</th>
                <th className="px-4 py-3 text-right font-medium">Missing %</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 bg-white">
              {profile.column_profiles.map((column, index) => (
                <tr key={`${column.name}-${index}`}>
                  <td className="whitespace-nowrap px-4 py-3 font-medium text-slate-800">
                    {column.name}
                  </td>
                  <td className="whitespace-nowrap px-4 py-3 text-slate-500">{column.dtype}</td>
                  <td className="px-4 py-3 text-right text-slate-600">
                    {column.missing_count.toLocaleString()}
                  </td>
                  <td className="px-4 py-3 text-right text-slate-600">
                    {column.missing_percentage.toFixed(2)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {profile.numeric_statistics.length > 0 && (
        <div>
          <h4 className="mb-2 text-sm font-semibold text-slate-800">Numerical statistics</h4>
          <div className="overflow-x-auto rounded-xl border border-slate-200">
            <table className="min-w-full divide-y divide-slate-200 text-left text-sm">
              <thead className="bg-slate-50 text-xs uppercase tracking-wide text-slate-500">
                <tr>
                  <th className="px-4 py-3 font-medium">Column</th>
                  <th className="px-4 py-3 text-right font-medium">Mean</th>
                  <th className="px-4 py-3 text-right font-medium">Min</th>
                  <th className="px-4 py-3 text-right font-medium">Max</th>
                  <th className="px-4 py-3 text-right font-medium">Std. dev.</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 bg-white">
                {profile.numeric_statistics.map((statistic, index) => (
                  <tr key={`${statistic.column}-${index}`}>
                    <td className="whitespace-nowrap px-4 py-3 font-medium text-slate-800">
                      {statistic.column}
                    </td>
                    <td className="px-4 py-3 text-right text-slate-600">
                      {formatStatistic(statistic.mean)}
                    </td>
                    <td className="px-4 py-3 text-right text-slate-600">
                      {formatStatistic(statistic.minimum)}
                    </td>
                    <td className="px-4 py-3 text-right text-slate-600">
                      {formatStatistic(statistic.maximum)}
                    </td>
                    <td className="px-4 py-3 text-right text-slate-600">
                      {formatStatistic(statistic.standard_deviation)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <div>
        <h3 className="text-lg font-semibold text-slate-900">Cleaning summary</h3>
        <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-3">
          <div className="rounded-xl border border-slate-200 p-3.5">
            <p className="text-xs text-slate-500">Cleaned rows</p>
            <p className="mt-1 font-semibold text-slate-900">
              {preprocessing.cleaned_rows.toLocaleString()}
            </p>
          </div>
          <div className="rounded-xl border border-slate-200 p-3.5">
            <p className="text-xs text-slate-500">Missing after</p>
            <p className="mt-1 font-semibold text-slate-900">
              {preprocessing.missing_values_after.toLocaleString()}
            </p>
          </div>
          <div className="col-span-2 rounded-xl border border-slate-200 p-3.5 sm:col-span-1">
            <p className="text-xs text-slate-500">Duplicates removed</p>
            <p className="mt-1 font-semibold text-slate-900">
              {preprocessing.duplicate_rows_removed.toLocaleString()}
            </p>
          </div>
        </div>

        <ul className="mt-3 space-y-2">
          {preprocessing.operations.map((operation, index) => (
            <li
              key={`${operation.operation}-${operation.column ?? 'dataset'}-${index}`}
              className="flex items-start justify-between gap-4 rounded-lg bg-slate-50 px-3.5 py-3 text-sm"
            >
              <div>
                {operation.column && (
                  <p className="font-medium text-slate-800">{operation.column}</p>
                )}
                <p className="leading-5 text-slate-600">{operation.description}</p>
              </div>
              <span className="shrink-0 rounded-full bg-white px-2.5 py-1 text-xs font-medium text-slate-600 shadow-sm">
                {operation.affected.toLocaleString()} affected
              </span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  )
}

export default DatasetReport
