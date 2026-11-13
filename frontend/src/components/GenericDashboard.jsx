function GenericDashboard({ result }) {
  const summary = result.summary
  return (
    <section className="space-y-6" aria-labelledby="generic-analytics-heading">
      <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-7">
        <p className="text-xs font-semibold uppercase tracking-[0.18em] text-emerald-700">Dataset analytics</p>
        <h2 id="generic-analytics-heading" className="mt-1 text-xl font-semibold text-slate-900">Schema-aware overview</h2>
        <div className="mt-5 grid grid-cols-2 gap-3 lg:grid-cols-4">
          {[
            ['Rows', summary.rows], ['Columns', summary.columns],
            ['Missing values', summary.missing_values], ['Duplicate rows', summary.duplicate_rows],
          ].map(([label, value]) => (
            <article key={label} className="rounded-xl border border-slate-200 bg-slate-50 p-4">
              <p className="text-xs text-slate-500">{label}</p>
              <p className="mt-1 text-2xl font-semibold text-slate-900">{value.toLocaleString()}</p>
            </article>
          ))}
        </div>
      </div>

      {summary.numeric.length > 0 && (
        <div className="overflow-x-auto rounded-2xl border border-slate-200 bg-white p-5 sm:p-7">
          <h3 className="font-semibold text-slate-900">Numeric summaries</h3>
          <table className="mt-4 min-w-full text-left text-sm">
            <thead className="border-b text-xs uppercase text-slate-500"><tr>{['Column', 'Count', 'Sum', 'Average', 'Minimum', 'Maximum'].map((item) => <th key={item} className="px-3 py-2">{item}</th>)}</tr></thead>
            <tbody>{summary.numeric.map((item) => <tr key={item.column} className="border-b border-slate-100"><td className="px-3 py-3 font-medium">{item.column}</td>{['count', 'sum', 'mean', 'minimum', 'maximum'].map((key) => <td key={key} className="px-3 py-3">{item[key]?.toLocaleString(undefined, { maximumFractionDigits: 2 }) ?? 'N/A'}</td>)}</tr>)}</tbody>
          </table>
        </div>
      )}

      <div className="rounded-2xl border border-emerald-200 bg-emerald-50 p-5 sm:p-6">
        <h3 className="font-semibold text-emerald-950">Suggested questions</h3>
        <ul className="mt-3 grid gap-2 text-sm text-emerald-900 sm:grid-cols-2">
          {result.suggested_questions.map((question) => <li key={question} className="rounded-lg bg-white px-3 py-2">{question}</li>)}
        </ul>
      </div>

      {result.charts.categorical_distributions.map((chart) => {
        const maximum = Math.max(...chart.values.map((item) => item.count), 1)
        return <article key={chart.column} className="rounded-2xl border border-slate-200 bg-white p-5 sm:p-7"><h3 className="font-semibold text-slate-900">{chart.column} distribution</h3><div className="mt-4 space-y-3">{chart.values.map((item) => <div key={item.label}><div className="flex justify-between gap-4 text-xs"><span className="truncate text-slate-600">{item.label}</span><span className="font-medium">{item.count.toLocaleString()}</span></div><div className="mt-1 h-2 rounded-full bg-slate-100"><div className="h-2 rounded-full bg-emerald-600" style={{ width: `${(item.count / maximum) * 100}%` }} /></div></div>)}</div></article>
      })}
    </section>
  )
}

export default GenericDashboard
