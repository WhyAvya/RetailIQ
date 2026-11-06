import { useMemo } from 'react'
import PlotlyChart from './PlotlyChart.jsx'

function ChartState({ uploadId, state, children }) {
  if (!uploadId) {
    return (
      <div className="flex h-80 items-center justify-center rounded-xl border border-dashed border-slate-300 bg-slate-50 px-5 text-center text-sm text-slate-500">
        Select a dataset to view this chart.
      </div>
    )
  }
  if (state.isLoading) {
    return <div className="h-80 animate-pulse rounded-xl bg-slate-100" aria-label="Loading chart" />
  }
  if (state.error) {
    return (
      <p role="alert" className="flex h-80 items-center justify-center rounded-xl border border-red-200 bg-red-50 px-5 text-center text-sm text-red-700">
        {state.error}
      </p>
    )
  }
  if (state.result && !state.result.available) {
    return (
      <div className="flex h-80 flex-col items-center justify-center rounded-xl border border-amber-200 bg-amber-50 px-5 text-center">
        <p className="text-sm font-semibold text-amber-900">Chart unavailable</p>
        <p className="mt-1 text-xs leading-5 text-amber-700">{state.result.reason}</p>
      </div>
    )
  }
  if (!state.result) return null
  return children
}

function ChartNotes({ warnings }) {
  if (!warnings?.length) return null
  return (
    <ul className="mt-3 space-y-1 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs leading-5 text-amber-800">
      {warnings.map((warning, index) => (
        <li key={`${warning}-${index}`}>• {warning}</li>
      ))}
    </ul>
  )
}

function AnalyticsCharts({ uploadId, result, error, isLoading }) {
  const charts = result?.upload_id === uploadId ? result.charts : null
  const monthly = { result: charts?.monthly_revenue, error, isLoading }
  const regional = { result: charts?.sales_by_region, error, isLoading }
  const category = { result: charts?.category_performance, error, isLoading }
  const profit = { result: charts?.profit_analysis, error, isLoading }

  const monthlyPlot = useMemo(() => {
    if (!monthly.result?.available) return null
    return {
      data: [
        {
          type: 'scatter',
          mode: 'lines+markers',
          x: monthly.result.data.map((item) => `${item.month}-01`),
          y: monthly.result.data.map((item) => item.revenue),
          customdata: monthly.result.data.map((item) => item.label),
          line: { color: '#047857', width: 3 },
          marker: { color: '#10b981', size: 7 },
          hovertemplate: '%{customdata}<br>Revenue: %{y:,.2f}<extra></extra>',
        },
      ],
      layout: {
        margin: { l: 62, r: 18, t: 18, b: 55 },
        hovermode: 'x unified',
        xaxis: { title: 'Month', type: 'date', tickformat: '%b %Y', gridcolor: '#e2e8f0' },
        yaxis: { title: 'Revenue', tickformat: ',.2f', gridcolor: '#e2e8f0' },
      },
    }
  }, [monthly.result])

  const regionalPlot = useMemo(() => {
    if (!regional.result?.available) return null
    return {
      data: [
        {
          type: 'bar',
          x: regional.result.data.map((item) => item.region),
          y: regional.result.data.map((item) => item.revenue),
          marker: { color: '#0f766e' },
          hovertemplate: '%{x}<br>Revenue: %{y:,.2f}<extra></extra>',
        },
      ],
      layout: {
        margin: { l: 62, r: 18, t: 18, b: 55 },
        xaxis: { title: 'Region', gridcolor: '#e2e8f0' },
        yaxis: { title: 'Revenue', tickformat: ',.2f', gridcolor: '#e2e8f0' },
      },
    }
  }, [regional.result])

  const categoryPlot = useMemo(() => {
    if (!category.result?.available) return null
    const categories = category.result.data.map((item) => item.category)
    return {
      data: [
        {
          type: 'bar',
          name: 'Revenue',
          x: categories,
          y: category.result.data.map((item) => item.revenue),
          marker: { color: '#0f766e' },
          hovertemplate: '%{x}<br>Revenue: %{y:,.2f}<extra></extra>',
        },
        {
          type: 'bar',
          name: 'Profit',
          x: categories,
          y: category.result.data.map((item) => item.profit),
          marker: { color: '#6366f1' },
          hovertemplate: '%{x}<br>Profit: %{y:,.2f}<extra></extra>',
        },
      ],
      layout: {
        barmode: 'group',
        margin: { l: 62, r: 18, t: 18, b: 70 },
        legend: { orientation: 'h', y: 1.12 },
        xaxis: { title: 'Category', automargin: true, gridcolor: '#e2e8f0' },
        yaxis: { title: 'Value', tickformat: ',.2f', gridcolor: '#e2e8f0' },
      },
    }
  }, [category.result])

  const profitPlot = useMemo(() => {
    if (!profit.result?.available) return null
    const segments = profit.result.data.map((item) => item.segment)
    return {
      data: [
        {
          type: 'bar',
          name: 'Profit',
          x: segments,
          y: profit.result.data.map((item) => item.profit),
          customdata: profit.result.data.map((item) => item.status),
          marker: {
            color: profit.result.data.map((item) => {
              if (item.status === 'profitable') return '#059669'
              if (item.status === 'loss-making') return '#dc2626'
              if (item.status === 'break-even') return '#64748b'
              return '#cbd5e1'
            }),
          },
          hovertemplate: '%{x}<br>Profit: %{y:,.2f}<br>Status: %{customdata}<extra></extra>',
        },
        {
          type: 'scatter',
          mode: 'lines+markers',
          name: 'Profit margin',
          x: segments,
          y: profit.result.data.map((item) => item.profit_margin),
          yaxis: 'y2',
          line: { color: '#f59e0b', width: 2 },
          marker: { color: '#f59e0b', size: 7 },
          connectgaps: false,
          hovertemplate: '%{x}<br>Profit margin: %{y:,.2f}%<extra></extra>',
        },
      ],
      layout: {
        margin: { l: 62, r: 62, t: 18, b: 90 },
        legend: { orientation: 'h', y: 1.12 },
        xaxis: {
          title: profit.result.dimension || 'Business segment',
          automargin: true,
          tickangle: -35,
          gridcolor: '#e2e8f0',
        },
        yaxis: { title: 'Profit', tickformat: ',.2f', gridcolor: '#e2e8f0' },
        yaxis2: {
          title: 'Profit margin (%)',
          overlaying: 'y',
          side: 'right',
          tickformat: ',.2f',
          showgrid: false,
        },
      },
    }
  }, [profit.result])

  return (
    <section aria-labelledby="charts-heading">
      <div className="mb-5">
        <p className="text-xs font-semibold uppercase tracking-[0.18em] text-emerald-700">
          Revenue views
        </p>
        <h2 id="charts-heading" className="mt-1 text-xl font-semibold text-slate-900">
          Sales trends
        </h2>
        <p className="mt-1.5 text-sm leading-6 text-slate-500">
          Interactive monthly and regional revenue views for the selected dataset.
        </p>
      </div>

      <div className="grid gap-5 xl:grid-cols-2">
        <article className="min-w-0 rounded-2xl border border-slate-200 bg-white p-4 shadow-[0_18px_55px_-30px_rgba(15,23,42,0.25)] sm:p-6">
          <h3 className="text-base font-semibold text-slate-900">Monthly revenue</h3>
          <p className="mt-1 text-xs leading-5 text-slate-500">Revenue grouped by calendar month.</p>
          <div className="mt-3">
            <ChartState uploadId={uploadId} state={monthly}>
              {monthlyPlot && (
                <PlotlyChart
                  data={monthlyPlot.data}
                  layout={monthlyPlot.layout}
                  ariaLabel="Monthly revenue line chart"
                />
              )}
            </ChartState>
          </div>
          <ChartNotes warnings={monthly.result?.warnings} />
        </article>

        <article className="min-w-0 rounded-2xl border border-slate-200 bg-white p-4 shadow-[0_18px_55px_-30px_rgba(15,23,42,0.25)] sm:p-6">
          <h3 className="text-base font-semibold text-slate-900">Sales by region</h3>
          <p className="mt-1 text-xs leading-5 text-slate-500">Regions sorted from highest to lowest revenue.</p>
          <div className="mt-3">
            <ChartState uploadId={uploadId} state={regional}>
              {regionalPlot && (
                <PlotlyChart
                  data={regionalPlot.data}
                  layout={regionalPlot.layout}
                  ariaLabel="Regional revenue bar chart"
                />
              )}
            </ChartState>
          </div>
          <ChartNotes warnings={regional.result?.warnings} />
        </article>
      </div>

      <div className="mt-8 mb-5">
        <p className="text-xs font-semibold uppercase tracking-[0.18em] text-emerald-700">
          Performance views
        </p>
        <h2 className="mt-1 text-xl font-semibold text-slate-900">
          Product profitability
        </h2>
        <p className="mt-1.5 text-sm leading-6 text-slate-500">
          Revenue and profit comparisons for the selected dataset.
        </p>
      </div>

      <div className="grid gap-5 xl:grid-cols-2">
        <article className="min-w-0 rounded-2xl border border-slate-200 bg-white p-4 shadow-[0_18px_55px_-30px_rgba(15,23,42,0.25)] sm:p-6">
          <h3 className="text-base font-semibold text-slate-900">Category performance</h3>
          <p className="mt-1 text-xs leading-5 text-slate-500">
            Revenue and profit grouped by product category.
          </p>
          <div className="mt-3">
            <ChartState uploadId={uploadId} state={category}>
              {categoryPlot && (
                <PlotlyChart
                  data={categoryPlot.data}
                  layout={categoryPlot.layout}
                  ariaLabel="Category revenue and profit grouped bar chart"
                />
              )}
            </ChartState>
          </div>
          <ChartNotes warnings={category.result?.warnings} />
        </article>

        <article className="min-w-0 rounded-2xl border border-slate-200 bg-white p-4 shadow-[0_18px_55px_-30px_rgba(15,23,42,0.25)] sm:p-6">
          <h3 className="text-base font-semibold text-slate-900">Profit analysis</h3>
          <p className="mt-1 text-xs leading-5 text-slate-500">
            Green and red bars highlight profitable and loss-making segments.
          </p>
          <div className="mt-3">
            <ChartState uploadId={uploadId} state={profit}>
              {profitPlot && (
                <PlotlyChart
                  data={profitPlot.data}
                  layout={profitPlot.layout}
                  ariaLabel="Profit and profit margin by business segment chart"
                />
              )}
            </ChartState>
          </div>
          <ChartNotes warnings={profit.result?.warnings} />
        </article>
      </div>
    </section>
  )
}

export default AnalyticsCharts
