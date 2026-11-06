import { useEffect, useRef } from 'react'

function PlotlyChart({ data, layout, ariaLabel }) {
  const chartRef = useRef(null)
  const plotlyRef = useRef(null)

  useEffect(() => {
    const chart = chartRef.current
    if (!chart) return undefined
    const controller = new AbortController()

    async function renderChart() {
      const { default: Plotly } = await import('plotly.js-basic-dist-min')
      if (controller.signal.aborted) return
      plotlyRef.current = Plotly
      await Plotly.react(
        chart,
        data,
        {
          autosize: true,
          paper_bgcolor: 'transparent',
          plot_bgcolor: 'transparent',
          font: { family: 'Inter, ui-sans-serif, system-ui, sans-serif', color: '#475569' },
          ...layout,
        },
        {
          responsive: true,
          displaylogo: false,
          modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        },
      )
    }

    renderChart()

    return () => {
      controller.abort()
      plotlyRef.current?.purge(chart)
    }
  }, [data, layout])

  return <div ref={chartRef} role="img" aria-label={ariaLabel} className="h-80 w-full" />
}

export default PlotlyChart
