import { useState } from 'react'
import AnalyticsDashboard from './components/AnalyticsDashboard.jsx'
import CsvUpload from './components/CsvUpload.jsx'
import UploadHistory from './components/UploadHistory.jsx'

function App() {
  const [historyRefresh, setHistoryRefresh] = useState(0)
  const [selectedUploadId, setSelectedUploadId] = useState(null)

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <header className="border-b border-slate-200 bg-white/90 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-5 py-4 sm:px-8">
          <div className="flex items-center gap-3">
            <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-emerald-700 text-sm font-bold text-white shadow-sm">
              R
            </span>
            <div>
              <p className="text-base font-semibold tracking-tight text-slate-900">RetailIQ</p>
              <p className="text-xs text-slate-500">Retail analytics workspace</p>
            </div>
          </div>
          <span className="rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 text-xs font-medium text-emerald-700">
            Data workspace
          </span>
        </div>
      </header>

      <main className="relative overflow-hidden">
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-x-0 top-0 h-72 bg-[radial-gradient(circle_at_top,rgba(16,185,129,0.10),transparent_62%)]"
        />
        <div className="relative mx-auto max-w-6xl px-5 py-12 sm:px-8 sm:py-16 lg:py-24">
          <div className="grid gap-10 lg:grid-cols-[0.9fr_1.1fr] lg:items-center lg:gap-16">
            <section className="max-w-xl">
              <p className="mb-4 inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 shadow-sm">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
                Phase 3 - Retail analytics
              </p>
              <h1 className="text-4xl font-semibold tracking-tight text-slate-950 sm:text-5xl sm:leading-[1.1]">
                Turn retail data into clearer decisions.
              </h1>
              <p className="mt-5 max-w-lg text-base leading-7 text-slate-600 sm:text-lg">
                Start by adding your sales dataset. RetailIQ will use this workspace to prepare
                future insights, trends, and natural-language analysis.
              </p>
              <div className="mt-8 grid max-w-md grid-cols-2 gap-3 text-sm text-slate-600">
                <div className="rounded-xl border border-slate-200 bg-white/70 p-3.5">
                  <p className="font-semibold text-slate-800">CSV format</p>
                  <p className="mt-1 text-xs leading-5 text-slate-500">Simple tabular retail data</p>
                </div>
                <div className="rounded-xl border border-slate-200 bg-white/70 p-3.5">
                  <p className="font-semibold text-slate-800">Original retained</p>
                  <p className="mt-1 text-xs leading-5 text-slate-500">Source CSV stays unchanged</p>
                </div>
              </div>
            </section>

            <CsvUpload onUploadComplete={() => setHistoryRefresh((value) => value + 1)} />
          </div>

          <div className="mt-10 lg:mt-14">
            <UploadHistory
              refreshToken={historyRefresh}
              onDatasetSelected={setSelectedUploadId}
            />
          </div>

          <div className="mt-10">
            <AnalyticsDashboard
              key={selectedUploadId || 'no-dataset'}
              uploadId={selectedUploadId}
            />
          </div>
        </div>
      </main>
    </div>
  )
}

export default App
