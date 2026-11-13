import { useEffect, useState } from 'react'
import { clearSessionData, fetchCurrencyMetadata } from './api/retailiqApi.js'
import AnalyticsDashboard from './components/AnalyticsDashboard.jsx'
import CsvUpload from './components/CsvUpload.jsx'
import EvaluationDashboard from './components/EvaluationDashboard.jsx'
import RagChatbot from './components/RagChatbot.jsx'
import UploadHistory from './components/UploadHistory.jsx'
import { initialCurrencySelection } from './components/currency.js'
import BrandLogo from './components/BrandLogo.jsx'
import Reveal from './components/Reveal.jsx'

function App() {
  const [historyRefresh, setHistoryRefresh] = useState(0)
  const [selectedDataset, setSelectedDataset] = useState(null)
  const [activeView, setActiveView] = useState('workspace')
  const [currencyByDataset, setCurrencyByDataset] = useState({})
  const selectedId = selectedDataset?.uploadId || ''
  const currency = selectedId ? currencyByDataset[selectedId] : null

  useEffect(() => {
    if (!selectedId) return undefined
    const controller = new AbortController()
    fetchCurrencyMetadata(selectedId, { signal: controller.signal })
      .then((result) => {
        if (result.upload_id !== selectedId) return
        setCurrencyByDataset((current) => ({
          ...current,
          [selectedId]: {
            metadata: result.currency,
            selection: initialCurrencySelection(result.currency, current[selectedId]?.selection?.source === 'user' ? current[selectedId].selection : null),
            isLoading: false,
            error: '',
          },
        }))
      })
      .catch((error) => {
        if (error.name !== 'AbortError') {
          setCurrencyByDataset((current) => ({
            ...current,
            [selectedId]: {
              ...current[selectedId],
              selection: current[selectedId]?.selection || { code: '', source: 'unspecified' },
              isLoading: false,
              error: error.message || 'Currency metadata could not be loaded.',
            },
          }))
        }
      })
    return () => controller.abort()
  }, [selectedId])

  const selectCurrency = (code) => {
    if (!selectedId) return
    setCurrencyByDataset((current) => ({
      ...current,
      [selectedId]: {
        ...current[selectedId],
        selection: { code, source: 'user' },
      },
    }))
  }

  const clearData = async () => {
    if (!window.confirm('Clear this session\'s uploaded CSV and search index?')) return
    await clearSessionData()
    setSelectedDataset(null)
    setCurrencyByDataset({})
    setHistoryRefresh((value) => value + 1)
  }

  return (
    <div className="min-h-screen bg-[#F5F8F3] text-[#1F2937]">
      <header className="motion-header-enter sticky top-0 z-30 border-b border-emerald-950/10 bg-white/95 shadow-[0_1px_12px_rgba(6,78,59,0.06)] backdrop-blur">
        <div className="mx-auto flex max-w-7xl flex-col items-start gap-3 px-4 py-3 sm:flex-row sm:items-center sm:justify-between sm:px-8">
          <BrandLogo />
          <div className="flex w-full items-center gap-2 sm:w-auto"><nav aria-label="Primary navigation" className="flex flex-1 rounded-xl border border-emerald-950/10 bg-[#F5F8F3] p-1 text-xs font-semibold sm:flex-none sm:text-sm">
            {[
              ['workspace', 'Workspace'],
              ['chatbot', 'Ask AI'],
              ['evaluation', 'Evaluation'],
            ].map(([view, label]) => (
              <button key={view} type="button" aria-current={activeView === view ? 'page' : undefined} onClick={() => setActiveView(view)} className={`flex-1 rounded-lg px-4 py-2.5 transition sm:flex-none ${activeView === view ? 'bg-[#064E3B] text-white shadow-sm' : 'text-slate-600 hover:bg-white hover:text-[#064E3B]'}`}>
                {label}
              </button>
            ))}
          </nav><button type="button" onClick={clearData} className="rounded-lg border border-red-200 px-3 py-2 text-xs font-semibold text-red-700 transition hover:bg-red-50">Clear my data</button></div>
        </div>
      </header>

      <main className="overflow-x-clip">
        <div className="mx-auto max-w-7xl px-4 py-8 sm:px-8 sm:py-10 lg:py-12">
          {selectedDataset && (
            <div className="motion-content-enter mb-6 flex min-w-0 flex-col gap-2 rounded-xl border border-emerald-200 bg-white px-4 py-3 shadow-sm sm:flex-row sm:items-center sm:justify-between">
              <div className="min-w-0">
                <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-emerald-700">Active dataset</p>
                <p className="truncate text-sm font-semibold text-slate-800">{selectedDataset.filename}</p>
              </div>
              {selectedDataset.status && (
                <span className={`w-fit rounded-full px-2.5 py-1 text-xs font-semibold capitalize ${
                  selectedDataset.status === 'ready' || selectedDataset.status === 'processed'
                    ? 'bg-emerald-100 text-emerald-800'
                    : selectedDataset.status === 'failed'
                      ? 'bg-red-100 text-red-700'
                      : 'bg-amber-100 text-amber-800'
                }`}>
                  Index: {selectedDataset.status}
                </span>
              )}
            </div>
          )}
          {activeView === 'workspace' ? (
            <>
          <div className="grid gap-8 lg:grid-cols-[0.85fr_1.15fr] lg:items-center lg:gap-14">
            <section className="motion-copy-enter max-w-xl">
              <p className="mb-4 inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 shadow-sm">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
                CSV analytics + grounded AI
              </p>
              <h1 className="text-4xl font-semibold tracking-tight text-slate-950 sm:text-5xl sm:leading-[1.1]">
                Understand any tabular dataset faster.
              </h1>
              <p className="mt-5 max-w-lg text-base leading-7 text-slate-600 sm:text-lg">
                Upload a CSV to explore exact summaries, relevant charts, and grounded
                natural-language answers with source-row citations.
              </p>
              <div className="mt-8 grid max-w-md grid-cols-2 gap-3 text-sm text-slate-600">
                <div className="rounded-xl border border-slate-200 bg-white/70 p-3.5">
                  <p className="font-semibold text-slate-800">CSV format</p>
                  <p className="mt-1 text-xs leading-5 text-slate-500">Tabular data with a header row</p>
                </div>
                <div className="rounded-xl border border-slate-200 bg-white/70 p-3.5">
                  <p className="font-semibold text-slate-800">Temporary session</p>
                  <p className="mt-1 text-xs leading-5 text-slate-500">Inactive data expires after 30 minutes</p>
                </div>
              </div>
            </section>

            <Reveal revealKey="workspace-upload" delay={70} className="min-w-0">
              <CsvUpload onUploadComplete={(data) => { setSelectedDataset({ uploadId: data.upload_id, filename: data.filename, status: data.status }); setHistoryRefresh((value) => value + 1) }} />
            </Reveal>
          </div>

          <Reveal revealKey="workspace-datasets" delay={55} className="mt-10 lg:mt-14">
            <UploadHistory
              refreshToken={historyRefresh}
              onDatasetSelected={(uploadId, filename, status) => setSelectedDataset(uploadId ? { uploadId, filename, status } : null)}
            />
          </Reveal>

          <Reveal revealKey="workspace-analytics" delay={55} className="mt-10">
            <AnalyticsDashboard
              key={selectedDataset?.uploadId || 'no-dataset'}
              uploadId={selectedDataset?.uploadId || null}
              currency={currency}
              onCurrencyChange={selectCurrency}
            />
          </Reveal>
            </>
          ) : activeView === 'chatbot' ? (
            <Reveal revealKey="ask-ai-view">
              <RagChatbot selectedDataset={selectedDataset} onDatasetSelected={setSelectedDataset} currency={currency} onCurrencyChange={selectCurrency} />
            </Reveal>
          ) : (
            <Reveal revealKey="evaluation-view">
              <EvaluationDashboard />
            </Reveal>
          )}
        </div>
      </main>
    </div>
  )
}

export default App
