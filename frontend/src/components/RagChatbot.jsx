import { useEffect, useReducer, useRef, useState } from 'react'
import { askRetailIQ, fetchUploads, friendlyRagError, retryIndexing } from '../api/retailiqApi.js'
import { chatReducer, initialChatState } from './chatState.js'
import { indexingStatusText, isRagReady } from './indexingStatus.js'
import CurrencySelector from './CurrencySelector.jsx'
import MarkdownAnswer from './MarkdownAnswer.js'
import { formatAnalyticsAnswer, isMonetaryCalculation } from './currency.js'

const EXAMPLE_QUESTIONS = [
  'Which technology accessory orders had strong profits?',
  'Show me examples of furniture products that generated losses.',
  'What is the total revenue?',
  'Which region generated the highest revenue?',
]

function SourceList({ sources }) {
  if (!sources?.length) return null
  return (
    <details className="mt-4 rounded-xl border border-emerald-200 bg-emerald-50/60">
      <summary className="cursor-pointer px-4 py-3 text-sm font-semibold text-emerald-900">
        Sources ({sources.length})
      </summary>
      <div className="space-y-3 border-t border-slate-200 p-3 sm:p-4">
        {sources.map((source) => (
          <article key={source.chunk_id} className="motion-surface rounded-lg border border-emerald-100 bg-white p-3.5 shadow-sm">
            <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500">
              <span className="rounded-md bg-emerald-100 px-2 py-1 font-semibold text-emerald-800">Source row {source.source_row ?? 'unknown'}</span>
              {source.metadata?.category && <span>{source.metadata.category}</span>}
              {source.metadata?.sub_category && <span>• {source.metadata.sub_category}</span>}
              {source.metadata?.region && <span>• {source.metadata.region}</span>}
            </div>
            <p className="mt-2 whitespace-pre-wrap break-words text-xs leading-5 text-slate-600">
              {source.document_excerpt}
            </p>
            {typeof source.similarity_score === 'number' && (
              <p className="mt-2 text-[11px] text-slate-400">
                Cosine similarity: {source.similarity_score.toFixed(4)}
              </p>
            )}
          </article>
        ))}
      </div>
    </details>
  )
}

function ChatMessage({ message, currency }) {
  const blockedMonetaryAnswer = (
    message.mode === 'analytics'
    && currency?.metadata?.status === 'multiple'
    && isMonetaryCalculation(message.calculation)
  )
  const answer = message.mode === 'analytics'
    ? formatAnalyticsAnswer(message.answer, message.calculation, currency?.selection?.code || '')
    : message.answer
  return (
    <article className="motion-content-enter space-y-3">
      <div className="ml-auto max-w-[88%] rounded-2xl rounded-br-md bg-emerald-700 px-4 py-3 text-sm leading-6 text-white sm:max-w-[75%]">
        {message.question}
      </div>
      <div className={`max-w-[95%] rounded-2xl rounded-bl-md border px-4 py-4 sm:max-w-[86%] ${
        message.error
          ? 'border-red-200 bg-red-50 text-red-800'
          : message.mode === 'unsupported'
            ? 'border-amber-200 bg-amber-50 text-amber-900'
            : 'border-slate-200 bg-white text-slate-700 shadow-sm'
      }`}>
        {!message.error && (
          <p className="mb-2 text-[11px] font-semibold uppercase tracking-[0.14em] text-emerald-700">
            {message.mode === 'analytics' ? 'Calculated from dataset analytics' : message.mode === 'semantic_rag' ? 'AI answer' : 'Limited support'}
          </p>
        )}
        {blockedMonetaryAnswer ? (
          <p role="alert" className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm leading-6 text-amber-800">
            This aggregate monetary answer is hidden because the dataset contains multiple currencies. Filter to one currency before aggregating sales or profit.
          </p>
        ) : message.error ? (
          <p className="break-words text-sm leading-6">{message.error}</p>
        ) : (
          <MarkdownAnswer content={answer} />
        )}
        {message.mode === 'analytics' && message.calculation && (
          <p className="mt-3 text-xs text-slate-500">
            Calculation: {message.calculation.intent?.replaceAll('_', ' ')}
          </p>
        )}
        {message.mode === 'semantic_rag' && <SourceList sources={message.sources} />}
      </div>
    </article>
  )
}

function RagChatbot({ selectedDataset, onDatasetSelected, currency, onCurrencyChange }) {
  const [uploads, setUploads] = useState([])
  const [uploadsError, setUploadsError] = useState('')
  const [question, setQuestion] = useState('')
  const [uploadsRefresh, setUploadsRefresh] = useState(0)
  const [state, dispatch] = useReducer(chatReducer, initialChatState)
  const requestController = useRef(null)
  const endRef = useRef(null)
  const selectedId = selectedDataset?.uploadId || ''
  const selectedUpload = uploads.find((item) => item.upload_id === selectedId)
  const ragReady = isRagReady(selectedUpload)

  useEffect(() => {
    const controller = new AbortController()
    let refreshTimer
    fetchUploads({ signal: controller.signal })
      .then((items) => {
        setUploads(items)
        if (items.some((item) => item.status === 'pending' || item.status === 'indexing')) {
          refreshTimer = setTimeout(() => setUploadsRefresh((value) => value + 1), 2000)
        }
      })
      .catch((error) => {
        if (error.name !== 'AbortError') setUploadsError(error.message)
      })
    return () => {
      controller.abort()
      clearTimeout(refreshTimer)
    }
  }, [uploadsRefresh])

  useEffect(() => {
    requestController.current?.abort()
    dispatch({ type: 'dataset_changed', datasetId: selectedId || null })
  }, [selectedId])

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
  }, [state.messages, state.pending])

  useEffect(() => () => requestController.current?.abort(), [])

  const chooseDataset = (event) => {
    const upload = uploads.find((item) => item.upload_id === event.target.value)
    setQuestion('')
    onDatasetSelected?.(upload ? { uploadId: upload.upload_id, filename: upload.filename, status: upload.status } : null)
  }

  const submitQuestion = async () => {
    const trimmed = question.trim()
    if (!trimmed || !selectedId || !ragReady || state.pending || requestController.current) return

    const controller = new AbortController()
    requestController.current = controller
    dispatch({ type: 'request_started' })
    setQuestion('')

    try {
      const result = await askRetailIQ(selectedId, trimmed, 3, { signal: controller.signal })
      dispatch({
        type: 'request_succeeded',
        message: {
          id: globalThis.crypto?.randomUUID?.() || `${Date.now()}`,
          question: trimmed,
          ...result,
        },
      })
    } catch (error) {
      if (error.name !== 'AbortError') {
        dispatch({
          type: 'request_failed',
          message: {
            id: globalThis.crypto?.randomUUID?.() || `${Date.now()}`,
            question: trimmed,
            error: friendlyRagError(error),
          },
        })
      }
    } finally {
      if (requestController.current === controller) requestController.current = null
    }
  }

  const retrySelectedIndexing = async () => {
    if (!selectedId) return
    try {
      await retryIndexing(selectedId)
      setUploads((items) => items.map((item) => (
        item.upload_id === selectedId ? { ...item, status: 'pending' } : item
      )))
      setUploadsRefresh((value) => value + 1)
    } catch (error) {
      setUploadsError(error.message)
    }
  }

  const handleKeyDown = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      submitQuestion()
    }
  }

  return (
    <section aria-labelledby="chatbot-heading" className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-[0_18px_55px_-30px_rgba(6,78,59,0.28)]">
      <div className="border-b border-slate-200 p-5 sm:p-7">
        <p className="text-xs font-semibold uppercase tracking-[0.18em] text-emerald-700">RetailIQ assistant</p>
        <div className="mt-1 flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <h1 id="chatbot-heading" className="text-2xl font-semibold text-slate-950">Ask questions about your retail data</h1>
            <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-500">
              Exact numerical questions use dataset analytics. Record-level questions use retrieved CSV evidence and the configured AI model.
            </p>
          </div>
          {state.messages.length > 0 && (
            <button type="button" onClick={() => dispatch({ type: 'clear' })} className="shrink-0 rounded-lg border border-slate-200 px-3 py-2 text-xs font-semibold text-slate-600 hover:bg-slate-50">
              Clear chat
            </button>
          )}
        </div>

        <label className="mt-5 block text-sm font-medium text-slate-700" htmlFor="chat-dataset">Dataset</label>
        <select id="chat-dataset" value={selectedId} onChange={chooseDataset} className="mt-2 w-full rounded-xl border border-slate-300 bg-white px-3.5 py-3 text-sm text-slate-700 outline-none focus:border-emerald-500 focus:ring-2 focus:ring-emerald-100">
          <option value="">Select an uploaded dataset</option>
          {uploads.map((upload) => (
            <option key={upload.upload_id} value={upload.upload_id} disabled={!isRagReady(upload)}>
              {upload.filename} ({upload.status})
            </option>
          ))}
        </select>
        {uploadsError && <p role="alert" className="mt-2 text-sm text-red-700">{uploadsError}</p>}
        {selectedDataset && (
          <div className="mt-3 flex flex-wrap items-center gap-2 rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 text-xs">
            <span className="font-semibold text-emerald-900">Selected: {selectedDataset.filename}</span>
            {selectedUpload?.status && <span className="capitalize text-emerald-700">Index {selectedUpload.status}</span>}
          </div>
        )}
        {selectedId && currency && (
          <div className="mt-4">
            <CurrencySelector currency={currency} onChange={onCurrencyChange} />
          </div>
        )}
        {selectedId && selectedUpload && !ragReady && (
          <div className="mt-3 flex items-center justify-between gap-3 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800">
            <span>{indexingStatusText(selectedUpload.status)}</span>
            {selectedUpload.status === 'failed' && (
              <button type="button" onClick={retrySelectedIndexing} className="font-semibold underline">Retry</button>
            )}
          </div>
        )}
      </div>

      <div className="flex min-h-[34rem] flex-col">
        <div className="max-h-[38rem] flex-1 space-y-6 overflow-y-auto bg-slate-50/60 p-4 sm:p-6">
          {state.messages.length === 0 && !state.pending && (
            <div className="mx-auto flex max-w-xl flex-col items-center py-12 text-center">
              <img src="/favicon-48x48.png" alt="" className="h-12 w-12 object-contain" />
              <h2 className="mt-4 font-semibold text-slate-800">Start with a question</h2>
              <p className="mt-2 text-sm leading-6 text-slate-500">Each question is processed independently. Select a dataset, then try one of these examples.</p>
              <div className="mt-5 flex flex-wrap justify-center gap-2">
                {EXAMPLE_QUESTIONS.map((example) => (
                  <button key={example} type="button" disabled={!selectedId || !ragReady} onClick={() => setQuestion(example)} className="rounded-full border border-slate-200 bg-white px-3 py-2 text-xs text-slate-600 hover:border-emerald-300 disabled:cursor-not-allowed disabled:opacity-50">
                    {example}
                  </button>
                ))}
              </div>
            </div>
          )}
          {state.messages.map((message) => <ChatMessage key={message.id} message={message} currency={currency} />)}
          {state.pending && (
            <div role="status" className="w-fit rounded-xl border border-emerald-100 bg-white px-4 py-3 text-sm text-slate-600 shadow-sm">
              <span className="mr-2 inline-block h-2 w-2 animate-pulse rounded-full bg-emerald-500" />
              RetailIQ is working on your question...
            </div>
          )}
          <div ref={endRef} />
        </div>

        <div className="border-t border-slate-200 bg-white p-4 sm:p-5">
          {!selectedId && <p className="mb-2 text-xs text-amber-700">Select a ready dataset before asking a question.</p>}
          <div className="flex flex-col gap-3 sm:flex-row sm:items-end">
            <textarea value={question} onChange={(event) => setQuestion(event.target.value)} onKeyDown={handleKeyDown} disabled={!selectedId || !ragReady || state.pending} rows={2} placeholder="Ask about products, profits, revenue, orders, regions..." className="min-h-14 flex-1 resize-y rounded-xl border border-slate-300 px-3.5 py-3 text-sm leading-6 outline-none focus:border-emerald-500 focus:ring-2 focus:ring-emerald-100 disabled:bg-slate-100" />
            <button type="button" onClick={submitQuestion} disabled={!selectedId || !ragReady || !question.trim() || state.pending} className="rounded-xl bg-[#064E3B] px-6 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-emerald-800 disabled:cursor-not-allowed disabled:bg-slate-300">
              {state.pending ? 'Sending...' : 'Send'}
            </button>
          </div>
          <p className="mt-2 text-[11px] text-slate-400">Enter to send • Shift+Enter for a new line • Questions are independent</p>
        </div>
      </div>
    </section>
  )
}

export default RagChatbot
