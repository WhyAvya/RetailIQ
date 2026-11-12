import { useEffect, useRef, useState } from 'react'

const UPLOADS_URL = 'http://localhost:8000/api/uploads'

function formatFileSize(bytes) {
  if (bytes < 1024) return `${bytes} bytes`
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`
}

function formatUploadDate(value) {
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(new Date(value))
}

async function responseData(response, fallbackMessage) {
  const data = await response.json().catch(() => ({}))
  if (!response.ok) {
    throw new Error(typeof data.detail === 'string' ? data.detail : fallbackMessage)
  }
  return data
}

function UploadHistory({ refreshToken, onDatasetSelected }) {
  const [uploads, setUploads] = useState([])
  const [historyError, setHistoryError] = useState('')
  const [isLoadingHistory, setIsLoadingHistory] = useState(true)
  const [selectedId, setSelectedId] = useState('')
  const [preview, setPreview] = useState(null)
  const [previewError, setPreviewError] = useState('')
  const [isLoadingPreview, setIsLoadingPreview] = useState(false)
  const previewControllerRef = useRef(null)

  useEffect(() => {
    const controller = new AbortController()
    let refreshTimer

    async function loadHistory() {
      setIsLoadingHistory(true)
      setHistoryError('')

      try {
        const response = await fetch(UPLOADS_URL, { signal: controller.signal })
        const data = await responseData(response, 'Upload history could not be loaded.')
        const items = Array.isArray(data.uploads) ? data.uploads : []
        setUploads(items)
        if (items.some((item) => item.status === 'pending' || item.status === 'indexing')) {
          refreshTimer = setTimeout(loadHistory, 2000)
        }
      } catch (error) {
        if (error.name !== 'AbortError') {
          setHistoryError(error.message || 'Upload history could not be loaded.')
        }
      } finally {
        if (!controller.signal.aborted) setIsLoadingHistory(false)
      }
    }

    loadHistory()
    return () => {
      controller.abort()
      clearTimeout(refreshTimer)
    }
  }, [refreshToken])

  useEffect(
    () => () => {
      previewControllerRef.current?.abort()
    },
    [],
  )

  const loadPreview = async (uploadId) => {
    previewControllerRef.current?.abort()
    const controller = new AbortController()
    previewControllerRef.current = controller
    setSelectedId('')
    onDatasetSelected?.(null)
    setPreview(null)
    setPreviewError('')
    setIsLoadingPreview(true)

    try {
      const response = await fetch(`${UPLOADS_URL}/${uploadId}/preview`, {
        signal: controller.signal,
      })
      const data = await responseData(response, 'The dataset preview could not be loaded.')
      if (previewControllerRef.current !== controller) return
      setSelectedId(uploadId)
      onDatasetSelected?.(uploadId, data.filename)
      setPreview(data)
    } catch (error) {
      if (error.name !== 'AbortError' && previewControllerRef.current === controller) {
        setPreviewError(error.message || 'The dataset preview could not be loaded.')
      }
    } finally {
      if (previewControllerRef.current === controller) {
        setIsLoadingPreview(false)
      }
    }
  }

  return (
    <section
      aria-labelledby="history-heading"
      className="rounded-2xl border border-slate-200 bg-white p-5 shadow-[0_18px_55px_-30px_rgba(15,23,42,0.25)] sm:p-7"
    >
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-emerald-700">
            Saved datasets
          </p>
          <h2 id="history-heading" className="mt-1 text-xl font-semibold text-slate-900">
            Select a dataset
          </h2>
          <p className="mt-1.5 text-sm leading-6 text-slate-500">
            Choose an uploaded dataset for the next analytics steps and verify its preview.
          </p>
        </div>
        <p className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs leading-5 text-amber-800 sm:max-w-xs">
          Development only: authentication and per-user data isolation are not implemented.
        </p>
      </div>

      {historyError && (
        <p role="alert" className="mt-5 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          {historyError}
        </p>
      )}

      {isLoadingHistory && <p className="mt-5 text-sm text-slate-500">Loading upload history...</p>}

      {!isLoadingHistory && !historyError && uploads.length === 0 && (
        <div className="mt-5 rounded-xl border border-dashed border-slate-300 bg-slate-50 px-5 py-8 text-center">
          <p className="text-sm font-medium text-slate-700">No uploads have been saved yet.</p>
          <p className="mt-1 text-xs text-slate-500">Your next successful CSV upload will appear here.</p>
        </div>
      )}

      {uploads.length > 0 && (
        <div className="mt-5 grid gap-5 lg:grid-cols-[minmax(0,0.8fr)_minmax(0,1.2fr)]">
          <div className="space-y-2">
            {uploads.map((upload) => (
              <button
                key={upload.upload_id}
                type="button"
                aria-pressed={selectedId === upload.upload_id}
                onClick={() => loadPreview(upload.upload_id)}
                className={`w-full rounded-xl border p-4 text-left transition focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500 focus-visible:ring-offset-2 ${
                  selectedId === upload.upload_id
                    ? 'border-emerald-400 bg-emerald-50'
                    : 'border-slate-200 bg-white hover:border-emerald-300 hover:bg-slate-50'
                }`}
              >
                <div className="flex items-start justify-between gap-3">
                  <p className="min-w-0 truncate text-sm font-semibold text-slate-800">
                    {upload.filename}
                  </p>
                  <span className="flex shrink-0 flex-wrap justify-end gap-1.5">
                    {selectedId === upload.upload_id && (
                      <span className="rounded-full bg-emerald-700 px-2.5 py-1 text-xs font-medium text-white">
                        Selected
                      </span>
                    )}
                    <span className={`rounded-full px-2.5 py-1 text-xs font-medium capitalize ${
                      upload.status === 'failed'
                        ? 'bg-red-100 text-red-700'
                        : upload.status === 'ready' || upload.status === 'processed'
                          ? 'bg-emerald-100 text-emerald-700'
                          : 'bg-amber-100 text-amber-700'
                    }`}>
                      {upload.status}
                    </span>
                  </span>
                </div>
                <p className="mt-2 text-xs text-slate-500">
                  {formatUploadDate(upload.uploaded_at)} · {formatFileSize(upload.size)}
                </p>
              </button>
            ))}
          </div>

          <div className="min-w-0 rounded-xl border border-slate-200 bg-slate-50/60 p-4">
            {!selectedId && (
              <div className="flex min-h-40 items-center justify-center text-center text-sm text-slate-500">
                Choose a saved upload to select it and view its dataset preview.
              </div>
            )}
            {isLoadingPreview && <p className="text-sm text-slate-500">Loading preview...</p>}
            {previewError && <p role="alert" className="text-sm text-red-700">{previewError}</p>}
            {preview && (
              <div>
                <div className="mb-3">
                  <p className="mb-1 text-xs font-semibold uppercase tracking-[0.14em] text-emerald-700">
                    Selected dataset
                  </p>
                  <h3 className="truncate text-sm font-semibold text-slate-900">{preview.filename}</h3>
                  <p className="mt-1 text-xs text-slate-500">
                    {preview.columns.length} columns · {preview.preview_row_count} preview rows
                  </p>
                </div>
                <div className="overflow-x-auto rounded-lg border border-slate-200 bg-white">
                  <table className="min-w-full divide-y divide-slate-200 text-left text-xs">
                    <thead className="bg-slate-50 text-slate-600">
                      <tr>
                        {preview.columns.map((column, index) => (
                          <th key={`${column}-${index}`} className="whitespace-nowrap px-3 py-2.5 font-semibold">
                            {column}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100 text-slate-600">
                      {preview.rows.map((row, rowIndex) => (
                        <tr key={rowIndex}>
                          {preview.columns.map((column, columnIndex) => (
                            <td key={`${column}-${columnIndex}`} className="max-w-52 truncate whitespace-nowrap px-3 py-2.5">
                              {row[column] === null ? '—' : String(row[column])}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </section>
  )
}

export default UploadHistory
