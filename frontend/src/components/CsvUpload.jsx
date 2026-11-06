import { useRef, useState } from 'react'
import DatasetReport from './DatasetReport.jsx'

const UPLOAD_URL = 'http://localhost:8000/api/upload'

function formatFileSize(bytes) {
  if (bytes === 0) return '0 bytes'

  const units = ['bytes', 'KB', 'MB', 'GB']
  const unitIndex = Math.min(
    Math.floor(Math.log(bytes) / Math.log(1024)),
    units.length - 1,
  )
  const size = bytes / 1024 ** unitIndex

  return `${size.toFixed(unitIndex === 0 || size >= 10 ? 0 : 1)} ${units[unitIndex]}`
}

function UploadIcon() {
  return (
    <svg
      aria-hidden="true"
      className="h-6 w-6"
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth="1.8"
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5M5 13.5v4.25A2.25 2.25 0 007.25 20h9.5A2.25 2.25 0 0019 17.75V13.5"
      />
    </svg>
  )
}

function FileIcon() {
  return (
    <svg
      aria-hidden="true"
      className="h-5 w-5"
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth="1.8"
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M14.25 2.25H6.75A2.25 2.25 0 004.5 4.5v15a2.25 2.25 0 002.25 2.25h10.5a2.25 2.25 0 002.25-2.25V7.5m-5.25-5.25L19.5 7.5m-5.25-5.25V7.5h5.25M8 13h8m-8 4h5"
      />
    </svg>
  )
}

function CsvUpload({ onUploadComplete }) {
  const inputRef = useRef(null)
  const [selectedFile, setSelectedFile] = useState(null)
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')
  const [isDragging, setIsDragging] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [uploadResult, setUploadResult] = useState(null)

  const selectFile = (files) => {
    if (isUploading) return

    setMessage('')
    setUploadResult(null)

    if (files.length > 1) {
      setSelectedFile(null)
      setError('Please select one CSV file at a time.')
      return
    }

    const file = files[0]
    if (!file) return

    if (!file.name.toLowerCase().endsWith('.csv')) {
      setSelectedFile(null)
      setError('That file is not a CSV. Please choose a file ending in .csv.')
      return
    }

    setSelectedFile(file)
    setError('')
  }

  const handleInputChange = (event) => {
    selectFile(event.target.files)
    event.target.value = ''
  }

  const handleDrop = (event) => {
    event.preventDefault()
    setIsDragging(false)
    selectFile(event.dataTransfer.files)
  }

  const removeFile = () => {
    setSelectedFile(null)
    setError('')
    setMessage('')
    setUploadResult(null)
  }

  const handleUpload = async () => {
    if (!selectedFile || isUploading) return

    const formData = new FormData()
    formData.append('file', selectedFile)

    setIsUploading(true)
    setError('')
    setMessage('')
    setUploadResult(null)

    try {
      const response = await fetch(UPLOAD_URL, {
        method: 'POST',
        body: formData,
      })
      const data = await response.json().catch(() => ({}))

      if (!response.ok) {
        const detail =
          typeof data.detail === 'string'
            ? data.detail
            : 'The server could not process this file.'
        throw new Error(detail)
      }

      setMessage(
        `${data.filename} was received successfully (${formatFileSize(data.size)}). ` +
          `${data.rows.toLocaleString()} rows and ${data.columns.toLocaleString()} columns were found.`,
      )
      setUploadResult(data)
      onUploadComplete?.(data)
    } catch (uploadError) {
      setError(
        uploadError instanceof Error
          ? uploadError.message
          : 'The upload failed. Please try again.',
      )
    } finally {
      setIsUploading(false)
    }
  }

  return (
    <section
      aria-labelledby="upload-heading"
      className="rounded-2xl border border-slate-200 bg-white p-5 shadow-[0_18px_55px_-30px_rgba(15,23,42,0.35)] sm:p-7"
    >
      <div className="mb-5">
        <p className="mb-2 text-xs font-semibold uppercase tracking-[0.18em] text-emerald-700">
          Dataset upload
        </p>
        <h2 id="upload-heading" className="text-xl font-semibold text-slate-900">
          Add your retail data
        </h2>
        <p className="mt-1.5 text-sm leading-6 text-slate-500">
          Select one CSV file to prepare it for analysis.
        </p>
      </div>

      <div
        role="button"
        tabIndex={isUploading ? -1 : 0}
        aria-label="Choose a CSV file"
        aria-disabled={isUploading}
        onClick={() => !isUploading && inputRef.current?.click()}
        onKeyDown={(event) => {
          if (!isUploading && (event.key === 'Enter' || event.key === ' ')) {
            event.preventDefault()
            inputRef.current?.click()
          }
        }}
        onDragEnter={(event) => {
          event.preventDefault()
          setIsDragging(true)
        }}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        className={`group cursor-pointer rounded-xl border-2 border-dashed px-5 py-10 text-center transition sm:py-12 ${
          isDragging
            ? 'border-emerald-500 bg-emerald-50'
            : isUploading
              ? 'cursor-wait border-slate-200 bg-slate-50 opacity-70'
              : 'border-slate-300 bg-slate-50/70 hover:border-emerald-400 hover:bg-emerald-50/50'
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".csv,text/csv"
          disabled={isUploading}
          onChange={handleInputChange}
          className="sr-only"
        />
        <span className="mx-auto flex h-12 w-12 items-center justify-center rounded-xl border border-slate-200 bg-white text-emerald-700 shadow-sm transition group-hover:-translate-y-0.5 group-hover:shadow-md">
          <UploadIcon />
        </span>
        <p className="mt-4 text-sm font-semibold text-slate-800 sm:text-base">
          Drop your CSV file here
        </p>
        <p className="mt-1.5 text-sm text-slate-500">
          or <span className="font-medium text-emerald-700">browse your computer</span>
        </p>
        <p className="mt-3 text-xs text-slate-400">CSV files only - One file at a time</p>
      </div>

      {error && (
        <div
          role="alert"
          className="mt-4 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700"
        >
          {error}
        </div>
      )}

      {selectedFile && (
        <div className="mt-4 flex items-center gap-3 rounded-xl border border-slate-200 bg-white p-3.5">
          <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-emerald-50 text-emerald-700">
            <FileIcon />
          </span>
          <div className="min-w-0 flex-1">
            <p className="truncate text-sm font-medium text-slate-800">{selectedFile.name}</p>
            <p className="mt-0.5 text-xs text-slate-500">{formatFileSize(selectedFile.size)}</p>
          </div>
          <button
            type="button"
            disabled={isUploading}
            onClick={removeFile}
            className="rounded-lg px-3 py-2 text-sm font-medium text-slate-500 transition hover:bg-slate-100 hover:text-slate-800 focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500 focus-visible:ring-offset-2 disabled:cursor-wait disabled:opacity-50"
          >
            Remove
          </button>
        </div>
      )}

      <button
        type="button"
        disabled={!selectedFile || isUploading}
        onClick={handleUpload}
        aria-busy={isUploading}
        className="mt-5 w-full rounded-xl bg-emerald-700 px-4 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-emerald-800 focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-600 focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:bg-slate-200 disabled:text-slate-400 disabled:shadow-none"
      >
        {isUploading ? 'Uploading...' : 'Upload CSV'}
      </button>

      {message && (
        <p
          role="status"
          className="mt-4 rounded-lg border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm leading-6 text-emerald-700"
        >
          {message}
        </p>
      )}

      {uploadResult && (
        <DatasetReport
          profile={uploadResult.profile}
          preprocessing={uploadResult.preprocessing}
        />
      )}
    </section>
  )
}

export default CsvUpload
