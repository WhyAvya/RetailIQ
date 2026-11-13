import { useEffect, useMemo, useState } from 'react'
import AnalyticsCharts from './AnalyticsCharts.jsx'
import DashboardFilters from './DashboardFilters.jsx'
import KpiDashboard from './KpiDashboard.jsx'
import CurrencySelector from './CurrencySelector.jsx'

const ANALYTICS_URL = 'http://localhost:8000/api/analytics'
const EMPTY_FILTERS = {
  region: '',
  category: '',
  startDate: '',
  endDate: '',
}

async function responseData(response, fallbackMessage) {
  const data = await response.json().catch(() => ({}))
  if (!response.ok) {
    throw new Error(typeof data.detail === 'string' ? data.detail : fallbackMessage)
  }
  return data
}

function buildFilterQuery(filters) {
  const parameters = new URLSearchParams()
  if (filters.region) parameters.set('region', filters.region)
  if (filters.category) parameters.set('category', filters.category)
  if (filters.startDate) parameters.set('start_date', filters.startDate)
  if (filters.endDate) parameters.set('end_date', filters.endDate)
  return parameters.toString()
}

function AnalyticsDashboard({ uploadId, currency, onCurrencyChange }) {
  const [options, setOptions] = useState(null)
  const [optionsError, setOptionsError] = useState('')
  const [isLoadingOptions, setIsLoadingOptions] = useState(false)
  const [draft, setDraft] = useState(EMPTY_FILTERS)
  const [applied, setApplied] = useState(EMPTY_FILTERS)
  const [validationError, setValidationError] = useState('')
  const [result, setResult] = useState(null)
  const [analyticsError, setAnalyticsError] = useState('')
  const [isLoadingAnalytics, setIsLoadingAnalytics] = useState(false)
  const filterQuery = useMemo(() => buildFilterQuery(applied), [applied])

  useEffect(() => {
    if (!uploadId) return undefined
    const controller = new AbortController()

    async function loadOptions() {
      setIsLoadingOptions(true)
      setOptionsError('')
      try {
        const response = await fetch(`${ANALYTICS_URL}/${uploadId}/filter-options`, {
          signal: controller.signal,
        })
        const data = await responseData(response, 'Filter options could not be loaded.')
        if (!controller.signal.aborted && data.upload_id === uploadId) setOptions(data.options)
      } catch (error) {
        if (error.name !== 'AbortError') {
          setOptionsError(error.message || 'Filter options could not be loaded.')
        }
      } finally {
        if (!controller.signal.aborted) setIsLoadingOptions(false)
      }
    }

    loadOptions()
    return () => controller.abort()
  }, [uploadId])

  useEffect(() => {
    if (!uploadId) return undefined
    const controller = new AbortController()
    let isCurrentRequest = true

    async function loadDashboard() {
      setIsLoadingAnalytics(true)
      setAnalyticsError('')
      setResult(null)
      try {
        const suffix = filterQuery ? `?${filterQuery}` : ''
        const response = await fetch(`${ANALYTICS_URL}/${uploadId}/dashboard${suffix}`, {
          signal: controller.signal,
        })
        const data = await responseData(response, 'Dashboard analytics could not be loaded.')
        if (isCurrentRequest && data.upload_id === uploadId) setResult(data)
      } catch (error) {
        if (isCurrentRequest && error.name !== 'AbortError') {
          setAnalyticsError(error.message || 'Dashboard analytics could not be loaded.')
        }
      } finally {
        if (isCurrentRequest) setIsLoadingAnalytics(false)
      }
    }

    loadDashboard()
    return () => {
      isCurrentRequest = false
      controller.abort()
    }
  }, [filterQuery, uploadId])

  const changeFilter = (name, value) => {
    setValidationError('')
    setDraft((current) => ({ ...current, [name]: value }))
  }

  const applyFilters = (event) => {
    event.preventDefault()
    if (draft.startDate && draft.endDate && draft.startDate > draft.endDate) {
      setValidationError('Start date must be on or before end date.')
      return
    }
    setValidationError('')
    setApplied({ ...draft })
  }

  const resetFilters = () => {
    setValidationError('')
    setDraft({ ...EMPTY_FILTERS })
    setApplied({ ...EMPTY_FILTERS })
  }

  return (
    <>
      {uploadId && currency && (
        <div className="mb-6">
          <CurrencySelector currency={currency} onChange={onCurrencyChange} />
        </div>
      )}
      <DashboardFilters
        uploadId={uploadId}
        options={options}
        optionsError={optionsError}
        isLoadingOptions={isLoadingOptions}
        draft={draft}
        applied={applied}
        validationError={validationError}
        filterContext={result?.upload_id === uploadId ? result.filter_context : null}
        onChange={changeFilter}
        onApply={applyFilters}
        onReset={resetFilters}
      />

      <div className="mt-10">
        <KpiDashboard
          uploadId={uploadId}
          result={result}
          error={analyticsError}
          isLoading={isLoadingAnalytics}
          currency={currency}
        />
      </div>

      <div className="mt-10">
        <AnalyticsCharts
          uploadId={uploadId}
          result={result}
          error={analyticsError}
          isLoading={isLoadingAnalytics}
          currency={currency}
        />
      </div>
    </>
  )
}

export default AnalyticsDashboard
