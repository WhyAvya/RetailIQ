export const CURRENCY_OPTIONS = [
  'USD', 'INR', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD',
  'CHF', 'CNY', 'HKD', 'KRW', 'NZD', 'SGD', 'ZAR',
]

export function initialCurrencySelection(metadata, savedSelection) {
  if (savedSelection) return savedSelection
  if (metadata?.status === 'detected' && metadata.detected_code) {
    return { code: metadata.detected_code, source: 'detected' }
  }
  return { code: '', source: 'unspecified' }
}

export function formatMonetaryValue(value, currencyCode, includeUnspecifiedLabel = false) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return 'Unavailable'
  if (!currencyCode) {
    const formatted = new Intl.NumberFormat(undefined, {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value)
    return includeUnspecifiedLabel ? `${formatted} (currency not specified)` : formatted
  }
  return new Intl.NumberFormat(undefined, {
    style: 'currency',
    currency: currencyCode,
  }).format(value)
}

export function plotlyCurrencyFormat(currencyCode) {
  if (!currencyCode) return { tickprefix: '', tickformat: ',.2f' }
  const formatter = new Intl.NumberFormat(undefined, {
    style: 'currency',
    currency: currencyCode,
  })
  const options = formatter.resolvedOptions()
  const symbol = formatter.formatToParts(0).find((part) => part.type === 'currency')?.value || currencyCode
  return {
    tickprefix: symbol,
    tickformat: `,.${options.maximumFractionDigits}f`,
  }
}

export function isMonetaryCalculation(calculation) {
  if (!calculation) return false
  const intent = calculation.intent || ''
  if (['total_revenue', 'total_profit', 'average_order_value', 'highest_revenue_region', 'lowest_profit_category'].includes(intent)) {
    return true
  }
  const plan = calculation.plan || {}
  return ['sales', 'profit'].includes(plan.metric) || ['sales', 'profit'].includes(plan.ranking_metric)
}

export function monetaryCalculationValues(calculation) {
  if (!isMonetaryCalculation(calculation)) return []
  const values = []
  if (typeof calculation.value === 'number') values.push(calculation.value)
  for (const key of ['revenue', 'profit']) {
    if (typeof calculation[key] === 'number') values.push(calculation[key])
  }
  for (const record of calculation.records || []) {
    for (const key of ['Sales', 'Profit', 'revenue', 'profit', 'value']) {
      if (typeof record[key] === 'number') values.push(record[key])
    }
  }
  return [...new Set(values)]
}

export function formatAnalyticsAnswer(answer, calculation, currencyCode) {
  if (!isMonetaryCalculation(calculation) || typeof answer !== 'string') return answer
  return monetaryCalculationValues(calculation)
    .sort((left, right) => String(right).length - String(left).length)
    .reduce((formatted, value) => {
      const raw = Number.isInteger(value)
        ? value.toLocaleString('en-US')
        : value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
      return formatted.replaceAll(raw, formatMonetaryValue(value, currencyCode, !currencyCode))
    }, answer)
}
