import assert from 'node:assert/strict'
import test from 'node:test'
import {
  formatMonetaryValue,
  formatAnalyticsAnswer,
  initialCurrencySelection,
  isMonetaryCalculation,
  monetaryCalculationValues,
  plotlyCurrencyFormat,
} from './currency.js'

test('detected USD and INR become dataset defaults while missing metadata stays unspecified', () => {
  assert.deepEqual(initialCurrencySelection({ status: 'detected', detected_code: 'USD' }), { code: 'USD', source: 'detected' })
  assert.deepEqual(initialCurrencySelection({ status: 'detected', detected_code: 'INR' }), { code: 'INR', source: 'detected' })
  assert.deepEqual(initialCurrencySelection({ status: 'unspecified', detected_code: null }), { code: '', source: 'unspecified' })
})

test('saved user selection wins when switching back to a dataset', () => {
  const saved = { code: 'EUR', source: 'user' }
  assert.equal(initialCurrencySelection({ status: 'detected', detected_code: 'USD' }, saved), saved)
})

test('currency formatting uses Intl and unspecified values are explicit', () => {
  assert.match(formatMonetaryValue(2502, 'USD'), /\$.*2[,.]502|2[,.]502.*\$/)
  assert.match(formatMonetaryValue(2502, 'INR'), /₹|INR/)
  assert.match(formatMonetaryValue(2502, 'EUR'), /€|EUR/)
  assert.match(formatMonetaryValue(2502, '', true), /currency not specified/)
  assert.equal(plotlyCurrencyFormat('USD').tickprefix.length > 0, true)
})

test('only monetary analytics values are selected for currency presentation', () => {
  const monetary = {
    intent: 'structured_query',
    value: 2502,
    plan: { metric: 'sales' },
    records: [{ State: 'West', Sales: 1200 }],
  }
  assert.equal(isMonetaryCalculation(monetary), true)
  assert.deepEqual(monetaryCalculationValues(monetary), [2502, 1200])
  assert.equal(isMonetaryCalculation({ intent: 'number_of_orders', value: 4 }), false)
  assert.match(
    formatAnalyticsAnswer('Exact dataset calculation: total sales is 2,502.00.', monetary, 'USD'),
    /\$/,
  )
  assert.equal(
    formatAnalyticsAnswer('Number of orders is 4.', { intent: 'number_of_orders', value: 4 }, 'USD'),
    'Number of orders is 4.',
  )
})
