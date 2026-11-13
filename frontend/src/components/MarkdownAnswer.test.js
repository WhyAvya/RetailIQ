import assert from 'node:assert/strict'
import test from 'node:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { MarkdownAnswer } from './MarkdownAnswer.js'

function render(content) {
  return renderToStaticMarkup(createElement(MarkdownAnswer, { content }))
}

test('renders headings, bold text, lists, and GFM tables', () => {
  const html = render('# Summary\n\n**Strong result**\n\n- One\n- Two\n\n| Region | Sales |\n|---|---:|\n| West | 25 |')

  assert.match(html, /<h1>Summary<\/h1>/)
  assert.match(html, /<strong>Strong result<\/strong>/)
  assert.match(html, /<ul>/)
  assert.match(html, /<table>/)
})

test('suppresses raw HTML, scripts, event handlers, and unsafe image markup', () => {
  const html = render('Safe response text.\n\n<script>alert(1)</script>\n\n<img src=x onerror="alert(2)">\n\n[bad](javascript:alert(3))')

  assert.doesNotMatch(html, /<script|<img|onerror|javascript:/i)
  assert.match(html, /Safe response text/)
})
