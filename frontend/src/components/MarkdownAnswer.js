import { createElement } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'


function SafeLink(props) {
  return createElement('a', {
    ...props,
    target: '_blank',
    rel: 'noopener noreferrer',
  })
}

export function MarkdownAnswer({ content }) {
  return createElement(
    'div',
    { className: 'markdown-answer break-words text-sm leading-6' },
    createElement(
      ReactMarkdown,
      {
        remarkPlugins: [remarkGfm],
        skipHtml: true,
        disallowedElements: ['img'],
        unwrapDisallowed: true,
        components: { a: SafeLink },
      },
      content || '',
    ),
  )
}

export default MarkdownAnswer
