import { useLayoutEffect, useRef, useState } from 'react'

const revealedKeys = new Set()
const listeners = new WeakMap()
let sharedObserver = null
let observedCount = 0

function observerInstance() {
  if (sharedObserver) return sharedObserver
  sharedObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return
      const reveal = listeners.get(entry.target)
      if (!reveal) return
      listeners.delete(entry.target)
      sharedObserver.unobserve(entry.target)
      observedCount -= 1
      reveal()
    })
    if (observedCount === 0) {
      sharedObserver.disconnect()
      sharedObserver = null
    }
  }, { threshold: 0.08, rootMargin: '0px 0px -32px' })
  return sharedObserver
}

function observeOnce(node, reveal) {
  const observer = observerInstance()
  listeners.set(node, reveal)
  observedCount += 1
  observer.observe(node)
  return () => {
    if (!listeners.has(node)) return
    listeners.delete(node)
    observer.unobserve(node)
    observedCount -= 1
    if (observedCount === 0 && sharedObserver === observer) {
      observer.disconnect()
      sharedObserver = null
    }
  }
}

function Reveal({ as: Element = 'div', children, className = '', delay = 0, revealKey, ...props }) {
  const nodeRef = useRef(null)
  const mountedRef = useRef(false)
  const [shouldAnimate] = useState(() => (
    typeof IntersectionObserver !== 'undefined'
      && !window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
      && !(revealKey && revealedKeys.has(revealKey))
  ))
  const [revealed, setRevealed] = useState(!shouldAnimate)

  useLayoutEffect(() => {
    if (!shouldAnimate || !nodeRef.current) return undefined

    mountedRef.current = true
    let stopObserving = () => {}
    const reveal = () => {
      if (!mountedRef.current) return
      if (revealKey) revealedKeys.add(revealKey)
      setRevealed(true)
    }

    try {
      stopObserving = observeOnce(nodeRef.current, reveal)
    } catch {
      queueMicrotask(reveal)
    }

    return () => {
      mountedRef.current = false
      stopObserving()
    }
  }, [revealKey, shouldAnimate])

  return (
    <Element
      {...props}
      ref={nodeRef}
      className={`reveal-shell ${shouldAnimate ? 'reveal-ready' : ''} ${revealed ? 'reveal-visible' : ''} ${className}`}
      style={{ '--reveal-delay': `${delay}ms` }}
    >
      {children}
    </Element>
  )
}

export default Reveal
