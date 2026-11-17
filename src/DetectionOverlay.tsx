import { useEffect, useRef, useState } from 'react'
import { projectBox, summarizeLabels, type VisionState } from './detection'
import type { ReferencedObject } from './visionRespond'

// Objects the last answer referred to, highlighted over the camera view.
export type AnswerHighlight = { id: number; frameWidth: number; frameHeight: number; objects: ReferencedObject[] }

const HIGHLIGHT_VISIBLE_MS = 15_000
// The backend only treats detections at 0.5 or higher as "seen"; weaker
// references are drawn dashed with a "?" (the answer hedges them too).
const UNCERTAIN_BELOW = 0.5

const MAX_DRAWN_BOXES = 15
// Boxes describe the moment the frame was captured; the live video moves on,
// so they fade out after a while. The "Seen" summary stays until the next frame.
const BOX_VISIBLE_MS = 12_000
// The camera LED, "Seen" summary, and conversation menu sit in the top ~58 px.
// Labels of boxes that start there are pushed down inside the box.
const TOP_CONTROLS_BAND = 58
const LABEL_ROOM = 120

type DetectionOverlayProps = {
  vision: VisionState
  highlight?: AnswerHighlight | null
  mirrored: boolean
}

function DetectionOverlay({ vision, highlight = null, mirrored }: DetectionOverlayProps) {
  const layerRef = useRef<HTMLDivElement>(null)
  const [size, setSize] = useState({ width: 0, height: 0 })
  const [hiddenFrameId, setHiddenFrameId] = useState<number | null>(null)
  const [hiddenHighlightId, setHiddenHighlightId] = useState<number | null>(null)

  const highlightId = highlight?.id ?? null
  useEffect(() => {
    if (highlightId === null) return
    const timer = window.setTimeout(() => setHiddenHighlightId(highlightId), HIGHLIGHT_VISIBLE_MS)
    return () => window.clearTimeout(timer)
  }, [highlightId])
  const activeHighlight = highlight && hiddenHighlightId !== highlight.id ? highlight : null
  // Boxes are relative to the unmirrored frame; projectBox mirrors them.
  const highlighted = activeHighlight
    ? activeHighlight.objects.slice(0, MAX_DRAWN_BOXES).flatMap((object, index) => {
      const view = projectBox(object.box, activeHighlight.frameWidth, activeHighlight.frameHeight, size.width, size.height, mirrored)
      return view ? [{ object, view, key: `answer-${activeHighlight.id}-${index}` }] : []
    })
    : []

  useEffect(() => {
    const layer = layerRef.current
    if (!layer) return
    const measure = () => setSize({ width: layer.clientWidth, height: layer.clientHeight })
    measure()
    if (typeof ResizeObserver === 'undefined') {
      window.addEventListener('resize', measure)
      return () => window.removeEventListener('resize', measure)
    }
    const observer = new ResizeObserver(measure)
    observer.observe(layer)
    return () => observer.disconnect()
  }, [])

  const readyFrameId = vision.status === 'ready' ? vision.frameId : null
  useEffect(() => {
    if (readyFrameId === null) return
    const timer = window.setTimeout(() => setHiddenFrameId(readyFrameId), BOX_VISIBLE_MS)
    return () => window.clearTimeout(timer)
  }, [readyFrameId])

  const showBoxes = vision.status === 'ready' && hiddenFrameId !== vision.frameId
  const boxes = vision.status === 'ready' && showBoxes
    ? vision.result.detections.slice(0, MAX_DRAWN_BOXES).flatMap((detection, index) => {
      const view = projectBox(
        detection.box,
        vision.result.image_width,
        vision.result.image_height,
        size.width,
        size.height,
        mirrored,
      )
      return view ? [{ detection, view, key: `${vision.frameId}-${index}` }] : []
    })
    : []

  return (
    <>
      <div ref={layerRef} className={`detection-layer${highlighted.length ? ' has-highlight' : ''}`} aria-hidden="true">
        {highlighted.map(({ object, view, key }) => (
          <div
            key={key}
            className={`detection-box is-referenced${object.confidence < UNCERTAIN_BELOW ? ' is-uncertain' : ''}`}
            style={{ left: view.left, top: view.top, width: view.width, height: view.height }}
          >
            <span
              className={[
                view.top < TOP_CONTROLS_BAND ? 'is-inside' : '',
                view.left + LABEL_ROOM > size.width ? 'is-right' : '',
              ].join(' ').trim() || undefined}
              style={view.top < TOP_CONTROLS_BAND
                ? { top: Math.max(2, Math.min(TOP_CONTROLS_BAND - view.top, view.height - 20)) }
                : undefined}
            >
              {object.confidence < UNCERTAIN_BELOW ? `${object.label}?` : object.label}
            </span>
          </div>
        ))}
        {boxes.map(({ detection, view, key }) => (
          <div
            key={key}
            className={`detection-box${detection.label === 'person' ? ' is-person' : ''}`}
            style={{ left: view.left, top: view.top, width: view.width, height: view.height }}
          >
            <span
              className={[
                view.top < TOP_CONTROLS_BAND ? 'is-inside' : '',
                // Keep labels of boxes at the right edge on screen.
                view.left + LABEL_ROOM > size.width ? 'is-right' : '',
              ].join(' ').trim() || undefined}
              style={view.top < TOP_CONTROLS_BAND
                ? { top: Math.max(2, Math.min(TOP_CONTROLS_BAND - view.top, view.height - 20)) }
                : undefined}
            >
              {detection.label} {Math.round(detection.confidence * 100)}%
            </span>
          </div>
        ))}
      </div>

      {vision.status !== 'idle' && (
        <div
          className={`vision-summary is-${vision.status}`}
          role="status"
          aria-live="polite"
          title={vision.status === 'error' ? vision.detail : vision.status === 'ready'
            ? `Detected ${Math.round(vision.elapsedMs)} ms after capture (model ${Math.round(vision.result.inference_ms)} ms)`
            : undefined}
        >
          {vision.status === 'detecting' && <span>Looking…</span>}
          {vision.status === 'error' && <span>{vision.message}</span>}
          {vision.status === 'ready' && (
            <>
              <span className="vision-seen">
                {vision.result.detections.length
                  ? `Seen: ${summarizeLabels(vision.result.detections).join(', ')}`
                  : 'Nothing recognized'}
              </span>
              <span className={`vision-person${vision.result.person_present ? ' is-present' : ''}`}>
                {vision.result.person_present ? 'Person in view' : 'No person'}
              </span>
            </>
          )}
        </div>
      )}
    </>
  )
}

export default DetectionOverlay
