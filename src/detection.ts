// Object detection helpers: the /api/detect client and the pure math used to
// draw boxes over the camera view. The math lives here (not in a component)
// so it can be unit-tested without a browser: see detection.test.ts.

export type DetectionBox = {
  // Fractions (0-1) of the uploaded frame, top-left origin.
  x: number
  y: number
  width: number
  height: number
}

export type Detection = {
  label: string
  confidence: number
  box: DetectionBox
}

export type DetectionResult = {
  detections: Detection[]
  person_present: boolean
  image_width: number
  image_height: number
  inference_ms: number
}

// Latest detection state kept in App. Phase 4 reads the 'ready' variant (and
// can await App's frameDetectionRef promise) to send detections with a question.
export type VisionState =
  | { status: 'idle' }
  | { status: 'detecting'; frameId: number }
  // elapsedMs: capture -> result (JPEG encode + queue + upload + inference + JSON).
  | { status: 'ready'; frameId: number; capturedAt: number; result: DetectionResult; elapsedMs: number }
  | { status: 'error'; frameId: number; message: string; detail: string }

export type ViewBox = { left: number; top: number; width: number; height: number }

export const DETECT_TIMEOUT_MS = 20_000
export const FRAME_JPEG_QUALITY = 0.85

export class DetectionRequestError extends Error {
  // Short text for the UI; `detail` keeps the full server message (e.g. the
  // weight download command) for the tooltip and logs.
  constructor(message: string, readonly detail: string, readonly status?: number) {
    super(message)
    this.name = 'DetectionRequestError'
  }
}

/**
 * Where a frame of frameWidth x frameHeight lands inside a viewWidth x
 * viewHeight element with `object-fit: cover; object-position: center`.
 * The frame is scaled up until it covers the element, then centered, so the
 * overflowing edges (offsetX/offsetY negative) are cropped away.
 */
export function coverLayout(frameWidth: number, frameHeight: number, viewWidth: number, viewHeight: number) {
  const scale = Math.max(viewWidth / frameWidth, viewHeight / frameHeight)
  const width = frameWidth * scale
  const height = frameHeight * scale
  return { scale, width, height, offsetX: (viewWidth - width) / 2, offsetY: (viewHeight - height) / 2 }
}

/**
 * Map a normalized box from the captured frame to pixels in the displayed
 * element. `mirrored` matches the video's CSS `transform: scaleX(-1)` (the
 * captured canvas frame itself is not mirrored). Boxes are clipped to the
 * visible area; a box that is cropped away entirely returns null.
 */
export function projectBox(
  box: DetectionBox,
  frameWidth: number,
  frameHeight: number,
  viewWidth: number,
  viewHeight: number,
  mirrored: boolean,
): ViewBox | null {
  if (!(frameWidth > 0 && frameHeight > 0 && viewWidth > 0 && viewHeight > 0)) return null
  const layout = coverLayout(frameWidth, frameHeight, viewWidth, viewHeight)
  const width = box.width * layout.width
  const height = box.height * layout.height
  let left = layout.offsetX + box.x * layout.width
  const top = layout.offsetY + box.y * layout.height
  if (mirrored) left = viewWidth - left - width

  const x1 = Math.max(0, left)
  const y1 = Math.max(0, top)
  const x2 = Math.min(viewWidth, left + width)
  const y2 = Math.min(viewHeight, top + height)
  if (x2 - x1 < 2 || y2 - y1 < 2) return null
  return { left: x1, top: y1, width: x2 - x1, height: y2 - y1 }
}

/** "laptop ×2", "person", ... ordered by best confidence (the API sorts detections). */
export function summarizeLabels(detections: Detection[], maxItems = 6): string[] {
  const counts = new Map<string, number>()
  for (const detection of detections) counts.set(detection.label, (counts.get(detection.label) ?? 0) + 1)
  const items = [...counts].map(([label, count]) => (count > 1 ? `${label} ×${count}` : label))
  if (items.length <= maxItems) return items
  return [...items.slice(0, maxItems), `+${items.length - maxItems} more`]
}

function isDetectionResult(value: unknown): value is DetectionResult {
  if (!value || typeof value !== 'object') return false
  const result = value as Partial<DetectionResult>
  return Array.isArray(result.detections)
    && typeof result.person_present === 'boolean'
    && typeof result.image_width === 'number'
    && typeof result.image_height === 'number'
}

function detailText(detail: unknown): string {
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) return detail.map((item) => (item as { msg?: string })?.msg ?? '').filter(Boolean).join('; ')
  return ''
}

/** Turn a failed /api/detect response into a short UI message plus the full detail. */
export function describeDetectError(status: number, body: unknown): DetectionRequestError {
  const detail = detailText((body as { detail?: unknown } | null)?.detail) || `HTTP ${status}`
  let message = 'Detection failed'
  if (status === 503) message = detail.includes('download_vision_model') ? 'Detector model missing' : 'Detector unavailable'
  else if (status === 413) message = 'Frame too large to analyze'
  else if (status === 400 || status === 415 || status === 422) message = 'Frame could not be read'
  else if (status === 502) message = 'Detector error'
  return new DetectionRequestError(message, detail, status)
}

/** POST one JPEG frame to the local detector. Throws DetectionRequestError. */
export async function detectFrame(frame: Blob, fetcher: typeof fetch = fetch): Promise<DetectionResult> {
  const form = new FormData()
  form.append('file', frame, 'frame.jpg')
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), DETECT_TIMEOUT_MS)
  let response: Response
  try {
    // No Content-Type header: fetch adds the multipart boundary itself.
    response = await fetcher('/api/detect', { method: 'POST', body: form, signal: controller.signal })
  } catch (error) {
    const timedOut = error instanceof DOMException && error.name === 'AbortError'
    throw new DetectionRequestError(
      timedOut ? 'Detection timed out' : 'Detector not reachable',
      timedOut ? `No answer from /api/detect within ${DETECT_TIMEOUT_MS / 1000} s.` : 'Is the backend running?',
    )
  } finally {
    clearTimeout(timer)
  }
  let body: unknown = null
  try {
    body = await response.json()
  } catch {
    body = null
  }
  if (!response.ok) throw describeDetectError(response.status, body)
  if (!isDetectionResult(body)) throw new DetectionRequestError('Detection failed', 'Unexpected response from /api/detect.')
  return body
}

/** canvas.toBlob as a promise (JPEG). Rejects if the browser cannot encode the frame. */
export function canvasToJpeg(canvas: HTMLCanvasElement, quality = FRAME_JPEG_QUALITY): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob)
      else reject(new DetectionRequestError('Frame could not be encoded', 'canvas.toBlob returned null.'))
    }, 'image/jpeg', quality)
  })
}
