// Client for POST /api/respond/vision (Phase 4): sends the spoken question with
// the captured frame and its detections, and describes the structured answer
// for the UI. Pure helpers only, so they can be tested with plain Node
// (see visionRespond.test.ts).

import type { DetectionBox, DetectionResult } from './detection'

export type AnswerRoute = 'text' | 'detections' | 'vlm' | 'no_vision'

export type ReferencedObject = {
  label: string // natural word ("phone")
  detector_label: string // COCO label ("cell phone")
  confidence: number
  box: DetectionBox // normalized to the unmirrored frame, like /api/detect
  position: string // already worded for the mirrored preview
}

export type VisionAnswer = {
  response: string
  memory_warning: string | null
  route: AnswerRoute
  used_vision: boolean
  objects_referenced: ReferencedObject[]
  confidence: 'high' | 'medium' | 'low' | null
  reason: string // debug text: tooltip only, never shown as the answer
  vision_warning: string | null
  latency_ms: number
}

// The VLM route can take minutes on the 2-core laptop (two attempts of up to
// VLM_TIMEOUT_SECONDS=180 each, plus queueing). Give the server time to time
// out first and report a real error.
export const RESPOND_TIMEOUT_MS = 450_000
// After this long without an answer the UI says it is "looking closely".
export const SLOW_ANSWER_MS = 5_000

export class RespondError extends Error {
  constructor(message: string, readonly kind: 'http' | 'network' | 'timeout' | 'cancelled' | 'bad-response', readonly status?: number) {
    super(message)
    this.name = 'RespondError'
  }
}

/** The multipart body the backend expects. The browser sets the boundary. */
export function buildVisionForm(
  transcript: string,
  sessionId: string,
  detection: DetectionResult | null,
  frame: Blob | null,
): FormData {
  const form = new FormData()
  form.append('transcript', transcript)
  form.append('session_id', sessionId)
  if (detection) form.append('detections', JSON.stringify(detection))
  if (frame) form.append('file', frame, 'frame.jpg')
  return form
}

function detailText(detail: unknown): string {
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) return detail.map((item) => (item as { msg?: string })?.msg ?? '').filter(Boolean).join('; ')
  return ''
}

const ROUTES: AnswerRoute[] = ['text', 'detections', 'vlm', 'no_vision']

function isVisionAnswer(value: unknown): value is VisionAnswer {
  if (!value || typeof value !== 'object') return false
  const answer = value as Partial<VisionAnswer>
  return typeof answer.response === 'string'
    && ROUTES.includes(answer.route as AnswerRoute)
    && Array.isArray(answer.objects_referenced)
}

/**
 * POST the question. `signal` lets the user stop waiting (the server may still
 * finish, and then saves the turn). Throws RespondError.
 */
export async function askVision(
  form: FormData,
  options: { signal?: AbortSignal; fetcher?: typeof fetch; timeoutMs?: number } = {},
): Promise<VisionAnswer> {
  const fetcher = options.fetcher ?? fetch
  const timeoutMs = options.timeoutMs ?? RESPOND_TIMEOUT_MS
  const controller = new AbortController()
  let timedOut = false
  const timer = setTimeout(() => { timedOut = true; controller.abort() }, timeoutMs)
  const onCancel = () => controller.abort()
  options.signal?.addEventListener('abort', onCancel)
  if (options.signal?.aborted) controller.abort()

  let response: Response
  try {
    response = await fetcher('/api/respond/vision', { method: 'POST', body: form, signal: controller.signal })
  } catch {
    if (timedOut) throw new RespondError(`No answer after ${Math.round(timeoutMs / 60_000 * 10) / 10} minutes. The computer may be low on memory.`, 'timeout')
    if (controller.signal.aborted) throw new RespondError('Stopped waiting for the answer.', 'cancelled')
    throw new RespondError('Jarvis backend not reachable. Is it running?', 'network')
  } finally {
    clearTimeout(timer)
    options.signal?.removeEventListener('abort', onCancel)
  }

  let body: unknown = null
  try {
    body = await response.json()
  } catch {
    body = null
  }
  if (!response.ok) {
    const detail = detailText((body as { detail?: unknown } | null)?.detail)
    throw new RespondError(detail || `Local AI response failed (HTTP ${response.status}).`, 'http', response.status)
  }
  if (!isVisionAnswer(body)) throw new RespondError('Unexpected response from /api/respond/vision.', 'bad-response')
  return body
}

export type RouteTag = { text: string; tone: 'vision' | 'memory' | 'warning'; title: string }

/** Small tag that says plainly where the answer came from. */
export function routeTag(answer: Pick<VisionAnswer, 'route' | 'confidence' | 'vision_warning' | 'reason'>): RouteTag {
  const confidence = answer.confidence ? ` · ${answer.confidence} confidence` : ''
  const debug = answer.reason ? ` (${answer.reason})` : ''
  switch (answer.route) {
    case 'detections':
      return answer.vision_warning
        ? { text: `Camera, detector only${confidence}`, tone: 'warning', title: `The vision model failed, so this comes from the object detector only.${debug}` }
        : { text: `Looked at camera${confidence}`, tone: 'vision', title: `Answered from the object detector's results for the captured frame.${debug}` }
    case 'vlm':
      return answer.vision_warning
        ? { text: `Camera, detector only${confidence}`, tone: 'warning', title: `The close look failed, so this comes from the object detector only.${debug}` }
        : { text: `Looked closely${confidence}`, tone: 'vision', title: `Answered by the local vision model looking at the captured frame.${debug}` }
    case 'no_vision':
      return { text: 'No camera data', tone: 'warning', title: `The question needed the camera, but no frame or detections were sent.${debug}` }
    default:
      return { text: 'From memory', tone: 'memory', title: `Answered by the language model with conversation memory; the camera was not needed.${debug}` }
  }
}

/** "0:07", "2:31" */
export function formatElapsed(ms: number): string {
  const seconds = Math.max(0, Math.floor(ms / 1000))
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`
}

/**
 * Ask the backend which route the question will take (no model runs, about a
 * millisecond). Resolves to true when the slow vision model will be used,
 * false when not, and null when the preview failed (the UI then falls back to
 * time-based wording). Never rejects.
 */
export async function previewUsesVisionModel(
  transcript: string,
  detection: DetectionResult | null,
  hasImage: boolean,
  fetcher: typeof fetch = fetch,
): Promise<boolean | null> {
  try {
    const response = await fetcher('/api/respond/route', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ transcript, has_image: hasImage, detections: detection }),
    })
    if (!response.ok) return null
    const body = await response.json() as { uses_vision_model?: unknown }
    return typeof body.uses_vision_model === 'boolean' ? body.uses_vision_model : null
  } catch {
    return null
  }
}

// Offer "Stop waiting" after this long even when no vision model is used (the
// text route took 29-60 s on this laptop under memory pressure).
export const OFFER_STOP_MS = 15_000

export type WaitStatus = { text: string; closeLook: boolean; canStop: boolean }

/**
 * Status line while waiting for the answer. `usesVisionModel` comes from
 * previewUsesVisionModel (null = unknown). "Looking closely" is only claimed
 * when the backend said the vision model will run.
 */
export function waitingMessage(elapsedMs: number, hasFrame: boolean, usesVisionModel: boolean | null): WaitStatus {
  const clock = formatElapsed(elapsedMs)
  if (usesVisionModel) {
    return {
      text: `Looking closely at the camera… ${clock} (this can take a few minutes on this computer)`,
      closeLook: true,
      canStop: true,
    }
  }
  if (elapsedMs < SLOW_ANSWER_MS) return { text: 'Thinking', closeLook: false, canStop: false }
  const hint = usesVisionModel === null && hasFrame ? ' (a close look at the camera can take a few minutes)' : ''
  return { text: `Still working… ${clock}${hint}`, closeLook: false, canStop: elapsedMs >= OFFER_STOP_MS || hint !== '' }
}
