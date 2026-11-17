// Dependency-free tests for the /api/respond/vision client helpers.
// Run: npm run test:frontend
import type { DetectionResult } from './detection.js'
import {
  askVision,
  buildVisionForm,
  formatElapsed,
  RespondError,
  routeTag,
  OFFER_STOP_MS,
  previewUsesVisionModel,
  SLOW_ANSWER_MS,
  waitingMessage,
  type VisionAnswer,
} from './visionRespond.js'

let failures = 0
let passed = 0

async function test(name: string, body: () => void | Promise<void>) {
  try {
    await body()
    passed += 1
    console.log(`ok   ${name}`)
  } catch (error) {
    failures += 1
    console.log(`FAIL ${name}\n     ${error instanceof Error ? error.message : String(error)}`)
  }
}

function equal(actual: unknown, expected: unknown, label = '') {
  const a = JSON.stringify(actual)
  const e = JSON.stringify(expected)
  if (a !== e) throw new Error(`${label} expected ${e}, got ${a}`)
}

function ok(condition: unknown, label: string) {
  if (!condition) throw new Error(label)
}

const ANSWER: VisionAnswer = {
  response: 'The bottle is on the right side of the screen.',
  memory_warning: null,
  route: 'detections',
  used_vision: true,
  objects_referenced: [{
    label: 'bottle', detector_label: 'bottle', confidence: 0.7,
    box: { x: 0.2, y: 0.4, width: 0.1, height: 0.2 }, position: 'on the right side of the screen',
  }],
  confidence: 'high',
  reason: 'locate from detections',
  vision_warning: null,
  latency_ms: 5900,
}

const DETECTION: DetectionResult = {
  detections: [{ label: 'bottle', confidence: 0.7, box: { x: 0.2, y: 0.4, width: 0.1, height: 0.2 } }],
  person_present: false, image_width: 640, image_height: 480, inference_ms: 120,
}

// Node 17 has no FormData, Blob or Response: minimal stand-ins (browsers have the real ones).
const globals = globalThis as unknown as { FormData?: unknown }
if (typeof globals.FormData === 'undefined') {
  globals.FormData = class {
    private entries = new Map<string, unknown>()
    append(name: string, value: unknown, filename?: string) {
      const size = (value as { size?: number }).size
      this.entries.set(name, filename === undefined ? value : { name: filename, size, file: value })
    }
    get(name: string) { return this.entries.has(name) ? this.entries.get(name) : null }
    has(name: string) { return this.entries.has(name) }
  }
}

function jsonResponse(status: number, body: unknown): Response {
  return { ok: status >= 200 && status < 300, status, json: async () => body } as unknown as Response
}

async function expectError(promise: Promise<unknown>, kind: string, messagePart = '') {
  try {
    await promise
  } catch (error) {
    ok(error instanceof RespondError, `expected RespondError, got ${String(error)}`)
    equal((error as RespondError).kind, kind, 'kind')
    if (messagePart) ok((error as Error).message.includes(messagePart), `message "${(error as Error).message}" lacks "${messagePart}"`)
    return
  }
  throw new Error('should have thrown')
}

async function main() {
  await test('buildVisionForm sends transcript, session, detections and frame', async () => {
    const frame = { size: 3, type: 'image/jpeg' } as Blob
    const form = buildVisionForm('Where is my bottle?', 'session-1', DETECTION, frame)
    equal(form.get('transcript'), 'Where is my bottle?')
    equal(form.get('session_id'), 'session-1')
    equal(JSON.parse(String(form.get('detections'))), DETECTION)
    const file = form.get('file') as File
    equal(file.name, 'frame.jpg')
    equal(file.size, 3)
  })

  await test('buildVisionForm leaves out missing detections and frame', () => {
    const form = buildVisionForm('Hi', 's', null, null)
    equal(form.has('detections'), false)
    equal(form.has('file'), false)
  })

  await test('askVision posts multipart without a manual Content-Type', async () => {
    let seen: RequestInit | undefined
    let url = ''
    const fetcher = (async (input: string, init?: RequestInit) => {
      url = input
      seen = init
      return jsonResponse(200, ANSWER)
    }) as unknown as typeof fetch
    const answer = await askVision(buildVisionForm('q', 's', null, null), { fetcher })
    equal(url, '/api/respond/vision')
    equal(seen?.method, 'POST')
    ok(seen?.body instanceof FormData, 'body is FormData')
    equal(seen?.headers, undefined, 'headers')
    equal(answer.route, 'detections')
  })

  await test('askVision surfaces the server detail on errors', async () => {
    const fetcher = (async () => jsonResponse(503, { detail: 'Ollama is not running.' })) as unknown as typeof fetch
    await expectError(askVision(new FormData(), { fetcher }), 'http', 'Ollama is not running.')
    const list = (async () => jsonResponse(422, { detail: [{ msg: 'Field required' }] })) as unknown as typeof fetch
    await expectError(askVision(new FormData(), { fetcher: list }), 'http', 'Field required')
  })

  await test('askVision rejects a malformed success body', async () => {
    const fetcher = (async () => jsonResponse(200, { response: 'hi' })) as unknown as typeof fetch
    await expectError(askVision(new FormData(), { fetcher }), 'bad-response')
  })

  await test('askVision reports an unreachable backend', async () => {
    const fetcher = (async () => { throw new TypeError('Failed to fetch') }) as unknown as typeof fetch
    await expectError(askVision(new FormData(), { fetcher }), 'network')
  })

  // A fetch that only settles when aborted, like a VLM call that takes minutes.
  const hanging = ((_: string, init?: RequestInit) => new Promise((_resolve, reject) => {
    init?.signal?.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')))
  })) as unknown as typeof fetch

  await test('askVision stops waiting when the user cancels', async () => {
    const user = new AbortController()
    const pending = askVision(new FormData(), { fetcher: hanging, signal: user.signal })
    user.abort()
    await expectError(pending, 'cancelled')
  })

  await test('askVision times out with a plain message', async () => {
    await expectError(askVision(new FormData(), { fetcher: hanging, timeoutMs: 20 }), 'timeout', 'No answer after')
  })

  await test('routeTag says where the answer came from', () => {
    equal(routeTag(ANSWER).text, 'Looked at camera · high confidence')
    equal(routeTag({ ...ANSWER, route: 'vlm', confidence: 'medium' }).text, 'Looked closely · medium confidence')
    equal(routeTag({ ...ANSWER, route: 'vlm', confidence: 'low', vision_warning: 'timed out' }).tone, 'warning')
    equal(routeTag({ ...ANSWER, route: 'text', confidence: null }).text, 'From memory')
    equal(routeTag({ ...ANSWER, route: 'no_vision', confidence: null }).text, 'No camera data')
    // The debug reason is only a tooltip.
    ok(routeTag(ANSWER).title.includes('locate from detections'), 'reason in title')
    ok(!routeTag(ANSWER).text.includes('locate'), 'reason not in text')
  })

  await test('waitingMessage only claims a close look when the vision model runs', () => {
    const vlm = waitingMessage(1000, true, true)
    ok(vlm.closeLook && vlm.canStop, 'vlm: close look + stop')
    ok(vlm.text.startsWith('Looking closely at the camera… 0:01'), vlm.text)
    equal(waitingMessage(1000, true, false), { text: 'Thinking', closeLook: false, canStop: false })
    const slowText = waitingMessage(SLOW_ANSWER_MS + 2000, true, false)
    ok(!slowText.closeLook && !slowText.canStop, 'detector/text route: no close-look claim, no stop yet')
    ok(slowText.text.startsWith('Still working… 0:07'), slowText.text)
    ok(waitingMessage(OFFER_STOP_MS, true, false).canStop, 'stop offered after a while')
    const unknown = waitingMessage(SLOW_ANSWER_MS, true, null)
    ok(unknown.canStop && unknown.text.includes('can take a few minutes'), 'unknown route with frame hints')
    ok(!waitingMessage(SLOW_ANSWER_MS, false, null).text.includes('camera'), 'no frame: no camera hint')
  })

  await test('previewUsesVisionModel posts JSON and never throws', async () => {
    let sent: RequestInit | undefined
    const fetcher = (async (_: string, init?: RequestInit) => {
      sent = init
      return jsonResponse(200, { route: 'vlm', uses_vision_model: true, reason: 'x' })
    }) as unknown as typeof fetch
    equal(await previewUsesVisionModel('What color is it?', DETECTION, true, fetcher), true)
    equal(JSON.parse(String(sent?.body)), { transcript: 'What color is it?', has_image: true, detections: DETECTION })
    const failing = (async () => jsonResponse(500, {})) as unknown as typeof fetch
    equal(await previewUsesVisionModel('q', null, false, failing), null)
    const offline = (async () => { throw new TypeError('Failed to fetch') }) as unknown as typeof fetch
    equal(await previewUsesVisionModel('q', null, false, offline), null)
  })

  await test('formatElapsed', () => {
    equal(formatElapsed(0), '0:00')
    equal(formatElapsed(7_900), '0:07')
    equal(formatElapsed(151_000), '2:31')
    equal(formatElapsed(-5), '0:00')
  })

  console.log(`\n${passed} passed, ${failures} failed`)
  if (failures) throw new Error(`${failures} frontend test(s) failed`)
}

void main().catch((error) => {
  console.error(error instanceof Error ? error.message : error)
  ;(globalThis as unknown as { process: { exitCode: number } }).process.exitCode = 1
})
