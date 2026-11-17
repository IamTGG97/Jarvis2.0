// Dependency-free unit tests for the detection helpers (no Vitest/Jest needed).
// Run: npm run test:frontend  (compiles with tsconfig.test.json, runs with Node).
import {
  coverLayout,
  describeDetectError,
  detectFrame,
  projectBox,
  summarizeLabels,
  type Detection,
  type DetectionResult,
} from './detection.js'

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

function near(actual: number, expected: number, label = '', tolerance = 1e-6) {
  if (Math.abs(actual - expected) > tolerance) throw new Error(`${label} expected ${expected}, got ${actual}`)
}

function det(label: string, confidence: number, x = 0, y = 0, width = 0.1, height = 0.1): Detection {
  return { label, confidence, box: { x, y, width, height } }
}

async function main() {
  await test('cover layout crops the sides of a 4:3 frame in a tall phone view', () => {
    // 640x480 frame in a 390x844 portrait viewport: height decides the scale.
    const layout = coverLayout(640, 480, 390, 844)
    near(layout.scale, 844 / 480, 'scale')
    near(layout.height, 844, 'height')
    near(layout.width, 640 * (844 / 480), 'width')
    near(layout.offsetX, (390 - 640 * (844 / 480)) / 2, 'offsetX')
    near(layout.offsetY, 0, 'offsetY')
  })

  await test('cover layout crops top and bottom of a 4:3 frame in a wide desktop view', () => {
    const layout = coverLayout(640, 480, 1600, 900)
    near(layout.scale, 2.5, 'scale')
    near(layout.offsetX, 0, 'offsetX')
    near(layout.offsetY, (900 - 1200) / 2, 'offsetY')
  })

  await test('same aspect ratio is a plain multiply', () => {
    const view = projectBox({ x: 0.25, y: 0.5, width: 0.5, height: 0.25 }, 640, 480, 1280, 960, false)
    equal(view, { left: 320, top: 480, width: 640, height: 240 })
  })

  await test('mirroring flips x around the view center and keeps size', () => {
    const view = projectBox({ x: 0.1, y: 0.2, width: 0.3, height: 0.4 }, 640, 480, 1280, 960, true)
    // Unmirrored left = 128, width = 384 -> mirrored left = 1280 - 128 - 384.
    equal(view, { left: 768, top: 192, width: 384, height: 384 })
  })

  await test('center of the frame stays at the center of the view, mirrored or not', () => {
    const box = { x: 0.45, y: 0.45, width: 0.1, height: 0.1 }
    for (const mirrored of [false, true]) {
      const view = projectBox(box, 640, 480, 390, 844, mirrored)
      if (!view) throw new Error('box vanished')
      near(view.left + view.width / 2, 195, `center x (mirrored=${mirrored})`, 1e-6)
      near(view.top + view.height / 2, 422, `center y (mirrored=${mirrored})`, 1e-6)
    }
  })

  await test('boxes in the cropped margin are dropped and partial ones clipped', () => {
    // In a 390x844 view the 640x480 frame is 1125.33 px wide, offset -367.67:
    // the left 32.7% of the frame is cropped away.
    equal(projectBox({ x: 0, y: 0.1, width: 0.2, height: 0.2 }, 640, 480, 390, 844, false), null)
    const partial = projectBox({ x: 0.3, y: 0.1, width: 0.1, height: 0.2 }, 640, 480, 390, 844, false)
    if (!partial) throw new Error('partial box vanished')
    near(partial.left, 0, 'clipped left')
    // Visible width = right edge of the box (offset + 0.4 * frame width).
    near(partial.width, -367.6666667 + 0.4 * 1125.3333333, 'clipped width', 1e-3)
    // Mirrored, the same frame-left box is cropped off the right edge instead.
    equal(projectBox({ x: 0, y: 0.1, width: 0.2, height: 0.2 }, 640, 480, 390, 844, true), null)
  })

  await test('invalid sizes give no box instead of NaN', () => {
    equal(projectBox({ x: 0, y: 0, width: 1, height: 1 }, 0, 480, 390, 844, false), null)
    equal(projectBox({ x: 0, y: 0, width: 1, height: 1 }, 640, 480, 0, 0, false), null)
  })

  await test('summary groups labels in confidence order with counts', () => {
    const detections = [det('person', 0.9), det('laptop', 0.88), det('laptop', 0.7), det('cup', 0.5), det('person', 0.4)]
    equal(summarizeLabels(detections), ['person ×2', 'laptop ×2', 'cup'])
    equal(summarizeLabels([]), [])
  })

  await test('summary caps the list', () => {
    const labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    equal(summarizeLabels(labels.map((label, i) => det(label, 1 - i / 10)), 3), ['a', 'b', 'c', '+5 more'])
  })

  await test('error mapping keeps the server detail and gives a short message', () => {
    const missing = describeDetectError(503, { detail: 'weights missing. Run: backend\\download_vision_model.py' })
    equal(missing.message, 'Detector model missing')
    equal(missing.status, 503)
    if (!missing.detail.includes('download_vision_model')) throw new Error('detail lost')
    equal(describeDetectError(413, { detail: 'too big' }).message, 'Frame too large to analyze')
    equal(describeDetectError(502, null).detail, 'HTTP 502')
    equal(describeDetectError(422, { detail: [{ msg: 'Field required' }] }).detail, 'Field required')
  })

  // detectFrame with a fake fetch. Node 17 lacks FormData, so stub it if needed.
  const globals = globalThis as unknown as { FormData?: unknown }
  if (typeof globals.FormData === 'undefined') {
    globals.FormData = class {
      fields: unknown[] = []
      append(...args: unknown[]) { this.fields.push(args) }
    }
  }
  const sample: DetectionResult = {
    detections: [det('person', 0.9, 0.1, 0.1, 0.5, 0.8)],
    person_present: true,
    image_width: 640,
    image_height: 480,
    inference_ms: 120.5,
  }
  const fakeFetch = (status: number, body: unknown, calls: unknown[] = []) => (async (...args: unknown[]) => {
    calls.push(args)
    return { ok: status >= 200 && status < 300, status, json: async () => body }
  }) as unknown as typeof fetch

  await test('detectFrame posts multipart to /api/detect and returns the result', async () => {
    const calls: unknown[][] = []
    const result = await detectFrame({} as Blob, fakeFetch(200, sample, calls))
    equal(result, sample)
    const [url, init] = calls[0] as [string, RequestInit]
    equal(url, '/api/detect')
    equal(init.method, 'POST')
    if (init.headers) throw new Error('must not set Content-Type; fetch adds the boundary')
  })

  await test('detectFrame turns HTTP errors into DetectionRequestError', async () => {
    try {
      await detectFrame({} as Blob, fakeFetch(503, { detail: 'Run download_vision_model.py' }))
      throw new Error('should have thrown')
    } catch (error) {
      equal((error as Error).name, 'DetectionRequestError')
      equal((error as Error).message, 'Detector model missing')
    }
  })

  await test('detectFrame rejects a malformed success body', async () => {
    try {
      await detectFrame({} as Blob, fakeFetch(200, { detections: 'nope' }))
      throw new Error('should have thrown')
    } catch (error) {
      equal((error as Error).message, 'Detection failed')
    }
  })

  await test('detectFrame reports an unreachable backend', async () => {
    const failing = (async () => { throw new TypeError('Failed to fetch') }) as unknown as typeof fetch
    try {
      await detectFrame({} as Blob, failing)
      throw new Error('should have thrown')
    } catch (error) {
      equal((error as Error).message, 'Detector not reachable')
    }
  })

  console.log(`\n${passed} passed, ${failures} failed`)
  if (failures) throw new Error(`${failures} frontend test(s) failed`)
}

void main().catch((error) => {
  console.error(error instanceof Error ? error.message : error)
  // Signal failure to npm without needing Node's type definitions.
  ;(globalThis as unknown as { process: { exitCode: number } }).process.exitCode = 1
})
