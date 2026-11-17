import { useEffect, useRef, useState } from 'react'
import './App.css'
import DetectionOverlay from './DetectionOverlay'
import SessionMenu from './SessionMenu'
import {
  canvasToJpeg,
  detectFrame,
  DetectionRequestError,
  type DetectionResult,
  type VisionState,
} from './detection'
import { fetchSessionTurns, SESSION_STORAGE_KEY } from './sessionsApi'

// The captured still, kept in memory only (never stored or saved). Phase 4 can
// send this same JPEG to a vision-language model instead of re-capturing.
type CapturedFrame = { frameId: number; blob: Blob; width: number; height: number; capturedAt: number }

function App() {
  const [sessionId, setSessionId] = useState(() => {
    const existingSessionId = window.localStorage.getItem(SESSION_STORAGE_KEY)
    if (existingSessionId) return existingSessionId

    const newSessionId = window.crypto.randomUUID()
    window.localStorage.setItem(SESSION_STORAGE_KEY, newSessionId)
    return newSessionId
  })
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const microphoneStreamRef = useRef<MediaStream | null>(null)
  const recorderRef = useRef<MediaRecorder | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  const speechFrameRef = useRef<number | null>(null)
  const silenceTimerRef = useRef<number | null>(null)
  const detectedSpeechRef = useRef(false)
  const isRecordingRef = useRef(false)
  const [cameraStatus, setCameraStatus] = useState<'requesting' | 'active' | 'error'>('requesting')
  const [cameraMessage, setCameraMessage] = useState('Waiting for camera permission')
  const [isVideoReady, setIsVideoReady] = useState(false)
  const [speechState, setSpeechState] = useState<'idle' | 'listening' | 'transcribing' | 'thinking'>('idle')
  const [speechMessage, setSpeechMessage] = useState('')
  const [transcript, setTranscript] = useState('')
  const [response, setResponse] = useState('')
  const [capturedResolution, setCapturedResolution] = useState('')
  const [memoryWarning, setMemoryWarning] = useState('')
  const sessionSwitchRef = useRef(0)
  // Object detection for the frame captured when the mic is pressed.
  const [vision, setVision] = useState<VisionState>({ status: 'idle' })
  const frameIdRef = useRef(0)
  const capturedFrameRef = useRef<CapturedFrame | null>(null)
  // Never rejects: resolves to the frame's detections, or null on failure or
  // when a newer frame superseded it. Phase 4: await this before /api/respond.
  const frameDetectionRef = useRef<{ frameId: number; promise: Promise<DetectionResult | null> } | null>(null)
  // The backend runs one detection at a time, so requests are chained.
  const detectChainRef = useRef<Promise<unknown>>(Promise.resolve())

  function startFrameDetection(canvas: HTMLCanvasElement) {
    const frameId = ++frameIdRef.current
    const startedAt = performance.now()
    const capturedAt = Date.now()
    const { width, height } = canvas
    setVision({ status: 'detecting', frameId })

    // toBlob copies the canvas now and encodes off the main thread, so the
    // microphone starts without waiting for the JPEG or the upload.
    const outcome = canvasToJpeg(canvas).then((blob) => {
      capturedFrameRef.current = { frameId, blob, width, height, capturedAt }
      const run = detectChainRef.current.then(() => (
        // Skip frames that a newer capture replaced while they were queued.
        frameId === frameIdRef.current ? detectFrame(blob) : null
      ))
      detectChainRef.current = run.catch(() => undefined)
      return run
    })

    const promise = outcome.then((result) => {
      if (result && frameId === frameIdRef.current) {
        setVision({ status: 'ready', frameId, capturedAt, result, elapsedMs: performance.now() - startedAt })
      }
      return result
    }, (error: unknown) => {
      const failure = error instanceof DetectionRequestError
        ? error
        : new DetectionRequestError('Detection failed', error instanceof Error ? error.message : String(error))
      console.warn('Object detection failed:', failure.detail)
      if (frameId === frameIdRef.current) {
        setVision({ status: 'error', frameId, message: failure.message, detail: failure.detail })
      }
      return null
    })
    frameDetectionRef.current = { frameId, promise }
  }

  async function switchSession(nextSessionId: string) {
    const switchId = ++sessionSwitchRef.current
    window.localStorage.setItem(SESSION_STORAGE_KEY, nextSessionId)
    setSessionId(nextSessionId)
    setTranscript('')
    setResponse('')
    setMemoryWarning('')
    setSpeechMessage('')
    try {
      const turns = await fetchSessionTurns(nextSessionId)
      if (switchId !== sessionSwitchRef.current) return
      const lastTurn = turns[turns.length - 1]
      if (lastTurn) {
        setTranscript(lastTurn.transcript)
        setResponse(lastTurn.response)
      } else {
        setSpeechMessage('New conversation')
      }
    } catch (error) {
      if (switchId !== sessionSwitchRef.current) return
      setSpeechMessage(error instanceof Error ? error.message : 'Could not load the conversation.')
    }
  }

  const finishUtterance = () => {
    if (!isRecordingRef.current) return
    isRecordingRef.current = false
    if (silenceTimerRef.current !== null) window.clearTimeout(silenceTimerRef.current)
    silenceTimerRef.current = null
    if (speechFrameRef.current !== null) cancelAnimationFrame(speechFrameRef.current)
    speechFrameRef.current = null
    setSpeechState('transcribing')
    setSpeechMessage('Transcribing')
    recorderRef.current?.stop()
    microphoneStreamRef.current?.getTracks().forEach((track) => track.stop())
    microphoneStreamRef.current = null
    void audioContextRef.current?.close()
    audioContextRef.current = null
  }

  async function transcribeBatch(audio: Blob) {
    if (!audio.size) {
      setSpeechMessage('No audio captured. Tap to try again.')
      setSpeechState('idle')
      return
    }

    const formData = new FormData()
    formData.append('file', audio, audio.type.includes('mp4') ? 'utterance.mp4' : 'utterance.webm')

    try {
      const response = await fetch('/api/transcribe', { method: 'POST', body: formData })
      const result = await response.json() as { transcript?: string; detail?: string }
      if (!response.ok) throw new Error(result.detail || 'Local transcription failed.')
      const words = result.transcript?.trim()
      if (!words) {
        setTranscript('')
        setSpeechMessage('No speech detected. Tap to try again.')
        return
      }

      setTranscript(words)
      setResponse('')
      setSpeechState('thinking')
      setSpeechMessage('Thinking')

      const answerResponse = await fetch('/api/respond', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ transcript: words, session_id: sessionId }),
      })
      const answer = await answerResponse.json() as { response?: string; detail?: string; memory_warning?: string | null }
      if (!answerResponse.ok) throw new Error(answer.detail || 'Local AI response failed.')
      setResponse(answer.response?.trim() || '')
      setMemoryWarning(answer.memory_warning || '')
      setSpeechMessage('')
    } catch (error) {
      setSpeechMessage(error instanceof Error ? error.message : 'Local speech service unavailable.')
    } finally {
      setSpeechState('idle')
    }
  }

  useEffect(() => {
    let isCurrent = true
    let cameraStream: MediaStream | undefined
    const videoElement = videoRef.current

    async function startCamera() {
      if (!navigator.mediaDevices?.getUserMedia) {
        setCameraStatus('error')
        setCameraMessage('Camera access requires localhost or a secure HTTPS connection.')
        return
      }

      try {
        cameraStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false })
        if (!isCurrent) {
          cameraStream.getTracks().forEach((track) => track.stop())
          return
        }

        if (videoElement) videoElement.srcObject = cameraStream
        setCameraStatus('active')
        setCameraMessage('Camera active')
      } catch (error) {
        setCameraStatus('error')
        if (error instanceof DOMException && error.name === 'NotFoundError') {
          setCameraMessage('No camera was found. Connect a camera and refresh the page.')
        } else if (error instanceof DOMException && error.name === 'NotAllowedError') {
          setCameraMessage('Camera permission was blocked. Allow access in your browser, then refresh.')
        } else {
          setCameraMessage('Could not start the camera. Check browser permissions and try again.')
        }
      }
    }

    void startCamera()

    return () => {
      isCurrent = false
      if (silenceTimerRef.current !== null) window.clearTimeout(silenceTimerRef.current)
      if (speechFrameRef.current !== null) cancelAnimationFrame(speechFrameRef.current)
      recorderRef.current?.stream.getTracks().forEach((track) => track.stop())
      cameraStream?.getTracks().forEach((track) => track.stop())
      microphoneStreamRef.current?.getTracks().forEach((track) => track.stop())
      void audioContextRef.current?.close()
      if (videoElement) videoElement.srcObject = null
    }
  }, [])

  async function toggleMicrophone() {
    if (speechState === 'listening') {
      finishUtterance()
      return
    }
    if (speechState === 'transcribing' || speechState === 'thinking') return

    const video = videoRef.current
    const canvas = canvasRef.current
    const context = canvas?.getContext('2d')
    if (!video || !canvas || !context || !video.videoWidth || !video.videoHeight) {
      setSpeechMessage('Camera is not ready to capture a frame yet')
      return
    }

    const scale = Math.min(1, 1024 / Math.max(video.videoWidth, video.videoHeight))
    canvas.width = Math.round(video.videoWidth * scale)
    canvas.height = Math.round(video.videoHeight * scale)
    context.drawImage(video, 0, 0, canvas.width, canvas.height)
    setCapturedResolution(`${canvas.width} × ${canvas.height}`)
    // Runs in parallel with listening; a detection failure only shows a note.
    startFrameDetection(canvas)

    try {
      microphoneStreamRef.current = await navigator.mediaDevices.getUserMedia({ audio: true, video: false })
      if (!('MediaRecorder' in window)) throw new Error('Audio recording is not supported in this browser')

      const recorder = new MediaRecorder(microphoneStreamRef.current)
      recorderRef.current = recorder
      audioChunksRef.current = []
      setTranscript('')
      setResponse('')
      setMemoryWarning('')
      setSpeechMessage('Listening')
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) audioChunksRef.current.push(event.data)
      }
      recorder.onstop = () => {
        const audio = new Blob(audioChunksRef.current, { type: recorder.mimeType || 'audio/webm' })
        void transcribeBatch(audio)
      }
      recorder.start()
      isRecordingRef.current = true

      const audioContext = new AudioContext()
      audioContextRef.current = audioContext
      const source = audioContext.createMediaStreamSource(microphoneStreamRef.current)
      const analyser = audioContext.createAnalyser()
      analyser.fftSize = 512
      source.connect(analyser)
      const samples = new Uint8Array(analyser.fftSize)
      detectedSpeechRef.current = false
      setSpeechState('listening')
      setSpeechMessage('Listening')

      const detectVoiceActivity = () => {
        analyser.getByteTimeDomainData(samples)
        const rms = Math.sqrt(samples.reduce((total, sample) => total + (sample - 128) ** 2, 0) / samples.length)
        if (rms > 8) {
          detectedSpeechRef.current = true
          setSpeechMessage('Listening')
          if (silenceTimerRef.current !== null) window.clearTimeout(silenceTimerRef.current)
          silenceTimerRef.current = null
        } else if (detectedSpeechRef.current && silenceTimerRef.current === null) {
          silenceTimerRef.current = window.setTimeout(finishUtterance, 1200)
        }
        if (recorder.state === 'recording') speechFrameRef.current = requestAnimationFrame(detectVoiceActivity)
      }
      speechFrameRef.current = requestAnimationFrame(detectVoiceActivity)
    } catch (error) {
      microphoneStreamRef.current?.getTracks().forEach((track) => track.stop())
      microphoneStreamRef.current = null
      setSpeechState('idle')
      setSpeechMessage(error instanceof DOMException && error.name === 'NotAllowedError'
        ? 'Microphone permission was blocked'
        : error instanceof Error ? error.message : 'Could not start the microphone')
    }
  }

  return (
    <main className="app-shell">
      <section className="camera-scene" aria-label="Jarvis camera view">
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          onLoadedMetadata={() => setIsVideoReady(true)}
          aria-label="Live camera preview"
        />
        {cameraStatus === 'active' && <DetectionOverlay vision={vision} mirrored />}
        {cameraStatus !== 'active' && (
          <div className="camera-placeholder" role="status">
            <strong>{cameraStatus === 'requesting' ? 'Camera access' : 'Camera unavailable'}</strong>
            <p>{cameraMessage}</p>
          </div>
        )}

        <span className="camera-led" aria-label={cameraStatus === 'active' ? 'Camera active' : 'Camera inactive'}>
          <i className={cameraStatus === 'active' ? 'is-active' : ''} />
        </span>

        <SessionMenu
          activeSessionId={sessionId}
          disabled={speechState !== 'idle'}
          onSelectSession={(nextSessionId) => void switchSession(nextSessionId)}
        />

        {(transcript || response || speechMessage || memoryWarning) && (
          <div className="voice-feedback" aria-live="polite" role="status">
            {transcript && (
              <p className="voice-turn">
                <span>You</span>
                {transcript}
              </p>
            )}
            {response && (
              <p className="voice-turn voice-turn-answer">
                <span>Jarvis</span>
                {response}
              </p>
            )}
            {memoryWarning && <p className="voice-status voice-warning">{memoryWarning}</p>}
            {speechMessage && <p className="voice-status">{speechMessage}</p>}
          </div>
        )}

        <button
          className={`microphone-button${speechState === 'listening' ? ' is-listening' : ''}`}
          type="button"
          onClick={() => void toggleMicrophone()}
          disabled={cameraStatus !== 'active' || !isVideoReady || speechState === 'transcribing'}
          aria-label={speechState === 'listening' ? 'Stop listening' : speechState === 'transcribing' ? 'Transcribing speech' : speechState === 'thinking' ? 'Generating response' : 'Start listening'}
          aria-description={capturedResolution ? 'A still frame was captured and checked for objects locally.' : undefined}
          aria-pressed={speechState === 'listening'}
          title={`${speechState === 'listening' ? 'Stop listening' : 'Start listening'}${capturedResolution ? ` · captured ${capturedResolution} frame` : ''}`}
        >
          {speechState === 'listening' ? (
            <svg viewBox="0 0 24 24" aria-hidden="true"><rect x="7" y="7" width="10" height="10" rx="1.5" /></svg>
          ) : (
            <svg viewBox="0 0 24 24" aria-hidden="true"><rect x="9" y="3" width="6" height="12" rx="3" /><path d="M5.5 11.5a6.5 6.5 0 0 0 13 0M12 18v3m-4 0h8" /></svg>
          )}
          <span className="sr-only">{speechState === 'listening' ? 'Listening' : speechState === 'transcribing' ? 'Transcribing' : speechState === 'thinking' ? 'Thinking' : 'Microphone idle'}</span>
        </button>

        <footer className="wordmark">JARVIS <span>2.0</span></footer>
      </section>
      <canvas ref={canvasRef} className="capture-canvas" aria-hidden="true" />
    </main>
  )
}

export default App
