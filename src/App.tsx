import { useEffect, useRef, useState } from 'react'
import './App.css'

type SpeechRecognitionResultLike = ArrayLike<{ transcript: string }> & { isFinal?: boolean }

type SpeechRecognitionEventLike = Event & {
  resultIndex: number
  results: ArrayLike<SpeechRecognitionResultLike>
}

type SpeechRecognitionLike = {
  continuous: boolean
  interimResults: boolean
  lang: string
  onresult: ((event: SpeechRecognitionEventLike) => void) | null
  onerror: (() => void) | null
  start: () => void
  stop: () => void
}

type SpeechRecognitionWindow = Window & {
  SpeechRecognition?: new () => SpeechRecognitionLike
  webkitSpeechRecognition?: new () => SpeechRecognitionLike
}

function App() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const microphoneStreamRef = useRef<MediaStream | null>(null)
  const recorderRef = useRef<MediaRecorder | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const speechRecognitionRef = useRef<SpeechRecognitionLike | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  const speechFrameRef = useRef<number | null>(null)
  const silenceTimerRef = useRef<number | null>(null)
  const detectedSpeechRef = useRef(false)
  const isRecordingRef = useRef(false)
  const finalTranscriptRef = useRef('')
  const [cameraStatus, setCameraStatus] = useState<'requesting' | 'active' | 'error'>('requesting')
  const [cameraMessage, setCameraMessage] = useState('Waiting for camera permission')
  const [isVideoReady, setIsVideoReady] = useState(false)
  const [speechState, setSpeechState] = useState<'idle' | 'listening' | 'transcribing'>('idle')
  const [speechMessage, setSpeechMessage] = useState('')
  const [transcript, setTranscript] = useState('')
  const [capturedFrame, setCapturedFrame] = useState('')
  const [capturedResolution, setCapturedResolution] = useState('')

  const finishUtterance = () => {
    if (!isRecordingRef.current) return
    isRecordingRef.current = false
    if (silenceTimerRef.current !== null) window.clearTimeout(silenceTimerRef.current)
    silenceTimerRef.current = null
    if (speechFrameRef.current !== null) cancelAnimationFrame(speechFrameRef.current)
    speechFrameRef.current = null
    speechRecognitionRef.current?.stop()
    setSpeechState('transcribing')
    setSpeechMessage('Transcribing')
    recorderRef.current?.stop()
    microphoneStreamRef.current?.getTracks().forEach((track) => track.stop())
    microphoneStreamRef.current = null
    void audioContextRef.current?.close()
    audioContextRef.current = null
  }

  async function transcribeBatch(audio: Blob) {
    const speechFallback = finalTranscriptRef.current.trim()
    if (!audio.size) {
      setSpeechMessage(speechFallback || 'No audio captured. Tap to try again.')
      setSpeechState('idle')
      return
    }

    const formData = new FormData()
    formData.append('file', audio, audio.type.includes('mp4') ? 'utterance.mp4' : 'utterance.webm')

    try {
      const response = await fetch('/api/transcribe', { method: 'POST', body: formData })
      const result = await response.json() as { transcript?: string; detail?: string }
      if (!response.ok) throw new Error(result.detail || 'Speech service unavailable')
      const words = result.transcript?.trim() || speechFallback
      setTranscript(words)
      setSpeechMessage(words ? '' : 'No speech detected. Tap to try again.')
    } catch (error) {
      setTranscript(speechFallback)
      setSpeechMessage(speechFallback ? '' : error instanceof Error ? error.message : 'Speech service unavailable.')
    }
    setSpeechState('idle')
  }

  useEffect(() => {
    let isCurrent = true
    let cameraStream: MediaStream | undefined

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

        if (videoRef.current) videoRef.current.srcObject = cameraStream
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
      speechRecognitionRef.current?.stop()
      recorderRef.current?.stream.getTracks().forEach((track) => track.stop())
      cameraStream?.getTracks().forEach((track) => track.stop())
      microphoneStreamRef.current?.getTracks().forEach((track) => track.stop())
      void audioContextRef.current?.close()
      if (videoRef.current) videoRef.current.srcObject = null
    }
  }, [])

  async function toggleMicrophone() {
    if (speechState === 'listening') {
      finishUtterance()
      return
    }
    if (speechState === 'transcribing') return

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
    setCapturedFrame(canvas.toDataURL('image/jpeg', 0.9))
    setCapturedResolution(`${canvas.width} × ${canvas.height}`)

    try {
      microphoneStreamRef.current = await navigator.mediaDevices.getUserMedia({ audio: true, video: false })
      if (!('MediaRecorder' in window)) throw new Error('Audio recording is not supported in this browser')

      const recorder = new MediaRecorder(microphoneStreamRef.current)
      recorderRef.current = recorder
      audioChunksRef.current = []
      finalTranscriptRef.current = ''
      setTranscript('')
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

      const recognitionWindow = window as SpeechRecognitionWindow
      const SpeechApi = recognitionWindow.SpeechRecognition || recognitionWindow.webkitSpeechRecognition
      if (SpeechApi) {
        const recognition = new SpeechApi()
        speechRecognitionRef.current = recognition
        recognition.continuous = true
        recognition.interimResults = true
        recognition.lang = 'en-US'
        recognition.onerror = () => undefined
        recognition.onresult = (event) => {
          let interim = ''
          for (let index = event.resultIndex; index < event.results.length; index += 1) {
            const phrase = event.results[index][0]?.transcript ?? ''
            if (event.results[index].isFinal) finalTranscriptRef.current += phrase
            else interim += phrase
          }
          setTranscript(`${finalTranscriptRef.current}${interim}`.trim())
        }
        recognition.start()
      }
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
        {cameraStatus !== 'active' && (
          <div className="camera-placeholder" role="status">
            <strong>{cameraStatus === 'requesting' ? 'Camera access' : 'Camera unavailable'}</strong>
            <p>{cameraMessage}</p>
          </div>
        )}

        <span className="camera-led" aria-label={cameraStatus === 'active' ? 'Camera active' : 'Camera inactive'}>
          <i className={cameraStatus === 'active' ? 'is-active' : ''} />
        </span>

        {speechMessage && (
          <p className="voice-feedback" aria-live="polite">{transcript || speechMessage}</p>
        )}
        {!speechMessage && transcript && (
          <p className="voice-feedback" aria-live="polite">{transcript}</p>
        )}

        <button
          className={`microphone-button${speechState === 'listening' ? ' is-listening' : ''}`}
          type="button"
          onClick={() => void toggleMicrophone()}
          disabled={cameraStatus !== 'active' || !isVideoReady || speechState === 'transcribing'}
          aria-label={speechState === 'listening' ? 'Stop listening' : speechState === 'transcribing' ? 'Transcribing speech' : 'Start listening'}
          aria-description={capturedFrame ? 'A still frame has been captured locally.' : undefined}
          aria-pressed={speechState === 'listening'}
          title={`${speechState === 'listening' ? 'Stop listening' : 'Start listening'}${capturedResolution ? ` · captured ${capturedResolution} frame` : ''}`}
        >
          {speechState === 'listening' ? (
            <svg viewBox="0 0 24 24" aria-hidden="true"><rect x="7" y="7" width="10" height="10" rx="1.5" /></svg>
          ) : (
            <svg viewBox="0 0 24 24" aria-hidden="true"><rect x="9" y="3" width="6" height="12" rx="3" /><path d="M5.5 11.5a6.5 6.5 0 0 0 13 0M12 18v3m-4 0h8" /></svg>
          )}
          <span className="sr-only">{speechState === 'listening' ? 'Listening' : speechState === 'transcribing' ? 'Transcribing' : 'Microphone idle'}</span>
        </button>

        <footer className="wordmark">JARVIS <span>2.0</span></footer>
      </section>
      <canvas ref={canvasRef} className="capture-canvas" aria-hidden="true" />
    </main>
  )
}

export default App
