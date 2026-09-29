import { useEffect, useRef, useState } from 'react'
import './App.css'

function App() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const microphoneStreamRef = useRef<MediaStream | null>(null)
  const [cameraStatus, setCameraStatus] = useState<'requesting' | 'active' | 'error'>('requesting')
  const [cameraMessage, setCameraMessage] = useState('Waiting for camera permission')
  const [isVideoReady, setIsVideoReady] = useState(false)
  const [isMicrophoneActive, setIsMicrophoneActive] = useState(false)
  const [microphoneMessage, setMicrophoneMessage] = useState('Microphone is off')
  const [capturedFrame, setCapturedFrame] = useState('')
  const [capturedResolution, setCapturedResolution] = useState('')

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
      cameraStream?.getTracks().forEach((track) => track.stop())
      microphoneStreamRef.current?.getTracks().forEach((track) => track.stop())
      if (videoRef.current) videoRef.current.srcObject = null
    }
  }, [])

  async function toggleMicrophone() {
    if (isMicrophoneActive) {
      microphoneStreamRef.current?.getTracks().forEach((track) => track.stop())
      microphoneStreamRef.current = null
      setIsMicrophoneActive(false)
      setMicrophoneMessage('Microphone is off')
      return
    }

    const video = videoRef.current
    const canvas = canvasRef.current
    const context = canvas?.getContext('2d')
    if (!video || !canvas || !context || !video.videoWidth || !video.videoHeight) {
      setMicrophoneMessage('Camera is not ready to capture a frame yet')
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
      setIsMicrophoneActive(true)
      setMicrophoneMessage('Microphone active · frame captured')
    } catch (error) {
      if (error instanceof DOMException && error.name === 'NotAllowedError') {
        setMicrophoneMessage('Microphone permission blocked. Allow access in your browser settings.')
      } else {
        setMicrophoneMessage('Could not start the microphone. Check your audio input and try again.')
      }
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

        <button
          className={`microphone-button${isMicrophoneActive ? ' is-listening' : ''}`}
          type="button"
          onClick={() => void toggleMicrophone()}
          disabled={cameraStatus !== 'active' || !isVideoReady}
          aria-label={isMicrophoneActive ? 'Stop microphone' : 'Start microphone'}
          aria-pressed={isMicrophoneActive}
          title={`${isMicrophoneActive ? 'Stop' : 'Start'} microphone${capturedResolution ? ` · captured ${capturedResolution} frame` : ''}`}
        >
          {isMicrophoneActive ? (
            <svg viewBox="0 0 24 24" aria-hidden="true"><rect x="7" y="7" width="10" height="10" rx="1.5" /></svg>
          ) : (
            <svg viewBox="0 0 24 24" aria-hidden="true"><rect x="9" y="3" width="6" height="12" rx="3" /><path d="M5.5 11.5a6.5 6.5 0 0 0 13 0M12 18v3m-4 0h8" /></svg>
          )}
          <span className="sr-only">{isMicrophoneActive ? 'Microphone on' : 'Microphone off'}{microphoneMessage}</span>
        </button>

        <footer className="wordmark">JARVIS <span>2.0</span></footer>
      </section>
      <canvas ref={canvasRef} className="capture-canvas" aria-hidden="true" />
    </main>
  )
}

export default App
