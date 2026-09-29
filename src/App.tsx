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
      <header className="topbar">
        <a className="wordmark" href="#home">jarvis<span>2.0</span></a>
        <p className="topbar-note">MULTIMODAL ASSISTANT <span>·</span> PROTOTYPE</p>
      </header>

      <section className="camera-workspace" id="home" aria-labelledby="page-title">
        <div className="workspace-heading">
          <div>
            <p className="eyebrow">PHASE 01 <span>/</span> CAMERA</p>
            <h1 id="page-title">A window to<br /><em>what’s around you.</em></h1>
          </div>
          <p className="intro-copy">Your camera stays on this device. A frame is only captured when you ask.</p>
        </div>

        <section className="camera-panel" aria-label="Live camera preview">
          <div className="camera-view">
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
                <span className="camera-symbol">◉</span>
                <strong>{cameraStatus === 'requesting' ? 'Waiting for camera' : 'Camera unavailable'}</strong>
                <p>{cameraMessage}</p>
              </div>
            )}
            <div className="view-corner view-corner-tl" />
            <div className="view-corner view-corner-tr" />
            <div className="view-corner view-corner-bl" />
            <div className="view-corner view-corner-br" />
          </div>
          <div className="camera-caption">
            <span className="camera-indicator">
              <i className={cameraStatus === 'active' ? 'is-active' : ''} />
              {cameraStatus === 'active' ? 'CAMERA ACTIVE' : cameraStatus === 'error' ? 'CAMERA OFFLINE' : 'REQUESTING ACCESS'}
            </span>
            <span className="camera-caption-note">LOCAL PREVIEW <b>·</b> NO RECORDING</span>
          </div>
        </section>

        <section className="capture-controls" aria-label="Capture controls">
          <div className="microphone-control">
            <button
              className={`microphone-button${isMicrophoneActive ? ' is-listening' : ''}`}
              type="button"
              onClick={() => void toggleMicrophone()}
              disabled={cameraStatus !== 'active' || !isVideoReady}
              aria-pressed={isMicrophoneActive}
            >
              <span className="microphone-button-icon" aria-hidden="true">{isMicrophoneActive ? '■' : '⌁'}</span>
              {isMicrophoneActive ? 'Stop listening' : 'Start listening'}
            </button>
            <span className={`microphone-status${isMicrophoneActive ? ' is-listening' : ''}`} role="status">
              <i />{microphoneMessage}
            </span>
          </div>
          {capturedFrame && (
            <div className="captured-frame">
              <img src={capturedFrame} alt="Most recently captured camera frame" />
              <span>CAPTURED FRAME <b>{capturedResolution}</b></span>
            </div>
          )}
        </section>

        <footer className="privacy-note"><span>01</span> THE CAMERA FEED IS NOT UPLOADED OR SAVED.</footer>
      </section>
      <canvas ref={canvasRef} className="capture-canvas" aria-hidden="true" />
    </main>
  )
}

export default App
