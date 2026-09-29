import { useEffect, useRef, useState } from 'react'
import './App.css'

function App() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const [cameraStatus, setCameraStatus] = useState<'requesting' | 'active' | 'error'>('requesting')
  const [cameraMessage, setCameraMessage] = useState('Waiting for camera permission')

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
      if (videoRef.current) videoRef.current.srcObject = null
    }
  }, [])

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
            <video ref={videoRef} autoPlay muted playsInline aria-label="Live camera preview" />
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

        <footer className="privacy-note"><span>01</span> THE CAMERA FEED IS NOT UPLOADED OR SAVED.</footer>
      </section>
    </main>
  )
}

export default App
